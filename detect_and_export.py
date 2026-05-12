#!/usr/bin/env python3
"""
detect_and_export.py - Detect sensitive text in video frames and export detections.

Extracts frames from an MP4, runs EasyOCR on every Nth frame, matches against
a list of sensitive strings (plus IPv4 regex), groups detections into tracks,
and writes a detections.json file.

Mac-optimized: multiprocessing frame batching, lazy sequential frame reads,
frame downscaling for OCR, incremental checkpoint saving, Apple Silicon MPS
support, and scene-change skipping.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# IPv4 pattern used as a special "sensitive" entry
# ---------------------------------------------------------------------------
IPV4_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.4
PADDING_PX = 10
IOU_THRESHOLD = 0.30
MAX_FRAME_GAP = 10
KEYFRAME_MIN_MOVE = 3   # pixels - store keyframe only if position changed this much
TRACK_PADDING_FRAMES = 2  # frames added before frame_start and after frame_end
CHECKPOINT_INTERVAL = 500  # save checkpoint every N processed frames
MEMORY_CAP = 50  # max frames held in memory at once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sensitive_strings(path: str) -> list[str]:
    """Return lower-stripped sensitive strings from a file (one per line)."""
    strings = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip().lower()
            if s:
                strings.append(s)
    return strings


def normalize(text: str) -> str:
    return text.strip().lower()


def matches_sensitive(detected_word: str, sensitive_list: list[str]) -> str | None:
    """
    Return the first matching sensitive string, or None.
    Match if:
      - any sensitive string is a substring of detected_word, OR
      - detected_word is a substring of a sensitive string (catches partial OCR)
    Also matches IPv4 pattern.
    """
    norm = normalize(detected_word)

    # IPv4 check
    if IPV4_PATTERN.search(norm):
        match = IPV4_PATTERN.search(norm)
        return match.group(0) if match else "ipv4"

    for s in sensitive_list:
        if s in norm or norm in s:
            return s
    return None


def easyocr_bbox_to_xywh(
    bbox: list, frame_w: int, frame_h: int, scale: float = 1.0
) -> tuple[int, int, int, int]:
    """
    Convert EasyOCR bbox ([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) to (x, y, w, h)
    with 10px padding clamped to frame bounds.

    If scale != 1.0, coordinates are multiplied back up to the original resolution
    (use when OCR was run on a downscaled frame).
    """
    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    x1 = int(min(xs) / scale)
    y1 = int(min(ys) / scale)
    x2 = int(max(xs) / scale)
    y2 = int(max(ys) / scale)

    x = max(0, x1 - PADDING_PX)
    y = max(0, y1 - PADDING_PX)
    w = min(frame_w, x2 + PADDING_PX) - x
    h = min(frame_h, y2 + PADDING_PX) - y
    return x, y, max(1, w), max(1, h)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def boxes_close_enough(a: tuple, b: tuple) -> bool:
    return iou(a, b) > IOU_THRESHOLD


def significant_move(a: tuple, b: tuple) -> bool:
    return (
        abs(a[0] - b[0]) > KEYFRAME_MIN_MOVE
        or abs(a[1] - b[1]) > KEYFRAME_MIN_MOVE
        or abs(a[2] - b[2]) > KEYFRAME_MIN_MOVE
        or abs(a[3] - b[3]) > KEYFRAME_MIN_MOVE
    )


# ---------------------------------------------------------------------------
# Track management
# ---------------------------------------------------------------------------

class Track:
    _next_id = 1

    def __init__(self, matched_string: str, frame: int, box: tuple):
        self.id = Track._next_id
        Track._next_id += 1
        self.matched_string = matched_string
        self.last_frame = frame
        self.last_box = box
        self.frame_start = frame
        self.frame_end = frame
        # keyframes: list of {"frame": int, "x": int, "y": int, "w": int, "h": int}
        self.keyframes: list[dict] = [self._make_kf(frame, box)]

    @staticmethod
    def _make_kf(frame: int, box: tuple) -> dict:
        x, y, w, h = box
        return {"frame": frame, "x": x, "y": y, "w": w, "h": h}

    def update(self, frame: int, box: tuple) -> None:
        if significant_move(self.last_box, box):
            self.keyframes.append(self._make_kf(frame, box))
        self.last_frame = frame
        self.last_box = box
        self.frame_end = frame

    def to_dict(self) -> dict:
        # Apply padding
        fs = max(0, self.frame_start - TRACK_PADDING_FRAMES)
        fe = self.frame_end + TRACK_PADDING_FRAMES
        return {
            "id": self.id,
            "matched_string": self.matched_string,
            "keyframes": self.keyframes,
            "frame_start": fs,
            "frame_end": fe,
        }


def find_or_create_track(
    active_tracks: list[Track],
    matched_string: str,
    frame: int,
    box: tuple,
) -> None:
    """Associate detection with an existing track or start a new one."""
    best: Track | None = None
    best_iou = 0.0

    for t in active_tracks:
        if t.matched_string != matched_string:
            continue
        if frame - t.last_frame > MAX_FRAME_GAP:
            continue
        score = iou(t.last_box, box)
        if score > best_iou:
            best_iou = score
            best = t

    if best is not None and best_iou > IOU_THRESHOLD:
        best.update(frame, box)
    else:
        active_tracks.append(Track(matched_string, frame, box))


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess - must be top-level for spawn)
# ---------------------------------------------------------------------------

def _worker_process_batch(args_tuple: tuple) -> list[dict]:
    """
    Process a batch of (frame_idx, frame_rgb) tuples with EasyOCR.

    This function is called in a worker process. Each worker creates its own
    EasyOCR Reader - readers cannot be shared across processes.

    Returns a list of detection dicts:
      {"frame_idx": int, "bbox": list, "text": str, "conf": float}
    """
    batch, sensitive_list, ocr_scale, use_gpu = args_tuple

    import easyocr  # must import inside worker for spawn safety

    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    detections = []
    for frame_idx, frame_rgb in batch:
        if ocr_scale != 1.0:
            h, w = frame_rgb.shape[:2]
            new_w = int(w * ocr_scale)
            new_h = int(h * ocr_scale)
            ocr_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            ocr_frame = frame_rgb

        results = reader.readtext(ocr_frame, detail=1)

        for bbox, text, conf in results:
            if conf < MIN_CONFIDENCE:
                continue
            matched = matches_sensitive(text, sensitive_list)
            if matched is None:
                continue
            detections.append({
                "frame_idx": frame_idx,
                "bbox": bbox,
                "text": text,
                "conf": float(conf),
                "matched": matched,
            })

    return detections


# ---------------------------------------------------------------------------
# Lazy frame generator - sequential reads (faster than seeking on Mac/APFS)
# ---------------------------------------------------------------------------

def frame_generator(video_path: str, skip_frames: int, start_frame: int = 0):
    """
    Yield (frame_idx, frame_rgb) for every skip_frames-th frame starting
    at start_frame.

    Uses sequential cap.read() to avoid slow random seeks on Mac/APFS.
    Discards frames that are not in our sample set by reading-and-ignoring.
    Never holds more than MEMORY_CAP frames decoded at once - yields one at
    a time so the caller controls buffering.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    current = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current < start_frame:
                current += 1
                continue

            if (current - start_frame) % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield current, frame_rgb

            current += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "detections_checkpoint.json"


def load_checkpoint() -> dict | None:
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(
    partial_detections: list[dict],
    last_frame: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    data = {
        "last_frame": last_frame,
        "fps": fps,
        "width": width,
        "height": height,
        "detections": partial_detections,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, total_frames, width, height) using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def run_detection(
    video_path: str,
    sensitive_list: list[str],
    skip_frames: int,
    use_gpu: bool,
    ocr_scale: float,
    num_workers: int,
    resume: bool,
    use_scene_change: bool,
) -> tuple[list[Track], float, int, int]:
    """Run OCR over the video and return completed tracks plus video metadata."""
    import multiprocessing
    import psutil

    fps, total_frames, width, height = get_video_info(video_path)
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames total")
    print(
        f"Processing every {skip_frames} frame(s) | workers={num_workers} | "
        f"ocr_scale={ocr_scale} | scene_change={use_scene_change}"
    )

    # --- Resume support ---
    start_frame = 0
    raw_detections: list[dict] = []
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            start_frame = checkpoint["last_frame"] + 1
            raw_detections = checkpoint.get("detections", [])
            print(f"Resuming from frame {start_frame} ({len(raw_detections)} detections loaded)")
        else:
            print("No checkpoint found, starting from the beginning.")

    frames_to_process = list(range(start_frame, total_frames, skip_frames))
    total_to_process = len(frames_to_process)
    print(f"Frames to process: {total_to_process}")

    # --- Collect frames into batches respecting memory cap ---
    # Each batch is a list of (frame_idx, frame_rgb) with max size MEMORY_CAP
    batch_size = max(1, MEMORY_CAP // max(1, num_workers))

    proc = psutil.Process()
    frames_processed_since_checkpoint = 0
    last_frame_saved = start_frame - 1
    scene_skipped = 0
    prev_gray: np.ndarray | None = None

    active_tracks: list[Track] = []
    completed_tracks: list[Track] = []

    with tqdm(total=total_to_process, unit="frame", desc="Detecting") as pbar:
        gen = frame_generator(video_path, skip_frames, start_frame)

        current_batch: list[tuple[int, np.ndarray]] = []
        batches_queued = 0

        def flush_batch(batch: list) -> None:
            nonlocal raw_detections, frames_processed_since_checkpoint, last_frame_saved

            if not batch:
                return

            if num_workers <= 1:
                # Single-process path: avoid Pool overhead
                result = _worker_process_batch(
                    (batch, sensitive_list, ocr_scale, use_gpu)
                )
                raw_detections.extend(result)
            else:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    # Split batch evenly across workers
                    sub_size = max(1, math.ceil(len(batch) / num_workers))
                    sub_batches = [
                        (batch[i:i + sub_size], sensitive_list, ocr_scale, use_gpu)
                        for i in range(0, len(batch), sub_size)
                    ]
                    for sub_result in pool.imap_unordered(
                        _worker_process_batch, sub_batches, chunksize=1
                    ):
                        raw_detections.extend(sub_result)

            frames_processed_since_checkpoint += len(batch)
            last_frame_saved = batch[-1][0]

            # Incremental checkpoint
            if frames_processed_since_checkpoint >= CHECKPOINT_INTERVAL:
                save_checkpoint(raw_detections, last_frame_saved, fps, width, height)
                frames_processed_since_checkpoint = 0

        for frame_idx, frame_rgb in gen:
            # Scene-change detection: skip near-duplicate frames
            if use_scene_change:
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    if diff.mean() < 2.0:
                        scene_skipped += 1
                        pbar.update(1)
                        # Do NOT update prev_gray - compare against last non-skipped frame
                        continue
                prev_gray = gray

            current_batch.append((frame_idx, frame_rgb))

            if len(current_batch) >= batch_size:
                flush_batch(current_batch)
                current_batch = []
                batches_queued += 1

            pbar.update(1)

            # Update progress postfix every 10 frames
            if (pbar.n % 10) == 0:
                mem_mb = proc.memory_info().rss / 1024 ** 2
                pbar.set_postfix({
                    "mem_MB": f"{mem_mb:.0f}",
                    "tracks": len(active_tracks) + len(completed_tracks),
                    "skipped": scene_skipped,
                })

        # Flush remaining frames
        flush_batch(current_batch)

    if use_scene_change:
        print(f"Scene-change: skipped {scene_skipped} near-duplicate frames")

    # --- Sort all raw detections by frame index then build tracks ---
    raw_detections.sort(key=lambda d: d["frame_idx"])

    for det in raw_detections:
        frame_idx = det["frame_idx"]
        bbox = det["bbox"]
        matched = det["matched"]
        box = easyocr_bbox_to_xywh(bbox, width, height, scale=ocr_scale)
        find_or_create_track(active_tracks, matched, frame_idx, box)

        # Expire stale tracks
        still_active = []
        for t in active_tracks:
            if frame_idx - t.last_frame > MAX_FRAME_GAP + skip_frames:
                completed_tracks.append(t)
            else:
                still_active.append(t)
        active_tracks = still_active

    # Move remaining active tracks to completed
    completed_tracks.extend(active_tracks)

    # Clean up checkpoint after a successful full run
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()

    return completed_tracks, fps, width, height


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect sensitive text in video and export detections.json"
    )
    parser.add_argument("--input", required=True, help="Input MP4 video path")
    parser.add_argument(
        "--strings", required=True, help="Path to sensitive strings file (one per line)"
    )
    parser.add_argument("--output", default="detections.json", help="Output JSON path")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Process every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable EasyOCR GPU mode (CUDA only)",
    )
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=0.5,
        dest="ocr_scale",
        help="Downscale factor for OCR input (default: 0.5 = half resolution). "
             "Lower = faster; 1.0 = full resolution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from detections_checkpoint.json if present",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        default=False,
        help="Try to use Apple Silicon MPS backend for PyTorch (falls back to CPU)",
    )
    parser.add_argument(
        "--scene-change",
        action="store_true",
        default=False,
        dest="scene_change",
        help="Skip OCR on frames with mean pixel diff < 2.0 vs previous frame "
             "(ideal for static terminal recordings)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (default: cpu_count - 1)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"ERROR: Input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.strings).exists():
        print(f"ERROR: Strings file not found: {args.strings}", file=sys.stderr)
        sys.exit(1)
    if not (0.1 <= args.ocr_scale <= 1.0):
        print("ERROR: --ocr-scale must be between 0.1 and 1.0", file=sys.stderr)
        sys.exit(1)

    # --- Apple Silicon MPS setup ---
    if args.mps:
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.set_default_device("mps")
                print("MPS backend enabled (Apple Silicon).")
            else:
                print("WARNING: MPS requested but not available. Falling back to CPU.")
        except ImportError:
            print("WARNING: torch not installed; --mps has no effect.")

    sensitive_list = load_sensitive_strings(args.strings)
    print(f"Loaded {len(sensitive_list)} sensitive string(s) (+ IPv4 pattern)")

    # Reset track ID counter for reproducibility
    Track._next_id = 1

    tracks, fps, width, height = run_detection(
        video_path=args.input,
        sensitive_list=sensitive_list,
        skip_frames=args.skip_frames,
        use_gpu=args.gpu,
        ocr_scale=args.ocr_scale,
        num_workers=args.workers,
        resume=args.resume,
        use_scene_change=args.scene_change,
    )

    # Build output structure
    output: dict[str, Any] = {
        "fps": fps,
        "width": width,
        "height": height,
        "tracks": [t.to_dict() for t in tracks],
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # Summary
    unique_strings = {t.matched_string for t in tracks}
    print(f"\nDone. Tracks found: {len(tracks)}, Unique strings matched: {len(unique_strings)}")
    if unique_strings:
        for s in sorted(unique_strings):
            count = sum(1 for t in tracks if t.matched_string == s)
            print(f"  '{s}': {count} track(s)")
    print(f"Detections written to: {args.output}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
