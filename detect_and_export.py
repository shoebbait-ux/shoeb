#!/usr/bin/env python3
"""
detect_and_export.py - Detect sensitive text in video frames and export detections.

Extracts frames from an MP4, runs EasyOCR on every Nth frame, matches against
a list of sensitive strings (plus IPv4 regex), groups detections into tracks,
and writes a detections.json file.
"""

import argparse
import json
import math
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
KEYFRAME_MIN_MOVE = 3  # pixels - store keyframe only if position changed this much
TRACK_PADDING_FRAMES = 2  # frames added before frame_start and after frame_end


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


def easyocr_bbox_to_xywh(bbox: list, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """
    Convert EasyOCR bbox ([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) to (x, y, w, h)
    with 10px padding clamped to frame bounds.
    """
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

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
# Main pipeline
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


def run_detection(
    video_path: str,
    sensitive_list: list[str],
    skip_frames: int,
    use_gpu: bool,
) -> tuple[list[Track], float, int, int]:
    """Run OCR over the video and return completed tracks plus video metadata."""
    import easyocr  # imported here so --help works without easyocr installed

    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    fps, total_frames, width, height = get_video_info(video_path)
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames total")
    print(f"Processing every {skip_frames} frame(s) with EasyOCR (GPU={use_gpu})")

    cap = cv2.VideoCapture(video_path)
    active_tracks: list[Track] = []
    completed_tracks: list[Track] = []

    frames_to_process = range(0, total_frames, skip_frames)

    with tqdm(total=len(frames_to_process), unit="frame", desc="Detecting") as pbar:
        for frame_idx in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(frame_rgb, detail=1)

            seen_strings_this_frame: set[str] = set()

            for bbox, text, conf in results:
                if conf < MIN_CONFIDENCE:
                    continue
                matched = matches_sensitive(text, sensitive_list)
                if matched is None:
                    continue
                box = easyocr_bbox_to_xywh(bbox, width, height)
                find_or_create_track(active_tracks, matched, frame_idx, box)
                seen_strings_this_frame.add(matched + str(box))

            # Expire tracks that haven't been seen for too long
            still_active: list[Track] = []
            for t in active_tracks:
                if frame_idx - t.last_frame > MAX_FRAME_GAP + skip_frames:
                    completed_tracks.append(t)
                else:
                    still_active.append(t)
            active_tracks = still_active

            pbar.update(1)

    cap.release()

    # Move remaining active tracks to completed
    completed_tracks.extend(active_tracks)

    return completed_tracks, fps, width, height


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
        help="Enable EasyOCR GPU mode",
    )
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"ERROR: Input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.strings).exists():
        print(f"ERROR: Strings file not found: {args.strings}", file=sys.stderr)
        sys.exit(1)

    sensitive_list = load_sensitive_strings(args.strings)
    print(f"Loaded {len(sensitive_list)} sensitive string(s) (+ IPv4 pattern)")

    # Reset track ID counter for reproducibility
    Track._next_id = 1

    tracks, fps, width, height = run_detection(
        args.input, sensitive_list, args.skip_frames, args.gpu
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
    main()
