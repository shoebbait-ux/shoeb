# Video Redaction Pipeline

Automatically detect sensitive text in video recordings and generate an After Effects script that blurs those regions frame-accurately.

---

## What it does

1. **`detect_and_export.py`** - Scans every Nth frame of an MP4 using EasyOCR, matches detected text against a list of sensitive strings (plus any IPv4 address), groups detections into smooth bounding-box tracks, and writes `detections.json`.
2. **`generate_ae_script.py`** - Reads `detections.json` and produces a `.jsx` ExtendScript file. Running that script inside After Effects creates a Fast Box Blur solid layer for every track and groups them into a `AUTO_REDACTIONS` pre-comp.

---

## Prerequisites

- **Python 3.9+**
- **ffmpeg** installed and available on your system `PATH` (used internally by OpenCV and ffmpeg-python)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **After Effects 2026** (for running the generated `.jsx`)
- An NVIDIA GPU + CUDA is optional but strongly recommended for fast OCR

---

## Installation

```bash
pip install -r requirements.txt
```

> First run will also download EasyOCR language models automatically (~300 MB).

---

## Step-by-step usage

### Step 1 - Prepare your sensitive strings

Edit `sensitive_strings.txt` and add one entry per line. The file already contains common entries. The detection script also automatically catches any IPv4 address in the video.

### Step 2 - Run detection

```bash
python detect_and_export.py \
    --input  recording.mp4 \
    --strings sensitive_strings.txt \
    --output  detections.json
```

This writes `detections.json` with all matched text tracks.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to the input MP4 |
| `--strings` | *(required)* | Path to sensitive strings file |
| `--output` | `detections.json` | Where to write results |
| `--skip-frames` | `2` | Process every Nth frame (higher = faster, may miss brief text) |
| `--gpu` | off | Enable EasyOCR GPU acceleration |

### Step 3 - Generate the After Effects script

```bash
python generate_ae_script.py \
    --detections detections.json \
    --output     redact.jsx
```

### Step 4 - Apply redactions in After Effects

1. Open After Effects 2026.
2. Open your project and import the original video if not already there.
3. Select the composition you want to redact in the Project panel, then double-click to make it the active composition.
4. Go to **File > Scripts > Run Script File...** and select `redact.jsx`.
5. Wait for the script to finish - it will show an alert:
   > *"Auto-redaction complete. X blur layer(s) created."*
6. All blur layers are automatically grouped into a pre-comp named **`AUTO_REDACTIONS`** above your footage layer.
7. Preview and render as normal.

---

## Tips for MobaXterm terminal recordings

Terminal recordings often have white text on dark backgrounds with high contrast - ideal for OCR. Recommended settings:

- Use `--skip-frames 2` (default) - terminal text is usually static for many frames so you won't miss detections.
- If the recording has low resolution or small fonts, try `--skip-frames 1` to catch every frame.
- If you get false positives (non-sensitive text being blurred), raise the confidence threshold by editing `MIN_CONFIDENCE` in `detect_and_export.py` from `0.4` to `0.5` or `0.6`.
- For very long recordings (1+ hour), `--skip-frames 4` or `--skip-frames 6` speeds things up significantly with minimal quality loss.

---

## GPU acceleration

Pass `--gpu` to `detect_and_export.py` to enable EasyOCR's CUDA backend:

```bash
python detect_and_export.py --input recording.mp4 --strings sensitive_strings.txt --gpu
```

Requires a CUDA-capable NVIDIA GPU with the appropriate PyTorch CUDA build installed. Without `--gpu`, the script runs on CPU (slower but works everywhere).

---

## Output format reference

`detections.json` structure:

```json
{
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "tracks": [
    {
      "id": 1,
      "matched_string": "pjsofttech",
      "keyframes": [
        {"frame": 120, "x": 100, "y": 200, "w": 150, "h": 25},
        {"frame": 125, "x": 102, "y": 200, "w": 150, "h": 25}
      ],
      "frame_start": 118,
      "frame_end": 132
    }
  ]
}
```

- `frame_start` / `frame_end` include a 2-frame padding on each side so blur fades in/out gracefully.
- `keyframes` records only positions that moved more than 3 px from the previous recorded position (to keep the file small while preserving smooth motion).
