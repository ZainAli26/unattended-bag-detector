# Unattended Bag Detection System

Real-time security system that detects bags in video, associates them with nearby people, and alerts when a bag is left unattended.

## Architecture

```
Video Frame → Detection (YOLOv8) → Tracking (BoTSORT) → Association → Unattended Logic → Visualization
```

- **Detection**: YOLOv8 pretrained on COCO — detects persons, backpacks, handbags, suitcases
- **Tracking**: BoTSORT via `model.track(persist=True)` for consistent IDs across frames
- **Association**: Hungarian algorithm with IoU + proximity scoring, hysteresis to prevent flickering
- **Alert state machine**: UNKNOWN → OWNED → SEPARATED → UNATTENDED (with configurable timeout)
- **Visualization**: Color-coded bounding boxes, association lines, countdown timers, alert banners

## Setup

```bash
pip install -r requirements.txt
```

The YOLOv8 model weights download automatically on first run.

## Usage

```bash
# Webcam with 30-second timeout
python main.py --source 0 --timeout 30

# Video file
python main.py --source path/to/video.mp4

# Save output video (headless)
python main.py --source video.mp4 --output result.mp4 --no-display

# Custom model and confidence
python main.py --source 0 --model yolov8s.pt --confidence 0.4
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` (webcam) | Video file path or camera index |
| `--timeout` | `30` | Seconds before a separated bag becomes unattended |
| `--model` | `yolov8n.pt` | YOLOv8 model variant |
| `--confidence` | `0.35` | Detection confidence threshold |
| `--no-display` | off | Run without GUI window |
| `--output` | none | Save annotated video to file |

Press **q** to quit.

## Visual Color Coding

| Color | Meaning |
|-------|---------|
| Green | Person |
| Blue | Owned bag (near its owner) |
| Yellow | Separated bag (owner walked away, timer counting) |
| Red | Unattended bag (timeout exceeded) |

## Configuration

All tuneable parameters are in `config.py`: model settings, class IDs, tracking history length, association thresholds, separation distance, timeout duration, and colors.

## Demo Videos

### Demo Video 1
https://github.com/user-attachments/assets/demo1.mp4

### Demo Video 2
https://github.com/user-attachments/assets/demo2.mp4
