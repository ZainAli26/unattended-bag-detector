# Unattended Bag Detection System

Real-time security system that detects bags in video, associates them with nearby people, and alerts when a bag is left unattended.

## Architecture

![Architecture](videos/architecture.png)

The system uses YOLOv8 pretrained on COCO to detect persons and bags (backpacks, handbags, suitcases) in each frame. BoTSORT assigns persistent track IDs across frames so objects can be followed over time. A Hungarian algorithm with IoU and proximity scoring links each bag to its nearest owner, with hysteresis to prevent flickering between people. A state machine tracks each bag through UNKNOWN → OWNED → SEPARATED → UNATTENDED states, triggering an alert after a configurable timeout. The output is visualized with color-coded bounding boxes, association lines, countdown timers, and alert banners.

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

![Demo 1](videos/demo1.gif)

### Demo Video 2

![Demo 2](videos/demo2.gif)
