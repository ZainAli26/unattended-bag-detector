import argparse
import sys

import cv2

import config
from detector import Detector
from tracker import TrackHistory
from association import AssociationEngine
from alert import AlertEngine
from visualizer import draw


def parse_args():
    p = argparse.ArgumentParser(description="Unattended bag detection system")
    p.add_argument("--source", default="0", help="Video file path or camera index (default: 0 for webcam)")
    p.add_argument("--timeout", type=float, default=None, help=f"Seconds before SEPARATED → UNATTENDED (default: {config.UNATTENDED_TIMEOUT})")
    p.add_argument("--model", default=None, help=f"YOLOv8 model name (default: {config.MODEL_NAME})")
    p.add_argument("--confidence", type=float, default=None, help=f"Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})")
    p.add_argument("--no-display", action="store_true", help="Run without GUI window (headless)")
    p.add_argument("--output", default=None, help="Path to save output video")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to config
    if args.timeout is not None:
        config.UNATTENDED_TIMEOUT = args.timeout
    if args.model is not None:
        config.MODEL_NAME = args.model
    if args.confidence is not None:
        config.CONFIDENCE_THRESHOLD = args.confidence

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{args.source}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize modules
    detector = Detector()
    track_history = TrackHistory()
    association = AssociationEngine()
    alert_engine = AlertEngine(fps=fps)

    print(f"Running on: {args.source} | timeout={config.UNATTENDED_TIMEOUT}s | model={config.MODEL_NAME}")
    print("Press 'q' to quit")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Detect + track
        people, bags = detector.detect_and_track(frame)

        # 2. Update track histories
        active_ids = set()
        for p in people:
            track_history.update(p["id"], p["bbox"])
            active_ids.add(p["id"])
        for b in bags:
            track_history.update(b["id"], b["bbox"])
            active_ids.add(b["id"])
        track_history.prune(active_ids)

        # 3. Associate bags to people
        bag_owner_map = association.update(people, bags)
        association.prune({b["id"] for b in bags})

        # 4. Update alert states
        bag_alerts = alert_engine.update(bags, people, bag_owner_map)
        alert_engine.prune({b["id"] for b in bags})

        # 5. Visualize
        frame = draw(frame, people, bags, bag_owner_map, bag_alerts)

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Unattended Bag Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
