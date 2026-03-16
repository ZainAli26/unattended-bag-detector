import math
from collections import defaultdict

from ultralytics import YOLO
import numpy as np

import config


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_bags(bags: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    if len(bags) <= 1:
        return bags
    # Sort by confidence descending
    bags_sorted = sorted(bags, key=lambda b: b["conf"], reverse=True)
    keep = []
    for bag in bags_sorted:
        duplicate = False
        for kept in keep:
            if _iou(bag["bbox"], kept["bbox"]) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(bag)
    return keep


class _BagMemory:

    def __init__(self):
        self.positions: dict[int, list[tuple]] = defaultdict(list)
        self.last_seen: dict[int, dict] = {}  # bag_id → last detection dict
        self.frames_missing: dict[int, int] = defaultdict(int)

    def update(self, bag: dict):
        bid = bag["id"]
        self.positions[bid].append(_center(bag["bbox"]))
        # Keep only recent positions
        if len(self.positions[bid]) > config.TRACK_HISTORY_LENGTH:
            self.positions[bid] = self.positions[bid][-config.TRACK_HISTORY_LENGTH:]
        self.last_seen[bid] = bag
        self.frames_missing[bid] = 0

    def is_stationary(self, bag_id: int) -> bool:
        pts = self.positions.get(bag_id, [])
        if len(pts) < config.BAG_STATIONARY_THRESH:
            return False
        recent = pts[-config.BAG_STATIONARY_THRESH:]
        cx0, cy0 = recent[0]
        for cx, cy in recent[1:]:
            if math.hypot(cx - cx0, cy - cy0) > config.BAG_STATIONARY_MAX_DRIFT:
                return False
        return True

    def get_phantoms(self, active_bag_ids: set) -> list[dict]:
        candidates = []
        to_remove = []
        for bid, last in self.last_seen.items():
            if bid in active_bag_ids:
                continue
            self.frames_missing[bid] += 1
            if self.frames_missing[bid] > config.BAG_PHANTOM_TTL:
                to_remove.append(bid)
                continue
            if self.is_stationary(bid):
                phantom = dict(last)
                phantom["phantom"] = True
                candidates.append(phantom)
        for bid in to_remove:
            self.positions.pop(bid, None)
            self.last_seen.pop(bid, None)
            self.frames_missing.pop(bid, None)

        phantoms = []
        used = set()
        # Sort by history length descending so the most-tracked bag wins
        candidates.sort(key=lambda p: len(self.positions.get(p["id"], [])), reverse=True)
        for p in candidates:
            pc = _center(p["bbox"])
            duplicate = False
            for kept in phantoms:
                kc = _center(kept["bbox"])
                if math.hypot(pc[0] - kc[0], pc[1] - kc[1]) < config.BAG_STATIONARY_MAX_DRIFT * 3:
                    duplicate = True
                    break
            if not duplicate:
                phantoms.append(p)
                used.add(p["id"])
        return phantoms


class Detector:
    def __init__(self):
        self.model = YOLO(config.MODEL_NAME)
        self._bag_memory = _BagMemory()

    def detect_and_track(self, frame: np.ndarray):
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_TYPE,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            classes=[config.PERSON_CLASS_ID] + list(config.BAG_CLASS_IDS),
            verbose=False,
        )

        people = []
        bags = []

        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                track_id = int(boxes.id[i])
                bbox = tuple(int(v) for v in boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i])

                entry = {"id": track_id, "bbox": bbox, "conf": conf}

                if cls_id == config.PERSON_CLASS_ID:
                    people.append(entry)
                elif cls_id in config.BAG_CLASS_IDS:
                    entry["class_id"] = cls_id
                    bags.append(entry)

        # Deduplicate overlapping bag detections (e.g. handbag + backpack on same object)
        bags = _dedup_bags(bags)

        # Update bag memory and get phantoms for lost stationary bags
        active_bag_ids = {b["id"] for b in bags}
        for b in bags:
            self._bag_memory.update(b)
        phantoms = self._bag_memory.get_phantoms(active_bag_ids)
        # Filter phantoms that overlap with live detections
        for phantom in phantoms:
            overlaps_live = any(
                _iou(phantom["bbox"], b["bbox"]) > 0.3 for b in bags
            )
            if not overlaps_live:
                bags.append(phantom)

        return people, bags
