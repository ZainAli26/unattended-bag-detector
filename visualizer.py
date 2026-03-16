"""Visualization — color-coded boxes, association lines, timers, alert overlay."""

import cv2
import numpy as np

import config
from alert import BagAlert, BagState


def _put_text(frame, text, org, color, scale=None):
    scale = scale or config.FONT_SCALE
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, config.LINE_THICKNESS)


_BAG_STATE_COLORS = {
    BagState.UNKNOWN: config.COLOR_OWNED,
    BagState.OWNED: config.COLOR_OWNED,
    BagState.SEPARATED: config.COLOR_SEPARATED,
    BagState.UNATTENDED: config.COLOR_UNATTENDED,
}

_BAG_LABEL = "Bag"


def draw(
    frame: np.ndarray,
    people: list[dict],
    bags: list[dict],
    bag_owner_map: dict[int, int | None],
    bag_alerts: dict[int, BagAlert],
) -> np.ndarray:
    """Draw all annotations on the frame (mutates in place and returns it)."""
    person_map = {p["id"]: p for p in people}

    # --- People ---
    for p in people:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_PERSON, config.LINE_THICKNESS)
        _put_text(frame, "Person", (x1, y1 - 8), config.COLOR_PERSON)

    # --- Bags ---
    for bag in bags:
        bid = bag["id"]
        alert = bag_alerts.get(bid)
        state = alert.state if alert else BagState.UNKNOWN
        color = _BAG_STATE_COLORS.get(state, config.COLOR_OWNED)
        x1, y1, x2, y2 = bag["bbox"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.LINE_THICKNESS)

        _put_text(frame, _BAG_LABEL, (x1, y1 - 8), color)

        # State + timer
        if state == BagState.SEPARATED and alert:
            timer_text = f"SEPARATED {alert.elapsed:.0f}s"
            _put_text(frame, timer_text, (x1, y2 + 18), config.COLOR_SEPARATED)
        elif state == BagState.UNATTENDED:
            _put_text(frame, "UNATTENDED", (x1, y2 + 18), config.COLOR_UNATTENDED, scale=0.7)

        # Association line to owner
        owner_id = bag_owner_map.get(bid)
        if owner_id is not None and owner_id in person_map:
            owner = person_map[owner_id]
            bag_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            owner_bottom = ((owner["bbox"][0] + owner["bbox"][2]) // 2, owner["bbox"][3])
            cv2.line(frame, bag_center, owner_bottom, color, 1, cv2.LINE_AA)

    # --- Global alert banner ---
    unattended_count = sum(
        1 for a in bag_alerts.values() if a.state == BagState.UNATTENDED
    )
    if unattended_count > 0:
        banner = f"ALERT: {unattended_count} UNATTENDED BAG(S)"
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
        _put_text(frame, banner, (10, 28), (255, 255, 255), scale=0.8)

    return frame
