"""Person-bag association via Hungarian algorithm + IoU/proximity scoring."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import config


def _iou(box_a: tuple, box_b: tuple) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _proximity_score(bag_center: tuple, person_bottom_center: tuple) -> float:
    dist = math.hypot(
        bag_center[0] - person_bottom_center[0],
        bag_center[1] - person_bottom_center[1],
    )
    if dist >= config.PROXIMITY_MAX_DISTANCE:
        return 0.0
    return 1.0 - dist / config.PROXIMITY_MAX_DISTANCE


def _bag_center(bbox: tuple) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _person_bottom_center(bbox: tuple) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


class AssociationEngine:

    def __init__(self):
        # bag_id → {"owner_id": int, "score": float}
        self.assignments: dict[int, dict] = {}

    def update(self, people: list[dict], bags: list[dict]) -> dict[int, int | None]:
        if not bags:
            return {}

        if not people:
            # No people visible — keep existing assignments but don't create new ones
            return {b["id"]: self.assignments.get(b["id"], {}).get("owner_id") for b in bags}

        n_bags = len(bags)
        n_people = len(people)

        # Build cost matrix (we maximize score, so cost = -score for Hungarian)
        score_matrix = np.zeros((n_bags, n_people))
        for i, bag in enumerate(bags):
            bc = _bag_center(bag["bbox"])
            for j, person in enumerate(people):
                iou_score = _iou(bag["bbox"], person["bbox"]) * 2.0
                prox_score = _proximity_score(bc, _person_bottom_center(person["bbox"]))
                score_matrix[i, j] = max(iou_score, prox_score)

        cost_matrix = -score_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        result: dict[int, int | None] = {b["id"]: None for b in bags}

        for r, c in zip(row_idx, col_idx):
            bag_id = bags[r]["id"]
            person_id = people[c]["id"]
            score = score_matrix[r, c]

            if score < config.ASSOCIATION_MIN_SCORE:
                continue

            prev = self.assignments.get(bag_id)
            if prev and prev["owner_id"] != person_id:
                # Hysteresis — only switch if new score clearly beats old
                if score < prev["score"] + config.OWNER_SWITCH_HYSTERESIS:
                    result[bag_id] = prev["owner_id"]
                    continue

            self.assignments[bag_id] = {"owner_id": person_id, "score": score}
            result[bag_id] = person_id

        # For unmatched bags, retain previous owner if any
        for bag in bags:
            bid = bag["id"]
            if result[bid] is None:
                prev = self.assignments.get(bid)
                if prev:
                    result[bid] = prev["owner_id"]

        return result

    def prune(self, active_bag_ids: set):
        stale = [bid for bid in self.assignments if bid not in active_bag_ids]
        for bid in stale:
            del self.assignments[bid]
