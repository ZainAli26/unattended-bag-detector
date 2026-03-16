"""Track history utilities — position history, stationarity checks."""

from collections import defaultdict, deque

import config


class TrackHistory:

    def __init__(self):
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.TRACK_HISTORY_LENGTH)
        )

    def update(self, track_id: int, bbox: tuple):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self._history[track_id].append((cx, cy))

    def get_center(self, track_id: int) -> tuple | None:
        h = self._history.get(track_id)
        if h:
            return h[-1]
        return None

    def get_bottom_center(self, bbox: tuple) -> tuple:
        return ((bbox[0] + bbox[2]) / 2, bbox[3])

    def prune(self, active_ids: set):
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]
