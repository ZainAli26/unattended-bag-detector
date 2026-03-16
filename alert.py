import math
from enum import Enum

import config


class BagState(Enum):
    UNKNOWN = "UNKNOWN"
    OWNED = "OWNED"
    SEPARATED = "SEPARATED"
    UNATTENDED = "UNATTENDED"


class BagAlert:

    def __init__(self):
        self.state = BagState.UNKNOWN
        self.separation_start: float | None = None  # video timestamp in seconds
        self.bbox: tuple | None = None  # last known position

    @property
    def elapsed(self) -> float:
        if self.separation_start is None:
            return 0.0
        return self._last_video_time - self.separation_start

    _last_video_time: float = 0.0


class AlertEngine:

    def __init__(self, fps: float = 30.0):
        self.alerts: dict[int, BagAlert] = {}
        self.fps = fps
        self._frame_count = 0
        # Recently expired alerts kept for state inheritance
        self._expired: list[BagAlert] = []

    @property
    def video_time(self) -> float:
        return self._frame_count / self.fps

    def _find_inherited_alert(self, bbox: tuple) -> BagAlert | None:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for old in self._expired:
            if old.bbox is None:
                continue
            ox = (old.bbox[0] + old.bbox[2]) / 2
            oy = (old.bbox[1] + old.bbox[3]) / 2
            if math.hypot(cx - ox, cy - oy) < config.BAG_STATIONARY_MAX_DRIFT * 3:
                return old
        return None

    def update(
        self,
        bags: list[dict],
        people: list[dict],
        bag_owner_map: dict[int, int | None],
    ) -> dict[int, BagAlert]:
        """Update state for all bags. Returns bag_id → BagAlert."""
        self._frame_count += 1
        now = self.video_time
        person_map = {p["id"]: p for p in people}

        for bag in bags:
            bid = bag["id"]
            if bid not in self.alerts:
                # Try to inherit from a nearby expired alert
                inherited = self._find_inherited_alert(bag["bbox"])
                if inherited and inherited.state in (BagState.SEPARATED, BagState.UNATTENDED):
                    alert = BagAlert()
                    alert.state = inherited.state
                    alert.separation_start = inherited.separation_start
                    self.alerts[bid] = alert
                else:
                    self.alerts[bid] = BagAlert()
            alert = self.alerts[bid]
            alert._last_video_time = now
            alert.bbox = bag["bbox"]
            owner_id = bag_owner_map.get(bid)

            if owner_id is None:
                # No owner ever assigned — stay UNKNOWN or keep SEPARATED/UNATTENDED
                if alert.state == BagState.UNKNOWN:
                    pass
                elif alert.state in (BagState.SEPARATED, BagState.UNATTENDED):
                    self._check_timeout(alert, now)
                continue

            owner = person_map.get(owner_id)
            if owner is None:
                # Owner left the frame
                self._mark_separated(alert, now)
                self._check_timeout(alert, now)
                continue

            # Owner is visible — check distance
            bag_cx = (bag["bbox"][0] + bag["bbox"][2]) / 2
            bag_cy = (bag["bbox"][1] + bag["bbox"][3]) / 2
            owner_bc = ((owner["bbox"][0] + owner["bbox"][2]) / 2, owner["bbox"][3])
            dist = math.hypot(bag_cx - owner_bc[0], bag_cy - owner_bc[1])

            if dist <= config.SEPARATION_DISTANCE:
                # Owner is near — bag is owned
                alert.state = BagState.OWNED
                alert.separation_start = None
            else:
                # Owner moved away
                self._mark_separated(alert, now)
                self._check_timeout(alert, now)

        return {bid: self.alerts[bid] for bid in self.alerts if any(b["id"] == bid for b in bags)}

    def _mark_separated(self, alert: BagAlert, now: float):
        if alert.state in (BagState.UNKNOWN, BagState.OWNED):
            alert.state = BagState.SEPARATED
            alert.separation_start = now

    def _check_timeout(self, alert: BagAlert, now: float):
        if alert.state == BagState.SEPARATED and alert.separation_start is not None:
            if now - alert.separation_start >= config.UNATTENDED_TIMEOUT:
                alert.state = BagState.UNATTENDED

    def prune(self, active_bag_ids: set):
        stale = [bid for bid in self.alerts if bid not in active_bag_ids]
        for bid in stale:
            alert = self.alerts.pop(bid)
            # Keep for inheritance if it had a meaningful state
            if alert.state in (BagState.SEPARATED, BagState.UNATTENDED):
                self._expired.append(alert)
        # Cap expired list size
        self._expired = self._expired[-20:]
