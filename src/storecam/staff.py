"""Staff versus customer by uniform colour, so staff are not counted in the queue.

Each person's torso (the middle of the box, below the head and above the counter) is reduced to
its median colour in CIE Lab and compared with the configured uniform colour. One frame is noisy
(the person bends, turns, or is partly hidden), so a track's colour is a vote over its recent
frames. Colour alone also matches customers in similar clothes (E14), so a track only counts as
staff once it has also spent some time in the staff zone, where customers rarely go.
This tells roles apart by clothing and place only; it does not identify anyone.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .events import Point, point_in_polygon

LIGHTNESS_WEIGHT = 0.5  # lighting and shadow change lightness more than hue, so it counts half


def torso_lab(image: np.ndarray, box: tuple[float, float, float, float]) -> tuple[float, float, float] | None:
    """Median OpenCV Lab colour (0..255 per channel) of the torso part of a person box."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = int(max(0, y1 + 0.2 * h)), int(max(0, y1 + 0.5 * h))
    tx1, tx2 = int(max(0, x1 + 0.25 * w)), int(max(0, x2 - 0.25 * w))
    patch = image[ty1:ty2, tx1:tx2]
    if patch.size < 3 * 50:  # too small or outside the frame to judge
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return tuple(float(v) for v in np.median(lab, axis=0))


def colour_distance(c: tuple[float, float, float], ref: tuple[float, float, float]) -> float:
    return math.sqrt((LIGHTNESS_WEIGHT * (c[0] - ref[0])) ** 2 + (c[1] - ref[1]) ** 2 + (c[2] - ref[2]) ** 2)


class StaffClassifier:
    """A track is staff when most of its last `window` torso colours match the uniform and it has
    spent at least `min_zone_seconds` in the staff zone so far."""

    def __init__(self, lab: tuple[float, float, float], max_distance: float, staff_zone: list[Point],
                 min_zone_seconds: float = 1.0, window: int = 20):
        self.ref = tuple(lab)
        self.max_distance = max_distance
        self.zone = staff_zone
        self.min_zone_seconds = min_zone_seconds
        self.window = window
        self.votes: dict[int, deque[bool]] = {}
        self.zone_seconds: dict[int, float] = {}
        self.last_t: dict[int, float] = {}

    def update(self, image: np.ndarray, tracks: dict[int, tuple], t: float) -> set[int]:
        """Returns the ids of the tracks currently judged to be staff."""
        staff = set()
        for tid, box in tracks.items():
            x1, _, x2, y2 = box
            dt = min(t - self.last_t.get(tid, t), 1.0)  # time since this track was last seen
            self.last_t[tid] = t
            if point_in_polygon(((x1 + x2) / 2, y2), self.zone):
                self.zone_seconds[tid] = self.zone_seconds.get(tid, 0.0) + dt
            c = torso_lab(image, box)
            votes = self.votes.setdefault(tid, deque(maxlen=self.window))
            if c is not None:
                votes.append(colour_distance(c, self.ref) <= self.max_distance)
            if (votes and sum(votes) * 2 > len(votes)
                    and self.zone_seconds.get(tid, 0.0) >= self.min_zone_seconds):
                staff.add(tid)
        return staff

    def forget(self, alive: set[int]) -> None:
        for tid in [t for t in self.votes if t not in alive]:
            self.votes.pop(tid, None)
            self.zone_seconds.pop(tid, None)
            self.last_t.pop(tid, None)
