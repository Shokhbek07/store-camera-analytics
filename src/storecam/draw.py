"""Drawing helpers: config geometry, tracks and counts over a frame, and face blurring."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Source

GREEN, RED, BLUE, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (230, 120, 40), (0, 220, 255), (255, 255, 255)


def _pts(poly) -> np.ndarray:
    return np.array(poly, dtype=np.int32).reshape(-1, 1, 2)


def geometry(frame: np.ndarray, source: Source) -> np.ndarray:
    """Draw the source's zones and door lines; zones are shaded so people stay visible."""
    overlay = frame.copy()
    for poly, colour in ((source.zones.queue, BLUE), (source.zones.staff, GREEN)):
        if poly:
            cv2.fillPoly(overlay, [_pts(poly)], colour)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    for poly, colour, label in ((source.zones.queue, BLUE, "queue"), (source.zones.staff, GREEN, "staff")):
        if poly:
            cv2.polylines(frame, [_pts(poly)], True, colour, 3)
            x, y = poly[0]
            cv2.putText(frame, label, (int(x) + 8, int(y) + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    for line in source.door_lines:
        a, b = tuple(map(int, line.a)), tuple(map(int, line.b))
        cv2.line(frame, a, b, RED, 4)
        cv2.circle(frame, tuple(map(int, line.inside)), 10, RED, -1)
        cv2.putText(frame, f"{line.name} (dot = inside)", (a[0], a[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    return frame


def blur_heads(frame: np.ndarray, boxes: np.ndarray, fraction: float = 0.3) -> np.ndarray:
    """Pixelate the top `fraction` of each person box, where the head is in an upright view.

    Used for every published frame. It is a privacy measure, not face detection: it relies on
    the person box, so a missed person is not blurred, and the caller should check output.
    """
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes.astype(int):
        bh = max(1, int((y2 - y1) * fraction))
        xa, xb = max(0, x1), min(w, x2)
        ya, yb = max(0, y1 - bh // 4), min(h, y1 + bh)
        if xb - xa < 2 or yb - ya < 2:
            continue
        roi = frame[ya:yb, xa:xb]
        small = cv2.resize(roi, (max(1, (xb - xa) // 12), max(1, (yb - ya) // 12)), interpolation=cv2.INTER_AREA)
        frame[ya:yb, xa:xb] = cv2.resize(small, (xb - xa, yb - ya), interpolation=cv2.INTER_NEAREST)
    return frame
