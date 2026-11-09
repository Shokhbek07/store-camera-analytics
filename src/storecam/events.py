"""Event logic: door line counting, per-second zone occupancy, and alerts.

Everything here is pure Python on track positions and timestamps, with no video or model,
so it can be unit tested with hand-made tracks.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

Point = tuple[float, float]


def point_in_polygon(p: Point, poly: list[Point]) -> bool:
    """Ray casting; points exactly on an edge may fall either way."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


@dataclass
class Crossing:
    track_id: int
    direction: str  # "in" or "out"
    t: float


class LineCounter:
    """Counts IN and OUT crossings of a door line with a dead band.

    A track's side is confirmed only when its foot point is more than `band` pixels from the
    line, within the line's extent, for `min_frames` consecutive frames. A crossing is counted
    when the confirmed side changes. A person who stops on the line, walks along it, or steps
    onto it and turns back never confirms the other side, so is never counted.

    An optional `door_region` covers the doorway itself, on the outside of the line, where
    people appear when they come in and vanish when they leave (E9). A track first seen there
    starts on the outside, so walking in past the line counts IN even though it was never seen
    beyond the band outside. A track that vanishes there without ever having been counted, and
    was not born there, is counted OUT when the tracker drops it: someone who waited on the
    outside side of the line and then left through the door.
    """

    def __init__(self, a: Point, b: Point, inside: Point, band: float = 20.0, min_frames: int = 2,
                 extent_margin: float = 0.1, door_region: list[Point] | None = None):
        self.a, self.b = a, b
        dx, dy = b[0] - a[0], b[1] - a[1]
        self.length = math.hypot(dx, dy)
        if self.length == 0:
            raise ValueError("door line has zero length")
        self.ux, self.uy = dx / self.length, dy / self.length
        # Unit normal pointing to the inside of the door.
        nx, ny = -self.uy, self.ux
        if (inside[0] - a[0]) * nx + (inside[1] - a[1]) * ny < 0:
            nx, ny = -nx, -ny
        self.nx, self.ny = nx, ny
        self.band = band
        self.min_frames = min_frames
        self.margin = extent_margin * self.length
        self.side: dict[int, int] = {}  # confirmed side: +1 inside, -1 outside
        self.pending: dict[int, tuple[int, int]] = {}  # candidate side and frames seen
        self.door_region = door_region
        self.last: dict[int, tuple[Point, float]] = {}  # last foot point and time per track
        self.born_in_door: set[int] = set()
        self.counted: set[int] = set()
        self.n_in = 0
        self.n_out = 0

    def _count(self, tid: int, direction: str, t: float) -> Crossing:
        if direction == "in":
            self.n_in += 1
        else:
            self.n_out += 1
        self.counted.add(tid)
        return Crossing(tid, direction, t)

    def signed_distance(self, p: Point) -> tuple[float, float]:
        """(distance along the normal, position along the line from a)."""
        rx, ry = p[0] - self.a[0], p[1] - self.a[1]
        return rx * self.nx + ry * self.ny, rx * self.ux + ry * self.uy

    def update(self, tracks: dict[int, Point], t: float) -> list[Crossing]:
        events = []
        for tid, p in tracks.items():
            if tid not in self.last and self.door_region and point_in_polygon(p, self.door_region):
                self.born_in_door.add(tid)
                self.side[tid] = -1
            self.last[tid] = (p, t)
            d, s = self.signed_distance(p)
            within = -self.margin <= s <= self.length + self.margin
            if not within or abs(d) <= self.band:
                self.pending.pop(tid, None)
                continue
            cand = 1 if d > 0 else -1
            prev_cand, n = self.pending.get(tid, (cand, 0))
            n = n + 1 if prev_cand == cand else 1
            self.pending[tid] = (cand, n)
            if n < self.min_frames:
                continue
            old = self.side.get(tid)
            self.side[tid] = cand
            if old is not None and old != cand:
                events.append(self._count(tid, "in" if cand > 0 else "out", t))
        return events

    def forget(self, alive: set[int]) -> list[Crossing]:
        """Drop state for tracks the tracker has removed; may count an exit through the door."""
        events = []
        for tid in [tid for tid in self.last if tid not in alive]:
            p, t = self.last.pop(tid)
            if (self.door_region and tid not in self.counted and tid not in self.born_in_door
                    and point_in_polygon(p, self.door_region)):
                events.append(self._count(tid, "out", t))
            self.counted.discard(tid)
            self.born_in_door.discard(tid)
        for d in (self.side, self.pending):
            for tid in list(d):
                if tid not in alive:
                    del d[tid]
        return events

    def reset(self) -> None:
        for d in (self.side, self.pending, self.last):
            d.clear()
        self.born_in_door.clear()
        self.counted.clear()


@dataclass
class SecondSample:
    second: int
    queue: int
    staff_present: bool


class ZoneSampler:
    """Reduces per-frame zone counts to one value per whole second.

    The queue count for a second is the median of its frames, rounded half up, and staff
    are present if they were present in at least half of its frames. Both damp single-frame
    detection misses.
    """

    def __init__(self, queue: list[Point], staff: list[Point] | None):
        self.queue_poly = queue
        self.staff_poly = staff
        self.current: int | None = None
        self.frames: list[tuple[int, bool]] = []

    def update(self, feet: list[Point], t: float) -> list[SecondSample]:
        sec = int(math.floor(t))
        out = []
        if self.current is not None and sec != self.current:
            out.append(self._flush())
        self.current = sec
        q = sum(1 for p in feet if point_in_polygon(p, self.queue_poly))
        s = self.staff_poly is not None and any(point_in_polygon(p, self.staff_poly) for p in feet)
        self.frames.append((q, s))
        return out

    def _flush(self) -> SecondSample:
        qs = [q for q, _ in self.frames]
        staff = sum(1 for _, s in self.frames if s) * 2 >= len(self.frames)
        sample = SecondSample(self.current, math.floor(statistics.median(qs) + 0.5), staff)
        self.frames = []
        return sample

    def finish(self) -> list[SecondSample]:
        if self.current is None or not self.frames:
            return []
        out = [self._flush()]
        self.current = None
        return out


@dataclass
class Alert:
    kind: str
    start: float  # when the condition began
    raised: float  # when the alert was raised (start + min_seconds)
    peak: int
    end: float | None = None
    ended_by: str | None = None


@dataclass
class AlertTracker:
    """Hysteresis on a per-second condition: start once, end once, no flicker.

    The alert is raised after the condition has held for `min_seconds` and ends after it has
    been false for `clear_seconds`. Its start is when the condition began and its end is when
    the condition last stopped, so the short gaps bridged by `clear_seconds` are inside it.
    """

    kind: str
    min_seconds: float
    clear_seconds: float
    onset: float | None = None
    false_since: float | None = None
    peak_so_far: int = 0
    active: Alert | None = None
    done: list[Alert] = field(default_factory=list)

    def update(self, t: float, condition: bool, value: int) -> list[tuple[str, Alert]]:
        events: list[tuple[str, Alert]] = []
        if condition:
            self.false_since = None
            if self.onset is None:
                self.onset = t
                self.peak_so_far = value
            self.peak_so_far = max(self.peak_so_far, value)
            if self.active is None and t - self.onset + 1 >= self.min_seconds:
                self.active = Alert(self.kind, self.onset, t, self.peak_so_far)
                events.append(("start", self.active))
            elif self.active is not None:
                self.active.peak = max(self.active.peak, value)
        else:
            if self.false_since is None:
                self.false_since = t
            if self.active is not None and t - self.false_since + 1 >= self.clear_seconds:
                events.append(("end", self._close(self.false_since, "cleared")))
            elif self.active is None:
                self.onset = None
        return events

    def _close(self, end: float, how: str) -> Alert:
        alert = self.active
        alert.end, alert.ended_by = end, how
        self.done.append(alert)
        self.active, self.onset, self.false_since = None, None, None
        return alert

    def finish(self, t: float) -> list[tuple[str, Alert]]:
        if self.active is None:
            return []
        end = self.false_since if self.false_since is not None else t
        return [("end", self._close(end, "stream_end"))]
