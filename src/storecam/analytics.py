"""Per-source analytics: tracks in, per-minute rows and alert events out."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import Source
from .events import Alert, AlertTracker, LineCounter, SecondSample, ZoneSampler, duplicate_boxes
from .staff import StaffClassifier

CSV_FIELDS = ["source", "minute", "minute_start_video_s", "minute_start_wall", "in", "out", "queue_avg",
              "queue_max", "empty_counter_s", "empty_counter_waiting_s", "seconds_observed"]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


class MinuteRow:
    def __init__(self, minute: int, wall: float):
        self.minute, self.wall = minute, wall
        self.n_in = self.n_out = 0
        self.queue: list[int] = []
        self.empty = self.empty_waiting = 0

    def as_dict(self, source: str) -> dict:
        return {"source": source, "minute": self.minute, "minute_start_video_s": self.minute * 60,
                "minute_start_wall": iso(self.wall), "in": self.n_in, "out": self.n_out,
                "queue_avg": round(sum(self.queue) / len(self.queue), 2) if self.queue else "",
                "queue_max": max(self.queue) if self.queue else "",
                "empty_counter_s": self.empty, "empty_counter_waiting_s": self.empty_waiting,
                "seconds_observed": len(self.queue)}


class SourceAnalytics:
    def __init__(self, source: Source, out_dir: Path, alerts_file):
        self.src = source
        self.lines = [LineCounter(dl.a, dl.b, dl.inside, dl.band_px, dl.min_frames, door_region=dl.door_region)
                      for dl in source.door_lines]
        z = source.zones
        self.zones = ZoneSampler(z.queue, z.staff) if z.queue else None
        sa = source.staff_appearance
        self.staff = (StaffClassifier(sa.lab, sa.max_distance, z.staff, sa.min_zone_seconds)
                      if sa and z.queue and z.staff else None)
        qa, ea = source.alerts.queue, source.alerts.empty_counter
        self.queue_alert = AlertTracker("queue", qa.min_seconds, qa.clear_seconds) if z.queue else None
        self.empty_alert = (AlertTracker("empty_counter", ea.min_seconds, ea.clear_seconds)
                            if z.queue and z.staff else None)
        self.alerts_file = alerts_file
        self.csv_path = out_dir / f"{source.name}_minutes.csv"
        self._csv = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=CSV_FIELDS, lineterminator="\n")
        self._writer.writeheader()
        self.rows: dict[int, MinuteRow] = {}  # minutes not yet written
        self.wall_at_t0: float | None = None
        self.last_t = 0.0
        self.crossings: list[dict] = []
        self.seconds: list[SecondSample] = []
        self.live = {"in": 0, "out": 0, "queue": 0, "staff_present": None, "second": None}
        self.staff_ids: set[int] = set()  # tracks judged to be staff in the latest frame

    # --------------------------------------------------------------------------------
    def _row_for(self, t: float) -> MinuteRow:
        minute = int(t // 60)
        if minute not in self.rows:
            self.rows[minute] = MinuteRow(minute, self.wall_at_t0 + minute * 60)
        return self.rows[minute]

    def _write_rows(self, upto_t: float | None) -> None:
        """Write finished minutes; zone seconds arrive about a second late, so wait 2 s."""
        for minute in sorted(self.rows):
            if upto_t is not None and (minute + 1) * 60 + 2 > upto_t:
                break
            self._writer.writerow(self.rows.pop(minute).as_dict(self.src.name))
        self._csv.flush()

    def _emit(self, event: str, alert: Alert) -> None:
        rec = {"event": event, "source": self.src.name, **asdict(alert),
               "start_wall": iso(self.wall_at_t0 + alert.start),
               "end_wall": iso(self.wall_at_t0 + alert.end) if alert.end is not None else None}
        self.alerts_file.write(json.dumps(rec) + "\n")
        self.alerts_file.flush()

    # --------------------------------------------------------------------------------
    def on_reconnect(self) -> None:
        for line in self.lines:
            line.reset()

    def update(self, t: float, wall: float, tracks: dict[int, tuple], alive: set[int],
               image=None) -> None:
        if self.wall_at_t0 is None:
            self.wall_at_t0 = wall - t
        self.last_t = t
        feet = {tid: ((x1 + x2) / 2, y2) for tid, (x1, y1, x2, y2) in tracks.items()}
        for line in self.lines:
            for c in line.update(feet, t) + line.forget(alive):
                row = self._row_for(c.t)
                if c.direction == "in":
                    row.n_in += 1
                    self.live["in"] += 1
                else:
                    row.n_out += 1
                    self.live["out"] += 1
                self.crossings.append({"t": round(c.t, 2), "track_id": c.track_id, "direction": c.direction})
        if self.zones is not None:
            if self.staff is not None and image is not None:
                self.staff_ids = self.staff.update(image, tracks, t)
                self.staff.forget(alive)
            dups = duplicate_boxes(tracks)
            skip = [tid in self.staff_ids or tid in dups for tid in feet]
            for sample in self.zones.update(list(feet.values()), t, skip):
                self._on_second(sample)
        self._row_for(t)  # a minute with no events still gets a row
        self._write_rows(t)

    def _on_second(self, s: SecondSample) -> None:
        self.seconds.append(s)
        row = self._row_for(s.second)
        row.queue.append(s.queue)
        empty = self.zones.staff_poly is not None and not s.staff_present
        waiting = empty and s.queue >= self.src.alerts.empty_counter.min_waiting
        row.empty += empty
        row.empty_waiting += waiting
        self.live.update(queue=s.queue, staff_present=s.staff_present, second=s.second)
        if self.queue_alert:
            for ev, a in self.queue_alert.update(s.second, s.queue >= self.src.alerts.queue.min_people, s.queue):
                self._emit(ev, a)
        if self.empty_alert:
            for ev, a in self.empty_alert.update(s.second, waiting, s.queue):
                self._emit(ev, a)

    def finish(self) -> None:
        if self.zones is not None:
            for sample in self.zones.finish():
                self._on_second(sample)
        for tracker in (self.queue_alert, self.empty_alert):
            if tracker:
                for ev, a in tracker.finish(math.floor(self.last_t) + 1):
                    self._emit(ev, a)
        self._write_rows(None)
        self._csv.close()

    def open_alerts(self) -> list[dict]:
        return [asdict(t.active) for t in (self.queue_alert, self.empty_alert) if t and t.active]
