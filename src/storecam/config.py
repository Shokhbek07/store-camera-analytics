"""The one config file: sources, door lines, zones, alert thresholds, model and output."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(Strict):
    weights: str = "models/yolo26s.pt"
    imgsz: int = 640
    conf: float = Field(0.25, ge=0, le=1)
    device: str = "auto"  # auto picks cuda, then mps, then cpu


class TrackerConfig(Strict):
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    match_thresh: float = 0.8
    lost_seconds: float = Field(1.5, gt=0)  # keep a lost track this long before dropping it
    fuse_score: bool = True


class DoorLine(Strict):
    name: str = "door"
    a: Point
    b: Point
    inside: Point  # any point on the inside of the door; sets the IN direction
    band_px: float = Field(20, ge=0)  # dead band on each side of the line
    min_frames: int = Field(2, ge=1)  # frames beyond the band before a side counts
    # Optional doorway polygon on the outside of the line: tracks born there start outside, and
    # tracks that vanish there without having been counted count as OUT (E9).
    door_region: list[Point] | None = None


class Zones(Strict):
    queue: list[Point] | None = None
    staff: list[Point] | None = None
    # Count a person in the queue only after this long in the queue zone, so people walking
    # through are left out (E19); 0 counts everyone in the zone.
    queue_min_dwell_s: float = Field(0, ge=0)

    @model_validator(mode="after")
    def _polygons(self):
        for name in ("queue", "staff"):
            poly = getattr(self, name)
            if poly is not None and len(poly) < 3:
                raise ValueError(f"zone {name} needs at least 3 points")
        return self


class QueueAlert(Strict):
    min_people: int = Field(3, ge=1)
    min_seconds: float = Field(10, gt=0)
    clear_seconds: float = Field(5, gt=0)


class EmptyCounterAlert(Strict):
    min_seconds: float = Field(10, gt=0)
    clear_seconds: float = Field(3, gt=0)
    min_waiting: int = Field(1, ge=1)


class Alerts(Strict):
    queue: QueueAlert = QueueAlert()
    empty_counter: EmptyCounterAlert = EmptyCounterAlert()


class Reconnect(Strict):
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0


class StaffAppearance(Strict):
    """Uniform colour that marks staff; measure it with scripts/calibrate_staff.py."""
    lab: tuple[float, float, float]  # OpenCV Lab, 0..255 per channel
    max_distance: float = Field(12, gt=0)
    min_zone_seconds: float = Field(1.0, ge=0)  # time a track must have spent in the staff zone


class Source(Strict):
    name: str
    url: str  # file path, rtsp:// or http(s):// URL
    realtime: bool = True  # pace files at their own frame rate, dropping frames if behind
    loop: bool = False  # restart a file at its end, to simulate a live camera
    door_lines: list[DoorLine] = []
    zones: Zones = Zones()
    alerts: Alerts = Alerts()
    staff_appearance: StaffAppearance | None = None  # if set, people in the uniform are not counted in the queue
    reconnect: Reconnect = Reconnect()


class Config(Strict):
    process_fps: float = Field(10, gt=0)  # frames analysed per second per source
    batch_wait_ms: float = 20  # how long the detector waits to fill a batch across sources
    model: ModelConfig = ModelConfig()
    tracker: TrackerConfig = TrackerConfig()
    output_dir: str = "runs/out"
    sources: list[Source]

    @model_validator(mode="after")
    def _unique_names(self):
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        return self


def load(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
