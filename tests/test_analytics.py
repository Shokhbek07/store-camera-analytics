import csv
import io

from storecam.analytics import SourceAnalytics
from storecam.config import DoorLine, Source

# Door line along y = 100, inside below it; the doorway region lies above it (outside).
DOOR = DoorLine(a=(0, 100), b=(200, 100), inside=(100, 200), band_px=20, min_frames=2,
                door_region=[(0, 0), (200, 0), (200, 90), (0, 90)])


def feet_box(x, y):
    return (x - 10, y - 50, x + 10, y)


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_exit_counted_late_lands_in_its_own_minute_and_only_once(tmp_path):
    # At 2 FPS, as in the CPU-only container, the tracker keeps a lost track for 15 frames
    # (7.5 s), so the doorway exit is counted long after the person was last seen at 59 s.
    a = SourceAnalytics(Source(name="door", url="x.mp4", door_lines=[DOOR]), tmp_path, io.StringIO())
    for t in (57.0, 57.5, 58.0):
        a.update(t, 1000 + t, {1: feet_box(100, 150)}, {1})  # inside, beyond the band
    a.update(59.0, 1059.0, {1: feet_box(100, 85)}, {1})  # last seen in the doorway
    t = 59.5
    while t < 66.5:
        a.update(t, 1000 + t, {}, {1})  # lost, not yet dropped
        t += 0.5
    a.update(66.5, 1066.5, {}, set())  # dropped: counted OUT at 59.0
    a.finish()
    got = rows(tmp_path / "door_minutes.csv")
    assert [r["minute"] for r in got] == ["0", "1"]
    assert (got[0]["out"], got[1]["out"]) == ("1", "0")
    assert a.crossings == [{"t": 59.0, "track_id": 1, "direction": "out"}]


def test_minute_rows_are_not_held_back_by_tracks_that_are_still_visible(tmp_path):
    a = SourceAnalytics(Source(name="door", url="x.mp4", door_lines=[DOOR]), tmp_path, io.StringIO())
    t = 50.0
    while t <= 63.0:
        a.update(t, 1000 + t, {1: feet_box(100, 150)}, {1})  # someone standing inside all along
        t += 0.5
    assert [r["minute"] for r in rows(tmp_path / "door_minutes.csv")] == ["0"]
