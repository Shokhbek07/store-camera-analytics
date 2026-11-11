import csv
import io

from storecam.analytics import SourceAnalytics
from storecam.config import DoorLine, Source, Zones

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


def test_a_reconnect_does_not_hand_queue_time_to_a_new_track_with_a_reused_id(tmp_path):
    # After a reconnect the tracker starts again at ID 1, so ID 1 is a different person.
    zones = Zones(queue=[(0, 0), (200, 0), (200, 200), (0, 200)], queue_min_dwell_s=2)
    a = SourceAnalytics(Source(name="counter", url="x.mp4", zones=zones), tmp_path, io.StringIO())
    t = 0.0
    while t < 10.0:
        a.update(t, 1000 + t, {1: feet_box(100, 100)}, {1})  # in the queue from 0 s
        t += 0.5
    a.on_reconnect()
    for t in (20.0, 20.5, 21.0):  # a new person, also ID 1, just arrived in the zone
        a.update(t, 1000 + t, {1: feet_box(100, 100)}, {1})
    a.finish()
    by_second = {s.second: s.queue for s in a.seconds}
    assert by_second[9] == 1
    assert by_second[20] == 0  # in the zone for under 2 s


def test_duplicate_box_threshold_comes_from_the_config(tmp_path):
    full, upper = (60, 0, 140, 190), (105, 10, 155, 100)  # upper box 70% inside the full one

    def queue_at_0(inside):
        zones = Zones(queue=[(0, 0), (200, 0), (200, 200), (0, 200)], duplicate_inside=inside)
        a = SourceAnalytics(Source(name="counter", url="x.mp4", zones=zones), tmp_path, io.StringIO())
        a.update(0.0, 1000.0, {1: full, 2: upper}, {1, 2})
        a.finish()
        return a.seconds[0].queue

    assert queue_at_0(0.85) == 2
    assert queue_at_0(0.6) == 1
