from storecam.events import AlertTracker, LineCounter, ZoneSampler, point_in_polygon

# A horizontal door line from (0, 100) to (200, 100); inside is below it (larger y).
LINE = dict(a=(0, 100), b=(200, 100), inside=(100, 150), band=10, min_frames=2)


def run(counter, ys, x=100.0, tid=1):
    events = []
    for i, y in enumerate(ys):
        events += counter.update({tid: (x, y)}, t=i)
    return [e.direction for e in events]


def test_walk_in_counts_once():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 80, 90, 100, 110, 120, 130, 140]) == ["in"]
    assert (c.n_in, c.n_out) == (1, 0)


def test_walk_out_counts_once():
    c = LineCounter(**LINE)
    assert run(c, [140, 130, 120, 110, 100, 90, 80, 70]) == ["out"]


def test_stop_on_line_is_not_counted():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 80, 95, 100, 104, 98, 102, 100, 99]) == []


def test_jitter_across_centre_is_not_counted():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 92, 108, 93, 107, 94, 106, 95]) == []


def test_turn_back_is_not_counted():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 85, 100, 108, 100, 85, 70, 60]) == []


def test_single_frame_glitch_does_not_flip_side():
    c = LineCounter(**LINE)
    # One frame beyond the band on the far side is not enough with min_frames=2.
    assert run(c, [60, 70, 80, 115, 80, 70]) == []


def test_in_then_out_counts_both():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 80, 120, 130, 120, 80, 70]) == ["in", "out"]


def test_walk_along_line_is_not_counted():
    c = LineCounter(**LINE)
    events = []
    for i, x in enumerate(range(-20, 240, 10)):
        y = 100 + (8 if i % 2 else -8)  # weaving inside the band
        events += c.update({1: (x, y)}, t=i)
    assert events == []


def test_crossing_outside_line_extent_is_not_counted():
    c = LineCounter(**LINE)
    assert run(c, [60, 70, 120, 130], x=400) == []


def test_track_born_inside_band_needs_a_real_crossing():
    c = LineCounter(**LINE)
    # Appears on the line (for example after an ID switch), then walks inside: no previous side.
    assert run(c, [100, 105, 120, 130]) == []


def test_tracks_are_independent():
    c = LineCounter(**LINE)
    ev = []
    for i, (y1, y2) in enumerate(zip([60, 80, 120, 130], [140, 120, 80, 70], strict=True)):
        ev += c.update({1: (50, y1), 2: (150, y2)}, t=i)
    assert sorted(e.direction for e in ev) == ["in", "out"]


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)


def test_zone_sampler_median_per_second():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    z = ZoneSampler(queue=sq, staff=[(20, 0), (30, 0), (30, 10), (20, 10)])
    out = []
    # Second 0: queue counts 2, 2, 0 (one missed frame) -> median 2; staff in 2 of 3 frames.
    out += z.update([(1, 1), (2, 2), (25, 5)], 0.0)
    out += z.update([(1, 1), (2, 2), (25, 5)], 0.33)
    out += z.update([], 0.66)
    out += z.update([(1, 1)], 1.0)  # closes second 0
    out += z.finish()
    assert [(s.second, s.queue, s.staff_present) for s in out] == [(0, 2, True), (1, 1, False)]


def test_alert_starts_once_and_bridges_short_gaps():
    a = AlertTracker("queue", min_seconds=3, clear_seconds=3)
    cond = [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    vals = [3, 4, 5, 3, 2, 6, 3, 2, 2, 3, 0, 0, 0, 0]
    events = []
    for t, (c, v) in enumerate(zip(cond, vals, strict=True)):
        events += a.update(t, bool(c), v)
    assert [k for k, _ in events] == ["start", "end"]
    alert = events[-1][1]
    assert (alert.start, alert.raised, alert.end, alert.peak) == (0, 2, 10, 6)


def test_alert_needs_full_duration():
    a = AlertTracker("queue", min_seconds=5, clear_seconds=2)
    events = []
    for t, c in enumerate([1, 1, 1, 1, 0, 1, 1, 1, 1, 0]):
        events += a.update(t, bool(c), 3)
    assert events == []


def test_open_alert_closed_at_stream_end():
    a = AlertTracker("empty_counter", min_seconds=2, clear_seconds=5)
    for t in range(4):
        a.update(t, True, 1)
    ((kind, alert),) = a.finish(4)
    assert (kind, alert.end, alert.ended_by) == ("end", 4, "stream_end")
