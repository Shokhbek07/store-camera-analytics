"""Try rules for who counts in the queue, without rerunning detection (E16).

1. `--dump` runs the detector and tracker once over the three counter clips at 10 fps and saves
   every track box with its torso colour to runs/queue_rules/ (a few minutes).
2. Without it, the saved tracks are replayed through the service's own staff rule, per-second
   sampling and queue alert, once per candidate rule and queue zone, and each is scored against
   both labellers' queue marks and the expected alerts.
The replay reproduces the full evaluation closely but not exactly (a few seconds differ), so a
chosen rule is then checked with `scripts/evaluate.py --rerun`.

Shokhbek's marks are scored per labelling round: "first" chose the rules, "blind review" tests them (E22).

Usage: uv run python scripts/queue_rules.py [--dump] [--set e16|e19] [--out FILE]
  --set e16 (default): every rule on the old zone, then zones with and without duplicate boxes.
  --set e19: on the chosen zone with duplicate boxes left out, rules for people queuing (E19).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config  # noqa: E402
from storecam.detect import Detector, Tracker  # noqa: E402
from storecam.events import AlertTracker, ZoneSampler, duplicate_boxes, point_in_polygon  # noqa: E402
from storecam.staff import colour_distance, torso_lab  # noqa: E402

CLIPS = ["counter_1650", "counter_1720", "counter_1735"]
DUMP = Path("runs/queue_rules")
REST = [[1050, 645], [1395, 570], [1395, 1072], [345, 1072]]
ZONES = {  # the queue zone's top edge on the left, where the tables and chairs are
    "edge 675 (before E16)": [[345, 675], *REST],
    "edge 700": [[345, 700], *REST],
    "edge 720": [[345, 720], *REST],
    "edge 730/725, in front of the chairs (chosen)": [[345, 730], [620, 725], *REST],
    "edge 750/740": [[345, 750], [620, 740], *REST],
}


def dump(cfg) -> None:
    det = Detector(cfg.model)
    DUMP.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        cap, tracker = cv2.VideoCapture(f"data/meva/{clip}.avi"), Tracker(cfg.tracker, 10)
        step = round((cap.get(cv2.CAP_PROP_FPS) or 30) / 10)
        rows, i = [], 0
        while cap.grab():
            if i % step == 0:
                _, img = cap.retrieve()
                for tid, box in tracker.update(det([img])[0], img).items():
                    c = torso_lab(img, box)
                    rows.append([round(i / (step * 10), 2), tid, *(round(v) for v in box), *(c or (None,) * 3)])
            i += 1
        (DUMP / f"{clip}.json").write_text(json.dumps(rows))
        print(clip, len(rows), "track boxes")


# Candidate rules: each returns the indices of boxes to leave out of the queue count in one frame.
def time_in_zone(seconds):
    def rule(t, boxes, feet, zone, state):
        out = set()
        for i, ((tid, _), f) in enumerate(zip(boxes, feet, strict=True)):
            s = state.setdefault(tid, {})
            if point_in_polygon(f, zone):
                s.setdefault("enter", t)
                s["last_in"] = t
            elif "last_in" in s and t - s["last_in"] > 1.0:
                s.pop("enter", None)
            if "enter" in s and t - s["enter"] < seconds:
                out.add(i)
        return out
    return rule


def standing_still(max_heights_per_s, window=1.0):
    def rule(t, boxes, feet, zone, state):
        out = set()
        for i, ((tid, (_, y1, _, y2)), f) in enumerate(zip(boxes, feet, strict=True)):
            h = state.setdefault(tid, deque())
            h.append((t, f))
            while t - h[0][0] > window:
                h.popleft()
            t0, f0 = h[0]
            if len(h) >= 3 and math.dist(f, f0) / max(t - t0, 1e-6) / max(y2 - y1, 1) > max_heights_per_s:
                out.add(i)
        return out
    return rule


def duplicates(t, boxes, feet, zone, state):
    dup = duplicate_boxes(dict(boxes))
    return {i for i, (tid, _) in enumerate(boxes) if tid in dup}


def both(*rules):
    """Leave out a box if any of the rules leaves it out; each rule keeps its own state."""
    def rule(t, boxes, feet, zone, state):
        return set().union(*(r(t, boxes, feet, zone, state.setdefault(k, {})) for k, r in enumerate(rules)))
    return rule


RULES = {
    "none": None,
    "at least 2 s in the zone": time_in_zone(2),
    "at least 3 s in the zone": time_in_zone(3),
    "at least 5 s in the zone": time_in_zone(5),
    "standing still (< 0.3 box heights/s)": standing_still(0.3),
    "standing still (< 0.5 box heights/s)": standing_still(0.5),
    "duplicate boxes left out": duplicates,
}
CHOSEN = "edge 730/725, in front of the chairs (chosen)"
E19 = {  # on CHOSEN; duplicates are always left out, as in the service since E16
    "duplicates (current)": duplicates,
    "duplicates + at least 2 s in the zone": both(duplicates, time_in_zone(2)),
    "duplicates + at least 3 s in the zone": both(duplicates, time_in_zone(3)),
    "duplicates + at least 5 s in the zone": both(duplicates, time_in_zone(5)),
    "duplicates + at least 8 s in the zone": both(duplicates, time_in_zone(8)),
    "duplicates + standing still (< 0.3 box heights/s)": both(duplicates, standing_still(0.3)),
    "duplicates + standing still (< 0.5 box heights/s)": both(duplicates, standing_still(0.5)),
    "duplicates + 3 s in the zone + standing still (< 0.5)": both(duplicates, time_in_zone(3), standing_still(0.5)),
}


def replay(clip, zone, rule, src):
    frames = defaultdict(list)
    for r in json.loads((DUMP / f"{clip}.json").read_text()):
        frames[r[0]].append(r)
    sa, qa = src.staff_appearance, src.alerts.queue
    votes, zone_s, last_t, state = defaultdict(lambda: deque(maxlen=20)), defaultdict(float), {}, {}
    sampler, seconds, t = ZoneSampler(zone, src.zones.staff), [], 0.0
    for t in sorted(frames):
        boxes, feet, skip = [], [], []
        for _, tid, x1, y1, x2, y2, lab_l, lab_a, lab_b in frames[t]:
            f = ((x1 + x2) / 2, y2)
            zone_s[tid] += min(t - last_t.get(tid, t), 1.0) if point_in_polygon(f, src.zones.staff) else 0.0
            last_t[tid] = t
            if lab_l is not None:
                votes[tid].append(colour_distance((lab_l, lab_a, lab_b), sa.lab) <= sa.max_distance)
            v = votes[tid]
            skip.append(bool(v) and sum(v) * 2 > len(v) and zone_s[tid] >= sa.min_zone_seconds)
            boxes.append((tid, (x1, y1, x2, y2)))
            feet.append(f)
        left_out = rule(t, boxes, feet, zone, state) if rule else set()
        seconds += sampler.update(feet, t, [s or i in left_out for i, s in enumerate(skip)])
    seconds += sampler.finish()
    alert, alerts = AlertTracker("queue", qa.min_seconds, qa.clear_seconds), []
    for s in seconds:
        alerts += [a for ev, a in alert.update(s.second, s.queue >= qa.min_people, s.queue) if ev == "end"]
    alerts += [a for ev, a in alert.finish(math.floor(t) + 1) if ev == "end"]
    return {s.second: s.queue for s in seconds}, [(a.start, a.end, a.peak) for a in alerts]


def score(zone, rule, src) -> dict:
    out = {"mae": {}, "alerts_found": 0, "alerts_expected": 0, "alerts_wrong": 0, "intervals": {}}
    for clip in CLIPS:
        per_s, alerts = replay(clip, zone, rule, src)
        for who, suffix in (("shokhbek", ".json"), ("claude", ".claude.json")):
            path = Path(f"labels/queue_{clip}{suffix}")
            if path.exists():
                marks = [m for m in json.loads(path.read_text())["labels"] if int(m["t"]) in per_s]
                groups = defaultdict(list)  # Shokhbek's marks per labelling round (E22)
                for m in marks:
                    groups[f"{clip} {who}" + (f" {m['round']}" if "round" in m else "")].append(m)
                for key, ms in groups.items():
                    out["mae"][key] = round(sum(abs(per_s[int(m["t"])] - m["queue"]) for m in ms) / len(ms), 2)
        expected = [a for a in json.loads(Path(f"labels/alerts_{clip}.claude.json").read_text())["labels"]
                    if a["kind"] == "queue"]
        overlaps = lambda a, b: min(a["end"] or 300, b[1]) > max(a["start"], b[0])  # noqa: E731
        out["alerts_expected"] += len(expected)
        out["alerts_found"] += sum(1 for a in expected if any(overlaps(a, b) for b in alerts))
        out["alerts_wrong"] += sum(1 for b in alerts if not any(overlaps(a, b) for a in expected))
        out["intervals"][clip] = alerts
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--set", choices=["e16", "e19"], default="e16")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = config.load("config.yaml")
    src = next(s for s in cfg.sources if s.name == "counter")
    if args.dump or not (DUMP / f"{CLIPS[0]}.json").exists():
        dump(cfg)
    rows = []
    if args.set == "e19":
        trials = [(CHOSEN, r, E19[r]) for r in E19]
    else:
        trials = [(z, r, RULES[r]) for r in RULES for z in list(ZONES)[:1]]
        trials += [(z, r, RULES[r]) for z in list(ZONES)[1:] for r in ("none", "duplicate boxes left out")]
    for zname, rname, rule in trials:
        res = score(ZONES[zname], rule, src)
        rows.append({"zone": zname, "rule": rname, **res})
        mae = ", ".join(f"{k} {v}" for k, v in res["mae"].items())
        print(f"{zname} | {rname} | MAE {mae} | alerts {res['alerts_found']}/{res['alerts_expected']} found, "
              f"{res['alerts_wrong']} wrong", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
