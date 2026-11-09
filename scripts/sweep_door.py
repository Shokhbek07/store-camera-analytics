"""Door counting variants against the hand counts (E9).

Each variant overrides a few config values, runs both door clips offline, and reports count
errors and event matching at 2 s and 3 s. The same two clips are used to choose and to measure,
so the chosen variant's numbers are optimistic; the report says so.

Usage: uv run python scripts/sweep_door.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate  # noqa: E402

from storecam import config  # noqa: E402

CLIPS = ["door_0309_1010", "door_0307_1720"]
# The door threshold, as an alternative to the inner edge of the mats (E9).
THRESHOLD = {"a": (880, 680), "b": (1450, 866)}
# The doorway: from 120 px beyond the threshold to 60 px in front of it, outside the mats line.
DOORWAY = [(880, 560), (1450, 746), (1450, 926), (880, 740)]
VARIANTS = {
    "baseline": {},
    "band40": {"band_px": 40},
    "frames4": {"min_frames": 4},
    "band40_frames4": {"band_px": 40, "min_frames": 4},
    "imgsz960": {"imgsz": 960},
    "imgsz960_band40_frames4": {"imgsz": 960, "band_px": 40, "min_frames": 4},
    "fps15_band40_frames4": {"process_fps": 15, "band_px": 40, "min_frames": 6},
    "threshold": {"line": THRESHOLD},
    "threshold_band10": {"line": THRESHOLD, "band_px": 10},
    "threshold_frames3": {"line": THRESHOLD, "min_frames": 3},
    "threshold_imgsz960": {"line": THRESHOLD, "imgsz": 960},
    "doorway": {"door_region": DOORWAY},
    "doorway_band40_frames4": {"door_region": DOORWAY, "band_px": 40, "min_frames": 4},
    "doorway_frames4": {"door_region": DOORWAY, "min_frames": 4},
    "doorway_imgsz960": {"door_region": DOORWAY, "imgsz": 960},
}


def variant_config(base, v: dict):
    cfg = base.model_copy(deep=True)
    if "imgsz" in v:
        cfg.model.imgsz = v["imgsz"]
    if "process_fps" in v:
        cfg.process_fps = v["process_fps"]
    door = next(s for s in cfg.sources if s.name == "door")
    for line in door.door_lines:
        if "line" in v:
            line.a, line.b = v["line"]["a"], v["line"]["b"]
        line.door_region = v.get("door_region", line.door_region)
        line.band_px = v.get("band_px", line.band_px)
        line.min_frames = v.get("min_frames", line.min_frames)
    return cfg


def main() -> None:
    base = config.load("config.yaml")
    rows = []
    for name, v in VARIANTS.items():
        cfg = variant_config(base, v)
        row = {"variant": name, **{k: x for k, x in v.items() if k not in ("line", "door_region")},
               "line": "threshold" if "line" in v else "mats", "door_region": "door_region" in v}
        abs_err = 0
        for stem in CLIPS:
            run_dir = evaluate.run_clip(cfg, "door", f"data/meva/{stem}.avi", Path("runs/sweep") / name / stem, False)
            s = json.loads((run_dir / "run_summary.json").read_text())["sources"]["door"]
            hand = json.loads(Path(f"labels/door_{stem}.json").read_text())["labels"]
            sys_in = sum(1 for c in s["crossings"] if c["direction"] == "in")
            sys_out = len(s["crossings"]) - sys_in
            h_in = sum(1 for e in hand if e["dir"] == "in")
            h_out = len(hand) - h_in
            abs_err += abs(sys_in - h_in) + abs(sys_out - h_out)
            row[stem] = f"{sys_in}/{sys_out} vs {h_in}/{h_out}"
            for window in (2.0, 3.0):
                evaluate.MATCH_S = window
                tp, fp, fn = evaluate.match_events(hand, s["crossings"])
                row[f"{stem} matched@{window:g}s"] = f"{tp} (+{fp} extra, -{fn} missed)"
        row["abs_count_error_total"] = abs_err
        rows.append(row)
        print(json.dumps(row))
    Path("results/sweep_door.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
