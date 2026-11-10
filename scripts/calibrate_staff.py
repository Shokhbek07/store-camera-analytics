"""Measure the staff uniform colour for `staff_appearance` in config.yaml.

Detects people once per second over a clip, keeps those standing in the source's staff zone, and
prints the median torso colour, plus how many staff-zone and queue-zone people it would match.
It assumes the staff zone mostly holds staff, so check the printed match rates before using it.

Usage: uv run python scripts/calibrate_staff.py SOURCE_NAME CLIP [--max-distance 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config  # noqa: E402
from storecam.detect import Detector  # noqa: E402
from storecam.events import point_in_polygon  # noqa: E402
from storecam.staff import colour_distance, torso_lab  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("clip")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-distance", type=float, default=12)
    args = p.parse_args()
    cfg = config.load(args.config)
    src = next(s for s in cfg.sources if s.name == args.source)
    if not (src.zones.staff and src.zones.queue):
        sys.exit(f"source {args.source} needs both a staff and a queue zone")
    det = Detector(cfg.model)
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    colours: dict[str, list] = {"staff": [], "queue": []}
    i = 0
    while cap.grab():
        if i % round(fps) == 0:
            _, img = cap.retrieve()
            for x1, y1, x2, y2 in det([img])[0].xyxy:
                foot = ((x1 + x2) / 2, y2)
                zone = ("staff" if point_in_polygon(foot, src.zones.staff)
                        else "queue" if point_in_polygon(foot, src.zones.queue) else None)
                c = torso_lab(img, (x1, y1, x2, y2)) if zone else None
                if c is not None:
                    colours[zone].append(c)
        i += 1
    if not colours["staff"]:
        sys.exit("nobody was detected in the staff zone")
    ref = tuple(round(float(v)) for v in np.median(np.array(colours["staff"]), axis=0))
    print(f"people seen once per second: {len(colours['staff'])} in the staff zone, "
          f"{len(colours['queue'])} in the queue zone")
    for zone, cs in colours.items():
        hit = sum(colour_distance(c, ref) <= args.max_distance for c in cs)
        print(f"{zone} zone: {hit / max(1, len(cs)):.0%} match the uniform colour")
    print("\nstaff_appearance:")
    print(f"  lab: [{ref[0]}, {ref[1]}, {ref[2]}]")
    print(f"  max_distance: {args.max_distance:g}")


if __name__ == "__main__":
    main()
