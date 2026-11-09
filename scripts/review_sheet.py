"""Build a review sheet so Shokhbek can check Claude's queue labels.

For every 10 s mark where Claude's label differs from Shokhbek's or from the system, this writes
the frame with the zones drawn to runs/review/<clip>_<t>.jpg and a table to runs/review/review.md
with an empty column for the reviewer's count. Frames stay local (runs/ is git-ignored).

Usage: uv run python scripts/review_sheet.py   (after scripts/evaluate.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config  # noqa: E402
from storecam.draw import geometry  # noqa: E402

OUT = Path("runs/review")
CLIPS = ["counter_1650", "counter_1720", "counter_1735"]


def labels(path: Path) -> dict[int, dict]:
    return {m["t"]: m for m in json.loads(path.read_text())["labels"]} if path.exists() else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path("results/eval/report.json").read_text())
    system = {(q["clip"], m["t"]): m["system"] for q in report["queue"] for m in q["per_mark"]}
    source = next(s for s in config.load("config.yaml").sources if s.name == "counter")
    rows = ["| Clip | Time | Shokhbek | Claude | System | Claude's note | Your count |",
            "|---|---|---|---|---|---|---|"]
    for clip in CLIPS:
        mine = labels(Path(f"labels/queue_{clip}.json"))
        claude = labels(Path(f"labels/queue_{clip}.claude.json"))
        cap = cv2.VideoCapture(f"data/meva/{clip}.avi")
        for t, m in sorted(claude.items()):
            sk = mine.get(t, {}).get("queue")
            sy = system.get((clip, t))
            if (sk is None or sk == m["queue"]) and sy == m["queue"]:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(OUT / f"{clip}_{t:03d}.jpg"), geometry(frame, source))
            note = "uncertain" if m.get("uncertain") else ""
            shown = "" if sk is None else sk
            rows.append(f"| {clip} | {t // 60}:{t % 60:02d} | {shown} | {m['queue']} | {sy} | {note} | |")
        cap.release()
    (OUT / "review.md").write_text(
        "# Queue labels to review\n\n"
        "Rule used by Claude: count every person whose feet are inside the queue outline.\n"
        "For each row open `<clip>_<seconds>.jpg` in this folder and write how many people you see queuing.\n\n"
        + "\n".join(rows) + "\n")
    print(f"{len(rows) - 2} marks to review in {OUT}/review.md")


if __name__ == "__main__":
    main()
