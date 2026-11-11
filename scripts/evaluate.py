"""Compare system output with the hand labels in labels/, clip by clip.

Runs the service offline (every sampled frame, no drops) on each evaluation clip as a single
source, using that source's lines, zones and thresholds from config.yaml, then compares:

- door: hand IN/OUT against system IN/OUT, plus event matching (same direction within 2 s);
- queue: the system's per-second queue count at each hand-labelled 10 s mark (mean absolute
  error) and staff presence agreement;
- alerts: expected alerts against system alerts of the same kind, matched by time overlap.

Usage:
    uv run python scripts/evaluate.py [--rerun] [--config config.yaml] [--out results/eval]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config  # noqa: E402

CLIPS = {
    "door": ["data/meva/door_0309_1010.avi", "data/meva/door_0307_1720.avi"],
    "counter": ["data/meva/counter_1650.avi", "data/meva/counter_1720.avi", "data/meva/counter_1735.avi"],
}
MATCH_S = 2.0
# Label files per labeller: labels/<mode>_<clip>.json by hand (Shokhbek), .claude.json by Claude.
LABELLERS = [("shokhbek", ".json"), ("claude", ".claude.json")]


def run_clip(cfg, source_name: str, video: str, out: Path, rerun: bool) -> Path:
    from storecam.service import Service

    if (out / "run_summary.json").exists() and not rerun:
        return out
    src = next(s for s in cfg.sources if s.name == source_name).model_copy(deep=True)
    src.url, src.realtime, src.loop = video, False, False
    one = cfg.model_copy(update={"sources": [src]})
    Service(one, out).run()
    return out


def match_events(hand: list[dict], system: list[dict]) -> tuple[int, int, int]:
    """Greedy one-to-one matching of crossings with the same direction within MATCH_S seconds."""
    used = set()
    tp = 0
    for h in sorted(hand, key=lambda e: e["t"]):
        best = None
        for j, s in enumerate(system):
            if j in used or s["direction"] != h["dir"]:
                continue
            dt = abs(s["t"] - h["t"])
            if dt <= MATCH_S and (best is None or dt < best[0]):
                best = (dt, j)
        if best:
            used.add(best[1])
            tp += 1
    return tp, len(system) - tp, len(hand) - tp


def overlap(a0, a1, b0, b1) -> bool:
    return min(a1, b1) > max(a0, b0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out_root = Path(args.out)
    report: dict = {"door": [], "queue": [], "alerts": []}
    for source_name, videos in CLIPS.items():
        for video in videos:
            stem = Path(video).stem
            run_dir = run_clip(cfg, source_name, video, out_root / stem, args.rerun)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            s = summary["sources"][source_name]
            duration = s["frames_processed"] / summary["process_fps_target"]

            for who, suffix in LABELLERS:
                compare(report, stem, who, suffix, s, run_dir, duration)
    agreement(report)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: [{kk: vv for kk, vv in r.items() if kk != "per_mark"} for r in v] for k, v in report.items()}, indent=1
        )
    )


def compare(report, stem, who, suffix, s, run_dir, duration) -> None:
    door_labels = Path(f"labels/door_{stem}{suffix}")
    if door_labels.exists():
        hand = json.loads(door_labels.read_text())["labels"]
        sys_in = sum(1 for c in s["crossings"] if c["direction"] == "in")
        sys_out = len(s["crossings"]) - sys_in
        h_in = sum(1 for e in hand if e["dir"] == "in")
        tp, fp, fn = match_events(hand, s["crossings"])
        report["door"].append(
            {
                "clip": stem,
                "labeller": who,
                "hand_in": h_in,
                "hand_out": len(hand) - h_in,
                "system_in": sys_in,
                "system_out": sys_out,
                "error_in": sys_in - h_in,
                "error_out": sys_out - (len(hand) - h_in),
                "matched": tp,
                "extra": fp,
                "missed": fn,
            }
        )

    queue_labels = Path(f"labels/queue_{stem}{suffix}")
    if queue_labels.exists():
        hand = json.loads(queue_labels.read_text())["labels"]
        per_s = {sec: (q, staff) for sec, q, staff in s["seconds"]}
        pairs = [(h, per_s[int(h["t"])]) for h in hand if int(h["t"]) in per_s]
        if pairs:
            errs = [sysq - h["queue"] for h, (sysq, _) in pairs]
            report["queue"].append(
                {
                    "clip": stem,
                    "labeller": who,
                    "marks": len(pairs),
                    "mae": round(sum(abs(e) for e in errs) / len(errs), 3),
                    "bias": round(sum(errs) / len(errs), 3),
                    "exact": sum(1 for e in errs if e == 0),
                    "hand_mean": round(sum(h["queue"] for h, _ in pairs) / len(pairs), 2),
                    # staff presence was not marked in every round (null)
                    "staff_marks": sum(1 for h, _ in pairs if h.get("staff_present") is not None),
                    "staff_agreement": sum(1 for h, (_, st) in pairs if st == h.get("staff_present")),
                    "per_mark": [
                        {
                            "t": h["t"],
                            "hand": h["queue"],
                            "system": q,
                            "hand_staff": h["staff_present"],
                            "system_staff": st,
                        }
                        for h, (q, st) in pairs
                    ],
                }
            )

    alert_labels = Path(f"labels/alerts_{stem}{suffix}")
    if alert_labels.exists():
        hand = json.loads(alert_labels.read_text())["labels"]
        system = [json.loads(line) for line in (run_dir / "alerts.jsonl").read_text().splitlines()]
        system = [a for a in system if a["event"] == "end"]
        for kind in ("queue", "empty_counter"):
            h = [a for a in hand if a["kind"] == kind]
            sy = [a for a in system if a["kind"] == kind]
            found = sum(
                1 for a in h if any(overlap(a["start"], a["end"] or duration, b["start"], b["end"]) for b in sy)
            )
            wrong = sum(
                1 for b in sy if not any(overlap(a["start"], a["end"] or duration, b["start"], b["end"]) for a in h)
            )
            report["alerts"].append(
                {
                    "clip": stem,
                    "labeller": who,
                    "kind": kind,
                    "expected": len(h),
                    "system": len(sy),
                    "found": found,
                    "missed": len(h) - found,
                    "wrong": wrong,
                    "system_intervals": [[b["start"], b["end"], b["peak"]] for b in sy],
                    "hand_intervals": [[a["start"], a["end"]] for a in h],
                }
            )


def agreement(report) -> None:
    """Queue counts where both labellers labelled the same mark of the same clip."""
    rows = {}
    for r in report["queue"]:
        rows.setdefault(r["clip"], {})[r["labeller"]] = {m["t"]: m["hand"] for m in r["per_mark"]}
    report["labeller_agreement"] = []
    for clip, by in rows.items():
        if len(by) < 2:
            continue
        a, b = by["shokhbek"], by["claude"]
        common = sorted(set(a) & set(b))
        diffs = [b[t] - a[t] for t in common]
        report["labeller_agreement"].append(
            {
                "clip": clip,
                "marks": len(common),
                "exact": sum(1 for d in diffs if d == 0),
                "mean_claude_minus_shokhbek": round(sum(diffs) / len(diffs), 3),
                "mean_abs_difference": round(sum(abs(d) for d in diffs) / len(diffs), 3),
            }
        )


if __name__ == "__main__":
    main()
