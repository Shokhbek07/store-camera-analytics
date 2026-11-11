"""Speed measurements for the README (E11).

1. Real time: the two configured sources at their own frame rate for 120 s; per-stream and total
   FPS against the 10 FPS target, dropped frames.
2. Throughput: 1, 2 and 4 sources processed as fast as possible for 40 s each, on the GPU (MPS)
   and on CPU, to show how far one machine is from its limit.

Usage: uv run python scripts/benchmark.py [--weights W] [--devices mps,cpu] [--out results/benchmark.json]

Comparing two detectors on a machine whose load comes and goes (E15):
  uv run python scripts/benchmark.py --compare models/yolo26s.pt,models/yolo26s_384x640_b2.onnx \
      --repeats 3 --realtime-seconds 60 --seconds 20 --out results/benchmark_compare.json
runs every configuration with each set of weights in turn, alternating which goes first, so both
see the same background load; it reports the median of the repeats. Every row records the
1-minute load average at its start and end.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config  # noqa: E402
from storecam.service import Service  # noqa: E402

CLIPS = ["data/meva/counter_1650.avi", "data/meva/door_0307_1720.avi", "data/meva/counter_1720.avi",
         "data/meva/door_0309_1010.avi"]


def run(base, n: int, device: str, realtime: bool, seconds: float, out: Path, weights: str | None = None) -> dict:
    cfg = base.model_copy(deep=True)
    cfg.model.device = device
    if weights:
        cfg.model.weights = weights
    sources = []
    templates = {s.name: s for s in cfg.sources}
    for i in range(n):
        clip = CLIPS[i % len(CLIPS)]
        src = templates["door" if "door" in clip else "counter"].model_copy(deep=True)
        src.name, src.url, src.realtime = f"s{i}_{Path(clip).stem}", clip, realtime
        sources.append(src)
    cfg.sources = sources
    load_start = os.getloadavg()[0]
    summary = Service(cfg, out, max_seconds=seconds).run()
    return {
        "load_1min_start": round(load_start, 1), "load_1min_end": round(os.getloadavg()[0], 1),
        "streams": n, "device": summary["detector"]["device"], "realtime": realtime,
        "weights": summary["detector"]["weights"], "runtime": summary["detector"].get("runtime"),
        "execution_provider": summary["detector"].get("execution_provider"),
        "imgsz": summary["detector"]["imgsz"], "precision": summary["detector"]["precision"],
        "gpu_memory_mb": summary["detector"].get("gpu_memory_mb"),
        "total_fps": summary["total_fps"], "mean_batch": summary["mean_batch_size"],
        "detector_ms_per_batch": summary["detector_ms_per_batch"],
        "per_stream_fps": [s["fps"] for s in summary["sources"].values()],
        "dropped": [s["frames_dropped"] for s in summary["sources"].values()],
    }


def configurations(devices: str, realtime_seconds: float, seconds: float) -> list[tuple]:
    """(name, streams, device, realtime, seconds): two real-time streams, then 1, 2 and 4 as fast as possible."""
    rows = [("realtime_2", 2, "auto", True, realtime_seconds)]
    return rows + [(f"{d}_{n}", n, d, False, seconds) for d in devices.split(",") for n in (1, 2, 4)]


def compare(base, weights: list[str], repeats: int, configs: list[tuple]) -> dict:
    rows = []
    for name, n, device, realtime, seconds in configs:
        for r in range(repeats):
            for w in weights if r % 2 == 0 else weights[::-1]:  # alternate which runs first
                row = run(base, n, device, realtime, seconds, Path(f"runs/bench/compare/{Path(w).stem}/{name}_r{r}"), w)
                rows.append({"config": name, "repeat": r, **row})
                print(json.dumps(rows[-1]), flush=True)
    medians = []
    for name, *_ in configs:
        for w in weights:
            got = [x for x in rows if x["config"] == name and x["weights"] == w]
            medians.append({"config": name, "weights": w, "runtime": got[0]["runtime"], "repeats": len(got),
                            "total_fps": statistics.median(x["total_fps"] for x in got),
                            "slowest_stream_fps": statistics.median(min(x["per_stream_fps"]) for x in got),
                            "detector_ms_per_batch": statistics.median(x["detector_ms_per_batch"] for x in got),
                            "load_1min": [x["load_1min_start"] for x in got]})
            print(json.dumps(medians[-1]), flush=True)
    return {"method": f"each configuration run {repeats} times per weights, alternating order; medians below",
            "medians": medians, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="override model.weights, e.g. an .onnx export")
    ap.add_argument("--compare", help="comma-separated weights to compare in alternation, e.g. a.pt,b.onnx")
    ap.add_argument("--repeats", type=int, default=3, help="with --compare: runs per configuration and weights")
    ap.add_argument("--realtime-seconds", type=float, default=120)
    ap.add_argument("--seconds", type=float, default=40, help="length of each as-fast-as-possible run")
    ap.add_argument("--devices", default="mps,cpu", help="devices for the throughput runs")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    base = config.load("config.yaml")
    configs = configurations(args.devices, args.realtime_seconds, args.seconds)
    if args.compare:
        result = compare(base, args.compare.split(","), args.repeats, configs)
    else:
        tag = Path(args.weights).stem if args.weights else "default"
        result = []
        for name, n, device, realtime, seconds in configs:
            result.append(run(base, n, device, realtime, seconds, Path(f"runs/bench/{tag}/{name}"), args.weights))
            print(json.dumps(result[-1]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
