"""Speed measurements for the README (E11).

1. Real time: the two configured sources at their own frame rate for 120 s; per-stream and total
   FPS against the 10 FPS target, dropped frames.
2. Throughput: 1, 2 and 4 sources processed as fast as possible for 40 s each, on the GPU (MPS)
   and on CPU, to show how far one machine is from its limit.

Usage: uv run python scripts/benchmark.py [--weights W] [--devices mps,cpu] [--out results/benchmark.json]
E15 runs it once with the PyTorch weights and once with an ONNX export (ONNX Runtime).
"""

from __future__ import annotations

import argparse
import json
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
    summary = Service(cfg, out, max_seconds=seconds).run()
    return {
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="override model.weights, e.g. an .onnx export")
    ap.add_argument("--devices", default="mps,cpu", help="devices for the throughput runs")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    base = config.load("config.yaml")
    tag = Path(args.weights).stem if args.weights else "default"
    rows = [run(base, 2, "auto", True, 120, Path(f"runs/bench/{tag}/realtime_2"), args.weights)]
    print(json.dumps(rows[-1]))
    for device in args.devices.split(","):
        for n in (1, 2, 4):
            rows.append(run(base, n, device, False, 40, Path(f"runs/bench/{tag}/{device}_{n}"), args.weights))
            print(json.dumps(rows[-1]))
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
