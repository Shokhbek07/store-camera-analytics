"""Detector-only comparison of PyTorch and ONNX Runtime (E15): speed, and agreement with a reference.

Runs each variant through the service's own Detector on real frame pairs (one counter frame and one
door frame per batch, as the service batches two sources), so letterboxing and NMS are included.
The first variant listed is the reference the others are compared with.

Usage:
  uv run python scripts/compare_detectors.py [--pairs 40] [--variants a,b,...] [--out FILE]
  uv run python scripts/compare_detectors.py --ort-threads      # ONNX Runtime CPU speed by thread count
  uv run python scripts/compare_detectors.py --coreml-units     # CoreML compute units, model only
Needs the exports from scripts/export_onnx.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam.config import ModelConfig  # noqa: E402
from storecam.detect import Detector  # noqa: E402

STATIC = "models/yolo26s_384x640_b2.onnx"
DYNAMIC = "models/yolo26s_dynamic.onnx"


def neural_network_format(det: Detector) -> None:
    """Put back the CoreML set-up Ultralytics chooses (NeuralNetwork format), which Detector replaces."""
    det.model.predictor.model.backend.session = ort.InferenceSession(det.cfg.weights, providers=[
        ("CoreMLExecutionProvider", {"flags": "COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES"}), "CPUExecutionProvider"])


VARIANTS = {  # name: (weights, device, change after loading)
    "pytorch_mps": ("models/yolo26s.pt", "mps", None),
    "pytorch_cpu": ("models/yolo26s.pt", "cpu", None),
    "ort_cpu_dynamic": (DYNAMIC, "cpu", None),
    "ort_cpu_static_b2": (STATIC, "cpu", None),
    "ort_coreml_neuralnetwork_static_b2": (STATIC, "mps", neural_network_format),
    "ort_coreml_mlprogram_static_b2": (STATIC, "mps", None),  # what Detector sets up on the Mac GPU
}


def frames(path: str, n: int, step: int = 15) -> list[np.ndarray]:
    cap, out, i = cv2.VideoCapture(path), [], 0
    while len(out) < n and cap.grab():
        if i % step == 0:
            out.append(cap.retrieve()[1])
        i += 1
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, :2], b[None, :, :2]).transpose(2, 0, 1)
    x2, y2 = np.minimum(a[:, None, 2:], b[None, :, 2:]).transpose(2, 0, 1)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def agreement(ref: list, got: list) -> dict:
    """Reference boxes with a box of IoU >= 0.9 in the other run, and their mean score difference."""
    matched, dconf = 0, []
    for r, g in zip(ref, got, strict=True):
        if len(r.xyxy) and len(g.xyxy):
            m = iou(r.xyxy, g.xyxy)
            for i in range(len(r.xyxy)):
                j = int(m[i].argmax())
                if m[i, j] >= 0.9:
                    matched += 1
                    dconf.append(abs(float(r.conf[i]) - float(g.conf[j])))
    return {"ref_boxes": sum(len(r.xyxy) for r in ref), "boxes": sum(len(g.xyxy) for g in got),
            "matched_iou90": matched, "mean_abs_conf_diff": round(float(np.mean(dconf)), 4) if dconf else None}


def compare(n_pairs: int, names: list[str]) -> list[dict]:
    a, b = frames("data/meva/counter_1650.avi", n_pairs), frames("data/meva/door_0307_1720.avi", n_pairs)
    pairs = [[x, y] for x, y in zip(a, b, strict=True)]
    rows, ref = [], None
    for name in names:
        weights, device, change = VARIANTS[name]
        det = Detector(ModelConfig(weights=weights, device=device))
        if change:
            change(det)
        for p in pairs[:3]:  # warm-up
            det(p)
        out, t0 = [], time.perf_counter()
        for p in pairs:
            out.extend(det(p))
        ms = 1000 * (time.perf_counter() - t0) / len(pairs)
        ref = ref or out
        row = {"variant": name, "ms_per_batch_of_2": round(ms, 1), "fps": round(2000 / ms, 1), **agreement(ref, out)}
        if det.onnx:
            row["providers"] = det.model.predictor.model.backend.session.get_providers()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def timed(session: ort.InferenceSession, x: np.ndarray, n: int) -> float:
    for _ in range(3):
        session.run(None, {"images": x})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"images": x})
    return 1000 * (time.perf_counter() - t0) / n


def ort_threads() -> list[dict]:
    x = np.random.default_rng(0).random((2, 3, 384, 640), dtype=np.float32)
    rows = []
    for threads in (0, 2, 4, 6, 8, 10, 14):  # 0 is ONNX Runtime's default
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(STATIC, so, providers=["CPUExecutionProvider"])
        rows.append({"threads": threads or "default", "ms_per_batch_of_2": round(timed(s, x, 15), 1)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def coreml_units() -> list[dict]:
    x = np.random.default_rng(0).random((2, 3, 384, 640), dtype=np.float32)
    ref = ort.InferenceSession(STATIC, providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    rows = []
    for units in ("CPUAndGPU", "ALL", "CPUAndNeuralEngine"):
        t0 = time.perf_counter()
        s = ort.InferenceSession(STATIC, providers=[("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": units, "RequireStaticInputShapes": "1"}),
            "CPUExecutionProvider"])
        load = time.perf_counter() - t0
        ms = timed(s, x, 30)
        y = s.run(None, {"images": x})[0]
        diff = float(np.abs(ref[:, 4:].max(1) - y[:, 4:].max(1)).max())
        rows.append({"units": units, "load_s": round(load, 1), "ms_per_batch_of_2": round(ms, 1),
                     "max_score_diff_vs_cpu": round(diff, 4)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--ort-threads", action="store_true")
    ap.add_argument("--coreml-units", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.ort_threads:
        rows = ort_threads()
    elif args.coreml_units:
        rows = coreml_units()
    else:
        rows = compare(args.pairs, args.variants.split(","))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
