"""Export the detector to ONNX for ONNX Runtime (D15, E15).

Writes two files next to the PyTorch weights:
- yolo26s_dynamic.onnx: any batch and image size, so frames are letterboxed exactly as in PyTorch.
- yolo26s_384x640_b{N}.onnx: a fixed 384x640 input (a 16:9 frame at 640 wide, as PyTorch pads it)
  and a fixed batch of N, which the CoreML execution provider needs to run the whole graph.
Both keep the same head and outside NMS as the PyTorch run, so detections should match closely.

Usage: uv run --group export python scripts/export_onnx.py [--weights models/yolo26s.pt] [--batch 2]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


def export(weights: Path, out: Path, **kwargs) -> None:
    with tempfile.TemporaryDirectory() as tmp:  # Ultralytics writes the .onnx next to its input
        src = Path(tmp) / weights.name
        shutil.copy(weights, src)
        onnx = YOLO(str(src)).export(format="onnx", simplify=True, **kwargs)
        shutil.move(onnx, out)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/yolo26s.pt")
    p.add_argument("--batch", type=int, default=2, help="fixed batch for the static export (the number of streams)")
    args = p.parse_args()
    w = Path(args.weights)
    export(w, w.with_name(f"{w.stem}_dynamic.onnx"), dynamic=True, imgsz=640)
    export(w, w.with_name(f"{w.stem}_384x640_b{args.batch}.onnx"), imgsz=(384, 640), batch=args.batch)


if __name__ == "__main__":
    main()
