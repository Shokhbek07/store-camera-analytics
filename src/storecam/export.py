"""Export the detector to ONNX for ONNX Runtime (D15, E15).

Two files are written next to the PyTorch weights:
- <stem>_dynamic.onnx: any batch and image size, so frames are letterboxed exactly as in PyTorch;
  the Docker image runs this one on CPU.
- <stem>_384x640_b<N>.onnx: a fixed 384x640 input (a 16:9 frame at 640 wide, padded as PyTorch
  pads it) and a fixed batch of N, which the CoreML execution provider on a Mac needs.
Both keep the same head and outside NMS as the PyTorch run, so detections can match exactly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def exports(weights: Path, batch: int) -> dict[Path, dict]:
    """Output path and Ultralytics export arguments for each file."""
    return {weights.with_name(f"{weights.stem}_dynamic.onnx"): {"dynamic": True, "imgsz": 640},
            weights.with_name(f"{weights.stem}_384x640_b{batch}.onnx"): {"imgsz": (384, 640), "batch": batch}}


def export_onnx(weights: Path, batch: int = 2, if_missing: bool = False) -> list[Path]:
    from ultralytics import YOLO

    written = []
    for out, kwargs in exports(weights, batch).items():
        if if_missing and out.exists():
            print(f"exists  {out}")
            continue
        with tempfile.TemporaryDirectory() as tmp:  # Ultralytics writes the .onnx next to its input
            src = Path(tmp) / weights.name
            shutil.copy(weights, src)
            onnx = YOLO(str(src)).export(format="onnx", simplify=True, **kwargs)
            shutil.move(onnx, out)
        print(f"wrote   {out}")
        written.append(out)
    return written
