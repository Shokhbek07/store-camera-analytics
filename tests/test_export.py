"""The ONNX export file names (the export itself needs the weights, so it is checked in E15)."""

from pathlib import Path

from storecam.export import exports


def test_export_names_and_shapes():
    files = exports(Path("models/yolo26s.pt"), batch=2)
    assert files == {Path("models/yolo26s_dynamic.onnx"): {"dynamic": True, "imgsz": 640},
                     Path("models/yolo26s_384x640_b2.onnx"): {"imgsz": (384, 640), "batch": 2}}
