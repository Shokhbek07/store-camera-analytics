"""Person detection, batched across sources, and one ByteTrack tracker per source."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnxruntime
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from .config import ModelConfig, TrackerConfig

PERSON = 0


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """PyTorch weights (.pt) or an ONNX export (.onnx, run by ONNX Runtime; see D15)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.onnx = cfg.weights.endswith(".onnx")
        self.model = YOLO(cfg.weights, task="detect")
        self.half = self.device.startswith("cuda") and not self.onnx
        if self.onnx:
            self._setup_onnx()

    def _setup_onnx(self) -> None:
        """Ultralytics builds the ONNX Runtime session on the first predict. On the Mac GPU it asks CoreML
        for the older NeuralNetwork format, which computes in FP16 and was slower in E15; rebuild the
        session as an FP32 MLProgram on CPU and GPU instead. On CPU its own session is kept."""
        self([np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), np.uint8)])
        backend = self.model.predictor.model.backend
        if self.device == "mps":
            backend.session = onnxruntime.InferenceSession(self.cfg.weights, providers=[
                ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                                             "RequireStaticInputShapes": "1"}),
                "CPUExecutionProvider"])
        self.session = backend.session

    def __call__(self, images: list[np.ndarray]):
        """One batched forward pass; returns Ultralytics Boxes (NumPy) per image."""
        extra = {"half": True} if self.half else {}  # FP16 only on CUDA; MPS and CPU run FP32
        results = self.model.predict(images, imgsz=self.cfg.imgsz, conf=self.cfg.conf, classes=[PERSON],
                                     device=self.device, verbose=False, **extra)
        return [r.boxes.cpu().numpy() for r in results]

    def describe(self) -> dict:
        info = {"weights": self.cfg.weights, "imgsz": self.cfg.imgsz, "device": self.device,
                "precision": "FP16" if self.half else "FP32"}
        if self.onnx:
            backend = self.model.predictor.model.backend
            info["runtime"] = f"onnxruntime {onnxruntime.__version__}"
            info["execution_provider"] = self.session.get_providers()[0]
            if not backend.dynamic:  # a fixed-size export letterboxes to its own size, not cfg.imgsz
                info["imgsz"] = list(backend.imgsz)
                info["export_batch"] = backend.batch
            return info
        info["runtime"] = f"pytorch {torch.__version__}"
        info["parameters_millions"] = round(sum(p.numel() for p in self.model.model.parameters()) / 1e6, 2)
        if self.device == "mps":
            info["gpu_memory_mb"] = round(torch.mps.driver_allocated_memory() / 1e6)
        elif self.device.startswith("cuda"):
            info["gpu_memory_mb"] = round(torch.cuda.max_memory_reserved() / 1e6)
            info["gpu"] = torch.cuda.get_device_name()
        return info


class Tracker:
    """ByteTrack for one source. Returns {track_id: (x1, y1, x2, y2)} for confirmed tracks."""

    def __init__(self, cfg: TrackerConfig, process_fps: float):
        self.cfg, self.fps = cfg, process_fps
        self.reset()

    def reset(self) -> None:
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=self.cfg.track_high_thresh,
                               track_low_thresh=self.cfg.track_low_thresh,
                               new_track_thresh=self.cfg.new_track_thresh,
                               track_buffer=max(1, round(self.cfg.lost_seconds * self.fps)),
                               match_thresh=self.cfg.match_thresh, fuse_score=self.cfg.fuse_score)
        self.bt = BYTETracker(args)

    def update(self, boxes, image: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
        out = self.bt.update(boxes, image)
        return {int(r[4]): (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in out}

    def alive_ids(self) -> set[int]:
        return {t.track_id for t in self.bt.tracked_stracks + self.bt.lost_stracks}
