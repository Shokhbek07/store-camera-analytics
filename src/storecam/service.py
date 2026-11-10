"""The service: readers per source, one batched detector, analytics and outputs per source."""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import draw
from .analytics import SourceAnalytics
from .config import Config
from .detect import Detector, Tracker
from .sources import Reader

log = logging.getLogger("storecam")


class Service:
    def __init__(self, cfg: Config, out_dir: Path | None = None, save_video: bool = False,
                 http_port: int | None = None, max_seconds: float | None = None, http_host: str = "127.0.0.1"):
        self.cfg = cfg
        self.out = Path(out_dir or cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.readers = {s.name: Reader(s, cfg.process_fps, self.stop) for s in cfg.sources}
        self.trackers = {s.name: Tracker(cfg.tracker, cfg.process_fps) for s in cfg.sources}
        self.sessions = {s.name: 0 for s in cfg.sources}
        self.alerts_file = (self.out / "alerts.jsonl").open("w")
        self.analytics = {s.name: SourceAnalytics(s, self.out, self.alerts_file) for s in cfg.sources}
        self.processed = {s.name: 0 for s in cfg.sources}
        self.save_video = save_video
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.http_port = http_port
        self.http_host = http_host  # 0.0.0.0 only inside a container, published to the host's 127.0.0.1
        self.max_seconds = max_seconds
        self.detector: Detector | None = None
        self.batches = 0
        self.infer_s = 0.0

    # Live status for the optional HTTP endpoint ------------------------------------------
    def status(self) -> dict:
        return {name: {**a.live, "open_alerts": a.open_alerts(), "connected": self.readers[name].connected,
                       "reconnects": self.readers[name].reconnects, "frames_processed": self.processed[name]}
                for name, a in self.analytics.items()}

    def _serve_http(self) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(service.status(), indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer((self.http_host, self.http_port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
        log.info("live status on http://%s:%d/", self.http_host, self.http_port)

    # Main loop -----------------------------------------------------------------------
    def run(self) -> dict:
        self.detector = Detector(self.cfg.model)
        log.info("detector %s", self.detector.describe())
        if self.http_port:
            self._serve_http()
        for r in self.readers.values():
            r.start()
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        started = time.monotonic()
        wait = self.cfg.batch_wait_ms / 1000
        while not self.stop.is_set():
            if all(r.finished for r in self.readers.values()):
                break
            if self.max_seconds and time.monotonic() - started > self.max_seconds:
                break
            # Wait briefly for frames so sources that are close in time share one batch.
            deadline = time.monotonic() + wait
            frames = []
            for r in self.readers.values():
                if r.finished:
                    continue
                r.wait_for_frame(max(0.0, deadline - time.monotonic()))
            for r in self.readers.values():
                f = r.take()
                if f is not None:
                    frames.append(f)
            if not frames:
                time.sleep(0.002)
                continue
            t0 = time.perf_counter()
            boxes = self.detector([f.image for f in frames])
            self.infer_s += time.perf_counter() - t0
            self.batches += 1
            for f, b in zip(frames, boxes, strict=True):
                self._process(f, b)
        elapsed = time.monotonic() - started
        self.stop.set()
        return self._finish(elapsed)

    def _process(self, f, boxes) -> None:
        name = f.source
        if f.session != self.sessions[name]:
            if self.sessions[name]:
                log.info("%s: new session %d, resetting tracker and line state", name, f.session)
                self.trackers[name].reset()
                self.analytics[name].on_reconnect()
            self.sessions[name] = f.session
        tracker = self.trackers[name]
        tracks = tracker.update(boxes, f.image)
        self.analytics[name].update(f.t, f.wall, tracks, tracker.alive_ids(), f.image)
        self.processed[name] += 1
        if self.save_video:
            self._write_frame(f, tracks)

    def _write_frame(self, f, tracks) -> None:
        src = next(s for s in self.cfg.sources if s.name == f.source)
        img = draw.blur_heads(f.image.copy(), _boxes(tracks))
        img = draw.geometry(img, src)
        staff_ids = self.analytics[f.source].staff_ids
        for tid, (x1, y1, x2, y2) in tracks.items():
            colour, label = ((60, 200, 60), f"{tid} staff") if tid in staff_ids else ((0, 220, 255), str(tid))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.circle(img, (int((x1 + x2) / 2), int(y2)), 6, colour, -1)
        live = self.analytics[f.source].live
        text = f"t={f.t:6.1f}s in={live['in']} out={live['out']} queue={live['queue']} staff={live['staff_present']}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        w = self.writers.get(f.source)
        if w is None:
            h, wd = img.shape[:2]
            w = cv2.VideoWriter(str(self.out / f"{f.source}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                self.cfg.process_fps, (wd, h))
            self.writers[f.source] = w
        w.write(img)

    def _finish(self, elapsed: float) -> dict:
        for a in self.analytics.values():
            a.finish()
        self.alerts_file.close()
        for w in self.writers.values():
            w.release()
        for r in self.readers.values():
            r.join(timeout=2)
        import platform
        import subprocess
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip() or platform.processor()
        except (OSError, subprocess.SubprocessError):
            cpu = platform.processor()
        summary = {
            "detector": self.detector.describe(),
            "machine": {"cpu": cpu, "platform": platform.platform()},
            "process_fps_target": self.cfg.process_fps,
            "elapsed_s": round(elapsed, 2),
            "batches": self.batches,
            "mean_batch_size": round(sum(self.processed.values()) / max(1, self.batches), 2),
            "detector_ms_per_batch": round(1000 * self.infer_s / max(1, self.batches), 2),
            "total_fps": round(sum(self.processed.values()) / elapsed, 2) if elapsed else None,
            "sources": {
                name: {"frames_processed": n, "fps": round(n / elapsed, 2) if elapsed else None,
                       "frames_decoded": self.readers[name].frames_read,
                       "frames_dropped": self.readers[name].frames_dropped,
                       "reconnects": self.readers[name].reconnects,
                       "crossings": self.analytics[name].crossings,
                       "seconds": [[x.second, x.queue, x.staff_present] for x in self.analytics[name].seconds]}
                for name, n in self.processed.items()},
        }
        (self.out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def _boxes(tracks):
    import numpy as np
    return np.array(list(tracks.values()), dtype=float).reshape(-1, 4)
