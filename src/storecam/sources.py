"""Video sources: one reader thread per source, with real-time pacing and reconnect.

Each reader decodes its own source and keeps only the newest sampled frame in a one-slot
mailbox. The detector takes whatever is newest, so a slow detector drops frames instead of
building a backlog, and a stalled or failed source never blocks the others.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Source

log = logging.getLogger("storecam.sources")


@dataclass
class Frame:
    source: str
    index: int  # sequence number within the source, across reconnects
    t: float  # seconds since the source first started (video time for files)
    wall: float  # wall-clock time.time() when the frame was read
    image: np.ndarray
    session: int  # increases on every reconnect, so trackers can reset


class Reader(threading.Thread):
    def __init__(self, source: Source, process_fps: float, stop: threading.Event,
                 opener=cv2.VideoCapture):
        super().__init__(name=f"reader-{source.name}", daemon=True)
        self.src = source
        self.process_fps = process_fps
        self.stop_event = stop
        self.opener = opener
        self._lock = threading.Condition()
        self._latest: Frame | None = None
        self.finished = False  # a file reached its end and does not loop
        self.connected = False
        self.reconnects = 0
        self.frames_read = 0
        self.frames_dropped = 0
        self.session = 0
        self._index = 0
        self._t_offset = 0.0  # video time carried across reopenings of a looped file
        self._first_start: float | None = None  # live streams: time since the first connection

    # Mailbox ---------------------------------------------------------------------------
    def _put(self, frame: Frame) -> None:
        with self._lock:
            if self._latest is not None:
                self.frames_dropped += 1
            self._latest = frame
            self._lock.notify_all()

    def take(self) -> Frame | None:
        with self._lock:
            f, self._latest = self._latest, None
            self._lock.notify_all()  # frees the reader in lockstep mode
            return f

    def wait_for_frame(self, timeout: float) -> bool:
        with self._lock:
            if self._latest is None:
                self._lock.wait(timeout)
            return self._latest is not None

    # Reading ---------------------------------------------------------------------------
    def _is_file(self) -> bool:
        return "://" not in self.src.url

    def run(self) -> None:
        delay = self.src.reconnect.initial_delay_s
        while not self.stop_event.is_set():
            cap = self.opener(self.src.url)
            if not cap.isOpened():
                log.warning("%s: cannot open %s; retrying in %.1fs", self.src.name, self.src.url, delay)
                cap.release()
                self.stop_event.wait(delay)
                delay = min(delay * 2, self.src.reconnect.max_delay_s)
                continue
            self.connected = True
            self.session += 1
            log.info("%s: opened (session %d)", self.src.name, self.session)
            ended_cleanly = self._read_loop(cap)
            cap.release()
            self.connected = False
            if ended_cleanly and self._is_file() and not self.src.loop:
                self.finished = True
                log.info("%s: end of file", self.src.name)
                return
            if ended_cleanly:
                delay = self.src.reconnect.initial_delay_s  # a looped file restarts at once
                continue
            self.reconnects += 1
            log.warning("%s: stream lost; reconnecting in %.1fs", self.src.name, delay)
            self.stop_event.wait(delay)
            delay = min(delay * 2, self.src.reconnect.max_delay_s)

    def _read_loop(self, cap) -> bool:
        """Read until the stream ends (True) or fails (False)."""
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps > 240:
            fps = 25.0
        stride = max(1, round(fps / self.process_fps))
        is_file = self._is_file()
        start_wall = time.monotonic()
        if self._first_start is None:
            self._first_start = start_wall
        n = 0
        last_t = 0.0
        while not self.stop_event.is_set():
            ok = cap.grab()
            if not ok:
                if is_file:
                    self._t_offset += last_t + 1.0 / fps
                return is_file  # end of a file is clean; a dead stream is a failure
            self.frames_read += 1
            t_media = n / fps if is_file else time.monotonic() - self._first_start
            last_t = t_media
            n += 1
            if (n - 1) % stride:
                continue
            if is_file and self.src.realtime:
                # Pace the file at its own frame rate, as a live camera would deliver it.
                ahead = t_media - (time.monotonic() - start_wall)
                if ahead > 0:
                    self.stop_event.wait(ahead)
            ok, image = cap.retrieve()
            if not ok:
                return False
            if not self.src.realtime:
                # Lockstep for offline evaluation: wait until the detector took the last frame.
                with self._lock:
                    while self._latest is not None and not self.stop_event.is_set():
                        self._lock.wait(0.1)
            t = self._t_offset + t_media if is_file else t_media
            self._put(Frame(self.src.name, self._index, t, time.time(), image, self.session))
            self._index += 1
        return True
