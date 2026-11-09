"""Reader behaviour with a fake capture: reconnect after failure, clean end of file, backoff."""

import threading
import time

import numpy as np

from storecam.config import Source
from storecam.sources import Reader


class FakeCapture:
    """Delivers `frames` frames, then fails (stream) or ends (file); `opens` counts constructions."""

    opens = 0

    def __init__(self, url, frames=6, fail_opens=0):
        FakeCapture.opens += 1
        self.ok_open = FakeCapture.opens > fail_opens
        self.left = frames

    def isOpened(self):
        return self.ok_open

    def get(self, prop):
        return 30.0

    def grab(self):
        if self.left <= 0:
            return False
        self.left -= 1
        return True

    def retrieve(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def drain(reader, seconds):
    got = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        f = reader.take()
        if f is not None:
            got.append(f)
        time.sleep(0.001)
    return got


def make(url, **kw):
    src = Source(name="s", url=url, realtime=False, reconnect={"initial_delay_s": 0.01, "max_delay_s": 0.04}, **kw)
    return src


def test_stream_reconnects_after_failure_and_bumps_session():
    FakeCapture.opens = 0
    stop = threading.Event()
    r = Reader(make("rtsp://127.0.0.1/x"), process_fps=30, stop=stop, opener=lambda u: FakeCapture(u, frames=3))
    r.start()
    frames = drain(r, 0.3)
    stop.set()
    r.join(1)
    assert r.reconnects >= 2
    assert len({f.session for f in frames}) >= 2  # trackers are reset on a new session
    assert [f.index for f in frames] == sorted(f.index for f in frames)


def test_file_ends_cleanly_without_reconnecting():
    FakeCapture.opens = 0
    stop = threading.Event()
    r = Reader(make("clip.avi"), process_fps=30, stop=stop, opener=lambda u: FakeCapture(u, frames=5))
    r.start()
    frames = drain(r, 0.2)
    r.join(1)
    assert r.finished and r.reconnects == 0
    assert len(frames) == 5  # lockstep mode delivers every sampled frame


def test_unopenable_source_keeps_retrying_with_backoff():
    FakeCapture.opens = 0
    stop = threading.Event()
    r = Reader(make("rtsp://127.0.0.1/x"), process_fps=30, stop=stop,
               opener=lambda u: FakeCapture(u, frames=2, fail_opens=1000))
    r.start()
    time.sleep(0.25)
    stop.set()
    r.join(1)
    # With delays of 0.01, 0.02, 0.04, 0.04, ... about 7 attempts fit in 0.25 s, not hundreds.
    assert 3 <= FakeCapture.opens <= 12
    assert not r.connected and not r.finished


def test_sampling_keeps_every_third_frame_at_10_of_30_fps():
    FakeCapture.opens = 0
    stop = threading.Event()
    r = Reader(make("clip.avi"), process_fps=10, stop=stop, opener=lambda u: FakeCapture(u, frames=9))
    r.start()
    frames = drain(r, 0.2)
    r.join(1)
    assert [round(f.t, 3) for f in frames] == [0.0, 0.1, 0.2]
