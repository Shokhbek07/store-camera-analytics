import sys

import pytest

import storecam

CONFIG = """
sources:
  - name: counter
    url: {url}
model:
  weights: {weights}
"""


def run_cli(monkeypatch, capsys, tmp_path, url, weights):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG.format(url=url, weights=weights))
    monkeypatch.setattr(sys, "argv", ["storecam", "run", str(cfg)])
    with pytest.raises(SystemExit) as exit_info:
        storecam.main()
    return exit_info.value.code, capsys.readouterr().err


def test_run_stops_with_a_hint_when_a_video_is_missing(monkeypatch, capsys, tmp_path):
    code, err = run_cli(monkeypatch, capsys, tmp_path, tmp_path / "missing.avi", tmp_path / "w.pt")
    assert code == 2 and "video not found" in err and "download.sh" in err


def test_run_stops_with_a_hint_when_onnx_weights_are_missing(monkeypatch, capsys, tmp_path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"")
    code, err = run_cli(monkeypatch, capsys, tmp_path, video, tmp_path / "w.onnx")
    assert code == 2 and "weights not found" in err and "export-onnx" in err


def test_stream_urls_are_not_checked_as_files(monkeypatch, capsys, tmp_path):
    code, err = run_cli(monkeypatch, capsys, tmp_path, "rtsp://cameras:8554/counter", tmp_path / "w.pt")
    assert code == 2 and "video not found" not in err and "download.sh" in err
