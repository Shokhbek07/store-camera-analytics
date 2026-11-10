"""The live endpoint: the page at /, the counts at /status.json, nothing else."""

import json
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from storecam.service import Service

STATUS = {"counter": {"in": 2, "out": 1, "queue": 4, "staff_present": False, "second": 30, "connected": True,
                      "reconnects": 0, "frames_processed": 300,
                      "open_alerts": [{"kind": "queue", "start": 12.0, "raised": 22.0, "peak": 5}]}}


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    fake = SimpleNamespace(status=lambda: STATUS, http_host="127.0.0.1", http_port=port)
    Service._serve_http(fake)  # only needs status() and the address
    return f"http://127.0.0.1:{port}"


def test_page_is_served_at_root(base_url):
    with urllib.request.urlopen(base_url + "/") as r:
        assert r.headers["Content-Type"].startswith("text/html")
        assert "status.json" in r.read().decode()


def test_status_json_has_the_live_counts(base_url):
    with urllib.request.urlopen(base_url + "/status.json") as r:
        assert r.headers["Content-Type"] == "application/json"
        assert json.loads(r.read()) == STATUS


def test_other_paths_are_not_found(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base_url + "/config.yaml")
    assert e.value.code == 404
