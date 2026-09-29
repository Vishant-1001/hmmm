"""End-to-end: real server + built UI + headless browser; displayed values must equal API values."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pw = pytest.importorskip("playwright")
from demo_walkthrough import chromium_path, walkthrough  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ROOT / "frontend" / "dist" / "index.html").exists() or not chromium_path()
    or not (ROOT / "artifacts" / "demo" / "registry.json").exists(),
    reason="needs built frontend, a Chromium binary and the demo bundle")


@pytest.fixture(scope="module")
def server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "forecast_bust.api.app:app", "--port", str(port)],
                            cwd=ROOT, env={**os.environ}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            urllib.request.urlopen(url + "/api/demo/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(10)


def test_full_demo_flow(server):
    out = walkthrough(server)
    assert all(c["ok"] for c in out["checks"]) and len(out["checks"]) >= 12
    assert out["reveal_requested_before_click"] is False
    for k, v in out["timings_ms"].items():
        assert v < 5000, f"{k} took {v} ms"
