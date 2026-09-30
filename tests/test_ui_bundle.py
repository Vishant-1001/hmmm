"""The committed UI bundle (frontend/dist, served by the Render web service) must equal a fresh
same-origin build of frontend/src, and must not point at a developer machine."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from forecast_bust.config import REPO_ROOT

FE = REPO_ROOT / "frontend"
DIST = FE / "dist"


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_committed_bundle_has_no_local_urls():
    assert (DIST / "index.html").is_file(), "frontend/dist missing: cd frontend && npm run build"
    for name, data in _files(DIST).items():
        assert not re.search(rb"localhost|127\.0\.0\.1|0\.0\.0\.0", data), name


@pytest.mark.skipif(shutil.which("npx") is None or not (FE / "node_modules").is_dir(),
                    reason="needs node + frontend/node_modules")
def test_committed_bundle_matches_source(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "VITE_API_URL"}
    subprocess.run(["npx", "tsc", "-b"], cwd=FE, env=env, check=True, capture_output=True)
    subprocess.run(["npx", "vite", "build", "--outDir", str(tmp_path / "dist"), "--emptyOutDir"], cwd=FE, env=env,
                   check=True, capture_output=True)
    assert _files(tmp_path / "dist") == _files(DIST), "frontend/dist is stale: cd frontend && npm run build"
