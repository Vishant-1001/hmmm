"""Configuration loading and repository paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

def _project_root() -> Path:
    """Project root holding config/, artifacts/, models/ and frontend/.

    config/ is runtime configuration, not package data, so it is never looked for beside
    site-packages. Resolution order: FBS_PROJECT_ROOT (set in the Docker image), then the
    first ancestor of this source file (in-tree / editable install), then of the working
    directory, that contains config/model.yaml.
    """
    env = os.environ.get("FBS_PROJECT_ROOT")
    if env:
        root = Path(env).resolve()
        if not (root / "config" / "model.yaml").is_file():
            raise FileNotFoundError(f"FBS_PROJECT_ROOT={env} does not contain config/model.yaml")
        return root
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / "config" / "model.yaml").is_file():
                return d
    raise FileNotFoundError("cannot locate the project root (a directory containing config/model.yaml); "
                            "set FBS_PROJECT_ROOT to the repository checkout")


REPO_ROOT = _project_root()
CONFIG_DIR = REPO_ROOT / "config"
# FBS_SPLIT=dev runs the whole pipeline on a development split (train 2018-2019,
# validation 2020, dev-test 2021) so that iteration never touches the 2022 test year.
DEV = os.environ.get("FBS_SPLIT", "final") == "dev"
# FBS_RUN names a run namespace (e.g. "v2") so a rebuilt pipeline never overwrites the
# artifacts, models or interim tables of an earlier final evaluation. Empty = original v1 paths.
RUN = os.environ.get("FBS_RUN", "")
_suffix = Path(RUN) / ("dev" if DEV else "")
ARTIFACT_DIR = Path(os.environ.get("FBS_ARTIFACT_DIR", REPO_ROOT / "artifacts")) / _suffix
DATA_DIR = Path(os.environ.get("FBS_DATA_DIR", REPO_ROOT / "data"))
MODEL_DIR = REPO_ROOT / "models" / _suffix
INTERIM_DIR = REPO_ROOT / "data" / "interim" / _suffix


def served_run() -> tuple[Path, str]:
    """(artifact dir, run name) the API/demo SERVE (read-only): FBS_ARTIFACT_DIR if set (run name from
    FBS_SERVE_RUN, default ""); else artifacts/<FBS_SERVE_RUN> (default v4) when that run's final metrics and
    demo bundle exist; else the v1 artifacts/. Read at call time, so an environment override applies to
    modules (re)imported afterwards. Pipelines still write to ARTIFACT_DIR."""
    env = os.environ.get("FBS_ARTIFACT_DIR")
    if env:
        return Path(env), os.environ.get("FBS_SERVE_RUN", "")
    run = os.environ.get("FBS_SERVE_RUN", "v4")
    p = REPO_ROOT / "artifacts" / run
    if run and (p / "metrics.json").is_file() and (p / "demo" / "registry.json").is_file():
        return p, run
    return REPO_ROOT / "artifacts", ""


SERVED_ARTIFACT_DIR, SERVED_RUN = served_run()


def run_config_name() -> str:
    """Model config file for this run namespace: config/model_<run>.yaml if present, else model.yaml."""
    return f"model_{RUN}" if RUN and (CONFIG_DIR / f"model_{RUN}.yaml").exists() else "model"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def data_config() -> dict:
    cfg = load_config("data")
    if DEV:
        cfg = {**cfg, "split": cfg["dev_split"]}
    return cfg


def model_config() -> dict:
    return load_config(run_config_name())


def cache_dir() -> Path:
    p = REPO_ROOT / data_config()["cache_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_json(obj):
    """Recursively replace NaN/inf with None so artifacts are strict JSON."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
