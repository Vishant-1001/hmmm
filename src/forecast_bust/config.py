"""Configuration loading and repository paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
# FBS_SPLIT=dev runs the whole pipeline on a development split (train 2018-2019,
# validation 2020, dev-test 2021) so that iteration never touches the 2022 test year.
DEV = os.environ.get("FBS_SPLIT", "final") == "dev"
_suffix = "dev" if DEV else ""
ARTIFACT_DIR = Path(os.environ.get("FBS_ARTIFACT_DIR", REPO_ROOT / "artifacts")) / _suffix
DATA_DIR = Path(os.environ.get("FBS_DATA_DIR", REPO_ROOT / "data"))
MODEL_DIR = REPO_ROOT / "models" / _suffix
INTERIM_DIR = REPO_ROOT / "data" / "interim" / _suffix


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
    return load_config("model")


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
