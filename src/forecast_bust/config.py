"""Configuration loading and repository paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
ARTIFACT_DIR = Path(os.environ.get("FBS_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
DATA_DIR = Path(os.environ.get("FBS_DATA_DIR", REPO_ROOT / "data"))
MODEL_DIR = REPO_ROOT / "models"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def data_config() -> dict:
    return load_config("data")


def model_config() -> dict:
    return load_config("model")


def cache_dir() -> Path:
    p = REPO_ROOT / data_config()["cache_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p
