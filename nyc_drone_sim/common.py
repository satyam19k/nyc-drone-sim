"""Shared helpers: config loading, paths, slugs. CPU only."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
FT_TO_M = 0.3048


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not s:
        raise ValueError(f"cannot build a location slug from {text!r}")
    return s


def scene_dir(cfg: dict, location_slug: str) -> Path:
    return REPO_ROOT / cfg["output"]["scenes_dir"] / location_slug


def altitude_m(alt_ft: float) -> float:
    return float(alt_ft) * FT_TO_M
