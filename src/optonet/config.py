"""Configuration loading utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    cfg["_path"] = str(path or DEFAULT_CONFIG)
    return cfg


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)
