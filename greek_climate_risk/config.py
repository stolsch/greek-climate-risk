"""Configuration helpers for the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
