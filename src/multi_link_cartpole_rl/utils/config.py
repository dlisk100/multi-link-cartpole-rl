"""Configuration helpers.

YAML configs keep early experiments readable. This helper imports PyYAML lazily so basic
package imports still work before dependencies are installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before loading YAML configs.") from exc

    with Path(path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")

    return data
