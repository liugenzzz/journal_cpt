from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else package_root() / "config.py"
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("_journal_cpt_config", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Invalid config module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = json.loads(getattr(module, "CONFIG_JSON"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config payload: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

