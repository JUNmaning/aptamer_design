# utils/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping/dict, got: {type(data)}")
    return data


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge upd into base (in-place) and return base.
    - If both values are dicts: recurse
    - Else: overwrite
    """
    for k, v in upd.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_set_kv(items: Iterable[str]) -> Iterable[Tuple[str, Any]]:
    """
    Parse --set key=value strings.
    Value is parsed using yaml.safe_load so that:
      "10" -> int, "0.1" -> float, "true" -> bool, "null" -> None, "[1,2]" -> list
    """
    for s in items:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got: {s}")
        key, raw = s.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise ValueError(f"Empty key in --set: {s}")
        val = yaml.safe_load(raw)  # type inference
        yield key, val


def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """
    Set cfg["a"]["b"]["c"] = value for dotted "a.b.c".
    Creates intermediate dicts if missing.
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("Empty dotted key")
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
