# utils/meta.py
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pathlib import Path

def update_meta_end(meta_path, status: str, best=None):
    meta_path = Path(meta_path)
    if meta_path.is_dir():
        meta_path = meta_path / "meta.json"



def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def write_resolved_config(run_dir: Union[str, Path], cfg: Dict[str, Any]) -> Path:
    """
    Save resolved config for reproducibility.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_resolved.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return path


def write_meta(
    run_dir: Union[str, Path],
    cfg: Dict[str, Any],
    ligand_file: Union[str, Path],
    status: str = "running",
) -> Path:
    """
    Create meta.json at run_dir/meta.json and return its path.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.json"
    meta: Dict[str, Any] = {
        "status": status,
        "start_time": _now_iso(),
        "end_time": None,
        "elapsed_seconds": None,
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
        "ligand": str(Path(ligand_file).expanduser().resolve()),
        "config_file": cfg.get("_config_file", ""),
        "cfg": cfg,  # 保留一份（可能较大，但最可复现）
        "best": None,
        "error": None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta_path


def update_meta_end(
    meta_path: Optional[Union[str, Path]] = None,
    *,
    run_dir: Optional[Union[str, Path]] = None,
    status: str = "done",
    best: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
) -> Path:
    """
    Update meta.json at the end of run.

    Compatible with two call styles:
      - update_meta_end(meta_path, status="done", best=...)
      - update_meta_end(run_dir=run_dir, status="done", best=...)

    Returns: meta_path
    """
    if meta_path is None:
        if run_dir is None:
            raise TypeError("update_meta_end requires meta_path or run_dir")
        meta_path = Path(run_dir).expanduser().resolve() / "meta.json"
    else:
        meta_path = Path(meta_path).expanduser().resolve()

    if not meta_path.exists():
        # 如果 meta.json 不存在，也要能写出来（避免收尾阶段再崩）
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {}
    else:
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    meta["status"] = status
    meta["end_time"] = _now_iso()
    if elapsed_seconds is not None:
        meta["elapsed_seconds"] = float(elapsed_seconds)
    if best is not None:
        meta["best"] = best
    if error is not None:
        meta["error"] = error

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta_path
