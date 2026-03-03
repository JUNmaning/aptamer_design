# utils/logging.py
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    run_dir: Path,
    name: str = "run",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Create a logger that writes to:
      - run_dir/logs/{name}.log
    and optionally also to console.

    Idempotent: if handlers already exist, it won't add duplicates.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # avoid double logging with root logger

    # If already configured, return
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(sh)

    logger.info(f"Logging initialized. log_file={log_file}")
    return logger
