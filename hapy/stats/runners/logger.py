"""
Shared logger for hapy runners.

Usage:
    from .logger import get_runner_logger
    logger = get_runner_logger()
    logger.error("message")

Writes to both stderr (WARNING+) and a rotating file (DEBUG+) at
~/.hapy/hapy_runners.log  (overridable via HAPY_LOG_FILE env var).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_LOGGER_NAME = "hapy.runners"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_runner_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    # --- stderr handler (WARNING and above) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(ch)

    # --- rotating file handler (DEBUG and above) ---
    log_path = Path(
        os.environ.get("HAPY_LOG_FILE", Path.cwd() / "logs" / "hapy_runners.log")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(fh)

    return logger