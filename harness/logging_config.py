"""Standard library logging setup for the eval harness."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"
_CONFIGURED = False


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    return int(getattr(logging, name, logging.INFO))


def setup_logging(
    level: str | int = DEFAULT_LEVEL,
    log_file: str | Path | None = None,
    *,
    console: bool = True,
    force: bool = False,
) -> Path | None:
    """
    Configure root logging once: optional console + optional file.

    Returns the log file path when a file handler is attached.
    """
    global _CONFIGURED
    root = logging.getLogger()
    resolved_level = _coerce_level(level)

    if _CONFIGURED and not force:
        root.setLevel(resolved_level)
        path: Path | None = None
        if log_file:
            path = Path(log_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            _ensure_file_handler(root, path, resolved_level)
        return path

    # Reset handlers when (re)configuring
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(resolved_level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(resolved_level)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    path = None
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_file_handler(root, path, resolved_level)

    # Keep third-party HTTP noise down unless user asked for DEBUG
    if resolved_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "Logging configured level=%s file=%s console=%s",
        logging.getLevelName(resolved_level),
        path,
        console,
    )
    return path


def add_log_file(log_file: str | Path, level: str | int | None = None) -> Path:
    """Attach (or replace) a file handler on the root logger."""
    root = logging.getLogger()
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = _coerce_level(level) if level is not None else root.level
    _ensure_file_handler(root, path, resolved)
    logging.getLogger(__name__).info("Logging to file: %s", path)
    return path


def _ensure_file_handler(root: logging.Logger, path: Path, level: int) -> None:
    abs_path = str(path.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = getattr(handler, "baseFilename", None)
            if existing and Path(existing).resolve() == path.resolve():
                handler.setLevel(level)
                return
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler = logging.FileHandler(abs_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (prefer module __name__)."""
    return logging.getLogger(name)
