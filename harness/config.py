"""Load optional eval.yaml / JSON config; CLI overrides win."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAMES = ("eval.yaml", "eval.yml", "eval.json")


def find_default_config(start: Path | None = None) -> Path | None:
    root = start or Path.cwd()
    for name in DEFAULT_CONFIG_NAMES:
        path = root / name
        if path.is_file():
            return path
    return None


def load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path }")
    return data


def merge_config_into_namespace(args: Any, cfg: dict[str, Any]) -> Any:
    """
    Apply config keys onto argparse namespace only where CLI left defaults/None.
    Explicit CLI non-default values win — we treat None as 'unset' for optional fields.
    """
    # Map config keys → argparse attr names
    key_map = {
        "base_url": "base_url",
        "host": "host",
        "port": "port",
        "api_key": "api_key",
        "model": "model",
        "compare": "compare",
        "preset": "preset",
        "datasets": "datasets",
        "only": "only",
        "out_root": "out_root",
        "run_id": "run_id",
        "dataset_limit": "dataset_limit",
        "judge": "judge",
        "judge_base_url": "judge_base_url",
        "judge_model": "judge_model",
        "judge_api_key": "judge_api_key",
        "limit": "limit",
        "seed": "seed",
        "mmlu_subjects": "mmlu_subjects",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "sleep": "sleep",
        "timeout": "timeout",
        "secure": "secure",
        "continue_on_error": "continue_on_error",
        "workers": "workers",
        "cache_dir": "cache_dir",
        "no_cache": "no_cache",
        "refresh_cache": "refresh_cache",
        "report": "report",
        "skip_hf_check": "skip_hf_check",
        "dataset_revisions": "dataset_revisions",
    }
    for cfg_key, attr in key_map.items():
        if cfg_key not in cfg:
            continue
        if not hasattr(args, attr):
            continue
        cur = getattr(args, attr)
        # booleans: only apply if still False default and config True? Better: apply if CLI didn't set.
        # We don't track "was set"; apply when current is None or for bool when False and config provided
        # only for None-able fields. For fields with defaults, config applies first then CLI
        # — caller should load config BEFORE parse or use two-phase.
        # Here: fill only if attr is None.
        if cur is None:
            setattr(args, attr, cfg[cfg_key])
    return args


def apply_config_defaults(cfg: dict[str, Any]) -> list[str]:
    """Convert config dict to argv-like defaults list for parse_args prefill — unused."""
    return []
