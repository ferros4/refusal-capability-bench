#!/usr/bin/env python3
"""
Single entrypoint: run refusal + capability evals and write a structured results tree.

Recommended (presets):
  python run_eval.py --model qwen3.6:35b-a3b-q8_0 --preset default
  python run_eval.py --model base --compare uncensored --preset compare --limit 50
  python run_eval.py --list-presets

Custom dataset mix (refusal + capability ids together):
  python run_eval.py --model m --datasets xstest,advbench,gsm8k,mmlu

Layout:
  results/<model_slug>/<timestamp>/
    meta.json
    summary.json
    refusal/
      base_cyber/ ...
    capability/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness import capability_eval, refusal_eval
from harness.api_client import DEFAULT_HOST, DEFAULT_PORT, resolve_base_url
from harness.capability_eval import USER_STOPPED as CAP_USER_STOPPED
from harness.logging_config import add_log_file, get_logger, setup_logging
from harness.refusal_eval import USER_STOPPED as REF_USER_STOPPED

log = get_logger(__name__)

USER_STOPPED = REF_USER_STOPPED


def slugify(name: str, max_len: int = 80) -> str:
    name = name.strip().lower()
    name = name.replace("hf.co/", "")
    name = re.sub(r"[^a-z0-9._+-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return (name or "model")[:max_len]


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def model_dir_name(model: str, compare: str | None = None) -> str:
    """Top-level folder under results/ for the model under test."""
    base = slugify(model, max_len=60)
    if compare:
        return f"{base }_vs_{slugify (compare ,max_len =40 )}"
    return base


def refusal_folder_name(dataset: str) -> str:
    """Map dataset ids to stable output folder names."""
    key = dataset.strip()
    aliases = {
        "builtin:cyber-overrefusal": "base_cyber",
        "cyber-overrefusal": "base_cyber",
        "builtin:generic-compliance": "generic_compliance",
        "generic-compliance": "generic_compliance",
    }
    if key in aliases:
        return aliases[key]
    if key.startswith("preset:"):
        return slugify(key)
    return slugify(key.replace("builtin:", "").replace("/", "_"))


def resolve_run_dir(
    out_root: Path, model: str, compare: str | None, run_id: str | None
) -> Path:
    """results/<model_slug>/[<compare>]/<timestamp>/"""
    model_root = out_root / model_dir_name(model, compare)
    ts = run_id or timestamp_utc()
    return model_root / ts


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_refusal_argv(
    *,
    base_url: str,
    api_key: str,
    model: str,
    dataset: str,
    out: Path,
    limit: int | None,
    judge: str,
    temperature: float,
    max_tokens: int,
    sleep: float,
    timeout: float,
    secure: bool,
    workers: int = 1,
    seed: int = 42,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    cache_dir: str | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
) -> list[str]:
    argv = [
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--model",
        model,
        "--dataset",
        dataset,
        "--out",
        str(out),
        "--judge",
        judge,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--sleep",
        str(sleep),
        "--timeout",
        str(timeout),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if secure:
        argv.append("--secure")
    if judge_base_url:
        argv.extend(["--judge-base-url", judge_base_url])
    if judge_model:
        argv.extend(["--judge-model", judge_model])
    if judge_api_key:
        argv.extend(["--judge-api-key", judge_api_key])
    if cache_dir:
        argv.extend(["--cache-dir", cache_dir])
    if no_cache:
        argv.append("--no-cache")
    if refresh_cache:
        argv.append("--refresh-cache")
    return argv


def _append_log_args(argv: list[str], args: argparse.Namespace, log_file: Path) -> list[str]:
    argv = list(argv)
    argv.extend(["--log-level", str(getattr(args, "log_level", "INFO"))])
    argv.extend(["--log-file", str(log_file)])
    return argv


def build_capability_argv(
    *,
    base_url: str,
    api_key: str,
    model: str,
    compare: str | None,
    out: Path,
    benches: str,
    limit: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    sleep: float,
    timeout: float,
    secure: bool,
    mmlu_subjects: str | None,
    workers: int = 1,
) -> list[str]:
    argv = [
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--model",
        model,
        "--out",
        str(out),
        "--benches",
        benches,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--sleep",
        str(sleep),
        "--timeout",
        str(timeout),
        "--workers",
        str(workers),
    ]
    if compare:
        argv.extend(["--compare", compare])
    if secure:
        argv.append("--secure")
    if mmlu_subjects:
        argv.extend(["--mmlu-subjects", mmlu_subjects])
    return argv


def _refusal_headlines(refusal_out: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, ds_map in refusal_out.items():
        out[model] = {
            ds: {
                "refusal_rate": info.get("refusal_rate"),
                "compliance_rate": info.get("compliance_rate"),
                "n": info.get("n"),
                "total_time_s": info.get("total_time_s"),
                "avg_tokens_per_sec": info.get("avg_tokens_per_sec"),
                "overall_tokens_per_sec": info.get("overall_tokens_per_sec"),
            }
            for ds, info in ds_map.items()
            if info
        }
    return out


def _capability_headlines(capability: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not capability.get("models"):
        return out
    for model, info in capability["models"].items():
        out[model] = {
            "overall_accuracy": info.get("accuracy"),
            "total_time_s": info.get("total_time_s"),
            "avg_tokens_per_sec": info.get("avg_tokens_per_sec"),
            "overall_tokens_per_sec": info.get("overall_tokens_per_sec"),
            "benches": {
                bench_name: {
                    "accuracy": bi.get("accuracy"),
                    "n": bi.get("n"),
                    "total_time_s": bi.get("total_time_s"),
                    "avg_tokens_per_sec": bi.get("avg_tokens_per_sec"),
                    "overall_tokens_per_sec": bi.get("overall_tokens_per_sec"),
                }
                for bench_name, bi in (info.get("benches") or {}).items()
            },
        }
    if capability.get("delta_compare_minus_base"):
        out["delta_compare_minus_base"] = capability["delta_compare_minus_base"]
    return out


def combine_summary(
    *,
    run_dir: Path,
    models: list[str],
    refusal_paths: dict[str, dict[str, Path]],
    capability_dir: Path | None,
    elapsed_s: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    refusal_out: dict[str, Any] = {}
    for model, ds_map in refusal_paths.items():
        refusal_out[model] = {}
        for ds_name, path in ds_map.items():
            refusal_out[model][ds_name] = load_json(path / "summary.json")

    capability = load_json(capability_dir / "summary.json") if capability_dir else {}

    combined = {
        "run_dir": str(run_dir),
        "models": models,
        "elapsed_s": round(elapsed_s, 2),
        "config": config,
        "refusal": refusal_out,
        "capability": capability,
    }

    combined["headlines"] = {
        "refusal": _refusal_headlines(refusal_out),
        "capability": _capability_headlines(capability),
    }
    return combined


def combine_multi_preset_summary(
    *,
    run_dir: Path,
    models: list[str],
    elapsed_s: float,
    config: dict[str, Any],
    preset_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-preset suite summaries into one top-level summary.json."""
    by_preset: dict[str, Any] = {}
    headlines_by_preset: dict[str, Any] = {}
    merged_refusal: dict[str, Any] = {}
    merged_cap_models: dict[str, Any] = {}

    for preset_id, summary in preset_summaries.items():
        by_preset[preset_id] = {
            "refusal": summary.get("refusal") or {},
            "capability": summary.get("capability") or {},
            "headlines": summary.get("headlines") or {},
            "elapsed_s": summary.get("elapsed_s"),
        }
        headlines_by_preset[preset_id] = summary.get("headlines") or {}

        for model, ds_map in (summary.get("refusal") or {}).items():
            merged_refusal.setdefault(model, {})
            for ds_name, info in ds_map.items():
                key = f"{preset_id}/{ds_name}"
                merged_refusal[model][key] = info

        cap = summary.get("capability") or {}
        for model, info in (cap.get("models") or {}).items():
            slot = merged_cap_models.setdefault(
                model,
                {
                    "n": 0,
                    "correct": 0,
                    "benches": {},
                    "total_time_s": 0.0,
                    "avg_tokens_per_sec": None,
                    "overall_tokens_per_sec": None,
                },
            )
            slot["n"] += int(info.get("n") or 0)
            slot["correct"] += int(info.get("correct") or 0)
            slot["total_time_s"] = round(
                float(slot["total_time_s"] or 0) + float(info.get("total_time_s") or 0),
                4,
            )
            for bench_name, bi in (info.get("benches") or {}).items():
                slot["benches"][f"{preset_id}/{bench_name}"] = bi
        if cap.get("delta_compare_minus_base"):
            merged_cap_models.setdefault("_meta", {})
            merged_cap_models["_meta"][f"delta_{preset_id}"] = cap[
                "delta_compare_minus_base"
            ]

    for model, info in list(merged_cap_models.items()):
        if model == "_meta":
            continue
        n = int(info.get("n") or 0)
        correct = int(info.get("correct") or 0)
        info["accuracy"] = round(correct / n, 4) if n else 0.0

    capability: dict[str, Any] = {"models": {}}
    meta = merged_cap_models.pop("_meta", None)
    for model, info in merged_cap_models.items():
        capability["models"][model] = info
    if meta:
        capability["delta_by_preset"] = meta

    combined = {
        "run_dir": str(run_dir),
        "models": models,
        "elapsed_s": round(elapsed_s, 2),
        "config": config,
        "presets": list(preset_summaries.keys()),
        "by_preset": by_preset,
        "refusal": merged_refusal,
        "capability": capability,
        "headlines": {
            "by_preset": headlines_by_preset,
            "refusal": _refusal_headlines(merged_refusal),
            "capability": _capability_headlines(capability),
        },
    }
    return combined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run refusal + capability evals into "
            "results/<model>/<timestamp>/{refusal,capability}/. "
            "Prefer --preset (recommended)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run_eval.py --model my-model --preset default\n"
            "  python run_eval.py --model my-model --preset refusal-only,coding\n"
            "  python run_eval.py --model base --compare uncen --preset compare --limit 50\n"
            "  python run_eval.py --model my-model --datasets xstest,gsm8k,mmlu\n"
            "  python run_eval.py --list-presets\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full OpenAI-compatible API root including /v1 (overrides --host/--port)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"API host when --base-url is omitted (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"API port when --base-url is omitted (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default=None, help="Primary model")
    parser.add_argument(
        "--compare", default=None, help="Optional second model (both suites)"
    )

    # --- Primary selection (presets recommended) ---
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Recommended: named suite preset(s) selecting refusal + capability datasets. "
            "Comma-separated to merge multiple presets (dataset lists unioned and "
            "deduped so each bench runs once), e.g. cyber,coding. "
            "Default if omitted: 'default'. See --list-presets."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Optional explicit mix of refusal and/or capability dataset ids "
            "(comma-separated), e.g. xstest,advbench,gsm8k,mmlu. "
            "When set, overrides the dataset lists from a single --preset "
            "(cannot combine with multiple presets)."
        ),
    )
    parser.add_argument(
        "--only",
        choices=["auto", "all", "refusal", "capability"],
        default="auto",
        help="Filter suites after preset/datasets resolve (default: auto from selection)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print recommended suite presets and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print all refusal + capability dataset ids and exit",
    )

    parser.add_argument(
        "--out-root",
        default="results",
        help="Root directory for runs (default: results)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional timestamp/folder name under the model dir (default: UTC timestamp)",
    )

    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Max samples per refusal dataset (overrides registry defaults)",
    )
    parser.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Separate OpenAI-compatible API for LLM judge",
    )
    parser.add_argument(
        "--judge-model", default=None, help="Judge model id (defaults to --model)"
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument(
        "--limit", type=int, default=50, help="Samples per capability dataset/bench"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu-subjects", default=None)
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel API workers per dataset"
    )
    parser.add_argument(
        "--cache-dir", default="cache/prompts", help="Prompt snapshot cache directory"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable prompt snapshot cache"
    )
    parser.add_argument(
        "--refresh-cache", action="store_true", help="Rewrite prompt cache entries"
    )
    parser.add_argument(
        "--report", action="store_true", help="Write report.html into the run directory"
    )
    parser.add_argument(
        "--skip-hf-check", action="store_true", help="Skip gated HF access preflight"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML/JSON config file (default: ./eval.yaml if present). CLI overrides config.",
    )

    # Shared
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--secure", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one suite/dataset fails, continue and record the error",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append logs to this file (default: <run-dir>/eval.log)",
    )
    return parser.parse_args(argv)


def _apply_file_config(args: argparse.Namespace) -> argparse.Namespace:
    from harness.config import find_default_config, load_config_file

    path = Path(args.config) if args.config else find_default_config()
    if path is None:
        return args
    cfg = load_config_file(path)
    # Config fills only unset / default-ish None fields; explicit CLI still wins for non-None.
    for key, val in cfg.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            continue
        cur = getattr(args, attr)
        if cur is None:
            setattr(args, attr, val)
        elif attr in {
            "workers",
            "limit",
            "seed",
            "dataset_limit",
            "temperature",
            "max_tokens",
            "timeout",
            "sleep",
        }:
            # numeric defaults always set by argparse — allow config only when matching common defaults
            defaults = {
                "workers": 1,
                "limit": 50,
                "seed": 42,
                "dataset_limit": None,
                "temperature": 0.0,
                "max_tokens": 1024,
                "timeout": 600.0,
                "sleep": 0.0,
            }
            if attr in defaults and cur == defaults[attr] and val is not None:
                setattr(args, attr, val)
        elif attr in {
            "judge",
            "preset",
            "only",
            "out_root",
            "base_url",
            "host",
            "api_key",
            "cache_dir",
        }:
            str_defaults = {
                "judge": "heuristic",
                "only": "auto",
                "out_root": "results",
                "api_key": "ollama",
                "cache_dir": "cache/prompts",
            }
            if attr in {"preset", "base_url", "host"} and cur is None:
                setattr(args, attr, val)
            elif attr in str_defaults and cur == str_defaults[attr]:
                setattr(args, attr, val)
        elif attr == "port" and cur is None and val is not None:
            setattr(args, attr, int(val))
        elif attr in {
            "report",
            "no_cache",
            "refresh_cache",
            "skip_hf_check",
            "continue_on_error",
            "secure",
        }:
            if cur is False and val:
                setattr(args, attr, bool(val))
    args._config_path = str(path)  # type: ignore[attr-defined]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args = _apply_file_config(args)

    # Console logging immediately; file handler attached once run_dir is known.
    setup_logging(level=args.log_level, log_file=args.log_file, console=True)
    log.debug("run_eval starting argv parsed")

    from harness.presets import (
        list_all_dataset_ids,
        list_suite_presets,
        resolve_suites,
    )
    from harness.safety import META_SAFETY_FIELDS

    if args.list_presets:
        print(
            json.dumps(
                {
                    "recommended": [
                        preset
                        for preset in list_suite_presets()
                        if preset["recommended"]
                    ],
                    "presets": list_suite_presets(),
                },
                indent=2,
            )
        )
        return 0

    if args.list_datasets:
        print(json.dumps(list_all_dataset_ids(), indent=2))
        return 0

    if not args.model:
        print(
            "--model is required (or pass --list-presets / --list-datasets)",
            file=sys.stderr,
        )
        return 2

    try:
        resolved_base_url = resolve_base_url(
            args.base_url,
            host=args.host,
            port=args.port,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        suites = resolve_suites(
            preset=args.preset, datasets=args.datasets, only=args.only
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    dataset_limit = args.dataset_limit
    models = [args.model]
    if args.compare:
        models.append(args.compare)

    run_dir = resolve_run_dir(
        Path(args.out_root), args.model, args.compare, args.run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else (run_dir / "eval.log")
    add_log_file(log_path, level=args.log_level)
    log.info(
        "run_eval start model=%s compare=%s run_dir=%s log_file=%s",
        args.model,
        args.compare,
        run_dir,
        log_path,
    )
    multi_preset = len(suites) > 1

    preset_ids = [suite.preset_id or "custom" for suite in suites]
    config = {
        "base_url": resolved_base_url,
        "host": args.host,
        "port": args.port,
        "models": models,
        "model_dir": str(run_dir.parent.name),
        "timestamp": run_dir.name,
        "preset": preset_ids[0] if not multi_preset else preset_ids,
        "presets": preset_ids,
        "preset_description": (
            suites[0].description
            if not multi_preset
            else "; ".join(
                f"{suite.preset_id or 'custom'}: {suite.description}" for suite in suites
            )
        ),
        "only": args.only,
        "datasets": {
            "by_preset": {
                (suite.preset_id or "custom"): {
                    "refusal": list(suite.refusal),
                    "capability": list(suite.capability),
                }
                for suite in suites
            },
            "refusal": sorted(
                {dataset_id for suite in suites for dataset_id in suite.refusal}
            ),
            "capability": sorted(
                {dataset_id for suite in suites for dataset_id in suite.capability}
            ),
        },
        "dataset_limit": dataset_limit,
        "judge": args.judge,
        "judge_base_url": args.judge_base_url,
        "judge_model": args.judge_model,
        "capability_limit": args.limit,
        "seed": args.seed,
        "workers": args.workers,
        "cache_dir": args.cache_dir,
        "no_cache": args.no_cache,
        "temperature": args.temperature,
        "config_path": getattr(args, "_config_path", None),
        "started_at": datetime.now(timezone.utc).isoformat(),
        **META_SAFETY_FIELDS,
    }
    # Single-preset back-compat: flat datasets lists match the one suite
    if not multi_preset:
        config["datasets"] = {
            "refusal": list(suites[0].refusal),
            "capability": list(suites[0].capability),
        }
        config["only"] = suites[0].only
        config["preset_description"] = suites[0].description
    (run_dir / "meta.json").write_text(json.dumps(config, indent=2))

    all_refusal_ids = sorted(
        {dataset_id for suite in suites for dataset_id in suite.refusal}
    )
    if all_refusal_ids and not args.skip_hf_check:
        from harness.hf_auth import check_hf_access

        problems = check_hf_access(all_refusal_ids, strict=True)
        if problems:
            for msg in problems:
                print(msg, file=sys.stderr)
            if not args.continue_on_error:
                config["errors"] = problems
                (run_dir / "meta.json").write_text(json.dumps(config, indent=2))
                return 2
            print(
                "Continuing despite HF check failures (--continue-on-error).",
                file=sys.stderr,
            )

    print(f"Run directory: {run_dir}")
    print(f"API: {resolved_base_url}")
    print(f"Presets ({len(suites)}): {', '.join(preset_ids)}")
    for suite in suites:
        print(
            f"  - {suite.preset_id or '(custom)'}: "
            f"refusal=[{', '.join(suite.refusal) or 'none'}] "
            f"capability=[{', '.join(suite.capability) or 'none'}]"
        )

    t0 = time.perf_counter()
    errors: list[str] = []
    user_stopped = False
    multi_model = len(models) > 1
    preset_summaries: dict[str, dict[str, Any]] = {}
    # Single-preset paths for back-compat finalize
    refusal_paths: dict[str, dict[str, Path]] = {}
    cap_dir: Path | None = None

    try:
        for suite in suites:
            if user_stopped:
                break
            preset_key = suite.preset_id or "custom"
            suite_root = run_dir / preset_key if multi_preset else run_dir
            suite_root.mkdir(parents=True, exist_ok=True)
            refusal_root = suite_root / "refusal"
            capability_dir = suite_root / "capability"
            suite_t0 = time.perf_counter()
            suite_refusal_paths: dict[str, dict[str, Path]] = {}
            suite_cap_dir: Path | None = None
            suite_errors: list[str] = []

            print(
                f"\n######## PRESET {preset_key} — {suite.description} ########"
            )

            only = suite.only
            refusal_ids = suite.refusal
            capability_ids = suite.capability

            # --- Refusal ---
            if only in ("all", "refusal") and refusal_ids:
                for model in models:
                    if user_stopped:
                        break
                    mslug = slugify(model)
                    suite_refusal_paths.setdefault(model, {})
                    for ds in refusal_ids:
                        folder = refusal_folder_name(ds)
                        out = (
                            refusal_root / folder / mslug
                            if multi_model
                            else refusal_root / folder
                        )
                        label = f"{preset_key}/{folder}" if multi_preset else folder
                        if user_stopped:
                            _write_skipped_refusal(out, model, ds, args.judge)
                            suite_refusal_paths[model][ds] = out
                            continue
                        print(
                            f"\n=== REFUSAL  preset={preset_key}  "
                            f"model={model}  dataset={ds}  -> {out} ==="
                        )
                        log.info(
                            "starting refusal dataset=%s model=%s out=%s",
                            ds,
                            model,
                            out,
                        )
                        r_argv = _append_log_args(
                            build_refusal_argv(
                                base_url=resolved_base_url,
                                api_key=args.api_key,
                                model=model,
                                dataset=ds,
                                out=out,
                                limit=dataset_limit,
                                judge=args.judge,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                sleep=args.sleep,
                                timeout=args.timeout,
                                secure=args.secure,
                                workers=args.workers,
                                seed=args.seed,
                                judge_base_url=args.judge_base_url,
                                judge_model=args.judge_model,
                                judge_api_key=args.judge_api_key,
                                cache_dir=args.cache_dir,
                                no_cache=args.no_cache,
                                refresh_cache=args.refresh_cache,
                            ),
                            args,
                            out / "eval.log",
                        )
                        try:
                            rc = refusal_eval.main(r_argv)
                            suite_refusal_paths[model][ds] = out
                            log.info(
                                "finished refusal dataset=%s model=%s rc=%s",
                                ds,
                                model,
                                rc,
                            )
                            if rc == 130:
                                user_stopped = True
                                suite_errors.append(f"{label}: {USER_STOPPED}")
                                break
                            if rc != 0:
                                raise RuntimeError(f"refusal_eval exited {rc}")
                        except KeyboardInterrupt:
                            user_stopped = True
                            suite_errors.append(f"{label}: {USER_STOPPED}")
                            log.warning(
                                "user stopped during refusal dataset=%s", ds
                            )
                            print(f"\n{USER_STOPPED}", file=sys.stderr)
                            if out.exists() and (out / "summary.json").exists():
                                suite_refusal_paths[model][ds] = out
                            else:
                                _write_skipped_refusal(out, model, ds, args.judge)
                                suite_refusal_paths[model][ds] = out
                            break
                        except Exception as exc:
                            msg = f"{label}/{mslug}: {type(exc).__name__}: {exc}"
                            suite_errors.append(msg)
                            log.exception(
                                "refusal dataset failed dataset=%s: %s", ds, exc
                            )
                            print(msg, file=sys.stderr)
                            if not args.continue_on_error:
                                errors.extend(suite_errors)
                                if multi_preset:
                                    _finalize_multi(
                                        run_dir,
                                        models,
                                        t0,
                                        config,
                                        errors,
                                        user_stopped,
                                        preset_summaries,
                                    )
                                else:
                                    _finalize(
                                        run_dir,
                                        models,
                                        suite_refusal_paths,
                                        None,
                                        t0,
                                        config,
                                        errors,
                                        user_stopped,
                                    )
                                return 1

            # --- Capability ---
            if only in ("all", "capability") and capability_ids:
                suite_cap_dir = capability_dir
                if user_stopped:
                    _write_skipped_capability(suite_cap_dir, models)
                else:
                    print(
                        f"\n=== CAPABILITY  preset={preset_key}  "
                        f"models={models}  benches={capability_ids} "
                        f"-> {suite_cap_dir} ==="
                    )
                    log.info(
                        "starting capability benches=%s out=%s",
                        capability_ids,
                        suite_cap_dir,
                    )
                    c_argv = _append_log_args(
                        build_capability_argv(
                            base_url=resolved_base_url,
                            api_key=args.api_key,
                            model=args.model,
                            compare=args.compare,
                            out=suite_cap_dir,
                            benches=",".join(capability_ids),
                            limit=args.limit,
                            seed=args.seed,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            sleep=args.sleep,
                            timeout=args.timeout,
                            secure=args.secure,
                            mmlu_subjects=args.mmlu_subjects,
                            workers=args.workers,
                        ),
                        args,
                        suite_cap_dir / "eval.log",
                    )
                    try:
                        rc = capability_eval.main(c_argv)
                        log.info("finished capability rc=%s", rc)
                        if rc == 130:
                            user_stopped = True
                            suite_errors.append(
                                f"{preset_key}/capability: {CAP_USER_STOPPED}"
                            )
                        elif rc != 0:
                            raise RuntimeError(f"capability_eval exited {rc}")
                    except KeyboardInterrupt:
                        user_stopped = True
                        suite_errors.append(f"{preset_key}/capability: {USER_STOPPED}")
                        log.warning("user stopped during capability")
                        print(f"\n{USER_STOPPED}", file=sys.stderr)
                        if not (suite_cap_dir / "summary.json").exists():
                            _write_skipped_capability(suite_cap_dir, models)
                    except Exception as exc:
                        msg = (
                            f"{preset_key}/capability: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        suite_errors.append(msg)
                        log.exception("capability failed: %s", exc)
                        print(msg, file=sys.stderr)
                        if not args.continue_on_error:
                            errors.extend(suite_errors)
                            if multi_preset:
                                _finalize_multi(
                                    run_dir,
                                    models,
                                    t0,
                                    config,
                                    errors,
                                    user_stopped,
                                    preset_summaries,
                                )
                            else:
                                _finalize(
                                    run_dir,
                                    models,
                                    suite_refusal_paths,
                                    suite_cap_dir,
                                    t0,
                                    config,
                                    errors,
                                    user_stopped,
                                )
                            return 1

            suite_config = {
                **config,
                "preset": preset_key,
                "preset_description": suite.description,
                "only": only,
                "datasets": {
                    "refusal": list(refusal_ids),
                    "capability": list(capability_ids),
                },
            }
            suite_summary = combine_summary(
                run_dir=suite_root,
                models=models,
                refusal_paths=suite_refusal_paths,
                capability_dir=suite_cap_dir,
                elapsed_s=time.perf_counter() - suite_t0,
                config=suite_config,
            )
            if multi_preset:
                (suite_root / "summary.json").write_text(
                    json.dumps(suite_summary, indent=2)
                )
            preset_summaries[preset_key] = suite_summary
            errors.extend(suite_errors)
            if not multi_preset:
                refusal_paths = suite_refusal_paths
                cap_dir = suite_cap_dir
    except KeyboardInterrupt:
        user_stopped = True
        errors.append(USER_STOPPED)
        print(f"\n{USER_STOPPED}", file=sys.stderr)

    if multi_preset:
        _finalize_multi(
            run_dir, models, t0, config, errors, user_stopped, preset_summaries
        )
    else:
        _finalize(
            run_dir,
            models,
            refusal_paths,
            cap_dir,
            t0,
            config,
            errors,
            user_stopped,
        )
    log.info(
        "run_eval finished errors=%s user_stopped=%s run_dir=%s",
        len(errors),
        user_stopped,
        run_dir,
    )

    if args.report:
        try:
            from harness.report import write_report

            report_path = write_report(run_dir)
            print(f"Wrote report: {report_path}")
        except Exception as exc:
            print(f"Report generation failed: {exc}", file=sys.stderr)

    if user_stopped:
        return 130
    return 1 if errors else 0


def _write_skipped_refusal(out: Path, model: str, dataset: str, judge: str) -> None:
    """Write a minimal refusal suite result when the suite never started."""
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": 0,
        "refusals": 0,
        "complies": 0,
        "refusal_rate": 0.0,
        "compliance_rate": 0.0,
        "errors": 1,
        "model": model,
        "dataset": dataset,
        "judge": judge,
        "interrupted": True,
        "user_stopped": True,
        "error": USER_STOPPED,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "results.jsonl").write_text("")
    with (out / "results.csv").open("w") as future:
        future.write(
            "sample_id,category,refused,judge,latency_s,error,prompt,response\n"
        )
        future.write(f",,true,{judge },0.0,{USER_STOPPED },,\n")


def _write_skipped_capability(out: Path, models: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "models": {
            model: {
                "n": 0,
                "correct": 0,
                "accuracy": 0.0,
                "benches": {},
                "error": USER_STOPPED,
            }
            for model in models
        },
        "interrupted": True,
        "user_stopped": True,
        "error": USER_STOPPED,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "results.jsonl").write_text("")
    with (out / "results.csv").open("w") as future:
        future.write(
            "bench,sample_id,model,correct,pred,gold,latency_s,error,response\n"
        )
        for model in models:
            future.write(f",,{model },False,,,0.0,{USER_STOPPED },\n")


def _finalize(
    run_dir: Path,
    models: list[str],
    refusal_paths: dict[str, dict[str, Path]],
    capability_dir: Path | None,
    t0: float,
    config: dict[str, Any],
    errors: list[str],
    user_stopped: bool = False,
) -> None:
    elapsed = time.perf_counter() - t0
    from harness.safety import META_SAFETY_FIELDS

    config = dict(config)
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config["errors"] = errors
    config["user_stopped"] = user_stopped
    config.update(META_SAFETY_FIELDS)
    (run_dir / "meta.json").write_text(json.dumps(config, indent=2))

    summary = combine_summary(
        run_dir=run_dir,
        models=models,
        refusal_paths=refusal_paths,
        capability_dir=capability_dir,
        elapsed_s=elapsed,
        config=config,
    )
    if user_stopped:
        summary["user_stopped"] = True
        summary["interrupted"] = True
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n========== RUN HEADLINES ==========")
    print(json.dumps(summary.get("headlines", {}), indent=2))
    print(f"\nCombined summary: {run_dir / 'summary.json'}")
    if user_stopped:
        print(f"\n{USER_STOPPED} — partial results saved.", file=sys.stderr)
    elif errors:
        print(f"Completed with {len(errors)} error(s).", file=sys.stderr)


def _finalize_multi(
    run_dir: Path,
    models: list[str],
    t0: float,
    config: dict[str, Any],
    errors: list[str],
    user_stopped: bool,
    preset_summaries: dict[str, dict[str, Any]],
) -> None:
    elapsed = time.perf_counter() - t0
    from harness.safety import META_SAFETY_FIELDS

    config = dict(config)
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config["errors"] = errors
    config["user_stopped"] = user_stopped
    config.update(META_SAFETY_FIELDS)
    (run_dir / "meta.json").write_text(json.dumps(config, indent=2))

    summary = combine_multi_preset_summary(
        run_dir=run_dir,
        models=models,
        elapsed_s=elapsed,
        config=config,
        preset_summaries=preset_summaries,
    )
    if user_stopped:
        summary["user_stopped"] = True
        summary["interrupted"] = True
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n========== RUN HEADLINES ==========")
    print(json.dumps(summary.get("headlines", {}), indent=2))
    print(f"\nCombined summary: {run_dir / 'summary.json'}")
    if user_stopped:
        print(f"\n{USER_STOPPED} — partial results saved.", file=sys.stderr)
    elif errors:
        print(f"Completed with {len(errors)} error(s).", file=sys.stderr)


def main_cli() -> None:
    """Console-script entrypoint."""
    raise SystemExit(main())


if __name__ == "__main__":
    main_cli()
