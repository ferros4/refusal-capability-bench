"""Timing / throughput helpers for eval summaries."""

from __future__ import annotations

from typing import Any, Iterable, Protocol


class TimedSample(Protocol):
    latency_s: float
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    tokens_per_sec: float
    tokens_estimated: bool
    error: str


def timing_stats(rows: Iterable[Any]) -> dict[str, Any]:
    """Aggregate wall time and token throughput for a list of trial-like objects."""
    items = list(rows)
    # Only count samples that actually ran (have latency or tokens); include zeros for failed gens
    ran = [
        row
        for row in items
        if getattr(row, "latency_s", 0) > 0 or getattr(row, "completion_tokens", 0) > 0
    ]
    # Prefer all non-stopped for totals if nothing "ran"
    pool = ran if ran else [row for row in items if not getattr(row, "error", "")]
    if not pool:
        pool = items

    total_time = sum(float(getattr(row, "latency_s", 0) or 0) for row in pool)
    total_completion = sum(
        int(getattr(row, "completion_tokens", 0) or 0) for row in pool
    )
    total_prompt = sum(int(getattr(row, "prompt_tokens", 0) or 0) for row in pool)
    total_tokens = sum(int(getattr(row, "total_tokens", 0) or 0) for row in pool)
    count = len(pool)
    tps_vals = [
        float(getattr(row, "tokens_per_sec", 0) or 0)
        for row in pool
        if float(getattr(row, "tokens_per_sec", 0) or 0) > 0
    ]
    estimated = any(bool(getattr(row, "tokens_estimated", False)) for row in pool)

    avg_tps = round(sum(tps_vals) / len(tps_vals), 3) if tps_vals else 0.0
    # Overall throughput: total completion tokens / total wall time
    overall_tps = round(total_completion / total_time, 3) if total_time > 0 else 0.0

    return {
        "n_timed": count,
        "total_time_s": round(total_time, 3),
        "avg_latency_s": round(total_time / count, 3) if count else 0.0,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_tokens_per_sec": avg_tps,
        "overall_tokens_per_sec": overall_tps,
        "tokens_estimated": estimated,
    }
