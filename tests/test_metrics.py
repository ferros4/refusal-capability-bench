from __future__ import annotations

from types import SimpleNamespace

from harness.metrics import timing_stats


def test_timing_stats():
    rows = [
        SimpleNamespace(
            latency_s=2.0,
            completion_tokens=100,
            prompt_tokens=10,
            total_tokens=110,
            tokens_per_sec=50.0,
            tokens_estimated=False,
            error="",
        ),
        SimpleNamespace(
            latency_s=1.0,
            completion_tokens=40,
            prompt_tokens=5,
            total_tokens=45,
            tokens_per_sec=40.0,
            tokens_estimated=False,
            error="",
        ),
    ]
    stats = timing_stats(rows)
    assert stats["total_time_s"] == 3.0
    assert stats["total_completion_tokens"] == 140
    assert stats["avg_latency_s"] == 1.5
    assert stats["avg_tokens_per_sec"] == 45.0
    assert stats["overall_tokens_per_sec"] == round(140 / 3.0, 3)
