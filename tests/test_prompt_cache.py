from __future__ import annotations

from pathlib import Path

from harness.prompt_cache import (
    cache_key,
    cache_path,
    load_cached_prompts,
    save_cached_prompts,
)


def test_prompt_cache_roundtrip(tmp_path: Path):
    key = cache_key(
        dataset_id="cyber-overrefusal", seed=42, limit=5, revision=None, split=None
    )
    path = cache_path(tmp_path, key)
    rows = [{"id": "1", "prompt": "hi", "category": "x"}]
    save_cached_prompts(path, rows, {"dataset_id": "cyber-overrefusal"})
    loaded = load_cached_prompts(path)
    assert loaded == rows
    assert path.with_suffix(".meta.json").exists()
