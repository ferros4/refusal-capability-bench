"""Seeded prompt-list snapshot cache for stable A/B compares."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_ROOT = Path("cache") / "prompts"


def cache_key(
    *,
    dataset_id: str,
    seed: int,
    limit: int | None,
    revision: str | None,
    split: str | None,
    extra: str = "",
) -> str:
    raw = f"{dataset_id }|seed={seed }|limit={limit }|rev={revision or ''}|split={split or ''}|{extra }"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset_id)[:60]
    return f"{safe }_{digest }"


def cache_path(root: Path, key: str) -> Path:
    return root / f"{key }.jsonl"


def load_cached_prompts(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_cached_prompts(
    path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
