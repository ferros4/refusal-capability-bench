"""Hugging Face auth / gated-dataset preflight checks."""

from __future__ import annotations

import os
from typing import Iterable

from harness.refusal_datasets import REGISTRY, DatasetSpec


def gated_specs_for_ids(dataset_ids: Iterable[str]) -> list[DatasetSpec]:
    out: list[DatasetSpec] = []
    for did in dataset_ids:
        spec = REGISTRY.get(did)
        if spec and spec.gated and spec.hf_path:
            out.append(spec)
    return out


def has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def check_hf_access(dataset_ids: Iterable[str], *, strict: bool = True) -> list[str]:
    """
    Return list of human-readable problems.
    If strict and problems non-empty, caller should abort.
    """
    gated = gated_specs_for_ids(dataset_ids)
    if not gated:
        return []

    problems: list[str] = []
    if not has_hf_token():
        names = ", ".join(sorted({spec.hf_path for spec in gated if spec.hf_path}))
        problems.append(
            "Gated Hugging Face datasets requested but no HF token found. "
            f"Run `huggingface-cli login` or set HF_TOKEN. Datasets: {names}"
        )
        if strict:
            return problems

    # Lightweight probe: try dataset info for first gated path
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        for spec in gated:
            assert spec.hf_path
            try:
                api.dataset_info(spec.hf_path, token=token)
            except Exception as exc:
                problems.append(
                    f"Cannot access gated dataset {spec.hf_path!r} (id={spec.id}): {exc}. "
                    "Accept the license on the Hub while logged in, then retry."
                )
                if strict:
                    break
    except ImportError:
        problems.append(
            "huggingface_hub not installed; cannot verify gated dataset access. "
            "pip install huggingface_hub"
        )

    return problems
