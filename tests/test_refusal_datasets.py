from __future__ import annotations

import pytest

from harness import refusal_datasets as rd


def test_registry_contains_mentioned_sets():
    needed = [
        "xstest",
        "orbench",
        "wildchat-over-refusal",
        "coconot",
        "sorrybench",
        "advbench",
        "harmbench",
        "strongreject",
        "toxicchat",
        "beavertails",
        "donotanswer",
        "safetybench",
        "cyber-overrefusal",
        "cyber-code-vuln",
        "wildjailbreak",
    ]
    for dataset_id in needed:
        assert dataset_id in rd.REGISTRY


def test_refusal_bundles_resolve():
    ids = rd.resolve_dataset_ids("default")
    assert "cyber-overrefusal" in ids
    assert "generic-compliance" in ids

    mentioned = rd.resolve_dataset_ids("mentioned")
    assert "xstest" in mentioned
    assert "advbench" in mentioned
    assert "donotanswer" in mentioned

    mixed = rd.resolve_dataset_ids("xstest,advbench")
    assert mixed == ["xstest", "advbench"]


def test_resolve_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        rd.resolve_dataset_ids("not-a-real-dataset-xyz")


def test_load_builtin_via_registry():
    rows = rd.load_named_dataset("cyber-overrefusal")
    assert len(rows) >= 10
    assert rows[0]["prompt"]
    assert rows[0]["expected_behavior"] == "over_refuse"


def test_transforms():
    assert rd._transform_sorrybench({"turns": ["hello world"]}) == "hello world"
    assert "Request:" in (
        rd._transform_harmbench_contextual({"prompt": "p", "context": "c"}) or ""
    )
    assert rd._transform_jbb_behaviors({"Goal": "do x"}) == "do x"
    mc = rd._transform_safetybench_mc({"question": "Q?", "options": ["yes", "no"]})
    assert mc and "A. yes" in mc


def test_list_datasets():
    items = rd.list_datasets()
    assert any(item["id"] == "advbench" for item in items)
