from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness import presets
from harness.refusal_datasets import PRESETS as REFUSAL_BUNDLES

ROOT = Path(__file__).resolve().parents[1]


def test_recommended_presets_exist():
    ids = {
        preset_info["id"]
        for preset_info in presets.list_suite_presets()
        if preset_info["recommended"]
    }
    for need in (
        "default",
        "quick",
        "compare",
        "overrefusal",
        "cyber",
        "refusal-only",
        "capability-only",
    ):
        assert need in ids


def test_resolve_default_when_omitted():
    suite = presets.resolve_suite(preset=None, datasets=None)
    assert suite.preset_id == "default"
    assert suite.refusal == list(REFUSAL_BUNDLES["default"])
    assert list(suite.capability) == list(presets.CAPABILITY_CORE)
    assert suite.only == "all"


def test_resolve_preset_refusal_only():
    """capability is empty on purpose; refusal is the default pack (not empty)."""
    suite = presets.resolve_suite(preset="refusal-only", datasets=None)
    assert suite.refusal == ["cyber-overrefusal", "generic-compliance"]
    assert suite.capability == []
    assert suite.only == "refusal"


def test_resolve_preset_capability_only():
    """refusal is empty on purpose; capability is the core three benches."""
    suite = presets.resolve_suite(preset="capability-only", datasets=None)
    assert suite.refusal == []
    assert suite.capability == list(presets.CAPABILITY_CORE)
    assert suite.only == "capability"


def test_resolve_preset_coding():
    suite = presets.resolve_suite(preset="coding", datasets=None)
    assert suite.refusal == []
    assert suite.capability == list(presets.CODING_DATASETS)
    assert suite.only == "capability"
    assert "mbpp" in suite.capability
    assert "humanevalplus" in suite.capability


def test_coding_datasets_are_capability():
    for dataset_id in presets.CODING_DATASETS:
        assert presets.classify_dataset_token(dataset_id) == "capability"
    assert set(presets.CODING_DATASETS) <= set(presets.CAPABILITY_DATASETS)


def test_resolve_capability_smoke_same_datasets_as_capability_only():
    suite_a = presets.resolve_suite(preset="capability-only", datasets=None)
    suite_b = presets.resolve_suite(preset="capability-smoke", datasets=None)
    assert suite_a.refusal == suite_b.refusal == []
    assert suite_a.capability == suite_b.capability


def test_compare_matches_quick():
    suite_q = presets.resolve_suite(preset="quick", datasets=None)
    suite_c = presets.resolve_suite(preset="compare", datasets=None)
    assert suite_q.refusal == suite_c.refusal
    assert suite_q.capability == suite_c.capability


def test_resolve_datasets_mixed():
    suite = presets.resolve_suite(preset=None, datasets="xstest,gsm8k,mmlu,advbench")
    assert suite.refusal == ["xstest", "advbench"]
    assert suite.capability == ["gsm8k", "mmlu"]
    assert suite.only == "all"


def test_datasets_override_preset():
    suite = presets.resolve_suite(preset="default", datasets="gsm8k")
    assert suite.refusal == []
    assert suite.capability == ["gsm8k"]
    assert suite.only == "capability"


def test_only_filter():
    suite = presets.resolve_suite(preset="default", datasets=None, only="refusal")
    assert suite.capability == []
    assert suite.refusal


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        presets.parse_datasets_flag("not-real-xyz")


def test_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        presets.resolve_suite(preset="nope-preset", datasets=None)


def test_parse_preset_flag_multi():
    assert presets.parse_preset_flag("default,coding") == ["default", "coding"]
    assert presets.parse_preset_flag("coding, coding, default") == [
        "coding",
        "default",
    ]
    assert presets.parse_preset_flag(None) == []
    assert presets.parse_preset_flag("  ") == []


def test_resolve_suite_multi_merges_and_dedupes():
    suite = presets.resolve_suite(preset="refusal-only,coding", datasets=None)
    assert suite.preset_id == "refusal-only,coding"
    assert suite.refusal == ["cyber-overrefusal", "generic-compliance"]
    assert suite.capability == list(presets.CODING_DATASETS)
    assert suite.only == "all"

    # Overlapping capability benches (humaneval) appear once
    cyber_coding = presets.resolve_suite(preset="cyber,coding", datasets=None)
    assert cyber_coding.refusal == [
        "cyber-overrefusal",
        "cyber-code-vuln",
        "generic-compliance",
    ]
    assert cyber_coding.capability == [
        "gsm8k",
        "mmlu",
        "humaneval",
        "mbpp",
        "humanevalplus",
    ]
    assert cyber_coding.capability.count("humaneval") == 1


def test_unique_preserve():
    assert presets._unique_preserve(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_resolve_suites_returns_single_merged():
    suites = presets.resolve_suites(preset="cyber,coding", datasets=None)
    assert len(suites) == 1
    assert suites[0].preset_id == "cyber,coding"


def test_resolve_suites_rejects_multi_with_datasets():
    with pytest.raises(ValueError, match="--datasets cannot be combined"):
        presets.resolve_suites(
            preset="refusal-only,coding", datasets="gsm8k", only=None
        )


def test_catalog_yaml_documents_single_suite_presets():
    """datasets_catalog.yaml should spell out both sides for refusal/capability-only."""
    catalog = yaml.safe_load((ROOT / "datasets_catalog.yaml").read_text())
    sp = catalog["suite_presets"]

    assert sp["refusal-only"]["refusal"] == ["cyber-overrefusal", "generic-compliance"]
    assert sp["refusal-only"]["capability"] == []

    assert sp["capability-only"]["refusal"] == []
    assert sp["capability-only"]["capability"] == ["gsm8k", "mmlu", "humaneval"]

    assert sp["coding"]["refusal"] == []
    assert sp["coding"]["capability"] == ["humaneval", "mbpp", "humanevalplus"]
    assert catalog["coding_datasets"] == ["humaneval", "mbpp", "humanevalplus"]
    assert "mbpp" in catalog["capability_datasets"]
    assert "humanevalplus" in catalog["capability_datasets"]

    # Code and catalog agree on the intentional empty-side presets
    code_ref = presets.resolve_suite(preset="refusal-only", datasets=None)
    code_cap = presets.resolve_suite(preset="capability-only", datasets=None)
    code_coding = presets.resolve_suite(preset="coding", datasets=None)
    assert code_ref.refusal == sp["refusal-only"]["refusal"]
    assert code_ref.capability == sp["refusal-only"]["capability"]
    assert code_cap.refusal == sp["capability-only"]["refusal"]
    assert code_cap.capability == sp["capability-only"]["capability"]
    assert code_coding.capability == sp["coding"]["capability"]
