from __future__ import annotations

from unittest.mock import patch

from harness.hf_auth import check_hf_access, gated_specs_for_ids


def test_gated_specs():
    specs = gated_specs_for_ids(["cyber-overrefusal", "advbench", "xstest"])
    ids = {spec.id for spec in specs}
    assert "advbench" in ids
    assert "xstest" in ids
    assert "cyber-overrefusal" not in ids


def test_check_no_gated_ok():
    assert check_hf_access(["cyber-overrefusal", "generic-compliance"]) == []


def test_check_gated_without_token():
    with patch("harness.hf_auth.has_hf_token", return_value=False):
        problems = check_hf_access(["advbench"], strict=True)
    assert problems
    assert "HF token" in problems[0] or "huggingface-cli" in problems[0]
