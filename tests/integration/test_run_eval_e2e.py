from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import run_eval
from harness.capability_eval import Sample


@pytest.mark.integration
def test_run_eval_refusal_only_layout(mock_llm_server: str, tmp_path: Path):
    rc = run_eval.main(
        [
            "--base-url",
            mock_llm_server,
            "--model",
            "mock-model",
            "--preset",
            "refusal-only",
            "--datasets",
            "generic-compliance",
            "--out-root",
            str(tmp_path),
            "--run-id",
            "itest1",
        ]
    )
    assert rc == 0
    run_dir = tmp_path / "mock-model" / "itest1"
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "generic_compliance" / "summary.json").exists()
    assert (run_dir / "generic_compliance" / "results.csv").exists()

    combined = json.loads((run_dir / "summary.json").read_text())
    headlines = combined["headlines"]["refusal"]["mock-model"]["generic-compliance"]
    assert headlines["n"] >= 3
    assert headlines["refusal_rate"] == 0.0
    assert "total_time_s" in headlines
    assert "avg_tokens_per_sec" in headlines


@pytest.mark.integration
def test_run_eval_all_suites_mocked_capability(mock_llm_server: str, tmp_path: Path):
    samples = [
        Sample(
            "gsm8k", "g1", "Solve the grade-school math problem.\n\nProblem: 10?", "10"
        ),
    ]

    def fake_gsm8k(limit, seed):
        return samples[: limit or 1]

    def fake_empty(limit, seed, subjects=None):
        return []

    with (
        patch("harness.capability_eval.load_gsm8k", side_effect=fake_gsm8k),
        patch("harness.capability_eval.load_mmlu", side_effect=fake_empty),
        patch("harness.capability_eval.load_humaneval", side_effect=fake_empty),
    ):
        rc = run_eval.main(
            [
                "--base-url",
                mock_llm_server,
                "--model",
                "mock-model",
                "--datasets",
                "generic-compliance,gsm8k",
                "--limit",
                "1",
                "--out-root",
                str(tmp_path),
                "--run-id",
                "itest_all",
            ]
        )
    assert rc == 0
    run_dir = tmp_path / "mock-model" / "itest_all"
    assert (run_dir / "generic_compliance" / "summary.json").exists()
    assert (run_dir / "capability" / "summary.json").exists()
    cap = json.loads((run_dir / "capability" / "summary.json").read_text())
    assert cap["models"]["mock-model"]["benches"]["gsm8k"]["n"] == 1
    assert "overall_tokens_per_sec" in cap["models"]["mock-model"]

    combined = json.loads((run_dir / "summary.json").read_text())
    assert "capability" in combined["headlines"]
    assert (
        combined["headlines"]["capability"]["mock-model"]["benches"]["gsm8k"][
            "accuracy"
        ]
        == 1.0
    )


@pytest.mark.integration
def test_run_eval_compare_nests_refusal(mock_llm_server: str, tmp_path: Path):
    rc = run_eval.main(
        [
            "--base-url",
            mock_llm_server,
            "--model",
            "base-mod",
            "--compare",
            "other-mod",
            "--datasets",
            "cyber-overrefusal",
            "--dataset-limit",
            "2",
            "--out-root",
            str(tmp_path),
            "--run-id",
            "cmp1",
        ]
    )
    assert rc == 0
    run_dir = tmp_path / "base-mod_vs_other-mod" / "cmp1"
    assert (run_dir / "base_cyber" / "base-mod" / "summary.json").exists()
    assert (run_dir / "base_cyber" / "other-mod" / "summary.json").exists()
