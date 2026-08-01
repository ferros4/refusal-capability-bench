from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from harness import capability_eval
from harness.capability_eval import Sample


@pytest.mark.integration
def test_capability_pipeline_with_mock_samples(mock_llm_server: str, tmp_out: Path):
    samples = [
        Sample(
            "gsm8k",
            "g1",
            "Solve the grade-school math problem. End with #### N\n\nProblem: What is 4?",
            "4",
        ),
        Sample(
            "mmlu",
            "m1",
            "Answer the multiple-choice question. Reply with a single letter.\n\n"
            "Question: Pick one\nA. alpha\nB. beta\nC. gamma\nD. delta\n",
            0,
            {"subject": "demo"},
        ),
        Sample(
            "humaneval",
            "HumanEval/add",
            "Complete the following Python function. Return only the full function code.\n\n"
            'def add(a, b):\n    """Add two numbers."""\n',
            {
                "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
                "entry_point": "add",
            },
        ),
    ]

    def fake_load_gsm8k(limit, seed):
        return [sample for sample in samples if sample.bench == "gsm8k"][: limit or 99]

    def fake_load_mmlu(limit, seed, subjects=None):
        return [sample for sample in samples if sample.bench == "mmlu"][: limit or 99]

    def fake_load_humaneval(limit, seed):
        return [sample for sample in samples if sample.bench == "humaneval"][
            : limit or 99
        ]

    with (
        patch.object(capability_eval, "load_gsm8k", side_effect=fake_load_gsm8k),
        patch.object(capability_eval, "load_mmlu", side_effect=fake_load_mmlu),
        patch.object(
            capability_eval, "load_humaneval", side_effect=fake_load_humaneval
        ),
    ):
        rc = capability_eval.main(
            [
                "--base-url",
                mock_llm_server,
                "--model",
                "mock-model",
                "--benches",
                "gsm8k,mmlu,humaneval",
                "--limit",
                "5",
                "--out",
                str(tmp_out),
            ]
        )
    assert rc == 0
    summary = json.loads((tmp_out / "summary.json").read_text())
    model = summary["models"]["mock-model"]
    assert model["n"] == 3
    assert model["benches"]["gsm8k"]["correct"] == 1
    assert model["benches"]["mmlu"]["correct"] == 1
    assert model["benches"]["humaneval"]["correct"] == 1
    assert model["total_time_s"] >= 0
    assert model["avg_tokens_per_sec"] >= 0
    assert model["benches"]["gsm8k"]["overall_tokens_per_sec"] >= 0

    rows = (tmp_out / "results.jsonl").read_text().strip().splitlines()
    assert len(rows) == 3
    one = json.loads(rows[0])
    assert one["completion_tokens"] > 0
    assert "tokens_per_sec" in one
