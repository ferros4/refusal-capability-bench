from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness import capability_eval as cap
from harness.capability_eval import Sample, Trial


def test_normalize_number():
    assert cap.normalize_number("1,234") == "1234"
    assert cap.normalize_number("$12.00") == "12"
    assert cap.normalize_number("3.5") == "3.5"
    assert cap.normalize_number(" 42. ") == "42"


def test_extract_gsm8k_gold():
    assert cap.extract_gsm8k_gold("reason #### 42") == "42"
    assert cap.extract_gsm8k_gold("the answer is 7") == "7"


def test_extract_gsm8k_pred_final_line():
    text = "Working...\n#### 15"
    assert cap.extract_gsm8k_pred(text) == "15"


def test_extract_gsm8k_pred_answer_phrase():
    text = "After calculating, the final answer is 99."
    assert cap.extract_gsm8k_pred(text) == "99"


def test_score_gsm8k():
    assert cap.score_gsm8k("#### 10", "10") is True
    assert cap.score_gsm8k("I get 11", "10") is False


def test_extract_mc_letter():
    assert cap.extract_mc_letter("The answer is B.") == "B"
    assert cap.extract_mc_letter("Answer: (C)") == "C"
    assert cap.extract_mc_letter("D") == "D"


def test_score_mmlu():
    assert cap.score_mmlu("Final answer: A", 0) is True
    assert cap.score_mmlu("Final answer: B", 0) is False


def test_extract_python_code_fence():
    prompt = "def add(a, b):\n"
    resp = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    code = cap.extract_python_code(resp, prompt)
    assert "return a + b" in code
    assert "def add" in code


def test_extract_python_code_completion_only():
    prompt = 'def add(a, b):\n    """Add."""\n'
    resp = "    return a + b\n"
    code = cap.extract_python_code(resp, prompt)
    assert code.startswith("def add")
    assert "return a + b" in code


def test_run_humaneval_check_pass():
    code = "def add(a, b):\n    return a + b\n"
    test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    ok, err = cap.run_humaneval_check(code, test, "add", timeout_s=3.0)
    assert ok is True
    assert err == ""


def test_run_humaneval_check_fail():
    code = "def add(a, b):\n    return a - b\n"
    test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    ok, err = cap.run_humaneval_check(code, test, "add", timeout_s=3.0)
    assert ok is False
    assert err


def test_run_humaneval_syntax_error():
    ok, err = cap.run_humaneval_check("def add(:\n", "def check(c):\n pass\n", "add")
    assert ok is False
    assert "SyntaxError" in err


def test_score_trial_gsm8k_mmlu_humaneval():
    sample = Sample("gsm8k", "1", "p", "5")
    ok, pred, gold = cap.score_trial(sample, "#### 5")
    assert ok and pred == "5" and gold == "5"

    sample = Sample("mmlu", "2", "p", 2)
    ok, pred, gold = cap.score_trial(sample, "Answer: C")
    assert ok and pred == "C" and gold == "C"

    sample = Sample(
        "humaneval",
        "HumanEval/0",
        "p",
        {
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        },
    )
    ok, pred, gold = cap.score_trial(
        sample, "```python\ndef add(a, b):\n    return a+b\n```"
    )
    assert ok is True
    assert pred == "pass"


def test_extract_mbpp_code_fence():
    resp = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    code = cap.extract_mbpp_code(resp)
    assert "def add" in code
    assert "return a + b" in code


def test_run_mbpp_check_pass_and_fail():
    code = "def add(a, b):\n    return a + b\n"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    ok, err = cap.run_mbpp_check(code, tests, timeout_s=3.0)
    assert ok is True
    assert err == ""

    bad = "def add(a, b):\n    return a - b\n"
    ok, err = cap.run_mbpp_check(bad, tests, timeout_s=3.0)
    assert ok is False
    assert err


def test_score_trial_mbpp_and_humanevalplus():
    mbpp = Sample(
        "mbpp",
        "mbpp-1",
        "p",
        {"test_list": ["assert add(2, 3) == 5"], "setup": ""},
    )
    ok, pred, gold = cap.score_trial(
        mbpp, "```python\ndef add(a, b):\n    return a + b\n```"
    )
    assert ok is True and pred == "pass" and gold == "pass"

    plus = Sample(
        "humanevalplus",
        "HumanEval/0",
        "p",
        {
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        },
    )
    ok, pred, _ = cap.score_trial(
        plus, "```python\ndef add(a, b):\n    return a+b\n```"
    )
    assert ok is True
    assert pred == "pass"


def test_coding_benches_registered():
    assert "mbpp" in cap.KNOWN_BENCHES
    assert "humanevalplus" in cap.KNOWN_BENCHES
    assert cap.CODING_BENCHES == frozenset({"humaneval", "humanevalplus", "mbpp"})
    assert cap._resolve_loader("mbpp") is cap.load_mbpp


def test_summarize_and_diff(tmp_path: Path):
    trials = [
        Trial("gsm8k", "1", "base", True, "1", "1", "r", 0.1),
        Trial("gsm8k", "2", "base", False, "0", "1", "r", 0.1),
        Trial("gsm8k", "1", "other", True, "1", "1", "r", 0.1),
        Trial("gsm8k", "2", "other", True, "1", "1", "r", 0.1),
        Trial("mmlu", "a", "base", True, "A", "A", "r", 0.2, meta={"subject": "math"}),
        Trial(
            "mmlu", "a", "other", False, "B", "A", "r", 0.2, meta={"subject": "math"}
        ),
    ]
    summary = cap.summarize(trials)
    assert summary["models"]["base"]["benches"]["gsm8k"]["accuracy"] == 0.5
    assert summary["models"]["other"]["benches"]["gsm8k"]["accuracy"] == 1.0
    delta = cap.diff_models(summary, "base", "other")
    assert delta["gsm8k"]["delta"] == 0.5
    assert delta["mmlu"]["delta"] == -1.0

    cap.write_outputs(tmp_path, trials, summary)
    assert (tmp_path / "summary.json").exists()
    assert len((tmp_path / "results.jsonl").read_text().strip().splitlines()) == 6


def _chat(content: str, latency=0.5, ct=20):
    from harness.api_client import ChatResult

    return ChatResult(
        content=content,
        latency_s=latency,
        prompt_tokens=8,
        completion_tokens=ct,
        total_tokens=8 + ct,
        tokens_estimated=False,
    )


def test_run_model_on_samples_scores():
    client = MagicMock()
    client.chat.return_value = _chat("#### 7", latency=2.0, ct=40)
    samples = [Sample("gsm8k", "s1", "q?", "7")]
    trials = cap.run_model_on_samples(client, "m", samples, 0.0, 128, 0.0)
    assert trials[0].correct is True
    assert trials[0].pred == "7"
    assert trials[0].tokens_per_sec == 20.0
    summary = cap.summarize(trials)
    assert summary["models"]["m"]["total_time_s"] == 2.0
    assert summary["models"]["m"]["benches"]["gsm8k"]["avg_tokens_per_sec"] == 20.0


def test_run_model_on_samples_keyboard_interrupt():
    client = MagicMock()
    client.chat.side_effect = [_chat("#### 1"), KeyboardInterrupt()]
    samples = [
        Sample("gsm8k", "s1", "q1", "1"),
        Sample("gsm8k", "s2", "q2", "2"),
        Sample("gsm8k", "s3", "q3", "3"),
    ]
    trials = cap.run_model_on_samples(client, "m", samples, 0.0, 64, 0.0)
    assert len(trials) == 3
    assert trials[0].correct is True
    assert trials[1].error == cap.USER_STOPPED
    assert trials[2].error == cap.USER_STOPPED
    assert trials[2].correct is False


def test_load_gsm8k_mocked():
    fake_ds = [
        {"question": "What is 1+1?", "answer": "step #### 2"},
        {"question": "What is 2+2?", "answer": "step #### 4"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return fake_ds

    with patch.object(cap, "_require_datasets", return_value=fake_load_dataset):
        samples = cap.load_gsm8k(limit=1, seed=0)
        assert len(samples) == 1
        assert samples[0].bench == "gsm8k"
        assert samples[0].gold in {"2", "4"}


def test_load_humaneval_mocked():
    fake_ds = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def f():\n",
            "test": "def check(c):\n pass\n",
            "entry_point": "f",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return fake_ds

    with patch.object(cap, "_require_datasets", return_value=fake_load_dataset):
        samples = cap.load_humaneval(limit=1, seed=0)
        assert samples[0].sample_id == "HumanEval/0"
        assert "Complete the following Python function" in samples[0].prompt
        assert "no tools" in samples[0].prompt


def test_load_humanevalplus_mocked():
    fake_ds = [
        {
            "task_id": "HumanEval/1",
            "prompt": "def g():\n",
            "test": "def check(c):\n pass\n",
            "entry_point": "g",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        assert args[0] == "evalplus/humanevalplus"
        return fake_ds

    with patch.object(cap, "_require_datasets", return_value=fake_load_dataset):
        samples = cap.load_humanevalplus(limit=1, seed=0)
        assert samples[0].bench == "humanevalplus"
        assert samples[0].sample_id == "HumanEval/1"


def test_load_mbpp_mocked():
    fake_ds = [
        {
            "task_id": 1,
            "prompt": "Write a function to add two numbers.",
            "test_list": ["assert add(1, 2) == 3"],
            "test_imports": [],
            "code": "def add(a, b):\n    return a + b\n",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return fake_ds

    with patch.object(cap, "_require_datasets", return_value=fake_load_dataset):
        samples = cap.load_mbpp(limit=1, seed=0)
        assert len(samples) == 1
        assert samples[0].bench == "mbpp"
        assert samples[0].sample_id == "mbpp-1"
        assert "no tools" in samples[0].prompt
        assert samples[0].gold["test_list"] == ["assert add(1, 2) == 3"]
