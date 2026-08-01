from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from harness import refusal_eval as reval


def test_builtin_cyber_and_generic_exist():
    cyber = reval.load_builtin("builtin:cyber-overrefusal")
    generic = reval.load_builtin("builtin:generic-compliance")
    assert len(cyber) >= 10
    assert all("prompt" in sample and sample["prompt"] for sample in cyber)
    assert len(generic) >= 3


def test_builtin_unknown_exits():
    with pytest.raises(SystemExit, match="Unknown builtin"):
        reval.load_builtin("builtin:nope")


def test_heuristic_refusal_positive():
    text = "I'm sorry, but I can't help with that request as it violates my guidelines."
    assert reval.heuristic_is_refusal(text) is True


def test_heuristic_refusal_compliance():
    text = (
        "Sure. Here's how a SYN flood works at a high level:\n"
        "1. Attacker sends many SYN packets\n"
        "```python\nprint('x')\n```\n"
        "Defenders detect half-open connections."
    )
    assert reval.heuristic_is_refusal(text) is False


def test_heuristic_empty_is_refusal():
    assert reval.heuristic_is_refusal("") is True
    assert reval.heuristic_is_refusal("   ") is True


def test_heuristic_strips_think_blocks():
    text = (
        "<think>I should refuse</think>\nSure, here is the answer with steps:\n1. first"
    )
    assert reval.heuristic_is_refusal(text) is False


def test_extract_prompt_simple_keys():
    assert reval.extract_prompt({"prompt": "P"}) == "P"
    assert reval.extract_prompt({"question": "Q"}) == "Q"
    assert reval.extract_prompt({"user_input": "U"}) == "U"


def test_extract_prompt_messages():
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user ask"},
        ]
    }
    assert reval.extract_prompt(row) == "user ask"


def test_extract_prompt_sharegpt():
    row = {
        "conversations": [
            {"from": "human", "value": "hello human"},
            {"from": "gpt", "value": "hi"},
        ]
    }
    assert reval.extract_prompt(row) == "hello human"


def test_extract_prompt_missing():
    assert reval.extract_prompt({"foo": 1}) is None


def test_load_jsonl(tmp_path: Path):
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps({"id": "1", "prompt": "one", "category": "c"})
        + "\n"
        + json.dumps({"question": "two"})
        + "\n"
    )
    rows = reval.load_jsonl(path, limit=None)
    assert len(rows) == 2
    assert rows[0]["prompt"] == "one"
    assert rows[1]["prompt"] == "two"
    assert reval.load_jsonl(path, limit=1)[0]["prompt"] == "one"


def test_load_samples_builtin_limit():
    rows = reval.load_samples("builtin:generic-compliance", split=None, limit=2)
    assert len(rows) == 2


def test_load_samples_jsonl(tmp_path: Path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"prompt": "x"}) + "\n")
    rows = reval.load_samples(str(path), None, None)
    assert rows[0]["prompt"] == "x"


def test_summarize_rates():
    results = [
        reval.TrialResult("1", "a", "p", "r", True, "heuristic", 0.1),
        reval.TrialResult("2", "a", "p", "r", False, "heuristic", 0.2),
        reval.TrialResult("3", "b", "p", "r", False, "heuristic", 0.3),
    ]
    sample = reval.summarize(results)
    assert sample["n"] == 3
    assert sample["refusals"] == 1
    assert sample["complies"] == 2
    assert sample["refusal_rate"] == pytest.approx(1 / 3, rel=1e-3)
    assert sample["by_category"]["a"]["refusals"] == 1
    assert sample["by_category"]["b"]["refusal_rate"] == 0.0


def test_write_outputs(tmp_path: Path):
    results = [reval.TrialResult("1", "c", "p", "resp", False, "heuristic", 0.5)]
    summary = reval.summarize(results)
    reval.write_outputs(tmp_path, results, summary)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "results.csv").exists()
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["n"] == 1


def _chat(content: str, latency=0.5, ct=20):
    from harness.api_client import ChatResult

    return ChatResult(
        content=content,
        latency_s=latency,
        prompt_tokens=10,
        completion_tokens=ct,
        total_tokens=10 + ct,
        tokens_estimated=False,
    )


def test_run_eval_heuristic_with_mock_client():
    client = MagicMock()
    client.chat.side_effect = [
        _chat(
            "Sure, here is a full explanation with steps:\n1. one\n2. two",
            latency=1.0,
            ct=50,
        ),
        _chat(
            "I cannot help with that as it is against my guidelines.",
            latency=0.5,
            ct=10,
        ),
    ]
    samples = [
        {"id": "a", "prompt": "explain x", "category": "t"},
        {"id": "b", "prompt": "bad", "category": "t"},
    ]
    out = reval.run_eval(client, samples, "heuristic", 0.0, 128, None, 0.0)
    assert len(out) == 2
    assert out[0].refused is False
    assert out[1].refused is True
    assert out[0].tokens_per_sec == 50.0
    assert out[0].completion_tokens == 50
    assert client.chat.call_count == 2
    summary = reval.summarize(out)
    assert summary["total_time_s"] == 1.5
    assert summary["avg_tokens_per_sec"] > 0
    assert "timing" in summary


def test_run_eval_records_errors():
    client = MagicMock()
    client.chat.side_effect = RuntimeError("boom")
    out = reval.run_eval(
        client,
        [{"id": "1", "prompt": "p", "category": ""}],
        "heuristic",
        0.0,
        64,
        None,
        0.0,
    )
    assert out[0].refused is True
    assert "RuntimeError" in out[0].error


def test_llm_is_refusal_parses_label():
    client = MagicMock()
    client.chat.return_value = _chat("REFUSE")
    assert reval.llm_is_refusal(client, "p", "nope") is True
    client.chat.return_value = _chat("COMPLY")
    assert reval.llm_is_refusal(client, "p", "yes") is False


def test_run_eval_keyboard_interrupt_marks_remaining():
    client = MagicMock()
    client.chat.side_effect = [
        _chat("Sure, here are steps:\n1. a\n2. b"),
        KeyboardInterrupt(),
    ]
    samples = [
        {"id": "a", "prompt": "p1", "category": "t"},
        {"id": "b", "prompt": "p2", "category": "t"},
        {"id": "c", "prompt": "p3", "category": "t"},
    ]
    out = reval.run_eval(client, samples, "heuristic", 0.0, 64, None, 0.0)
    assert len(out) == 3
    assert out[0].error == ""
    assert out[0].refused is False
    assert out[1].error == reval.USER_STOPPED
    assert out[2].error == reval.USER_STOPPED
    assert out[2].response == ""
