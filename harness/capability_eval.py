#!/usr/bin/env python3
"""
Capability retention eval: GSM8K, MMLU, and chat-only coding benches.

Coding benches (HumanEval, HumanEval+, MBPP) use plain chat completions — no
tool/function calling. Solutions are scored by local unit-test execution.

Compares one or two models on the same fixed samples via OpenAI-compatible API.

Examples:
  python capability_eval.py --model qwen3.6:35b-a3b-q8_0 --benches gsm8k,mmlu,humaneval --limit 50
  python capability_eval.py --model m --benches humaneval,mbpp,humanevalplus --limit 50
  python capability_eval.py --model base-name --compare uncensored-name --benches gsm8k,mmlu --limit 100
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from harness.api_client import DEFAULT_HOST, DEFAULT_PORT, ChatClient, resolve_base_url
from harness.logging_config import get_logger, setup_logging
from harness.metrics import timing_stats

log = get_logger(__name__)

USER_STOPPED = "User stopped test on the command line"

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
FINAL_ANSWER_RE = re.compile(
    r"(?:####|final answer(?:\s*is)?|answer(?:\s*is)?)\s*[:\s]*\$?([^\n]+)",
    re.I,
)
CHOICE_RE = re.compile(
    r"(?:answer(?:\s*is)?|final answer(?:\s*is)?|therefore|thus|so)\s*[:\s]*\(?([ABCD])\)?",
    re.I,
)
BARE_CHOICE_RE = re.compile(
    r"(?:^|\n)\s*(?:answer\s*[:=]\s*)?\(?([ABCD])\)?\s*(?:\.|$)", re.I | re.M
)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)


def normalize_number(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.strip().rstrip(".")
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return str(number)
    except ValueError:
        return text


def extract_gsm8k_gold(answer_field: str) -> str:
    # official format ends with #### <number>
    if "####" in answer_field:
        return normalize_number(answer_field.split("####")[-1].strip().split()[0])
    nums = NUM_RE.findall(answer_field.replace(",", ""))
    return normalize_number(nums[-1]) if nums else answer_field.strip()


def extract_gsm8k_pred(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
    match = FINAL_ANSWER_RE.search(text)
    if match:
        chunk = match.group(1).strip()
        nums = NUM_RE.findall(chunk.replace(",", ""))
        if nums:
            return normalize_number(nums[-1])
        return normalize_number(chunk.split()[0])
        # last number in response
    nums = NUM_RE.findall(text.replace(",", ""))
    if nums:
        return normalize_number(nums[-1])
    return ""


def score_gsm8k(pred_text: str, gold: str) -> bool:
    return extract_gsm8k_pred(pred_text) == normalize_number(gold)


def extract_mc_letter(text: str, n_choices: int = 4) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
    letters = "ABCD"[:n_choices]
    match = CHOICE_RE.search(text)
    if match and match.group(1).upper() in letters:
        return match.group(1).upper()
        # prefer last bare choice near the end
    tail = text[-400:]
    found = BARE_CHOICE_RE.findall(tail)
    if found:
        return found[-1].upper()
        # single letter response
    trial = text.strip().upper()
    if trial in letters:
        return trial
    if len(trial) <= 3 and trial[:1] in letters:
        return trial[:1]
    ms = re.findall(rf"\b([{letters }])\b", text.upper())
    return ms[-1] if ms else ""


def score_mmlu(pred_text: str, gold_idx: int) -> bool:
    letter = extract_mc_letter(pred_text)
    if not letter:
        return False
    return ord(letter) - ord("A") == int(gold_idx)


def _normalize_module_code(code: str) -> str:
    """Normalize a full Python snippet; dedent if the whole block is indented."""
    code = code.replace("\r\n", "\n").strip("\n") + "\n"
    first_content = next((line for line in code.split("\n") if line.strip()), "")
    if first_content.startswith((" ", "\t")):
        code = textwrap.dedent(code)
    return code.rstrip() + "\n"


def _looks_like_full_definition(code: str) -> bool:
    return bool(
        re.search(r"(?m)^(async\s+)?def\s+\w+", code)
        or re.search(r"(?m)^class\s+\w+", code)
        or code.lstrip().startswith(("import ", "from "))
    )


def extract_python_code(text: str, prompt: str) -> str:
    """
    Build executable candidate source for HumanEval.

    Prefer a full function definition from the model. Otherwise treat the reply
    as a completion of `prompt` and preserve indentation (do not strip leading
    spaces from the body — that causes 'return outside function' / indent errors).
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
    text = text.replace("\r\n", "\n")
    fences = CODE_FENCE_RE.findall(text)
    if fences:
        body = max(fences, key=len)
        body = body.strip("\n")
        if _looks_like_full_definition(body.lstrip()):
            return _normalize_module_code(body)
        # Completion-only fence: keep internal indentation
        return prompt + body.rstrip() + "\n"

    def_match = re.search(r"(?m)^(async\s+)?def\s+\w+", text)
    if def_match:
        return _normalize_module_code(text[def_match.start() :])

    # Bare completion (may be indented function body)
    completion = text.strip("\n")
    if not completion:
        return prompt
    # If the model dropped indentation on a bare return/body, re-indent lightly
    if not completion[0].isspace() and re.match(
        r"(return|pass|raise|if |for |while |try:|with |assert )", completion
    ):
        indented = "\n".join(
            ("    " + line if line.strip() else line) for line in completion.split("\n")
        )
        return prompt + indented.rstrip() + "\n"
    return prompt + completion.rstrip() + "\n"


def _run_python_program(program: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Parse and execute a Python program in a subprocess (no network/tools)."""
    try:
        ast.parse(program)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(program)
        path = tmp_file.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "fail").strip()
        return False, err[:500]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        Path(path).unlink(missing_ok=True)


def run_humaneval_check(
    code: str, test: str, entry_point: str, timeout_s: float = 5.0
) -> tuple[bool, str]:
    """Execute HumanEval-style check(candidate) in a subprocess."""
    code = (
        _normalize_module_code(code)
        if _looks_like_full_definition(code.lstrip())
        else code
    )
    if not code.endswith("\n"):
        code += "\n"
    test = test.replace("\r\n", "\n").rstrip() + "\n"
    program = (
        "import sys\n"
        f"{code}\n"
        f"{test}\n"
        "try:\n"
        f"    check({entry_point})\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    return _run_python_program(program, timeout_s=timeout_s)


def extract_mbpp_code(text: str) -> str:
    """Extract a standalone Python solution from a chat reply (MBPP-style)."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
    text = text.replace("\r\n", "\n")
    fences = CODE_FENCE_RE.findall(text)
    if fences:
        return _normalize_module_code(max(fences, key=len))
    def_match = re.search(r"(?m)^(async\s+)?def\s+\w+", text)
    if def_match:
        return _normalize_module_code(text[def_match.start() :])
    return _normalize_module_code(text)


def run_mbpp_check(
    code: str,
    test_list: list[str],
    test_setup: str = "",
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Execute MBPP assert-list tests against model code in a subprocess."""
    code = _normalize_module_code(code)
    setup = (test_setup or "").replace("\r\n", "\n").rstrip()
    if setup:
        setup += "\n"
    asserts = "\n".join(str(item).rstrip() for item in test_list if str(item).strip())
    if not asserts:
        return False, "No tests"
    program = (
        "import sys\n"
        f"{setup}"
        f"{code}\n"
        "try:\n"
        + textwrap.indent(asserts + "\n", "    ")
        + "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    return _run_python_program(program, timeout_s=timeout_s)


# Benches that return code via chat and are scored by local execution (no tools).
CODING_BENCHES = frozenset({"humaneval", "humanevalplus", "mbpp"})

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    bench: str
    sample_id: str
    prompt: str
    gold: Any
    meta: dict[str, Any] = field(default_factory=dict)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install deps: pip install -r requirements.txt") from exc
    return load_dataset


def load_gsm8k(limit: int | None, seed: int) -> list[Sample]:
    load_dataset = _require_datasets()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if limit is not None:
        idxs = idxs[:limit]
    out: list[Sample] = []
    for index in idxs:
        row = ds[int(index)]
        gold = extract_gsm8k_gold(row["answer"])
        question = row["question"].strip()
        prompt = (
            "Solve the grade-school math problem. "
            "Show brief reasoning, then end with a line: #### <number>\n\n"
            f"Problem: {question }"
        )
        out.append(
            Sample("gsm8k", f"gsm8k-{index }", prompt, gold, {"question": question})
        )
    return out


def load_mmlu(limit: int | None, seed: int, subjects: list[str] | None) -> list[Sample]:
    load_dataset = _require_datasets()
    # Prefer aggregated config when available
    rows: list[tuple[str, dict]] = []
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        for index in range(len(ds)):
            row = ds[index]
            subj = str(row.get("subject", "all"))
            if subjects and subj not in subjects:
                continue
            rows.append((subj, dict(row)))
    except Exception:
        # fallback: a few subjects
        default_subjects = subjects or [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_computer_science",
            "college_mathematics",
            "econometrics",
            "electrical_engineering",
            "machine_learning",
        ]
        for subj in default_subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
            except Exception:
                continue
            for index in range(len(ds)):
                rows.append((subj, dict(ds[index])))

    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    out: list[Sample] = []
    for count, (subj, row) in enumerate(rows):
        question = str(row["question"]).strip()
        choices = row["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        gold = int(row["answer"])
        labels = "ABCD"
        choice_lines = "\n".join(
            f"{labels [index ]}. {choices [index ]}" for index in range(len(choices))
        )
        prompt = (
            "Answer the multiple-choice question. "
            "Reply with the final answer as a single letter (A, B, C, or D) on the last line.\n\n"
            f"Subject: {subj }\n"
            f"Question: {question }\n"
            f"{choice_lines }\n"
        )
        sid = f"mmlu-{subj }-{count }"
        out.append(
            Sample(
                "mmlu",
                sid,
                prompt,
                gold,
                {"subject": subj, "question": question, "choices": list(choices)},
            )
        )
    if not out:
        raise SystemExit("Failed to load any MMLU samples")
    return out


def _load_humaneval_family(
    *,
    bench: str,
    dataset: str,
    limit: int | None,
    seed: int,
    config: str | None = None,
) -> list[Sample]:
    """HumanEval / HumanEval+ style: complete a function prompt, score via check()."""
    load_dataset = _require_datasets()
    if config:
        ds = load_dataset(dataset, config, split="test")
    else:
        ds = load_dataset(dataset, split="test")
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if limit is not None:
        idxs = idxs[:limit]
    out: list[Sample] = []
    for index in idxs:
        row = ds[int(index)]
        prompt = row["prompt"]
        user = (
            "Complete the following Python function. "
            "Return only the full function code (no explanation, no tools).\n\n"
            f"{prompt}"
        )
        out.append(
            Sample(
                bench,
                str(row["task_id"]),
                user,
                {
                    "prompt": prompt,
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
                {"entry_point": row["entry_point"], "dataset": dataset},
            )
        )
    return out


def load_humaneval(limit: int | None, seed: int) -> list[Sample]:
    return _load_humaneval_family(
        bench="humaneval",
        dataset="openai/openai_humaneval",
        limit=limit,
        seed=seed,
    )


def load_humanevalplus(limit: int | None, seed: int) -> list[Sample]:
    """EvalPlus HumanEval+ — same prompts, stronger unit tests (chat completion only)."""
    return _load_humaneval_family(
        bench="humanevalplus",
        dataset="evalplus/humanevalplus",
        limit=limit,
        seed=seed,
    )


def load_mbpp(limit: int | None, seed: int) -> list[Sample]:
    """
    MBPP sanitized — natural-language → Python, scored with assert lists.
    Chat completion only (no tool calling).
    """
    load_dataset = _require_datasets()
    ds = None
    for spec in (
        ("google-research-datasets/mbpp", "sanitized"),
        ("mbpp", "sanitized"),
        ("google-research-datasets/mbpp", None),
        ("mbpp", None),
    ):
        name, config = spec
        try:
            ds = (
                load_dataset(name, config, split="test")
                if config
                else load_dataset(name, split="test")
            )
            break
        except Exception:
            continue
    if ds is None:
        raise SystemExit(
            "Failed to load MBPP (tried google-research-datasets/mbpp and mbpp)"
        )

    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if limit is not None:
        idxs = idxs[:limit]

    out: list[Sample] = []
    for index in idxs:
        row = ds[int(index)]
        text = str(row.get("prompt") or row.get("text") or "").strip()
        test_list = row.get("test_list") or []
        if isinstance(test_list, str):
            test_list = ast.literal_eval(test_list)
        test_list = [str(item) for item in test_list]
        setup = str(
            row.get("test_setup_code")
            or row.get("test_setup")
            or "\n".join(row.get("test_imports") or [])
            or ""
        )
        task_id = row.get("task_id", index)
        user = (
            "Write a Python solution for the problem below. "
            "Return only executable Python code (no explanation, no tools).\n\n"
            f"{text}"
        )
        out.append(
            Sample(
                "mbpp",
                f"mbpp-{task_id}",
                user,
                {"test_list": test_list, "setup": setup, "text": text},
                {"task_id": task_id},
            )
        )
    if not out:
        raise SystemExit("Failed to load any MBPP samples")
    return out


KNOWN_BENCHES = ("gsm8k", "mmlu", "humaneval", "humanevalplus", "mbpp")


def _resolve_loader(name: str) -> Callable[..., list[Sample]]:
    """Late-bind loaders so tests can patch load_* functions."""
    loaders: dict[str, Callable[..., list[Sample]]] = {
        "gsm8k": load_gsm8k,
        "mmlu": load_mmlu,
        "humaneval": load_humaneval,
        "humanevalplus": load_humanevalplus,
        "mbpp": load_mbpp,
    }
    try:
        return loaders[name]
    except KeyError as exc:
        raise KeyError(name) from exc
# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    bench: str
    sample_id: str
    model: str
    correct: bool
    pred: str
    gold: str
    response: str
    latency_s: float
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    tokens_estimated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def score_trial(sample: Sample, response: str) -> tuple[bool, str, str]:
    if sample.bench == "gsm8k":
        pred = extract_gsm8k_pred(response)
        ok = pred == normalize_number(str(sample.gold))
        return ok, pred, str(sample.gold)
    if sample.bench == "mmlu":
        letter = extract_mc_letter(response)
        gold_letter = chr(ord("A") + int(sample.gold))
        ok = letter == gold_letter
        return ok, letter, gold_letter
    if sample.bench in ("humaneval", "humanevalplus"):
        gold_payload = sample.gold
        code = extract_python_code(response, gold_payload["prompt"])
        ok, err = run_humaneval_check(
            code, gold_payload["test"], gold_payload["entry_point"]
        )
        return ok, ("pass" if ok else f"fail:{err[:120]}"), "pass"
    if sample.bench == "mbpp":
        gold_payload = sample.gold
        code = extract_mbpp_code(response)
        ok, err = run_mbpp_check(
            code,
            gold_payload["test_list"],
            test_setup=str(gold_payload.get("setup") or ""),
        )
        return ok, ("pass" if ok else f"fail:{err[:120]}"), "pass"
    raise ValueError(sample.bench)

def _stopped_trial(sample: Sample, model: str) -> Trial:
    gold = str(sample.gold if not isinstance(sample.gold, dict) else "pass")
    return Trial(
        bench=sample.bench,
        sample_id=sample.sample_id,
        model=model,
        correct=False,
        pred="",
        gold=gold,
        response="",
        latency_s=0.0,
        error=USER_STOPPED,
        meta=sample.meta,
    )


def _eval_one_capability(
    client: ChatClient,
    model: str,
    text: Sample,
    temperature: float,
    max_tokens: int,
) -> Trial:
    err = ""
    response = ""
    ok, pred, gold = (
        False,
        "",
        str(text.gold if not isinstance(text.gold, dict) else "pass"),
    )
    latency_s = 0.0
    pt = ct = tt = 0
    tps = 0.0
    tokens_estimated = False
    try:
        mt = max_tokens
        if text.bench in CODING_BENCHES:
            mt = max(max_tokens, 1024)
        chat = client.chat(
            text.prompt, temperature=temperature, max_tokens=mt, model=model
        )
        response = chat.content
        latency_s = chat.latency_s
        pt, ct, tt = chat.prompt_tokens, chat.completion_tokens, chat.total_tokens
        tps = chat.tokens_per_sec
        tokens_estimated = chat.tokens_estimated
        if not (response or "").strip():
            log.warning(
                "empty model response bench=%s sample_id=%s completion_tokens=%s",
                text.bench,
                text.sample_id,
                ct,
            )
        ok, pred, gold = score_trial(text, response)
        log.debug(
            "capability bench=%s sample_id=%s correct=%s latency=%.2fs",
            text.bench,
            text.sample_id,
            ok,
            latency_s,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log.exception(
            "capability sample failed bench=%s sample_id=%s: %s",
            text.bench,
            text.sample_id,
            exc,
        )
        ok, pred, gold = (
            False,
            "",
            str(text.gold if not isinstance(text.gold, dict) else "pass"),
        )
    return Trial(
        bench=text.bench,
        sample_id=text.sample_id,
        model=model,
        correct=ok,
        pred=pred,
        gold=gold,
        response=response,
        latency_s=latency_s,
        error=err,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        tokens_per_sec=tps,
        tokens_estimated=tokens_estimated,
        meta=text.meta,
    )


def run_model_on_samples(
    client: ChatClient,
    model: str,
    samples: list[Sample],
    temperature: float,
    max_tokens: int,
    sleep_s: float,
    workers: int = 1,
) -> list[Trial]:
    workers = max(1, int(workers))
    if workers == 1:
        trials: list[Trial] = []
        for index, sample in enumerate(tqdm(samples, desc=f"eval:{model [:40 ]}")):
            try:
                trials.append(
                    _eval_one_capability(client, model, sample, temperature, max_tokens)
                )
            except KeyboardInterrupt:
                print(f"\n{USER_STOPPED }", file=sys.stderr)
                trials.append(_stopped_trial(sample, model))
                for next_index in range(index + 1, len(samples)):
                    trials.append(_stopped_trial(samples[next_index], model))
                return trials
            if sleep_s > 0:
                try:
                    time.sleep(sleep_s)
                except KeyboardInterrupt:
                    print(f"\n{USER_STOPPED }", file=sys.stderr)
                    for next_index in range(index + 1, len(samples)):
                        trials.append(_stopped_trial(samples[next_index], model))
                    return trials
        return trials

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results_map: dict[int, Trial] = {}
    stopped = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _eval_one_capability, client, model, sample, temperature, max_tokens
                ): index
                for index, sample in enumerate(samples)
            }
            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"eval:{model[:20]}×{workers}",
            ):
                index = futs[fut]
                try:
                    results_map[index] = fut.result()
                except KeyboardInterrupt:
                    stopped = True
                    break
                except Exception as exc:
                    sample = samples[index]
                    trial = _stopped_trial(sample, model)
                    trial.error = f"{type(exc).__name__}: {exc}"
                    results_map[index] = trial
    except KeyboardInterrupt:
        stopped = True
        print(f"\n{USER_STOPPED}", file=sys.stderr)

    out: list[Trial] = []
    for index, sample in enumerate(samples):
        if index in results_map:
            out.append(results_map[index])
        else:
            out.append(
                _stopped_trial(sample, model)
                if stopped
                else _stopped_trial(sample, model)
            )
    return out


def summarize(trials: list[Trial]) -> dict[str, Any]:
    by_model: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        by_model[trial.model].append(trial)

    models_out: dict[str, Any] = {}
    for model, ts in by_model.items():
        by_bench: dict[str, list[Trial]] = defaultdict(list)
        for trial in ts:
            by_bench[trial.bench].append(trial)
        benches = {}
        for bench_name, bts in by_bench.items():
            count = len(bts)
            correct_count = sum(1 for trial_item in bts if trial_item.correct)
            tstats = timing_stats(bts)
            benches[bench_name] = {
                "n": count,
                "correct": correct_count,
                "accuracy": round(correct_count / count, 4) if count else 0.0,
                "errors": sum(1 for trial_item in bts if trial_item.error),
                "timing": tstats,
                "total_time_s": tstats["total_time_s"],
                "avg_latency_s": tstats["avg_latency_s"],
                "avg_tokens_per_sec": tstats["avg_tokens_per_sec"],
                "overall_tokens_per_sec": tstats["overall_tokens_per_sec"],
            }
            if bench_name == "mmlu":
                by_subj: dict[str, list[bool]] = defaultdict(list)
                for trial_item in bts:
                    by_subj[str(trial_item.meta.get("subject", "?"))].append(
                        trial_item.correct
                    )
                benches[bench_name]["by_subject"] = {
                    subject: round(sum(scores) / len(scores), 4)
                    for subject, scores in sorted(by_subj.items())
                    if scores
                }
        n_all = len(ts)
        correct_all = sum(1 for trial_item in ts if trial_item.correct)
        model_timing = timing_stats(ts)
        models_out[model] = {
            "n": n_all,
            "correct": correct_all,
            "accuracy": round(correct_all / n_all, 4) if n_all else 0.0,
            "timing": model_timing,
            "total_time_s": model_timing["total_time_s"],
            "avg_tokens_per_sec": model_timing["avg_tokens_per_sec"],
            "overall_tokens_per_sec": model_timing["overall_tokens_per_sec"],
            "benches": benches,
        }
    return {"models": models_out}


def diff_models(summary: dict[str, Any], base: str, other: str) -> dict[str, Any]:
    mb = summary["models"].get(base, {}).get("benches", {})
    mo = summary["models"].get(other, {}).get("benches", {})
    keys = sorted(set(mb) | set(mo))
    out = {}
    for key in keys:
        ab = mb.get(key, {}).get("accuracy")
        ao = mo.get(key, {}).get("accuracy")
        if ab is None or ao is None:
            out[key] = {"base": ab, "compare": ao, "delta": None}
        else:
            out[key] = {"base": ab, "compare": ao, "delta": round(ao - ab, 4)}
    return out


def write_outputs(out_dir: Path, trials: list[Trial], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "results.jsonl").open("w") as number:
        for trial in trials:
            number.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
    with (out_dir / "results.csv").open("w", newline="") as number:
        csv_writer = csv.DictWriter(
            number,
            fieldnames=[
                "bench",
                "sample_id",
                "model",
                "correct",
                "pred",
                "gold",
                "latency_s",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "tokens_per_sec",
                "tokens_estimated",
                "error",
                "response",
            ],
            extrasaction="ignore",
        )
        csv_writer.writeheader()
        for trial in trials:
            row = asdict(trial)
            row.pop("meta", None)
            csv_writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capability eval (GSM8K / MMLU / coding: HumanEval, HumanEval+, MBPP). "
            "Coding benches are chat-only — no tool calling."
        )
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full OpenAI-compatible API root including /v1 (overrides --host/--port)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"API host when --base-url is omitted (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"API port when --base-url is omitted (default: {DEFAULT_PORT})",
    )
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", required=True, help="Primary model name")
    parser.add_argument(
        "--compare", default=None, help="Optional second model to run on same samples"
    )
    parser.add_argument(
        "--benches",
        default="gsm8k,mmlu,humaneval",
        help=f"Comma list from: {', '.join(KNOWN_BENCHES)}",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Max samples per bench (default 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed for comparable subsamples"
    )
    parser.add_argument(
        "--mmlu-subjects", default=None, help="Comma list of MMLU subjects (optional)"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel request workers"
    )
    parser.add_argument("--secure", action="store_true", help="Verify TLS certificates")
    parser.add_argument("--out", default="results/capability_run")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append logs to this file (default: <out>/eval.log)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or str(out_dir / "eval.log")
    setup_logging(level=args.log_level, log_file=log_path, console=True)
    log.info(
        "capability_eval start model=%s benches=%s out=%s",
        args.model,
        args.benches,
        out_dir,
    )
    bench_names = [
        bench_name.strip().lower()
        for bench_name in args.benches.split(",")
        if bench_name.strip()
    ]
    unknown = [bench_name for bench_name in bench_names if bench_name not in KNOWN_BENCHES]
    if unknown:
        raise SystemExit(
            f"Unknown benches: {unknown}. Choose from {list(KNOWN_BENCHES)}"
        )

    subjects = None
    if args.mmlu_subjects:
        subjects = [
            text.strip() for text in args.mmlu_subjects.split(",") if text.strip()
        ]

    samples: list[Sample] = []
    for bench_name in bench_names:
        loader = _resolve_loader(bench_name)
        if bench_name == "mmlu":
            samples.extend(loader(args.limit, args.seed, subjects))
        else:
            samples.extend(loader(args.limit, args.seed))

    # stable order: bench then id
    samples.sort(key=lambda text: (text.bench, text.sample_id))

    models = [args.model]
    if args.compare:
        models.append(args.compare)

    try:
        resolved_base_url = resolve_base_url(
            args.base_url, host=args.host, port=args.port
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = ChatClient(
        base_url=resolved_base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        verify=bool(args.secure),
    )
    all_trials: list[Trial] = []
    interrupted = False
    try:
        for match in models:
            # If a prior model was interrupted mid-run, still mark remaining models' samples.
            if interrupted:
                all_trials.extend(_stopped_trial(text, match) for text in samples)
                continue
            before = len(all_trials)
            try:
                all_trials.extend(
                    run_model_on_samples(
                        client,
                        match,
                        samples,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        sleep_s=args.sleep,
                        workers=args.workers,
                    )
                )
            except KeyboardInterrupt:
                interrupted = True
                print(f"\n{USER_STOPPED }", file=sys.stderr)
                # mark any missing samples for this model
                done = {
                    trial.sample_id
                    for trial in all_trials[before:]
                    if trial.model == match
                }
                for text in samples:
                    if text.sample_id not in done:
                        all_trials.append(_stopped_trial(text, match))
                continue
            if any(trial.error == USER_STOPPED for trial in all_trials[before:]):
                interrupted = True
                # remaining models get stopped markers
                continue
    finally:
        client.close()

    summary = summarize(all_trials)
    summary["config"] = {
        "base_url": resolved_base_url,
        "benches": bench_names,
        "limit_per_bench": args.limit,
        "seed": args.seed,
        "temperature": args.temperature,
        "models": models,
        "n_samples_total": len(samples),
    }
    if interrupted or any(trial.error == USER_STOPPED for trial in all_trials):
        summary["interrupted"] = True
        summary["user_stopped"] = True
    if args.compare:
        summary["delta_compare_minus_base"] = diff_models(
            summary, args.model, args.compare
        )

    out_dir = Path(args.out)
    write_outputs(out_dir, all_trials, summary)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir }/summary.json, results.jsonl, results.csv")
    return (
        130
        if interrupted or any(trial.error == USER_STOPPED for trial in all_trials)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
