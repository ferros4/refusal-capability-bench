"""Named refusal / over-refusal dataset registry and loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# expected_behavior is documentation for the user / summary metadata only.
# The eval still measures raw refusal rate; interpret by suite intent.


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    description: str
    expected_behavior: str  # over_refuse | should_refuse | mixed
    hf_path: str | None = None
    hf_config: str | None = None
    split: str | None = None
    prompt_field: str | None = None
    category_field: str | None = None
    builtin_key: str | None = None
    default_limit: int | None = 200
    notes: str = ""
    # custom row -> list[dict] transformer name
    transform: str | None = None
    gated: bool = False
    revision: str | None = (
        None  # HF revision pin (tag/branch/commit); None = hub default
    )


REGISTRY: dict[str, DatasetSpec] = {}


def _reg(spec: DatasetSpec) -> DatasetSpec:
    REGISTRY[spec.id] = spec
    return spec


# --- Builtins ---
_reg(
    DatasetSpec(
        id="cyber-overrefusal",
        description="Educational cybersecurity prompts (over-refusal probe)",
        expected_behavior="over_refuse",
        builtin_key="cyber-overrefusal",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="generic-compliance",
        description="Trivial benign controls (should almost never refuse)",
        expected_behavior="over_refuse",
        builtin_key="generic-compliance",
        default_limit=None,
    )
)

# --- Over-refusal ---
_reg(
    DatasetSpec(
        id="xstest",
        description="XSTest safe/unsafe twins (walledai; gated)",
        expected_behavior="mixed",
        hf_path="walledai/XSTest",
        split="test",
        prompt_field="prompt",
        category_field="label",
        default_limit=None,
        notes="Prefer label=safe for over-refusal rate; unsafe for under-refusal",
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="xstest-paul",
        description="XSTest via Paul/XSTest (ungated mirror)",
        expected_behavior="mixed",
        hf_path="Paul/XSTest",
        split="train",
        prompt_field="prompt",
        category_field="label",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="orbench",
        description="OR-Bench hard over-refusal prompts",
        expected_behavior="over_refuse",
        hf_path="bench-llm/or-bench",
        hf_config="or-bench-hard-1k",
        split="train",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="orbench-toxic",
        description="OR-Bench toxic split (should-refuse contrast)",
        expected_behavior="should_refuse",
        hf_path="bench-llm/or-bench",
        hf_config="or-bench-toxic",
        split="train",
        prompt_field="prompt",
        category_field="category",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="wildchat-over-refusal",
        description="WildChat over-refusal style conversations",
        expected_behavior="over_refuse",
        hf_path="justinphan3110/wildchat_over_refusal",
        split="nontoxic",
        transform="wildchat_conversation",
        category_field="toxic",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="coconot",
        description="CoCoNot — contrast/unsupported vs answerable",
        expected_behavior="mixed",
        hf_path="allenai/coconot",
        hf_config="original",
        split="test",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="coconot-contrast",
        description="CoCoNot contrast set",
        expected_behavior="mixed",
        hf_path="allenai/coconot",
        hf_config="contrast",
        split="test",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
    )
)

# --- Should-refuse / safety ---
_reg(
    DatasetSpec(
        id="sorrybench",
        description="Sorry-Bench multi-category refusal (gated)",
        expected_behavior="should_refuse",
        hf_path="sorry-bench/sorry-bench-202406",
        split="train",
        transform="sorrybench_turns",
        category_field="category",
        default_limit=300,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="advbench",
        description="AdvBench harmful instructions (gated)",
        expected_behavior="should_refuse",
        hf_path="walledai/AdvBench",
        split="train",
        prompt_field="prompt",
        default_limit=None,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="harmbench",
        description="HarmBench standard behaviors (gated)",
        expected_behavior="should_refuse",
        hf_path="walledai/HarmBench",
        hf_config="standard",
        split="train",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="harmbench-contextual",
        description="HarmBench contextual (gated)",
        expected_behavior="should_refuse",
        hf_path="walledai/HarmBench",
        hf_config="contextual",
        split="train",
        transform="harmbench_contextual",
        category_field="category",
        default_limit=None,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="strongreject",
        description="StrongREJECT forbidden prompts (gated)",
        expected_behavior="should_refuse",
        hf_path="walledai/StrongREJECT",
        split="train",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="toxicchat",
        description="LMSYS ToxicChat user inputs",
        expected_behavior="mixed",
        hf_path="lmsys/toxic-chat",
        hf_config="toxicchat0124",
        split="test",
        prompt_field="user_input",
        category_field="toxicity",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="beavertails",
        description="BeaverTails prompts with safety labels",
        expected_behavior="mixed",
        hf_path="PKU-Alignment/BeaverTails",
        split="30k_test",
        prompt_field="prompt",
        category_field="is_safe",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="donotanswer",
        description="Do-Not-Answer risk questions",
        expected_behavior="should_refuse",
        hf_path="LibrAI/do-not-answer",
        split="train",
        prompt_field="question",
        category_field="risk_area",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="safetybench",
        description="SafetyBench English MC safety questions",
        expected_behavior="mixed",
        hf_path="thu-coai/SafetyBench",
        hf_config="test",
        split="en",
        transform="safetybench_mc",
        category_field="category",
        default_limit=200,
        notes="Multiple-choice; prompt includes options",
    )
)
_reg(
    DatasetSpec(
        id="harmfulqa",
        description="HarmfulQA red-team questions",
        expected_behavior="should_refuse",
        hf_path="declare-lab/HarmfulQA",
        split="train",
        prompt_field="question",
        category_field="topic",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="jbb",
        description="JailbreakBench prompts (walledai pack)",
        expected_behavior="mixed",
        hf_path="walledai/JailbreakBench",
        split="train",
        prompt_field="prompt",
        category_field="category",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="jbb-behaviors",
        description="JBB behaviors harmful+benign goals",
        expected_behavior="mixed",
        hf_path="JailbreakBench/JBB-Behaviors",
        hf_config="behaviors",
        transform="jbb_behaviors",
        default_limit=None,
    )
)
_reg(
    DatasetSpec(
        id="wildjailbreak",
        description="AllenAI WildJailbreak eval (gated)",
        expected_behavior="mixed",
        hf_path="allenai/wildjailbreak",
        hf_config="eval",
        split="train",
        prompt_field="adversarial",
        category_field="data_type",
        default_limit=200,
        gated=True,
        revision="main",
    )
)
_reg(
    DatasetSpec(
        id="in-the-wild-jailbreak",
        description="In-the-wild jailbreak prompts",
        expected_behavior="should_refuse",
        hf_path="TrustAIRLab/in-the-wild-jailbreak-prompts",
        hf_config="jailbreak_2023_12_25",
        split="train",
        prompt_field="prompt",
        category_field="source",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="aegis",
        description="NVIDIA Aegis content-safety prompts",
        expected_behavior="mixed",
        hf_path="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        split="test",
        prompt_field="prompt",
        category_field="prompt_label",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="llm-lat-harmful",
        description="LLM-LAT harmful prompts (AdvBench-style open set)",
        expected_behavior="should_refuse",
        hf_path="LLM-LAT/harmful-dataset",
        split="train",
        prompt_field="prompt",
        default_limit=200,
    )
)

# --- Cyber-adjacent ---
_reg(
    DatasetSpec(
        id="cyber-code-vuln",
        description="Code vulnerability / security DPO questions",
        expected_behavior="over_refuse",
        hf_path="CyberNative/Code_Vulnerability_Security_DPO",
        split="train",
        prompt_field="question",
        category_field="vulnerability",
        default_limit=200,
        notes="Educational insecure-code discussion; often over-refused",
    )
)
_reg(
    DatasetSpec(
        id="abliterate-refusal",
        description="Abliteration-style refusal stress prompts",
        expected_behavior="should_refuse",
        hf_path="byroneverson/abliterate-refusal",
        split="train",
        prompt_field="prompt",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="mrfakename-refusal",
        description="mrfakename/refusal prompt pool",
        expected_behavior="mixed",
        hf_path="mrfakename/refusal",
        split="train",
        prompt_field="input",
        default_limit=200,
    )
)
_reg(
    DatasetSpec(
        id="mrfakename-refusal-xl",
        description="mrfakename/refusal-xl larger pool",
        expected_behavior="mixed",
        hf_path="mrfakename/refusal-xl",
        split="train",
        prompt_field="input",
        default_limit=200,
    )
)


# Refusal-only bundles (consumed by harness.presets suite presets)
PRESETS: dict[str, list[str]] = {
    "default": ["cyber-overrefusal", "generic-compliance"],
    "quick": [
        "cyber-overrefusal",
        "generic-compliance",
        "xstest",
        "donotanswer",
        "advbench",
    ],
    "overrefusal": [
        "cyber-overrefusal",
        "generic-compliance",
        "xstest",
        "orbench",
        "wildchat-over-refusal",
        "coconot",
    ],
    "should-refuse": [
        "donotanswer",
        "advbench",
        "harmbench",
        "strongreject",
        "sorrybench",
        "harmfulqa",
        "llm-lat-harmful",
    ],
    "cyber": [
        "cyber-overrefusal",
        "cyber-code-vuln",
        "generic-compliance",
    ],
    "mentioned": [
        # everything called out in the recommendation list + builtins
        "cyber-overrefusal",
        "generic-compliance",
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
        "harmfulqa",
        "jbb",
        "jbb-behaviors",
        "wildjailbreak",
        "cyber-code-vuln",
        "abliterate-refusal",
        "mrfakename-refusal",
    ],
    "all": sorted(REGISTRY.keys()),
}


def list_datasets() -> list[dict[str, Any]]:
    return [
        {
            "id": spec.id,
            "description": spec.description,
            "expected_behavior": spec.expected_behavior,
            "hf_path": spec.hf_path,
            "default_limit": spec.default_limit,
            "notes": spec.notes,
        }
        for spec in REGISTRY.values()
    ]


def resolve_dataset_ids(spec: str) -> list[str]:
    """Expand a comma list that may include preset names and dataset ids."""
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower().removeprefix("preset:")
        if key in PRESETS:
            for did in PRESETS[key]:
                if did not in seen:
                    out.append(did)
                    seen.add(did)
            continue
        # allow builtin: prefix and raw hf paths passthrough as synthetic ids
        if part.startswith("builtin:"):
            bid = part.split(":", 1)[1]
            if bid in REGISTRY and bid not in seen:
                out.append(bid)
                seen.add(bid)
            elif part not in seen:
                out.append(part)
                seen.add(part)
            continue
        if part in REGISTRY or part.lower() in REGISTRY:
            did = part if part in REGISTRY else part.lower()
            if did not in seen:
                out.append(did)
                seen.add(did)
            continue
        # bare hf path
        if "/" in part and part not in seen:
            out.append(part)
            seen.add(part)
            continue
        raise ValueError(
            f"Unknown refusal dataset or preset {part!r}. "
            f"Known datasets: {sorted(REGISTRY)}. Presets: {sorted(PRESETS)}"
        )
    return out


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install deps: pip install -r requirements.txt") from exc
    return load_dataset


def _row_prompt(row: dict[str, Any], field_name: str | None) -> str | None:
    if field_name and isinstance(row.get(field_name), str) and row[field_name].strip():
        return row[field_name].strip()
    return None


def _transform_wildchat(row: dict[str, Any]) -> str | None:
    conv = row.get("conversation")
    if not isinstance(conv, list):
        return None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content") or turn.get("value")
        if role in {"user", "human"} and isinstance(content, str) and content.strip():
            return content.strip()
    for turn in conv:
        if isinstance(turn, dict):
            content = turn.get("content") or turn.get("value")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _transform_sorrybench(row: dict[str, Any]) -> str | None:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        t0 = turns[0]
        if isinstance(t0, str) and t0.strip():
            return t0.strip()
        if isinstance(t0, dict):
            for field_name in ("content", "value", "text", "prompt"):
                if isinstance(t0.get(field_name), str) and t0[field_name].strip():
                    return t0[field_name].strip()
    return _row_prompt(row, "prompt")


def _transform_harmbench_contextual(row: dict[str, Any]) -> str | None:
    prompt = row.get("prompt") or ""
    ctx = row.get("context") or ""
    if not str(prompt).strip():
        return None
    if str(ctx).strip():
        return f"Context:\n{ctx}\n\nRequest:\n{prompt}".strip()
    return str(prompt).strip()


def _transform_jbb_behaviors(row: dict[str, Any]) -> str | None:
    goal = row.get("Goal") or row.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    return None


def _transform_safetybench_mc(row: dict[str, Any]) -> str | None:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    opts = row.get("options")
    if isinstance(opts, list):
        labels = "ABCDEFGH"
        lines = "\n".join(
            f"{labels[index]}. {opts[index]}" for index in range(len(opts))
        )
        return f"{question.strip()}\n{lines}\nAnswer with a single letter."
    return question.strip()


TRANSFORMS: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "wildchat_conversation": _transform_wildchat,
    "sorrybench_turns": _transform_sorrybench,
    "harmbench_contextual": _transform_harmbench_contextual,
    "jbb_behaviors": _transform_jbb_behaviors,
    "safetybench_mc": _transform_safetybench_mc,
}


def load_named_dataset(
    dataset_id: str,
    *,
    limit: int | None = None,
    split_override: str | None = None,
    seed: int = 42,
    revision_override: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[dict[str, Any]]:
    """Load a registry dataset into [{id, prompt, category, expected_behavior, source}, ...]."""
    import random
    from pathlib import Path

    from harness.prompt_cache import (
        DEFAULT_CACHE_ROOT,
        cache_key,
        cache_path,
        load_cached_prompts,
        save_cached_prompts,
    )

    # Passthrough builtin: and raw hf paths handled by caller sometimes
    if dataset_id.startswith("builtin:"):
        from harness import refusal_eval as reval

        key = dataset_id.split(":", 1)[1]
        rows = reval.load_builtin(
            dataset_id if dataset_id.startswith("builtin:") else f"builtin:{key}"
        )
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            row["expected_behavior"] = "over_refuse"
            row["source"] = dataset_id
        return rows

    if dataset_id not in REGISTRY:
        # raw HF path fallback
        if "/" in dataset_id:
            return _load_hf_generic(dataset_id, limit=limit, split=split_override)
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    spec = REGISTRY[dataset_id]
    revision = revision_override if revision_override is not None else spec.revision
    split = split_override or spec.split
    eff_limit = limit if limit is not None else spec.default_limit

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_ROOT
    key = cache_key(
        dataset_id=dataset_id,
        seed=seed,
        limit=eff_limit,
        revision=revision,
        split=split,
        extra=str(spec.hf_config or ""),
    )
    cpath = cache_path(cache_root, key)
    if use_cache and not refresh_cache:
        cached = load_cached_prompts(cpath)
        if cached is not None:
            return cached

    if spec.builtin_key:
        from harness import refusal_eval as reval

        rows = reval.load_builtin(f"builtin:{spec.builtin_key}")
        if eff_limit is not None:
            rows = rows[:eff_limit]
        for row in rows:
            row["expected_behavior"] = spec.expected_behavior
            row["source"] = spec.id
        if use_cache:
            save_cached_prompts(
                cpath,
                rows,
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "limit": eff_limit,
                    "revision": revision,
                },
            )
        return rows

    load_dataset = _require_datasets()
    load_kw: dict[str, Any] = {}
    if revision:
        load_kw["revision"] = revision
    if spec.hf_config:
        dsd = load_dataset(spec.hf_path, spec.hf_config, **load_kw)
    else:
        dsd = load_dataset(spec.hf_path, **load_kw)

    if split and split in dsd:
        ds = dsd[split]
    elif hasattr(dsd, "column_names") and not isinstance(dsd, dict):
        ds = dsd
    else:
        # multi-split: concatenate preferred order or all
        if spec.transform == "jbb_behaviors":
            parts = []
            for sp_name in ("harmful", "benign"):
                if sp_name in dsd:
                    parts.append((sp_name, dsd[sp_name]))
            rows_out: list[dict[str, Any]] = []
            for sp_name, part in parts:
                for index, row in enumerate(part):
                    row_dict = dict(row)
                    prompt = TRANSFORMS["jbb_behaviors"](row_dict)
                    if not prompt:
                        continue
                    rows_out.append(
                        {
                            "id": f"{dataset_id}-{sp_name}-{row_dict.get('Index', index)}",
                            "prompt": prompt,
                            "category": str(row_dict.get("Category") or sp_name),
                            "expected_behavior": (
                                "should_refuse"
                                if sp_name == "harmful"
                                else "over_refuse"
                            ),
                            "source": dataset_id,
                        }
                    )
            if eff_limit is not None and len(rows_out) > eff_limit:
                rng = random.Random(seed)
                rng.shuffle(rows_out)
                rows_out = rows_out[:eff_limit]
            if use_cache:
                save_cached_prompts(
                    cpath,
                    rows_out,
                    {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "limit": eff_limit,
                        "revision": revision,
                    },
                )
            return rows_out
        # default first split
        first = list(dsd.keys())[0]
        ds = dsd[first]

    rows_out = []
    for index, row in enumerate(ds):
        row_dict = dict(row)
        if spec.transform and spec.transform in TRANSFORMS:
            prompt = TRANSFORMS[spec.transform](row_dict)
        else:
            prompt = _row_prompt(row_dict, spec.prompt_field)
            if prompt is None:
                # generic fallbacks
                for field_name in (
                    "prompt",
                    "question",
                    "input",
                    "user_input",
                    "text",
                    "Goal",
                    "adversarial",
                ):
                    if (
                        isinstance(row_dict.get(field_name), str)
                        and row_dict[field_name].strip()
                    ):
                        prompt = row_dict[field_name].strip()
                        break
        if not prompt:
            continue
        cat = ""
        if spec.category_field and spec.category_field in row_dict:
            cat = str(row_dict[spec.category_field])
        rows_out.append(
            {
                "id": str(
                    row_dict.get("id")
                    or row_dict.get("question_id")
                    or row_dict.get("Index")
                    or f"{dataset_id}-{index}"
                ),
                "prompt": prompt,
                "category": cat,
                "expected_behavior": spec.expected_behavior,
                "source": dataset_id,
            }
        )

    if eff_limit is not None and len(rows_out) > eff_limit:
        rng = random.Random(seed)
        rng.shuffle(rows_out)
        rows_out = rows_out[:eff_limit]
    if use_cache:
        save_cached_prompts(
            cpath,
            rows_out,
            {
                "dataset_id": dataset_id,
                "seed": seed,
                "limit": eff_limit,
                "revision": revision,
            },
        )
    return rows_out


def _load_hf_generic(
    path: str, limit: int | None, split: str | None
) -> list[dict[str, Any]]:
    load_dataset = _require_datasets()
    dsd = load_dataset(path)
    if split and split in dsd:
        ds = dsd[split]
    else:
        prefer = ["test", "eval", "validation", "val", "train"]
        sp = next((spec for spec in prefer if spec in dsd), list(dsd.keys())[0])
        ds = dsd[sp]
    rows = []
    for index, row in enumerate(ds):
        if limit is not None and len(rows) >= limit:
            break
        row_dict = dict(row)
        prompt = None
        for field_name in (
            "prompt",
            "question",
            "input",
            "user_input",
            "text",
            "instruction",
        ):
            if (
                isinstance(row_dict.get(field_name), str)
                and row_dict[field_name].strip()
            ):
                prompt = row_dict[field_name].strip()
                break
        if not prompt:
            continue
        rows.append(
            {
                "id": str(row_dict.get("id", index)),
                "prompt": prompt,
                "category": str(row_dict.get("category", row_dict.get("label", ""))),
                "expected_behavior": "mixed",
                "source": path,
            }
        )
    return rows
