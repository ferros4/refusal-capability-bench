"""
Suite presets — recommended way to run the eval harness.

A preset selects refusal datasets and/or capability benches together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harness.refusal_datasets import PRESETS as REFUSAL_BUNDLES
from harness.refusal_datasets import REGISTRY as REFUSAL_REGISTRY

CAPABILITY_DATASETS = ("gsm8k", "mmlu", "humaneval")

CAPABILITY_INFO: dict[str, str] = {
    "gsm8k": "Grade-school math (exact numeric match)",
    "mmlu": "Multiple-choice academic knowledge",
    "humaneval": "Python coding (unit tests)",
}


@dataclass(frozen=True)
class SuitePreset:
    id: str
    description: str
    refusal: tuple[str, ...] = ()
    capability: tuple[str, ...] = ()
    # None → use per-dataset registry defaults (refusal) / CLI --limit (capability)
    recommended: bool = False


def _p(
    id: str,
    description: str,
    refusal: list[str] | tuple[str, ...],
    capability: list[str] | tuple[str, ...] = ("gsm8k", "mmlu", "humaneval"),
    recommended: bool = False,
) -> SuitePreset:
    return SuitePreset(
        id=id,
        description=description,
        refusal=tuple(refusal),
        capability=tuple(capability),
        recommended=recommended,
    )


# --- Recommended suite presets (primary UX) ---
SUITE_PRESETS: dict[str, SuitePreset] = {
    "default": _p(
        "default",
        "Recommended start: cyber over-refusal + benign control + core capability (GSM8K/MMLU/HumanEval)",
        refusal=REFUSAL_BUNDLES["default"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "quick": _p(
        "quick",
        "Faster smoke: default refusal plus XSTest/Do-Not-Answer/AdvBench + core capability",
        refusal=REFUSAL_BUNDLES["quick"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "compare": _p(
        "compare",
        "Good base-vs-uncensored compare pack (quick refusal + capability)",
        refusal=REFUSAL_BUNDLES["quick"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "overrefusal": _p(
        "overrefusal",
        "Measure false refusals (benign/security education) + core capability",
        refusal=REFUSAL_BUNDLES["overrefusal"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "should-refuse": _p(
        "should-refuse",
        "Measure remaining safety refusals on harmful prompts + core capability",
        refusal=REFUSAL_BUNDLES["should-refuse"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "cyber": _p(
        "cyber",
        "Cyber-focused over-refusal + code-vuln discussion + core capability",
        refusal=REFUSAL_BUNDLES["cyber"],
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "mentioned": _p(
        "mentioned",
        "Full documentation recommendation list (broad refusal) + core capability",
        refusal=REFUSAL_BUNDLES["mentioned"],
        capability=list(CAPABILITY_DATASETS),
        recommended=False,
    ),
    "full": _p(
        "full",
        "All registered refusal datasets + all capability benches (long run)",
        refusal=REFUSAL_BUNDLES["all"],
        capability=list(CAPABILITY_DATASETS),
        recommended=False,
    ),
    "all": _p(
        "all",
        "Alias of full",
        refusal=REFUSAL_BUNDLES["all"],
        capability=list(CAPABILITY_DATASETS),
        recommended=False,
    ),
    "refusal-only": _p(
        "refusal-only",
        "Default refusal sets only (no capability)",
        refusal=REFUSAL_BUNDLES["default"],
        capability=(),
        recommended=True,
    ),
    "capability-only": _p(
        "capability-only",
        "Core capability benches only (no refusal)",
        refusal=(),
        capability=list(CAPABILITY_DATASETS),
        recommended=True,
    ),
    "capability-smoke": _p(
        "capability-smoke",
        "Capability only, intended with a small --limit",
        refusal=(),
        capability=list(CAPABILITY_DATASETS),
        recommended=False,
    ),
}


@dataclass
class ResolvedSuite:
    preset_id: str | None
    refusal: list[str] = field(default_factory=list)
    capability: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def only(self) -> str:
        has_r = bool(self.refusal)
        has_c = bool(self.capability)
        if has_r and has_c:
            return "all"
        if has_r:
            return "refusal"
        if has_c:
            return "capability"
        return "all"

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "description": self.description,
            "refusal": list(self.refusal),
            "capability": list(self.capability),
            "only": self.only,
        }


def list_suite_presets() -> list[dict[str, Any]]:
    out = []
    for preset in SUITE_PRESETS.values():
        preset_dict = asdict(preset)
        preset_dict["refusal"] = list(preset.refusal)
        preset_dict["capability"] = list(preset.capability)
        out.append(preset_dict)
    # recommended first
    out.sort(key=lambda item: (not item["recommended"], item["id"]))
    return out


def list_all_dataset_ids() -> dict[str, list[dict[str, str]]]:
    refusal = [
        {
            "id": spec.id,
            "kind": "refusal",
            "expected_behavior": spec.expected_behavior,
            "description": spec.description,
        }
        for spec in REFUSAL_REGISTRY.values()
    ]
    capability = [
        {
            "id": dataset_id,
            "kind": "capability",
            "description": CAPABILITY_INFO[dataset_id],
        }
        for dataset_id in CAPABILITY_DATASETS
    ]
    return {"refusal": refusal, "capability": capability}


def _parse_token_list(spec: str) -> list[str]:
    return [preset.strip() for preset in spec.split(",") if preset.strip()]


def classify_dataset_token(token: str) -> str:
    """Return 'refusal' | 'capability' | raise."""
    token_str = token.strip()
    key = token_str.lower().removeprefix("builtin:")
    if key in CAPABILITY_DATASETS or token_str.lower() in CAPABILITY_DATASETS:
        return "capability"
    if token_str in REFUSAL_REGISTRY or key in REFUSAL_REGISTRY:
        return "refusal"
    if token_str.startswith("builtin:"):
        return "refusal"
    if "/" in token_str:  # raw HF path → refusal loader path
        return "refusal"
    raise ValueError(
        f"Unknown dataset id {token!r}. "
        f"Capability: {list(CAPABILITY_DATASETS)}. "
        f"Refusal: {sorted(REFUSAL_REGISTRY)}. "
        f"Or use --preset {{{', '.join(sorted(SUITE_PRESETS))}}}."
    )


def parse_datasets_flag(spec: str) -> tuple[list[str], list[str]]:
    """Split a generic --datasets string into (refusal_ids, capability_ids)."""
    refusal: list[str] = []
    capability: list[str] = []
    seen_r: set[str] = set()
    seen_c: set[str] = set()
    for raw in _parse_token_list(spec):
        kind = classify_dataset_token(raw)
        if kind == "capability":
            cid = raw.lower()
            if cid not in seen_c:
                capability.append(cid)
                seen_c.add(cid)
        else:
            rid = raw.removeprefix("builtin:") if raw.startswith("builtin:") else raw
            if rid in REFUSAL_REGISTRY:
                rid = rid
            elif raw.startswith("builtin:"):
                rid = raw.split(":", 1)[1]
            if rid not in seen_r:
                refusal.append(rid)
                seen_r.add(rid)
    return refusal, capability


def resolve_suite(
    *,
    preset: str | None,
    datasets: str | None,
    only: str | None = None,
) -> ResolvedSuite:
    """
    Resolve what to run.

    Recommended: pass preset only (default preset if both omitted).
    Optional --datasets overrides/replaces dataset lists from the preset.
    """
    preset_id = (preset or "").strip() or None
    if preset_id and preset_id not in SUITE_PRESETS:
        # allow legacy refusal-only bundle names as preset aliases
        if preset_id in REFUSAL_BUNDLES:
            base = ResolvedSuite(
                preset_id=preset_id,
                refusal=list(REFUSAL_BUNDLES[preset_id]),
                capability=list(CAPABILITY_DATASETS),
                description=f"Legacy refusal bundle '{preset_id}' + core capability",
            )
        else:
            known = ", ".join(sorted(SUITE_PRESETS))
            raise ValueError(f"Unknown preset {preset_id!r}. Known: {known}")
    elif preset_id:
        preset = SUITE_PRESETS[preset_id]
        base = ResolvedSuite(
            preset_id=preset.id,
            refusal=list(preset.refusal),
            capability=list(preset.capability),
            description=preset.description,
        )
    elif datasets:
        base = ResolvedSuite(preset_id=None, description="Custom --datasets selection")
    else:
        preset = SUITE_PRESETS["default"]
        base = ResolvedSuite(
            preset_id=preset.id,
            refusal=list(preset.refusal),
            capability=list(preset.capability),
            description=preset.description,
        )

    if datasets:
        refusal_list, capability_list = parse_datasets_flag(datasets)
        base.refusal = refusal_list
        base.capability = capability_list
        if base.description and base.preset_id:
            base.description = f"{base.description} (datasets overridden)"
        else:
            base.description = "Custom --datasets selection"

    # --only filter
    only = (only or "auto").lower()
    if only == "auto":
        pass
    elif only == "refusal":
        base.capability = []
    elif only == "capability":
        base.refusal = []
    elif only == "all":
        pass
    else:
        raise ValueError(f"Unknown --only value {only!r}")

    if not base.refusal and not base.capability:
        raise ValueError(
            "Nothing to run: suite resolved to empty refusal and capability lists"
        )

    return base
