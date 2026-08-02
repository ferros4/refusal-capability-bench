# Presets and datasets

Source of truth for suite presets: [`harness/presets.py`](../harness/presets.py)  
Human-readable catalog: [`datasets_catalog.yaml`](../datasets_catalog.yaml)  
Refusal registry: [`harness/refusal_datasets.py`](../harness/refusal_datasets.py)

```bash
refusal-capability-bench --list-presets
refusal-capability-bench --list-datasets
```

---

## Suite presets

Mirror of `suite_presets` in [`datasets_catalog.yaml`](../datasets_catalog.yaml).  
An **empty list on one side is intentional** (that suite is skipped).  
Outputs land under `results/<model>/<timestamp>/refusal/` or `…/capability/`.

| Preset | Rec. | Refusal | Capability | Notes |
|--------|:----:|---------|------------|-------|
| `default` | yes | `cyber-overrefusal`, `generic-compliance` | `gsm8k`, `mmlu`, `humaneval` | |
| `quick` | yes | `cyber-overrefusal`, `generic-compliance`, `xstest`, `donotanswer`, `advbench` | `gsm8k`, `mmlu`, `humaneval` | |
| `compare` | yes | same as `quick` | `gsm8k`, `mmlu`, `humaneval` | intended for `--compare` base vs uncensored |
| `overrefusal` | yes | `cyber-overrefusal`, `generic-compliance`, `xstest`, `orbench`, `wildchat-over-refusal`, `coconot` | `gsm8k`, `mmlu`, `humaneval` | |
| `should-refuse` | yes | `donotanswer`, `advbench`, `harmbench`, `strongreject`, `sorrybench`, `harmfulqa`, `llm-lat-harmful` | `gsm8k`, `mmlu`, `humaneval` | |
| `cyber` | yes | `cyber-overrefusal`, `cyber-code-vuln`, `generic-compliance` | `gsm8k`, `mmlu`, `humaneval` | |
| `refusal-only` | yes | `cyber-overrefusal`, `generic-compliance` | **`[]`** | default refusal pack; skip capability |
| `capability-only` | yes | **`[]`** | `gsm8k`, `mmlu`, `humaneval` | skip refusal |
| `capability-smoke` | | **`[]`** | `gsm8k`, `mmlu`, `humaneval` | same as capability-only; use a small `--limit` |
| `coding` | yes | **`[]`** | `humaneval`, `mbpp`, `humanevalplus` | chat-only coding; no tools |
| `coding-smoke` | | **`[]`** | `humaneval`, `mbpp`, `humanevalplus` | same as coding; use a small `--limit` |
| `mentioned` | | broad list — see `PRESETS["mentioned"]` in `refusal_datasets.py` | `gsm8k`, `mmlu`, `humaneval` | |
| `full` | | entire refusal registry | `gsm8k`, `mmlu`, `humaneval`, `mbpp`, `humanevalplus` | long run |
| `all` | | alias of `full` | same | |

```bash
refusal-capability-bench --model m --preset overrefusal
refusal-capability-bench --model m --preset should-refuse
refusal-capability-bench --model m --preset cyber
refusal-capability-bench --model m --preset refusal-only
refusal-capability-bench --model m --preset capability-only --limit 20
refusal-capability-bench --model m --preset coding --limit 20

# Multiple presets (comma-separated) — lists are merged and deduped
refusal-capability-bench --model m --preset cyber,coding --limit 20
refusal-capability-bench --model m --preset refusal-only,coding --limit 20
```

### Multiple `--preset` values

Comma-separated presets are **merged into one suite**:

1. Concatenate each preset’s refusal list and capability list (in the order given).
2. **Deduplicate** with order preserved (each dataset/bench id runs **at most once**).
3. Run the usual flat layout under `results/<model>/<timestamp>/{refusal,capability}/`.

Example: `--preset cyber,coding`

| Side | Result (deduped) |
|------|------------------|
| refusal | `cyber-overrefusal`, `cyber-code-vuln`, `generic-compliance` |
| capability | `gsm8k`, `mmlu`, `humaneval`, `mbpp`, `humanevalplus` |

`humaneval` appears in both `cyber` and `coding` but is executed **once**.

`meta.json` records `preset` as the joined id (e.g. `cyber,coding`) and the merged dataset lists.

---

## Custom `--datasets`

One flag mixes **refusal and capability** ids (overrides a **single** preset’s lists).  
Cannot be combined with multiple comma-separated `--preset` values.

```bash
refusal-capability-bench --model m --datasets xstest,advbench,gsm8k,mmlu
refusal-capability-bench --model m --preset default --datasets cyber-overrefusal,gsm8k
```

---

## Intent legend (refusal)

| `expected` | How to read `refusal_rate` |
|------------|----------------------------|
| `over_refuse` | **Lower is better** (benign / educational) |
| `should_refuse` | **Higher is better** (clearly disallowed) |
| `mixed` | Use `by_category` / labels in `summary.json` |

---

## Builtins (no download)

| ID | Expected | Notes |
|----|----------|-------|
| `cyber-overrefusal` | over_refuse | Folder alias: `refusal/base_cyber/` |
| `generic-compliance` | over_refuse | Should almost never refuse |

---

## Over-refusal

| ID | HF | Expected | Notes |
|----|-----|----------|-------|
| `xstest` | `walledai/XSTest` | mixed | **Gated** |
| `xstest-paul` | `Paul/XSTest` | mixed | Ungated mirror |
| `orbench` | `bench-llm/or-bench@or-bench-hard-1k` | over_refuse | |
| `orbench-toxic` | `bench-llm/or-bench@or-bench-toxic` | should_refuse | |
| `wildchat-over-refusal` | `justinphan3110/wildchat_over_refusal` | over_refuse | |
| `coconot` | `allenai/coconot@original` | mixed | |
| `coconot-contrast` | `allenai/coconot@contrast` | mixed | |

---

## Should-refuse / safety

| ID | HF | Expected | Notes |
|----|-----|----------|-------|
| `sorrybench` | `sorry-bench/sorry-bench-202406` | should_refuse | **Gated** |
| `advbench` | `walledai/AdvBench` | should_refuse | **Gated** |
| `harmbench` | `walledai/HarmBench@standard` | should_refuse | **Gated** |
| `harmbench-contextual` | `walledai/HarmBench@contextual` | should_refuse | **Gated** |
| `strongreject` | `walledai/StrongREJECT` | should_refuse | **Gated** |
| `toxicchat` | `lmsys/toxic-chat@toxicchat0124` | mixed | |
| `beavertails` | `PKU-Alignment/BeaverTails` | mixed | |
| `donotanswer` | `LibrAI/do-not-answer` | should_refuse | |
| `safetybench` | `thu-coai/SafetyBench@test/en` | mixed | EN MC |
| `harmfulqa` | `declare-lab/HarmfulQA` | should_refuse | |
| `jbb` | `walledai/JailbreakBench` | mixed | |
| `jbb-behaviors` | `JailbreakBench/JBB-Behaviors@behaviors` | mixed | |
| `wildjailbreak` | `allenai/wildjailbreak@eval` | mixed | **Gated** |
| `in-the-wild-jailbreak` | `TrustAIRLab/in-the-wild-jailbreak-prompts` | should_refuse | |
| `aegis` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | mixed | |
| `llm-lat-harmful` | `LLM-LAT/harmful-dataset` | should_refuse | |

---

## Cyber-adjacent

| ID | HF | Expected |
|----|-----|----------|
| `cyber-code-vuln` | `CyberNative/Code_Vulnerability_Security_DPO` | over_refuse |
| `abliterate-refusal` | `byroneverson/abliterate-refusal` | should_refuse |
| `mrfakename-refusal` | `mrfakename/refusal` | mixed |
| `mrfakename-refusal-xl` | `mrfakename/refusal-xl` | mixed |

---

## Gated Hugging Face access

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

Preflight runs automatically before gated loads. On failure the run aborts unless `--continue-on-error` or `--skip-hf-check`.

| Dataset | URL |
|---------|-----|
| XSTest | https://huggingface.co/datasets/walledai/XSTest |
| AdvBench | https://huggingface.co/datasets/walledai/AdvBench |
| HarmBench | https://huggingface.co/datasets/walledai/HarmBench |
| StrongREJECT | https://huggingface.co/datasets/walledai/StrongREJECT |
| Sorry-Bench | https://huggingface.co/datasets/sorry-bench/sorry-bench-202406 |
| WildJailbreak | https://huggingface.co/datasets/allenai/wildjailbreak |

---

## Capability benches

Catalog keys: `capability_core` (`gsm8k`, `mmlu`, `humaneval`),  
`coding_datasets` (`humaneval`, `mbpp`, `humanevalplus`),  
`capability_datasets` (union of both).

| Bench | Dataset | Kind | Metric |
|-------|---------|------|--------|
| `gsm8k` | `openai/gsm8k` | math | Exact number (`#### N`) |
| `mmlu` | `cais/mmlu` | knowledge | Letter A–D |
| `humaneval` | `openai/openai_humaneval` | coding | Unit tests in a subprocess |
| `mbpp` | `google-research-datasets/mbpp@sanitized` | coding | Assert-list tests in a subprocess |
| `humanevalplus` | `evalplus/humanevalplus` | coding | Stronger HumanEval+ unit tests |

### Coding benches (no tool calling)

Coding evals are **plain chat completions** only (`messages` user/system — no `tools` /
function-calling). The model returns source code in the reply; the harness extracts it
and runs local unit tests in a subprocess.

| Preset | Benches |
|--------|---------|
| `coding` / `coding-smoke` | `humaneval`, `mbpp`, `humanevalplus` |
| `--datasets humaneval,mbpp` | any mix of capability ids |

```bash
refusal-capability-bench --model m --preset coding --limit 25 --report
refusal-capability-bench --model m --datasets mbpp,humanevalplus --limit 20
```

Same `--seed` + `--limit` ⇒ stable A/B compares.  
**All coding benches execute model code locally** — sandbox untrusted models.

For uncensor checks: expect **↓ refusal** on `over_refuse`, **↑ or stable** on `should_refuse`, **≈ flat** capability.
