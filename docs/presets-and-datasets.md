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

An **empty list on one side is intentional** (that suite is skipped).

| Preset | Rec. | Refusal | Capability |
|--------|:----:|---------|------------|
| `default` | yes | `cyber-overrefusal`, `generic-compliance` | `gsm8k`, `mmlu`, `humaneval` |
| `quick` | yes | default + `xstest`, `donotanswer`, `advbench` | core 3 |
| `compare` | yes | same as `quick` | core 3 |
| `overrefusal` | yes | builtins + XSTest / OR-Bench / WildChat-OR / CoCoNot | core 3 |
| `should-refuse` | yes | Do-Not-Answer, AdvBench, HarmBench, StrongREJECT, … | core 3 |
| `cyber` | yes | `cyber-overrefusal`, `cyber-code-vuln`, `generic-compliance` | core 3 |
| `refusal-only` | yes | `cyber-overrefusal`, `generic-compliance` | **`[]` skip capability** |
| `capability-only` | yes | **`[]` skip refusal** | `gsm8k`, `mmlu`, `humaneval` |
| `capability-smoke` | | **`[]`** | same as capability-only (use small `--limit`) |
| `mentioned` | | broad recommendation list | core 3 |
| `full` / `all` | | entire refusal registry | core 3 |

```bash
refusal-capability-bench --model m --preset overrefusal
refusal-capability-bench --model m --preset should-refuse
refusal-capability-bench --model m --preset cyber
refusal-capability-bench --model m --preset refusal-only
refusal-capability-bench --model m --preset capability-only --limit 20
```

---

## Custom `--datasets`

One flag mixes **refusal and capability** ids (overrides preset lists):

```bash
refusal-capability-bench --model m --datasets xstest,advbench,gsm8k,mmlu
refusal-capability-bench --model m --preset default --datasets cyber-overrefusal,gsm8k
```

---

## Intent legend (refusal)

| `expected_behavior` | How to read `refusal_rate` |
|---------------------|----------------------------|
| `over_refuse` | **Lower is better** (benign / educational) |
| `should_refuse` | **Higher is better** (clearly disallowed) |
| `mixed` | Use `by_category` / labels in `summary.json` |

---

## Builtins (no download)

| ID | Intent | Notes |
|----|--------|-------|
| `cyber-overrefusal` | over_refuse | Folder alias: `refusal/base_cyber/` |
| `generic-compliance` | over_refuse | Should almost never refuse |

---

## Over-refusal

| ID | HF | Intent | Notes |
|----|-----|--------|-------|
| `xstest` | `walledai/XSTest` | mixed | **Gated**, `revision=main` |
| `xstest-paul` | `Paul/XSTest` | mixed | Ungated mirror |
| `orbench` | `bench-llm/or-bench` @ hard-1k | over_refuse | |
| `orbench-toxic` | `bench-llm/or-bench` @ toxic | should_refuse | |
| `wildchat-over-refusal` | `justinphan3110/wildchat_over_refusal` | over_refuse | |
| `coconot` / `coconot-contrast` | `allenai/coconot` | mixed | |

---

## Should-refuse / safety

| ID | HF | Notes |
|----|-----|-------|
| `sorrybench` | `sorry-bench/sorry-bench-202406` | **Gated** |
| `advbench` | `walledai/AdvBench` | **Gated** |
| `harmbench` / `harmbench-contextual` | `walledai/HarmBench` | **Gated** |
| `strongreject` | `walledai/StrongREJECT` | **Gated** |
| `donotanswer` | `LibrAI/do-not-answer` | |
| `toxicchat` | `lmsys/toxic-chat` | |
| `beavertails` | `PKU-Alignment/BeaverTails` | |
| `safetybench` | `thu-coai/SafetyBench` | EN MC |
| `harmfulqa` | `declare-lab/HarmfulQA` | |
| `jbb` / `jbb-behaviors` | JailbreakBench packs | |
| `wildjailbreak` | `allenai/wildjailbreak` | **Gated** |
| `in-the-wild-jailbreak` | TrustAIRLab | |
| `aegis` | NVIDIA Aegis 2.0 | |
| `llm-lat-harmful` | `LLM-LAT/harmful-dataset` | |

---

## Cyber-adjacent

| ID | HF | Intent |
|----|-----|--------|
| `cyber-code-vuln` | `CyberNative/Code_Vulnerability_Security_DPO` | over_refuse |
| `abliterate-refusal` | `byroneverson/abliterate-refusal` | should_refuse |
| `mrfakename-refusal` / `-xl` | mrfakename pools | mixed |

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

| Bench | Dataset | Metric |
|-------|---------|--------|
| `gsm8k` | `openai/gsm8k` | Exact number (`#### N`) |
| `mmlu` | `cais/mmlu` | Letter A–D |
| `humaneval` | `openai/openai_humaneval` | Unit tests in a subprocess |

Same `--seed` + `--limit` ⇒ stable A/B compares.  
**HumanEval** runs model code locally — sandbox untrusted models.

For uncensor checks: expect **↓ refusal** on `over_refuse`, **↑ or stable** on `should_refuse`, **≈ flat** capability.
