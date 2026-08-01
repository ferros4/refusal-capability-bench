# CLI reference

Entry points (after `uv sync --group dev`):

```bash
uv run refusal-capability-bench …
# or: source .venv/bin/activate && refusal-capability-bench …
python run_eval.py …    # from repo root with venv active
```

```bash
uv run refusal-capability-bench --help
uv run refusal-capability-bench --list-presets
uv run refusal-capability-bench --list-datasets
```

---

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | required* | Model id as the API lists it |
| `--preset` | `default` if omitted | **Recommended** suite selection |
| `--datasets` | none | Explicit refusal and/or capability ids (overrides preset lists) |
| `--compare` | none | Second model (refusal × both; capability head-to-head) |
| `--only` | `auto` | `auto` \| `all` \| `refusal` \| `capability` |
| `--out-root` | `results` | Parent directory for runs |
| `--run-id` | UTC timestamp | Folder under the model dir |
| `--dataset-limit` | registry default | Max samples **per refusal** dataset |
| `--limit` | `50` | Samples **per capability** bench |
| `--judge` | `heuristic` | `heuristic` \| `llm` |
| `--judge-base-url` | model API | Separate judge endpoint |
| `--judge-model` | `--model` | Judge model id |
| `--judge-api-key` | `--api-key` | Judge API key |
| `--workers` | `1` | Parallel sample requests |
| `--seed` | `42` | Shuffle seed (keep fixed for A/B) |
| `--cache-dir` | `cache/prompts` | Prompt snapshot cache |
| `--no-cache` | off | Disable cache reads/writes |
| `--refresh-cache` | off | Rebuild cache entries |
| `--report` | off | Write `report.html` |
| `--config` | `./eval.yaml` if present | YAML/JSON defaults |
| `--skip-hf-check` | off | Skip gated HF preflight |
| `--mmlu-subjects` | subsample all | Optional MMLU subject filter |
| `--base-url` | none | Full API root including `/v1` (overrides host/port) |
| `--host` | `127.0.0.1` | Used when `--base-url` omitted |
| `--port` | `11434` | Used when `--base-url` omitted (Ollama default) |
| `--api-key` | `ollama` | Bearer token |
| `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `1024` | Generation cap |
| `--sleep` | `0` | Delay between calls when `workers=1` |
| `--timeout` | `300` | HTTP timeout (seconds) |
| `--secure` | off | Verify TLS certificates |
| `--continue-on-error` | off | Continue after dataset/HF failures |
| `--list-presets` | — | Print suite presets |
| `--list-datasets` | — | Print refusal + capability ids |

\*Not required with `--list-presets` / `--list-datasets`.

---

## Config file (`eval.yaml`)

```bash
cp eval.yaml.example eval.yaml
# edit model, preset, workers, judge, …
refusal-capability-bench --preset default --report
```

If `./eval.yaml` exists (or `--config path`), values fill unset options. **CLI wins** when both are set.

Common keys: `host`, `port`, `base_url`, `model`, `compare`, `preset`, `datasets`, `limit`, `dataset_limit`, `seed`, `workers`, `judge`, `judge_model`, `judge_base_url`, `cache_dir`, `report`, `continue_on_error`.

See [`eval.yaml.example`](../eval.yaml.example).

---

## Examples

```bash
# Parallel + report
refusal-capability-bench --model m --preset compare --compare other --workers 4 --report

# External LLM judge
refusal-capability-bench --model m --preset quick --judge llm \
  --judge-model m --judge-base-url http://127.0.0.1:11434/v1

# Custom mix
refusal-capability-bench --model m --datasets xstest,advbench,gsm8k --limit 20 --dataset-limit 50

# Skip HF gated preflight (not recommended)
refusal-capability-bench --model m --preset quick --skip-hf-check
```

---

## Standalone suite runners

Still available for single-suite debugging:

```bash
python -m harness.refusal_eval \
  --model m --dataset xstest --limit 50 --workers 4 \
  --out results/manual/xstest

python -m harness.refusal_eval \
  --model m --dataset prompts_example.jsonl \
  --judge llm --judge-model m \
  --out results/manual/custom

python -m harness.capability_eval \
  --model base --compare uncensored \
  --benches gsm8k,mmlu,humaneval \
  --limit 50 --seed 42 --workers 4 \
  --out results/manual/cap_compare
```

See also [presets-and-datasets.md](presets-and-datasets.md) and [results-and-features.md](results-and-features.md).
