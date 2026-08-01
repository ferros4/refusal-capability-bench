# Development

## Project layout

```text
run_eval.py                 # CLI entry (also: refusal-capability-bench)
eval.yaml.example           # sample config → copy to eval.yaml
pyproject.toml              # package + console script + uv config
uv.lock                     # locked dependencies (uv)
requirements.txt            # optional pip export from uv
harness/
  api_client.py             # OpenAI-compatible HTTP client
  presets.py                # suite presets (source of truth)
  refusal_datasets.py       # refusal dataset registry + bundles
  refusal_eval.py           # refusal runner (standalone OK)
  capability_eval.py        # GSM8K / MMLU / coding (HumanEval, MBPP, HumanEval+)
  metrics.py                # timing / tokens-per-sec
  prompt_cache.py           # seeded prompt snapshots
  hf_auth.py                # gated HF preflight
  report.py                 # HTML report generator
  config.py                 # eval.yaml loader
  safety.py                 # research-use notice
tests/                      # unit + integration
docs/                       # extended documentation
datasets_catalog.yaml       # catalog mirror (docs only; not loaded at runtime)
cache/prompts/              # runtime prompt cache
results/                    # runtime outputs:
                            #   <model>/<timestamp>/refusal/<dataset>/
                            #   <model>/<timestamp>/capability/
```

## Setup (uv)

```bash
# https://docs.astral.sh/uv/getting-started/installation/
cd /path/to/refusal-capability-bench
uv sync --group dev
```

This creates `.venv`, installs the project editable, and syncs from `uv.lock`.

```bash
uv run refusal-capability-bench --help
# or
source .venv/bin/activate
refusal-capability-bench --help
```

Run commands from the **repo root** so `harness` imports resolve.

### Lockfile and dependencies

| File | Role |
|------|------|
| `pyproject.toml` | Project metadata, deps, `refusal-capability-bench` script |
| `uv.lock` | Locked transitive versions (commit this) |
| `requirements.txt` | Optional pip export (`uv export`); prefer `uv sync` |

```bash
# After changing dependencies in pyproject.toml:
uv lock
uv sync --group dev

# Refresh pip-oriented requirements.txt:
uv export --no-emit-project --extra dev --no-hashes -o requirements.txt
```

**pip-only:** `pip install -e ".[dev]"`.

## Tests

```bash
uv run pytest -q                              # unit + integration
uv run pytest -q tests --ignore=tests/integration
uv run pytest -q -m integration
```

| Path | Coverage |
|------|----------|
| `tests/test_*.py` | Unit (logic, cache, report, presets, auth, …) |
| `tests/integration/` | E2E vs in-process mock OpenAI server |
| `tests/integration/conftest.py` | Threaded mock `/v1/chat/completions` |

Integration covers: `ChatClient`, refusal builtins + JSONL, capability scoring, and `run_eval` layout (`refusal/` + `capability/` under the run dir).

## Package entry

```toml
# pyproject.toml
[project.scripts]
refusal-capability-bench = "run_eval:main_cli"

[dependency-groups]
dev = ["pytest>=8.0.0"]
```

```bash
uv sync --group dev
uv run refusal-capability-bench --help
```

## Notes for contributors

- Chat-only; no tool calling in the harness.
- Safety datasets include sensitive prompts by design.
- Prefer presets in docs and examples; keep advanced flags in [cli-reference.md](cli-reference.md).
- Source of truth for presets/datasets is Python code; YAML catalogs are documentation mirrors.
- Coding benches (`humaneval`, `mbpp`, `humanevalplus`) are chat-only — no tool/function calling; score via local unit tests.
