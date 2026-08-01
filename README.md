# Model eval harness (`refusal-capability-bench`)

Evaluate local / LAN LLMs over an **OpenAI-compatible API** (Ollama, llama.cpp, vLLM, …):

| Suite | Measures |
|-------|----------|
| **Refusal** | How often the model declines prompts |
| **Capability** | Whether core skills hold (GSM8K, MMLU, HumanEval) |

Default API (when `--base-url` is omitted): `http://127.0.0.1:11434/v1`  
(`--host` / `--port`, port default **11434**)

---

## Install and run (recommended path)

Uses **[uv](https://docs.astral.sh/uv/)** for the environment and lockfile (`uv.lock`).

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
cd /path/to/refusal-capability-bench
python3 -m venv .venv/
source .venv/bin/activate
uv sync

# Optional: gated Hugging Face datasets
uv run hf auth login

# Confirm the model server (use http:// unless TLS is really enabled)
curl http://127.0.0.1:11434/v1/models

# List presets, then run (uv run uses the project env)
uv run refusal-capability-bench --list-presets
uv run refusal-capability-bench --model YOUR_MODEL --preset default --report
```

Activate the venv instead of prefixing `uv run` if you prefer:

```bash
source .venv/bin/activate
refusal-capability-bench --model YOUR_MODEL --preset default --report
```

**pip-only** (no lock guarantees): `pip install -e ".[dev]"` or `pip install -r requirements.txt`.

### Optional config file

```bash
cp eval.yaml.example eval.yaml
# set model, preset, workers, …
refusal-capability-bench --preset default --report
```

CLI flags override `eval.yaml`. See [docs/cli-reference.md](docs/cli-reference.md).

---

## Common commands

```bash
# Base vs uncensored compare
refusal-capability-bench \
  --model qwen3.6:35b-a3b-q8_0 \
  --compare 'hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M' \
  --preset compare --limit 50 --workers 4 --report

# Faster smoke
refusal-capability-bench --model YOUR_MODEL --preset quick --report

# Refusal only / capability only
refusal-capability-bench --model YOUR_MODEL --preset refusal-only
refusal-capability-bench --model YOUR_MODEL --preset capability-only --limit 20
```

**Prefer `--preset`.** Start with `default` or `compare`.  
More presets and every dataset id: [docs/presets-and-datasets.md](docs/presets-and-datasets.md).

---

## Where results go

```text
results/<model>/<timestamp>/
  meta.json          # config + safety notice
  summary.json       # headlines (start here)
  report.html        # if you passed --report
  base_cyber/        # refusal outputs (per dataset)
  capability/        # GSM8K / MMLU / HumanEval
```

Open `summary.json` → `headlines`, or `report.html` in a browser.

Details (layout, metrics, Ctrl+C, cache, judges): [docs/results-and-features.md](docs/results-and-features.md).

---

## More documentation

| Doc | Contents |
|-----|----------|
| [docs/presets-and-datasets.md](docs/presets-and-datasets.md) | All presets, refusal/capability datasets, gated HF URLs |
| [docs/cli-reference.md](docs/cli-reference.md) | Full flag list, config keys, standalone runners |
| [docs/results-and-features.md](docs/results-and-features.md) | Output layout, tok/s, cache, report, safety, TLS |
| [docs/development.md](docs/development.md) | Layout, tests, packaging |
| [eval.yaml.example](eval.yaml.example) | Sample config file |
| [datasets_catalog.yaml](datasets_catalog.yaml) | Machine-oriented catalog mirror |

---

## Notes

- Chat-only (no tool calling).
- Some datasets contain sensitive prompts **by design** — research/eval use only.
- `SSL: WRONG_VERSION_NUMBER` → use `http://`, not `https://`, unless the server terminates TLS.
- Tests: `pytest -q` (see [docs/development.md](docs/development.md)).
