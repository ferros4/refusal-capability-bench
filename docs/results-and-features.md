# Results, metrics, and features

---

## Output layout

```text
results/
  <model_slug>/                         # or <model>_vs_<compare>
    <YYYYMMDDTHHMMSSZ>/                 # new folder every run
      meta.json                         # config, safety notice, errors, user_stopped
      summary.json                      # combined headlines — start here
      eval.log                          # run log (DEBUG/INFO/WARNING/ERROR)
      report.html                       # if --report
      refusal/                          # suite from presets (refusal list)
        base_cyber/                     # cyber-overrefusal
          summary.json
          results.jsonl
          results.csv
        generic_compliance/
        xstest/
        …
      capability/                       # suite from presets (capability list)
        summary.json
        results.jsonl
        results.csv

cache/prompts/
  <dataset>_<hash>.jsonl                # frozen prompt lists for stable compares
  <dataset>_<hash>.meta.json
```

### Multiple presets

`--preset cyber,coding` (comma-separated) **merges** the presets into one suite:

- Refusal and capability ids are concatenated in flag order, then **deduplicated** (order preserved).
- Each dataset/bench id is executed **at most once** (e.g. `humaneval` shared by `cyber` and `coding` runs once).
- Output layout is unchanged: flat `refusal/` and `capability/` under the timestamp.

```text
results/<model_slug>/<timestamp>/
  meta.json                    # preset: "cyber,coding"; merged datasets lists
  summary.json
  refusal/…                    # union of refusal datasets
  capability/                  # union of capability benches
```

With `--compare`, the top folder is `<model>_vs_<compare>`, and refusal folders nest per model:

```text
results/<model>_vs_<compare>/<timestamp>/
  refusal/
    base_cyber/<model_slug>/
    base_cyber/<compare_slug>/
  capability/                  # both models + delta_compare_minus_base
```

Refusal dataset → folder aliases (under `refusal/`):

| Dataset id | Folder |
|------------|--------|
| `cyber-overrefusal` | `base_cyber` |
| `generic-compliance` | `generic_compliance` |
| other registry ids | slugified id (e.g. `xstest`, `mrfakename_refusal`) |

Standalone `python -m harness.refusal_eval` / `capability_eval` write to whatever path you pass as `--out` (no automatic nesting).

---

## Reading results

**Start with** `summary.json` → `headlines`:

- `refusal.<model>.<dataset>.refusal_rate` / `compliance_rate`
- `refusal.<model>.<dataset>.total_time_s` / `avg_tokens_per_sec` / `overall_tokens_per_sec`
- `capability.<model>.benches.<bench>.accuracy` (+ timing fields)
- `capability.delta_compare_minus_base` when comparing

**`report.html`** (from `--report`): browser tables for refusal, capability, and deltas.

**`meta.json`**: run config, `research_use_notice`, errors, `user_stopped`.

---

## Timing and tokens/sec

| Field | Meaning |
|-------|---------|
| `latency_s` | Wall time for that generation |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | From API `usage` when present |
| `tokens_per_sec` | `completion_tokens / latency_s` |
| `tokens_estimated` | `true` if usage missing (≈4 chars/token fallback) |
| `total_time_s` | Sum of sample latencies (dataset/bench) |
| `avg_tokens_per_sec` | Mean of per-sample tok/s |
| `overall_tokens_per_sec` | `total_completion_tokens / total_time_s` |

Written into `summary.json`, `results.jsonl`, and `results.csv`.

---

## Logging

Stdlib logging goes to **stderr** and a file:

| Path | When |
|------|------|
| `results/<model>/<timestamp>/eval.log` | Default for unified `run_eval` |
| `<out>/eval.log` | Standalone `refusal_eval` / `capability_eval` |
| `--log-file PATH` | Override |

```bash
refusal-capability-bench --model m --preset default --log-level DEBUG
# or: --log-file logs/my-run.log --log-level INFO
```

Levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.  
`DEBUG` includes per-request stream stats; `WARNING` covers empty content, 502/503 retries, and reasoning fallbacks.

---

## API client (streaming)

Chat calls use **SSE streaming** (`stream: true`) against `/v1/chat/completions`.  
The client accumulates `delta.content` (and reasoning fields when present), then returns one `ChatResult`.

| Setting | Default | Notes |
|---------|---------|--------|
| `--timeout` | `300` | Full stream lifetime (connect + all chunks) |
| `--max-tokens` | `1024` | Passed through to the API |
| 502 / 503 | retry | Sleep 5s, up to 10 retries |

### Thinking / reasoning models

Some models (e.g. Qwen “thinking”) stream tokens into `reasoning_content` / similar fields and may leave `content` empty if they hit `--max-tokens` during reasoning. That produced blank `response` columns in `results.csv` for packs like `base_cyber`.

Behavior now:

1. Capture both `content` and reasoning deltas from the stream.
2. If `content` is empty but reasoning is not → use reasoning as the response and log a **WARNING** (suggest raising `--max-tokens`).
3. Prefer a higher cap for those models, e.g. `--max-tokens 4096` or `8192`.

---

## Feature details

### HF gated preflight

Before loading gated refusal datasets, the harness checks for an HF token and dataset access.

- Fail → abort (unless `--continue-on-error` or `--skip-hf-check`)
- Login: `huggingface-cli login` or `export HF_TOKEN=…`

### Prompt snapshot cache

Resolved prompt lists are stored under `cache/prompts/` (keyed by dataset id, seed, limit, revision, config).

- Stable A/B compares across runs
- `--no-cache` — disable
- `--refresh-cache` — rebuild entries
- `--cache-dir PATH` — custom location

### Dataset revision pins

Gated datasets in the registry pin `revision=main` so loads are explicit. Override via loader/config when you need a commit hash.

### Parallel workers

`--workers N` runs sample requests concurrently (thread pool). Use with care on a single GPU host (queueing still happens server-side).

### Judges

| Mode | Flag | Notes |
|------|------|-------|
| Heuristic | `--judge heuristic` (default) | Keyword/pattern refusal detector |
| LLM | `--judge llm` | Same or separate model labels `REFUSE` / `COMPLY` |
| External judge | `--judge-base-url` + `--judge-model` | Optional separate endpoint |

Always spot-check `results.csv` — soft refusals are easy to mis-label.

### HTML report

```bash
refusal-capability-bench --model m --preset default --report
# → results/<model>/<timestamp>/report.html
```

Or generate later:

```python
from harness.report import write_report
write_report("results/my-model/20260101T000000Z")
```

### Safety stamp

Every `meta.json` includes:

- `research_use_only: true`
- `research_use_notice` — sensitive prompts may be present; research/eval use only

### Ctrl+C

- Completed samples are saved
- Remaining rows: `error = "User stopped test on the command line"`
- `user_stopped: true` in meta/summaries
- Exit code **130**

---

## API / TLS

| Issue | Fix |
|-------|-----|
| `SSL: WRONG_VERSION_NUMBER` | Use `http://HOST:PORT/v1`, not `https://` |
| Self-signed HTTPS | Default `verify=False`; `--secure` only with a real cert |
| Override full URL | `--base-url http://HOST:PORT/v1` |
| Host / port | `--host 127.0.0.1 --port 11434` (defaults when `--base-url` omitted) |

Default when host/port omitted: `http://127.0.0.1:11434/v1`

---

## Coding benches

Preset `coding` (or `--datasets humaneval,mbpp,humanevalplus`) runs chat-only code
completion. The API request is a normal `/chat/completions` with user/system messages
only — **no tools / function calling**. Replies are scored by extracting Python and
running unit tests in a local subprocess (sandbox untrusted models).

| Bench | Role |
|-------|------|
| `humaneval` | Classic function completion |
| `mbpp` | NL problem → Python + assert list |
| `humanevalplus` | Same prompts as HumanEval, stronger tests |

---

## Interpreting uncensor compares

| Signal | Healthy uncensor |
|--------|------------------|
| `over_refuse` / cyber education | Refusal **down** |
| `should_refuse` / AdvBench-style | Refusal **up or stable** |
| Capability deltas | **≈ 0** (`delta_compare_minus_base`) |
