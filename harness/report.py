"""Generate a simple HTML report from a run directory summary.json."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from harness.safety import RESEARCH_USE_NOTICE


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f"<th>{_esc (header )}</th>" for header in headers)
    body = []
    for row in rows:
        tds = "".join(f"<td>{_esc (cell )}</td>" for cell in row)
        body.append(f"<tr>{tds }</tr>")
    return (
        f"<table><thead><tr>{th }</tr></thead><tbody>{''.join (body )}</tbody></table>"
    )


def build_html(run_dir: Path, summary: dict[str, Any] | None = None) -> str:
    run_dir = Path(run_dir)
    if summary is None:
        summary = json.loads((run_dir / "summary.json").read_text())
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    headlines = summary.get("headlines") or {}
    refusal = headlines.get("refusal") or {}
    capability = headlines.get("capability") or {}

    ref_rows: list[list[Any]] = []
    for model, ds_map in refusal.items():
        if not isinstance(ds_map, dict):
            continue
        for ds, info in ds_map.items():
            if not isinstance(info, dict):
                continue
            ref_rows.append(
                [
                    model,
                    ds,
                    info.get("n"),
                    info.get("refusal_rate"),
                    info.get("compliance_rate"),
                    info.get("total_time_s"),
                    info.get("avg_tokens_per_sec"),
                    info.get("overall_tokens_per_sec"),
                ]
            )

    cap_rows: list[list[Any]] = []
    for model, info in capability.items():
        if model == "delta_compare_minus_base" or not isinstance(info, dict):
            continue
        benches = info.get("benches") or {}
        if not benches:
            cap_rows.append(
                [
                    model,
                    "(overall)",
                    info.get("overall_accuracy"),
                    info.get("total_time_s"),
                    info.get("avg_tokens_per_sec"),
                    info.get("overall_tokens_per_sec"),
                    "",
                ]
            )
        for bench_name, bi in benches.items():
            if not isinstance(bi, dict):
                continue
            cap_rows.append(
                [
                    model,
                    bench_name,
                    bi.get("accuracy"),
                    bi.get("n"),
                    bi.get("total_time_s"),
                    bi.get("avg_tokens_per_sec"),
                    bi.get("overall_tokens_per_sec"),
                ]
            )

    delta = capability.get("delta_compare_minus_base") or summary.get(
        "capability", {}
    ).get("delta_compare_minus_base")
    delta_html = ""
    if isinstance(delta, dict) and delta:
        drows = [
            [key, info.get("base"), info.get("compare"), info.get("delta")]
            for key, info in delta.items()
            if isinstance(info, dict)
        ]
        delta_html = "<h2>Capability delta (compare − base)</h2>" + _table(
            ["bench", "base", "compare", "delta"], drows
        )

    notice = meta.get("research_use_notice") or RESEARCH_USE_NOTICE
    models = meta.get("models") or summary.get("models") or []
    preset = meta.get("preset") or ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Eval report — {_esc (run_dir .name )}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #122; background: #fafbfc; }}
  h1,h2 {{ color: #0b3d5c; }}
  table {{ border-collapse: collapse; margin: 1rem 0 2rem; width: 100%; max-width: 1100px; background: #fff; }}
  th, td {{ border: 1px solid #c5d0da; padding: 0.45rem 0.6rem; text-align: left; font-size: 0.92rem; }}
  th {{ background: #e8eef3; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
  .notice {{ background: #fff3cd; border: 1px solid #e0c36a; padding: 0.75rem 1rem; max-width: 900px; }}
  .meta {{ color: #445; font-size: 0.9rem; }}
  code {{ background: #eef2f5; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Eval report</h1>
<p class="meta">Run dir: <code>{_esc (run_dir )}</code><br/>
Models: {_esc (models )} · Preset: {_esc (preset )} · Elapsed s: {_esc (summary .get ("elapsed_s"))}</p>
<div class="notice"><strong>Notice:</strong> {_esc (notice )}</div>
<h2>Refusal</h2>
{_table (["model","dataset","n","refusal_rate","compliance_rate","total_time_s","avg_tok/s","overall_tok/s"],ref_rows )if ref_rows else "<p>No refusal headlines.</p>"}
<h2>Capability</h2>
{_table (["model","bench","accuracy/n","n_or_time","total_time_s","avg_tok/s","overall_tok/s"],cap_rows )if cap_rows else "<p>No capability headlines.</p>"}
{delta_html }
<p class="meta">Generated from <code>summary.json</code> / <code>meta.json</code>.</p>
</body>
</html>
"""


def write_report(run_dir: Path, out_name: str = "report.html") -> Path:
    run_dir = Path(run_dir)
    html_doc = build_html(run_dir)
    out = run_dir / out_name
    out.write_text(html_doc, encoding="utf-8")
    return out
