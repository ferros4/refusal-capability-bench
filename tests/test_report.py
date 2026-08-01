from __future__ import annotations

import json
from pathlib import Path

from harness.report import write_report
from harness.safety import RESEARCH_USE_NOTICE


def test_write_report(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(
        json.dumps(
            {
                "models": ["m"],
                "preset": "default",
                "research_use_notice": RESEARCH_USE_NOTICE,
            }
        )
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": 1.2,
                "headlines": {
                    "refusal": {
                        "m": {
                            "cyber-overrefusal": {
                                "n": 2,
                                "refusal_rate": 0.0,
                                "compliance_rate": 1.0,
                                "total_time_s": 0.5,
                                "avg_tokens_per_sec": 10.0,
                            }
                        }
                    },
                    "capability": {
                        "m": {
                            "overall_accuracy": 1.0,
                            "benches": {
                                "gsm8k": {"accuracy": 1.0, "n": 1, "total_time_s": 0.2}
                            },
                        }
                    },
                },
            }
        )
    )
    out = write_report(run)
    text = out.read_text()
    assert "Eval report" in text
    assert "cyber-overrefusal" in text
    assert "gsm8k" in text
    assert "Research" in text or "research" in text.lower()
