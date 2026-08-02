from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import run_eval


def test_slugify():
    assert run_eval.slugify("hf.co/Org/Model:Q4") == "org_model_q4"
    assert run_eval.slugify("  Foo Bar  ") == "foo_bar"
    assert run_eval.slugify("!!!") == "model"
    assert len(run_eval.slugify("a" * 200, max_len=20)) == 20


def test_model_dir_name():
    assert run_eval.model_dir_name("Qwen3.6:35b") == "qwen3.6_35b"
    assert "vs" in run_eval.model_dir_name("base", "other")


def test_refusal_folder_name():
    assert run_eval.refusal_folder_name("builtin:cyber-overrefusal") == "base_cyber"
    assert (
        run_eval.refusal_folder_name("builtin:generic-compliance")
        == "generic_compliance"
    )
    assert run_eval.refusal_folder_name("mrfakename/refusal") == "mrfakename_refusal"


def test_resolve_run_dir(tmp_path: Path):
    run_dir = run_eval.resolve_run_dir(tmp_path, "MyModel", None, "20260101T000000Z")
    assert run_dir == tmp_path / "mymodel" / "20260101T000000Z"
    d2 = run_eval.resolve_run_dir(tmp_path, "base", "uncen", "t1")
    assert d2 == tmp_path / "base_vs_uncen" / "t1"


def test_timestamp_utc_format():
    ts = run_eval.timestamp_utc()
    assert re.fullmatch(r"\d{8}T\d{6}Z", ts)


def test_build_refusal_argv_includes_secure_and_limit():
    argv = run_eval.build_refusal_argv(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        dataset="builtin:cyber-overrefusal",
        out=Path("/tmp/out"),
        limit=5,
        judge="heuristic",
        temperature=0.0,
        max_tokens=128,
        sleep=0.0,
        timeout=30.0,
        secure=True,
        workers=2,
        judge_model="judge-m",
    )
    assert "--secure" in argv
    assert argv[argv.index("--limit") + 1] == "5"
    assert argv[argv.index("--model") + 1] == "m"
    assert argv[argv.index("--workers") + 1] == "2"
    assert argv[argv.index("--judge-model") + 1] == "judge-m"


def test_build_capability_argv_compare_and_subjects():
    argv = run_eval.build_capability_argv(
        base_url="http://x/v1",
        api_key="k",
        model="base",
        compare="other",
        out=Path("/tmp/cap"),
        benches="gsm8k,mmlu",
        limit=10,
        seed=1,
        temperature=0.0,
        max_tokens=256,
        sleep=0.0,
        timeout=60.0,
        secure=False,
        mmlu_subjects="anatomy,astronomy",
    )
    assert argv[argv.index("--compare") + 1] == "other"
    assert argv[argv.index("--mmlu-subjects") + 1] == "anatomy,astronomy"
    assert "--secure" not in argv


def test_load_json_missing_and_present(tmp_path: Path):
    assert run_eval.load_json(tmp_path / "nope.json") == {}
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"x": 1}))
    assert run_eval.load_json(path) == {"x": 1}


def test_combine_summary_headlines(tmp_path: Path):
    run_dir = tmp_path / "mymodel" / "ts1"
    ref_dir = run_dir / "refusal" / "base_cyber"
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(
        json.dumps(
            {
                "refusal_rate": 0.1,
                "compliance_rate": 0.9,
                "n": 20,
                "total_time_s": 12.5,
                "avg_tokens_per_sec": 30.0,
            }
        )
    )
    cap_dir = run_dir / "capability"
    cap_dir.mkdir()
    (cap_dir / "summary.json").write_text(
        json.dumps(
            {
                "models": {
                    "base": {
                        "accuracy": 0.8,
                        "total_time_s": 9.0,
                        "avg_tokens_per_sec": 22.0,
                        "benches": {
                            "gsm8k": {
                                "accuracy": 0.75,
                                "n": 40,
                                "total_time_s": 4.0,
                                "avg_tokens_per_sec": 20.0,
                            }
                        },
                    }
                },
                "delta_compare_minus_base": {"gsm8k": {"delta": -0.05}},
            }
        )
    )

    combined = run_eval.combine_summary(
        run_dir=run_dir,
        models=["base"],
        refusal_paths={"base": {"builtin:cyber-overrefusal": ref_dir}},
        capability_dir=cap_dir,
        elapsed_s=12.34,
        config={"only": "all"},
    )
    assert combined["elapsed_s"] == 12.34
    headlines = combined["headlines"]
    assert (
        headlines["refusal"]["base"]["builtin:cyber-overrefusal"]["refusal_rate"] == 0.1
    )
    assert (
        headlines["refusal"]["base"]["builtin:cyber-overrefusal"]["total_time_s"]
        == 12.5
    )
    assert headlines["capability"]["base"]["benches"]["gsm8k"]["accuracy"] == 0.75


def test_main_preset_refusal_only(tmp_path: Path):
    seen_datasets: list[str] = []

    def fake_refusal_main(argv):
        ds = argv[argv.index("--dataset") + 1]
        seen_datasets.append(ds)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "refusal_rate": 0.0,
                    "compliance_rate": 1.0,
                    "n": 1,
                    "total_time_s": 0.1,
                }
            )
        )
        return 0

    with patch.object(run_eval.refusal_eval, "main", side_effect=fake_refusal_main):
        with patch.object(run_eval.capability_eval, "main") as cap_main:
            rc = run_eval.main(
                [
                    "--model",
                    "test-model",
                    "--preset",
                    "refusal-only",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "20260101T120000Z",
                ]
            )
    assert rc == 0
    cap_main.assert_not_called()
    run_dir = tmp_path / "test-model" / "20260101T120000Z"
    assert (run_dir / "refusal" / "base_cyber" / "summary.json").exists()
    assert (run_dir / "refusal" / "generic_compliance" / "summary.json").exists()
    assert not (run_dir / "capability").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["preset"] == "refusal-only"
    assert meta["datasets"]["refusal"] == ["cyber-overrefusal", "generic-compliance"]
    assert meta["datasets"]["capability"] == []
    assert set(seen_datasets) == {"cyber-overrefusal", "generic-compliance"}


def test_main_preset_capability_only(tmp_path: Path):
    def cap_main(argv):
        benches = argv[argv.index("--benches") + 1]
        assert benches == "gsm8k,mmlu,humaneval"
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "models": {
                        "cap-model": {
                            "accuracy": 1.0,
                            "benches": {"gsm8k": {"n": 1, "accuracy": 1.0}},
                        }
                    }
                }
            )
        )
        return 0

    with patch.object(run_eval.refusal_eval, "main") as ref_main:
        with patch.object(run_eval.capability_eval, "main", side_effect=cap_main):
            rc = run_eval.main(
                [
                    "--model",
                    "cap-model",
                    "--preset",
                    "capability-only",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "cap_only",
                    "--limit",
                    "1",
                ]
            )
    assert rc == 0
    ref_main.assert_not_called()
    run_dir = tmp_path / "cap-model" / "cap_only"
    assert (run_dir / "capability" / "summary.json").exists()
    assert not (run_dir / "refusal").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["preset"] == "capability-only"
    assert meta["datasets"]["refusal"] == []
    assert meta["datasets"]["capability"] == ["gsm8k", "mmlu", "humaneval"]
    assert meta.get("research_use_only") is True
    assert "research_use_notice" in meta


def test_main_multi_preset_merges_deduped_flat_layout(tmp_path: Path):
    """Comma-separated presets merge into one suite; overlapping benches run once."""
    refusal_datasets: list[str] = []
    cap_benches: list[str] = []

    def fake_refusal_main(argv):
        ds = argv[argv.index("--dataset") + 1]
        refusal_datasets.append(ds)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "refusal_rate": 0.0,
                    "compliance_rate": 1.0,
                    "n": 1,
                    "total_time_s": 0.1,
                }
            )
        )
        return 0

    def fake_cap_main(argv):
        benches = argv[argv.index("--benches") + 1]
        cap_benches.append(benches)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "models": {
                        "multi-model": {
                            "n": 1,
                            "correct": 1,
                            "accuracy": 1.0,
                            "benches": {
                                name: {"n": 1, "accuracy": 1.0}
                                for name in benches.split(",")
                            },
                        }
                    }
                }
            )
        )
        return 0

    with patch.object(run_eval.refusal_eval, "main", side_effect=fake_refusal_main):
        with patch.object(run_eval.capability_eval, "main", side_effect=fake_cap_main):
            rc = run_eval.main(
                [
                    "--model",
                    "multi-model",
                    "--preset",
                    "cyber,coding",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "multi_ts",
                    "--limit",
                    "1",
                ]
            )
    assert rc == 0
    run_dir = tmp_path / "multi-model" / "multi_ts"
    # Flat layout (not nested per preset)
    assert (run_dir / "refusal" / "base_cyber" / "summary.json").exists()
    assert (run_dir / "refusal" / "generic_compliance" / "summary.json").exists()
    assert (run_dir / "capability" / "summary.json").exists()
    assert not (run_dir / "cyber").exists()
    assert not (run_dir / "coding").exists()

    assert set(refusal_datasets) == {
        "cyber-overrefusal",
        "cyber-code-vuln",
        "generic-compliance",
    }
    assert len(cap_benches) == 1
    benches = cap_benches[0].split(",")
    assert benches == ["gsm8k", "mmlu", "humaneval", "mbpp", "humanevalplus"]
    assert benches.count("humaneval") == 1

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["preset"] == "cyber,coding"
    assert meta["datasets"]["capability"] == benches
    assert meta["datasets"]["refusal"] == [
        "cyber-overrefusal",
        "cyber-code-vuln",
        "generic-compliance",
    ]


def test_main_preset_coding(tmp_path: Path):
    seen_benches: list[str] = []

    def cap_main(argv):
        benches = argv[argv.index("--benches") + 1]
        seen_benches.append(benches)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "models": {
                        "code-model": {
                            "accuracy": 1.0,
                            "benches": {
                                "humaneval": {"n": 1, "accuracy": 1.0},
                                "mbpp": {"n": 1, "accuracy": 1.0},
                                "humanevalplus": {"n": 1, "accuracy": 1.0},
                            },
                        }
                    }
                }
            )
        )
        return 0

    with patch.object(run_eval.refusal_eval, "main") as ref_main:
        with patch.object(run_eval.capability_eval, "main", side_effect=cap_main):
            rc = run_eval.main(
                [
                    "--model",
                    "code-model",
                    "--preset",
                    "coding",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "coding_run",
                    "--limit",
                    "1",
                ]
            )
    assert rc == 0
    ref_main.assert_not_called()
    assert seen_benches == ["humaneval,mbpp,humanevalplus"]
    meta = json.loads(
        (tmp_path / "code-model" / "coding_run" / "meta.json").read_text()
    )
    assert meta["preset"] == "coding"
    assert meta["datasets"]["refusal"] == []
    assert meta["datasets"]["capability"] == [
        "humaneval",
        "mbpp",
        "humanevalplus",
    ]


def test_main_datasets_flag_mixed(tmp_path: Path):
    def fake_refusal_main(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps({"refusal_rate": 0.0, "n": 1, "compliance_rate": 1.0})
        )
        return 0

    def cap_main(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps({"models": {"m": {"accuracy": 1.0, "benches": {}}}})
        )
        return 0

    with patch.object(run_eval.refusal_eval, "main", side_effect=fake_refusal_main):
        with patch.object(run_eval.capability_eval, "main", side_effect=cap_main):
            rc = run_eval.main(
                [
                    "--model",
                    "m",
                    "--datasets",
                    "generic-compliance,gsm8k",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "mix1",
                    "--limit",
                    "1",
                ]
            )
    assert rc == 0
    meta = json.loads((tmp_path / "m" / "mix1" / "meta.json").read_text())
    assert meta["datasets"]["refusal"] == ["generic-compliance"]
    assert meta["datasets"]["capability"] == ["gsm8k"]


def test_list_presets_no_model():
    rc = run_eval.main(["--list-presets"])
    assert rc == 0


def test_main_user_stop_saves_partial(tmp_path: Path):
    def refusal_stop(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "refusal_rate": 0.0,
                    "compliance_rate": 1.0,
                    "n": 1,
                    "user_stopped": True,
                }
            )
        )
        (out / "results.csv").write_text(
            "sample_id,category,refused,judge,latency_s,error,prompt,response\n"
            f"1,t,False,heuristic,0.1,,p,r\n"
            f"2,t,True,heuristic,0.0,{run_eval.USER_STOPPED},p,\n"
        )
        return 130

    with patch.object(run_eval.refusal_eval, "main", side_effect=refusal_stop):
        rc = run_eval.main(
            [
                "--model",
                "stop-model",
                "--preset",
                "default",
                "--out-root",
                str(tmp_path),
                "--run-id",
                "stop_ts",
            ]
        )
    assert rc == 130
    run_dir = tmp_path / "stop-model" / "stop_ts"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["user_stopped"] is True
    assert (run_dir / "refusal" / "base_cyber" / "summary.json").exists()
    assert (run_dir / "capability" / "summary.json").exists()


def test_main_continue_on_error(tmp_path: Path):
    def cap_main(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"models": {}}))
        return 0

    with patch.object(run_eval.refusal_eval, "main", side_effect=RuntimeError("fail")):
        with patch.object(run_eval.capability_eval, "main", side_effect=cap_main):
            rc = run_eval.main(
                [
                    "--model",
                    "m",
                    "--preset",
                    "default",
                    "--out-root",
                    str(tmp_path),
                    "--run-id",
                    "err_run",
                    "--continue-on-error",
                    "--limit",
                    "1",
                ]
            )
    assert rc == 1
    meta = json.loads((tmp_path / "m" / "err_run" / "meta.json").read_text())
    assert meta["errors"]
