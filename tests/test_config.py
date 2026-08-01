from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from harness.config import load_config_file, merge_config_into_namespace


def test_load_yaml_config(tmp_path: Path):
    path = tmp_path / "eval.yaml"
    path.write_text("preset: quick\nworkers: 4\nmodel: mymodel\n")
    cfg = load_config_file(path)
    assert cfg["preset"] == "quick"
    assert cfg["workers"] == 4


def test_merge_fills_none_only():
    args = Namespace(preset=None, model="cli-model", workers=1)
    merge_config_into_namespace(
        args, {"preset": "default", "model": "cfg-model", "workers": 8}
    )
    assert args.preset == "default"
    assert args.model == "cli-model"  # already set
