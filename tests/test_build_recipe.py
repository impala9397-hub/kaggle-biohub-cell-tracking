"""kaggle/build_recipe.py + recipes.json + the base-notebook guard of build_r946.py. CPU only.

The recipe tests need nothing; the final build test needs the public base notebook (skips without it).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, skip_without_base

sys.path.insert(0, str(ROOT / "kaggle"))
import build_recipe  # noqa: E402

DATA = build_recipe.load()
FINAL = "final-headavg-pmax8"


def _clean_env(**extra) -> dict:
    keep = {k: v for k, v in os.environ.items()
            if not (k.startswith(("V1284", "TERTIARY_", "NOTEBOOK_PATCH", "SCRIPT_PATCH", "FAST_IO", "BIOHUB_"))
                    or k in ("OVERRIDES", "KERNEL_SLUG", "NO_ROBUSTNESS", "KAGGLE_OWNER", "ALLOW_BASE_MISMATCH"))}
    keep.update(extra)
    return keep


def test_every_recipe_resolves_and_its_patch_exists():
    slugs = set()
    for name in DATA["recipes"]:
        r = build_recipe.resolve(name)
        assert r["chain"][-1] == name
        patch = r["env"].get("NOTEBOOK_PATCH")
        if patch:
            assert (ROOT / patch).is_file(), patch
        assert r["slug"] not in slugs, f"duplicate slug {r['slug']}"
        slugs.add(r["slug"])


def test_recipes_point_at_recorded_submissions():
    rows = {r["ref"]: r for r in csv.DictReader((ROOT / "docs/data/submissions.csv").open())}
    for name, r in DATA["recipes"].items():
        sub = r["submission"]
        row = rows[str(sub["ref"])]
        assert row["name"] == sub["name"], (name, row["name"], sub["name"])
        assert row["public"] and row["private"], name


def test_final_recipe_environment():
    env = build_recipe.build_env(FINAL)
    ovr = json.loads(env["OVERRIDES"])
    assert env["KERNEL_SLUG"] == "ctg-lane-r-headavg-pmax8"
    assert env["TERTIARY_SEED_DATASET"] == "impala9397/ctg-seedC4-snap"
    assert env["NOTEBOOK_PATCH"] == "kaggle/x138_port_946.py"
    assert env["FAST_IO"] == "1" and env["V1284"] == "1"
    assert env["V1284_HEAD2_DATASET"] == "impala9397/ctg-coordhead-own32-r4s0"
    assert len(env["V1284_HEAD2_SHA256"]) == 64 and "V1284_HEAD_DATASET" not in env
    assert ovr["BIOHUB_SAFE_DIV_MAX_UM"] == "8.0"
    assert ovr["BIOHUB_MOTION_RELINK_FLOW_ITER"] == "1"      # the final selection runs one flow round, not iter2
    assert ovr["BIOHUB_RELINK_PROB_W"] == "32"
    assert ovr["BIOHUB_DIV_COMBO_MAX"] == "-2.6659"          # S86 threshold, not the S83 one
    assert "NO_ROBUSTNESS" not in env


def test_override_order_matches_the_submitted_kernels():
    """New env lines are inserted in OVERRIDES order, so the order is part of the build output."""
    keys = list(json.loads(build_recipe.build_env(FINAL)["OVERRIDES"]))
    gate = list(DATA["knob_sets"]["division_gate_s83"])
    flow = list(DATA["knob_sets"]["flow_relink_x138"])
    assert keys == gate + ["BIOHUB_RELINK_PROB_W"] + flow + ["BIOHUB_SAFE_DIV_MAX_UM"]


def test_headens5_lists_five_heads_and_keeps_the_public_head_first():
    env = build_recipe.build_env("final-headens5-pmax8")
    shas = env["V1284_HEAD2_SHA256S"].split(",")
    assert len(shas) == 5 and len(set(shas)) == 5 and all(len(s) == 64 for s in shas)
    assert "V1284_HEAD_DATASET" not in env  # head 1 stays the public V1284 head
    assert shas[0] == build_recipe.build_env("flow-v1284-headavg")["V1284_HEAD2_SHA256"]  # seed 0 = own32-r4s0


def _fake_base(tmp: Path, code: str) -> Path:
    d = tmp / "base"
    d.mkdir()
    nb = {"cells": [{"cell_type": "code", "source": [code], "metadata": {}, "outputs": [], "execution_count": None}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
    (d / "notebook.ipynb").write_text(json.dumps(nb))
    return d


def test_builder_refuses_a_different_base(tmp_path):
    base = _fake_base(tmp_path, "print('not the 0.946 notebook')\n")
    r = subprocess.run([sys.executable, str(ROOT / "kaggle/build_r946.py"), str(tmp_path / "out")], cwd=ROOT,
                       env=_clean_env(KERNEL_SLUG="x", BASE946_DIR=str(base)), capture_output=True, text=True)
    assert r.returncode != 0 and "sha256" in r.stderr
    assert not (tmp_path / "out").exists()


def test_builder_reports_a_missing_base(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "kaggle/build_r946.py"), str(tmp_path / "out")], cwd=ROOT,
                       env=_clean_env(KERNEL_SLUG="x", BASE946_DIR=str(tmp_path / "nowhere")), capture_output=True, text=True)
    assert r.returncode != 0 and "fetch_base946.sh" in r.stderr


def _build(tmp: Path, name: str, recipe: str) -> Path:
    skip_without_base()
    dst = tmp / name
    subprocess.run([sys.executable, str(ROOT / "kaggle/build_recipe.py"), recipe, str(dst)], cwd=ROOT,
                   env=_clean_env(**({"BASE946_DIR": os.environ["BASE946_DIR"]} if os.environ.get("BASE946_DIR") else {})),
                   check=True, capture_output=True, text=True)
    return dst


def test_final_build_is_deterministic_and_carries_the_recipe(tmp_path):
    a = _build(tmp_path, "a", FINAL)
    b = _build(tmp_path, "b", FINAL)
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    nb = json.loads((a / "notebook.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for line in ("os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = '8.0'", "os.environ['BIOHUB_RELINK_PROB_W'] = '32'",
                 "os.environ['BIOHUB_MOTION_RELINK_FLOW_ITER'] = '1'", "os.environ['BIOHUB_ALLOW_ARTIFACT_FALLBACK'] = '0'"):
        assert line in code, line
    assert "V1284 HEAD2:" in code and "FASTIO" in code and "TERTIARY PATCH APPLIED" in code
    meta = json.loads((a / "kernel-metadata.json").read_text())
    assert meta["id"] == "impala9397/ctg-lane-r-headavg-pmax8" and meta["is_private"] is True
    assert meta["dataset_sources"] == [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1", "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1", "impala9397/ctg-seedC4-snap",
        "anvithpothula/biohub-v1284-head-s075", "impala9397/ctg-coordhead-own32-r4s0"]
    compile(code, "notebook_cell", "exec")
