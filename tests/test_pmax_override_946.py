"""SAFE_DIV_MAX_UM 9 → 8 on the best-base recipe (iter2 + swapfixfree + own head, V1284=1) — CPU build check.

next-levers-20260926: the knob is the existing build_r946 OVERRIDES path (no new code). Checked here:
  1. without the override the build is byte-identical to itself rebuilt (deterministic) and sets 9.0 in both the env line
     and the notebook's _EXPECTED_NUMERIC guard;
  2. with BIOHUB_SAFE_DIV_MAX_UM=8.0 the notebook differs by exactly those two lines, metadata only by id/title.
The own head's sha256 is a fixed, well-formed fake: the builder only checks its format (the kernel checks the real file
at runtime). Needs the public base notebook (skips without it).
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
HEAD_SHA256 = "0123456789abcdef" * 4   # 64 lowercase hex characters, not the sha of any real file
BASE_OVR = {
    "BIOHUB_DIV_DIP_MODE": "4", "BIOHUB_DIV_COMBO_MAX": "-2.6659", "BIOHUB_DIV_DIV_MISSING": "-9.0",
    "BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed", "BIOHUB_MOTION_RELINK_FLOW_ITER": "2",
    "BIOHUB_SWAPFIX": "1",
    "BIOHUB_SWAPFIX_PARAMS": json.dumps({"mode": "hung", "w_p": 16.0, "stay": 0.0, "free_targets": True, "free_min_p": 0.3}),
}


def _build(tmp: Path, name: str, ovr: dict) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if k not in ("FAST_IO", "V1284", "BIOHUB_CSV_FLOAT")}
    env.update(TERTIARY_SEED_DATASET="impala9397/ctg-seedC4-snap", NOTEBOOK_PATCH="kaggle/swapfix_946.py",
               OVERRIDES=json.dumps(ovr), KERNEL_SLUG=name, FAST_IO="1", V1284="1",
               V1284_HEAD_DATASET="impala9397/ctg-coordhead-own32-r4s0", V1284_HEAD_SHA256=HEAD_SHA256)
    dst = tmp / name
    subprocess.run([sys.executable, str(LANE / "build_r946.py"), str(dst)], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return dst


def _lines(d: Path) -> list[str]:
    nb = json.loads((d / "notebook.ipynb").read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code").splitlines()


def test_pmax8_differs_by_env_line_and_guard_only(tmp_path):
    a = _build(tmp_path, "base", BASE_OVR)
    a2 = _build(tmp_path, "base2", BASE_OVR)
    b = _build(tmp_path, "pmax8", dict(BASE_OVR, BIOHUB_SAFE_DIV_MAX_UM="8.0"))
    assert _lines(a) == _lines(a2)
    la, lb = _lines(a), _lines(b)
    assert "os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = '9.0'" in la
    diff = [l for l in difflib.unified_diff(la, lb, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert len(diff) == 4, diff
    assert "-os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = '9.0'" in diff
    assert "+os.environ['BIOHUB_SAFE_DIV_MAX_UM'] = '8.0'" in diff
    guard = [l for l in diff if "_EXPECTED_NUMERIC" in l]
    assert len(guard) == 2 and "'BIOHUB_SAFE_DIV_MAX_UM': 8.0" in guard[1] and "'BIOHUB_SAFE_DIV_MAX_UM': 9.0" in guard[0]
    ma = json.loads((a / "kernel-metadata.json").read_text())
    mb = json.loads((b / "kernel-metadata.json").read_text())
    assert {k for k in ma if ma[k] != mb[k]} <= {"id", "title"}
