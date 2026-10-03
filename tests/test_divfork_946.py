"""kaggle/divfork_946.py — metric-aware division-fork rules (default off). CPU only.

Checked here:
  1. build: the flow-v1284 recipe with NOTEBOOK_PATCH=divfork_946.py differs from the same recipe with x138_port_946.py
     only by added lines (the helper block, the one call line, and the BIOHUB_DIVFORK env line when the knob is set);
     nothing is removed; metadata differs only in id/title. With the knob unset there is no env line.
  2. off: with BIOHUB_DIVFORK unset the helper returns its input list object unchanged (so the kernel output is identical).
  3. prune (geometric): a fork whose safe-division child sits within 7 um of another frame-t node (after flow back-projection)
     loses exactly that safe-division edge; a clean division is kept; no node is touched; every edge stays t -> t+1.
  4. chroma: a child brighter than 1.1 x the parent (integrated, background-subtracted) is cut; half-intensity daughters kept.
Test 1 needs the public base notebook (it skips without it); 2-4 are synthetic.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
FLOW_V1284_OVR = {
    "BIOHUB_DIV_DIP_MODE": "4", "BIOHUB_DIV_COMBO_MAX": "-2.6659", "BIOHUB_DIV_DIV_MISSING": "-9.0",
    "BIOHUB_DIV_Z_DIP_MU": "0.6187965957674368", "BIOHUB_DIV_Z_DIP_SD": "0.24116301719876107",
    "BIOHUB_DIV_Z_SYM_MU": "1.1339288035330837", "BIOHUB_DIV_Z_SYM_SD": "0.5528905816566464",
    "BIOHUB_DIV_Z_DIV_MU": "1.239899766683567", "BIOHUB_DIV_Z_DIV_SD": "2.1790329837769318",
    "BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed", "BIOHUB_MOTION_RELINK_FLOW_K": "12",
    "BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM": "40.0", "BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM": "1.5",
    "BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES": "4", "BIOHUB_MOTION_RELINK_FLOW_GATE": "1",
    "BIOHUB_MOTION_RELINK_FLOW_ITER": "1", "BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM": "0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT": "1", "BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT": "1.0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_COST": "0", "BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "7.0",
    "BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM": "0",
}   # = the flow-v1284 recipe's overrides.json (LB 0.961)


def _mod():
    spec = importlib.util.spec_from_file_location("divfork_946", LANE / "divfork_946.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _build(tmp: Path, name: str, patch: str, ovr: dict) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BIOHUB_", "V1284", "FAST_IO"))}
    env.update(TERTIARY_SEED_DATASET="impala9397/ctg-seedC4-snap", NOTEBOOK_PATCH=f"kaggle/{patch}",
               OVERRIDES=json.dumps(ovr), KERNEL_SLUG=name, FAST_IO="1", V1284="1")
    dst = tmp / name
    subprocess.run([sys.executable, str(LANE / "build_r946.py"), str(dst)], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return dst


def _lines(d: Path) -> list[str]:
    nb = json.loads((d / "notebook.ipynb").read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code").splitlines()


def _added_removed(a: list[str], b: list[str]):
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return [l[1:] for l in d if l[0] == "+"], [l[1:] for l in d if l[0] == "-"]


def test_build_adds_only_module_and_env_line(tmp_path):
    m = _mod()
    ref = _build(tmp_path, "ctg-lane-r-flow-v1284", "x138_port_946.py", FLOW_V1284_OVR)
    on = _build(tmp_path, "ctg-lane-r-flow-v1284-divprune", "divfork_946.py", dict(FLOW_V1284_OVR, BIOHUB_DIVFORK="prune,chroma"))
    off = _build(tmp_path, "ctg-lane-r-flow-v1284-divfork-off", "divfork_946.py", FLOW_V1284_OVR)
    helper = m.HELPER.splitlines()
    call = "    edges = _divfork_apply(nodes_by_id, edges, stats, dataset = dataset)"
    for d, env_lines in ((on, ["os.environ['BIOHUB_DIVFORK'] = 'prune,chroma'"]), (off, [])):
        added, removed = _added_removed(_lines(ref), _lines(d))
        assert removed == []
        # blank lines may be aligned to existing blank lines by difflib; every non-blank added line must be ours
        assert sorted(l for l in added if l.strip()) == sorted(l for l in env_lines + helper + [call] if l.strip())
        ma = json.loads((ref / "kernel-metadata.json").read_text())
        mb = json.loads((d / "kernel-metadata.json").read_text())
        assert {k for k in ma if ma[k] != mb[k]} <= {"id", "title"}
    # the env line comes before the helper reads it; the call sits right before the FINAL print
    lo = _lines(on)
    i_env = lo.index("os.environ['BIOHUB_DIVFORK'] = 'prune,chroma'")
    i_read = next(i for i, l in enumerate(lo) if l.startswith("_DIVFORK = tuple("))
    i_call = lo.index(call)
    assert i_env < i_read < i_call
    assert lo[i_call + 1] == "    print(f'[{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges')"
    compile("\n".join(lo), "divprune_cell", "exec")


def _toy(neighbour_behind: bool):
    """F(t=5) -> c1 (continuation) and -> c2 (safe-division); optional neighbour X(t=5) sitting where c2 came from."""
    vox = np.array([1.625, 0.40625, 0.40625])
    def node(i, t, um):
        z, y, x = np.asarray(um, float) / vox
        return {"node_id": i, "t": t, "z": z, "y": y, "x": x}
    nb = {1: node(1, 4, (20, 40, 40)), 2: node(2, 5, (20, 40, 40)), 3: node(3, 6, (20, 40, 36)),
          4: node(4, 6, (20, 40, 44)), 5: node(5, 7, (20, 40, 35)), 6: node(6, 7, (20, 40, 45))}
    ee = [{"source_id": 1, "target_id": 2}, {"source_id": 2, "target_id": 3}, {"source_id": 2, "target_id": 4, "safe_division": 1},
          {"source_id": 3, "target_id": 5}, {"source_id": 4, "target_id": 6}]
    if neighbour_behind:   # X at t=5, 1.5 um from c2's position, continuing to its own child Y at t=6
        nb[7] = node(7, 5, (20, 41.5, 44))
        nb[8] = node(8, 6, (20, 48, 44))
        ee.append({"source_id": 7, "target_id": 8})
    return nb, ee


def test_off_returns_input_object():
    ns = _mod().offline_ns({})
    nb, ee = _toy(True)
    stats = {}
    assert ns["_divfork_apply"](nb, ee, stats, dataset="x") is ee
    assert stats == {}


def test_prune_geometric():
    ns = _mod().offline_ns({"BIOHUB_DIVFORK": "prune"})
    nb, ee = _toy(True)
    ids = set(nb)
    out = ns["_divfork_apply"](nb, list(ee), {}, dataset="x")
    pairs = {(e["source_id"], e["target_id"]) for e in out}
    assert (2, 4) not in pairs and (2, 3) in pairs and len(out) == len(ee) - 1
    assert set(nb) == ids
    assert all(nb[b]["t"] == nb[a]["t"] + 1 for a, b in pairs)
    nb2, ee2 = _toy(False)
    out2 = ns["_divfork_apply"](nb2, list(ee2), {}, dataset="x")
    assert {(e["source_id"], e["target_id"]) for e in out2} == {(e["source_id"], e["target_id"]) for e in ee2}


def test_prune_chroma():
    m = _mod()
    vol_t = {}

    def blob(shape, centers):
        v = np.full(shape, 100.0)
        zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        for (z, y, x), amp in centers:
            v += amp * np.exp(-(((zz - z) * 1.625) ** 2 + ((yy - y) * 0.40625) ** 2 + ((xx - x) * 0.40625) ** 2) / (2 * 1.2 ** 2))
        return v.astype(np.uint16)

    nb, ee = _toy(False)
    shape = (30, 140, 140)
    def zyx(n):
        return (nb[n]["z"], nb[n]["y"], nb[n]["x"])
    for bright in (False, True):
        vol_t[5] = blob(shape, [(zyx(2), 1000.0)])
        vol_t[6] = blob(shape, [(zyx(3), 500.0), (zyx(4), 1600.0 if bright else 500.0)])
        ns = m.offline_ns({"BIOHUB_DIVFORK": "chroma"})
        ns["read_test_frame"] = lambda dataset, t, cache: vol_t[t]
        out = ns["_divfork_apply"](nb, list(ee), {}, dataset="toy")
        pairs = {(e["source_id"], e["target_id"]) for e in out}
        assert ((2, 4) in pairs) is (not bright)
