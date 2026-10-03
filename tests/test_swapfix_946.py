"""kaggle/swapfix_946.py — CPU checks (no Kaggle, no GPU).

1. build: NOTEBOOK_PATCH=swapfix_946 (knob unset) differs from NOTEBOOK_PATCH=x138_port_946 **only by added lines**, all
   of them inside the new text of the swapfix pairs. The only additions on the execution path are the two
   `if _SWF_ON:` guards (byte-identical output when off). A build with the knob on differs further only by env lines.
   Metadata is identical.
2. _swf_repair on synthetic graphs: it undoes two tracks that swapped targets in one frame; node set, edge count,
   in/out degrees and division edges are unchanged; it is deterministic; it changes nothing on an already optimal
   graph; synthetic nodes use p_syn.
The build tests need the public base notebook (they skip without it).
"""
from __future__ import annotations

import copy
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
sys.path.insert(0, str(LANE))
import swapfix_946 as swf  # noqa: E402

OVR = {"BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed", "BIOHUB_MOTION_RELINK_FLOW_ITER": "2"}


def _build(tmp: Path, name: str, patch: str, ovr: dict) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if k not in ("FAST_IO", "V1284", "BIOHUB_CSV_FLOAT")}
    env.update(TERTIARY_SEED_DATASET="impala9397/ctg-seedC4-snap", NOTEBOOK_PATCH=f"kaggle/{patch}",
               OVERRIDES=json.dumps(ovr), KERNEL_SLUG="ctg-swapfix-test", FAST_IO="1")
    dst = tmp / name
    subprocess.run([sys.executable, str(LANE / "build_r946.py"), str(dst)], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return dst


def _code(d: Path) -> str:
    nb = json.loads((d / "notebook.ipynb").read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    skip_without_base()
    tmp = tmp_path_factory.mktemp("swf")
    on = dict(OVR, BIOHUB_SWAPFIX="1", BIOHUB_SWAPFIX_PARAMS='{"w_p": 4.0}')
    return {"base": _build(tmp, "base", "x138_port_946.py", OVR),
            "off": _build(tmp, "off", "swapfix_946.py", OVR),
            "on": _build(tmp, "on", "swapfix_946.py", on)}


def _diff(a: str, b: str):
    d = [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in d if l.startswith("-")], [l[1:] for l in d if l.startswith("+")]


def test_off_adds_only_inert_lines(builds):
    removed, added = _diff(_code(builds["base"]), _code(builds["off"]))
    assert removed == []
    assert [n for n, _o, _new in swf.PAIRS[-len(swf.SWF_PAIRS):]] == [n for n, _o, _new in swf.SWF_PAIRS]
    assert not set(n for n, _o, _new in swf.FLX_PAIRS) & set(n for n, _o, _new in swf.PAIRS)   # offline-only knobs
    new_text = "".join(new for _n, _o, new in swf.SWF_PAIRS)
    assert added and all(l in new_text for l in added)
    # the only statements reachable inside filter_output_graph are the two guards
    body = [l for l in added if l.startswith("    ") and not l.startswith("        ")]
    fog = _code(builds["off"])
    i = fog.index("def filter_output_graph(")
    in_fog = [l for l in added if l in fog[i:]]
    assert [l for l in in_fog if l.strip() and not l.startswith("        ")] == ["    if _SWF_ON:", "    if _SWF_ON:"], body
    # knob env lines are set before the helper reads them
    on = _code(builds["on"])
    assert on.index("os.environ['BIOHUB_SWAPFIX'] =") < on.index("_SWF_ON = ")
    compile(fog, "off", "exec")
    for f in ("kernel-metadata.json",):
        assert (builds["base"] / f).read_bytes() == (builds["off"] / f).read_bytes()


def test_on_differs_only_by_env_lines(builds):
    removed, added = _diff(_code(builds["off"]), _code(builds["on"]))
    assert removed == []
    assert added and all("BIOHUB_SWAPFIX" in l for l in added), added
    compile(_code(builds["on"]), "on", "exec")


def test_flowx_off_equals_original_round_init():
    """flowx_resid rewrites the round init; with _FLX_RESID_UM == 0 (or round 0) _flx_keep is empty -> same sets."""
    old = [n for n in swf.FLX_PAIRS if n[0] == "flowx_resid"][0]
    g = {"_FLX_RESID_UM": 0.0, "source_ids": [3, 1, 2], "target_ids": [7, 8], "frame_matches": [(1, 7, 1.0, 0.5, "tight", 0.9)]}
    for round_index in (0, 1):
        a, b = dict(g, round_index=round_index), dict(g, round_index=round_index)
        exec(old[1].replace("            ", ""), a)
        exec(old[2].replace("            ", ""), b)
        assert (a["unmatched_sources"], a["unmatched_targets"], a["frame_matches"]) == \
               (b["unmatched_sources"], b["unmatched_targets"], b["frame_matches"])


def test_function_text_identical_in_notebook(builds):
    assert swf.FUNC_SRC in _code(builds["off"])


# ---------------------------------------------------------------- synthetic graphs
SCALE = np.array([1.0, 1.0, 1.0])


def _tracks(n_t=6, speed=3.0):
    """two parallel tracks moving +x at `speed` µm/frame, 6 µm apart in y, plus a dividing track far away."""
    nodes, edges, nid = {}, [], 0
    ids = {}
    for tr, y in ((0, 0.0), (1, 6.0)):
        for t in range(n_t):
            nodes[nid] = {"node_id": nid, "t": t, "z": 10.0, "y": 20.0 + y, "x": 20.0 + speed * t}
            ids[(tr, t)] = nid
            nid += 1
    for t in range(n_t - 1):
        for tr in (0, 1):
            edges.append({"source_id": ids[(tr, t)], "target_id": ids[(tr, t + 1)]})
    # a division far away: parent at t=2, daughters at t=3
    p = nid; nodes[p] = {"node_id": p, "t": 2, "z": 10.0, "y": 80.0, "x": 80.0}
    d1 = nid + 1; nodes[d1] = {"node_id": d1, "t": 3, "z": 10.0, "y": 78.0, "x": 80.0}
    d2 = nid + 2; nodes[d2] = {"node_id": d2, "t": 3, "z": 10.0, "y": 82.0, "x": 80.0}
    edges += [{"source_id": p, "target_id": d1, "safe_division": 1}, {"source_id": p, "target_id": d2, "safe_division": 1}]
    return nodes, edges, ids


def _table(nodes, edges, good=0.8):
    return {(int(e["source_id"]), int(e["target_id"])): good for e in edges}


def _degrees(edges):
    o, i = {}, {}
    for e in edges:
        o[e["source_id"]] = o.get(e["source_id"], 0) + 1
        i[e["target_id"]] = i.get(e["target_id"], 0) + 1
    return sorted(o.items()), sorted(i.items())


def test_swapped_identity_is_repaired():
    nodes, edges, ids = _tracks()
    table = _table(nodes, edges)
    # swap the two tracks' targets at t=2 -> t=3 (cross-over)
    a, b = ids[(0, 2)], ids[(1, 2)]
    for e in edges:
        if e["source_id"] == a and e["target_id"] == ids[(0, 3)]:
            e["target_id"] = ids[(1, 3)]
        elif e["source_id"] == b and e["target_id"] == ids[(1, 3)]:
            e["target_id"] = ids[(0, 3)]
    table[(a, ids[(1, 3)])] = 0.1
    table[(b, ids[(0, 3)])] = 0.1
    before = _degrees(edges)
    for mode in ("hung", "pair"):
        out, info = swf.repair(copy.deepcopy(nodes), copy.deepcopy(edges), table, set(nodes), SCALE, mode=mode,
                               w_vel=0.5, min_samples=1)
        got = {(e["source_id"], e["target_id"]) for e in out}
        assert (a, ids[(0, 3)]) in got and (b, ids[(1, 3)]) in got, mode
        assert info["swf_changed"] == 2
        assert len(out) == len(edges) and _degrees(out) == before
        assert [e for e in out if e.get("safe_division")] == [e for e in edges if e.get("safe_division")]


def test_optimal_graph_unchanged_and_deterministic():
    nodes, edges, _ids = _tracks()
    table = _table(nodes, edges)
    out1, info1 = swf.repair(copy.deepcopy(nodes), copy.deepcopy(edges), table, set(nodes), SCALE, min_samples=1)
    out2, info2 = swf.repair(copy.deepcopy(nodes), copy.deepcopy(edges), table, set(nodes), SCALE, min_samples=1)
    assert info1["swf_changed"] == 0 and info1 == info2
    assert [(e["source_id"], e["target_id"]) for e in out1] == [(e["source_id"], e["target_id"]) for e in edges]
    assert [(e["source_id"], e["target_id"]) for e in out1] == [(e["source_id"], e["target_id"]) for e in out2]


def test_synthetic_nodes_use_p_syn():
    nodes, edges, ids = _tracks()
    table = {}   # nothing captured
    out, info = swf.repair(copy.deepcopy(nodes), copy.deepcopy(edges), table, set(), SCALE, min_samples=1, w_p=4.0)
    assert info["swf_changed"] == 0


def test_free_target_keeps_counts():
    nodes, edges, ids = _tracks()
    # track 0 at t=3 is a start (no predecessor): its predecessor edge points to a wrong far node instead
    far = max(nodes) + 1
    nodes[far] = {"node_id": far, "t": 3, "z": 10.0, "y": 20.0, "x": 40.0}
    for e in edges:
        if e["source_id"] == ids[(0, 2)] and e["target_id"] == ids[(0, 3)]:
            e["target_id"] = far
    table = _table(nodes, edges)
    table[(ids[(0, 2)], far)] = 0.05
    table[(ids[(0, 2)], ids[(0, 3)])] = 0.9
    out, info = swf.repair(copy.deepcopy(nodes), copy.deepcopy(edges), table, set(nodes), SCALE, min_samples=1,
                           free_targets=True)
    got = {(e["source_id"], e["target_id"]) for e in out}
    assert (ids[(0, 2)], ids[(0, 3)]) in got and info["swf_free_used"] == 1
    assert len(out) == len(edges)
