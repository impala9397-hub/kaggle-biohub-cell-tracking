"""kaggle/x138_port_946.py — CPU checks on synthetic data (no GPU, no real videos).

1. On top of the N05 notebook text (relink_prob PAIRS), the x138 PAIRS only **add** lines, and none of them is an env
   assignment.
2. The x138 relink with flow off (= the BIOHUB_X138_FAST_RELINK path) gives the same edges as N05's motion_relink_edges
   (relink_prob bonus w=32 included).
3. Flow seed builds a neighbour displacement field (stats) and links along a coherent drift.
4. Readmit and gap fill add nodes/edges from a synthetic dump; _x138_relink_once drops the probability-table entries
   that contain new node ids.
5. The runtime script pairs match the organizer predict script once each and compile.
Tests 1-4 take the notebook text from the public base notebook and test 5 needs the organizer repository; each skips
without its input.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from conftest import BASE_NB, PREDICT_SCRIPT, skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
VOXEL = (1.625, 0.40625, 0.40625)

X138_FLOW_ENV = {
    "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed", "BIOHUB_MOTION_RELINK_FLOW_K": "12",
    "BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM": "40.0", "BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM": "1.5",
    "BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES": "4", "BIOHUB_MOTION_RELINK_FLOW_GATE": "1",
    "BIOHUB_MOTION_RELINK_FLOW_ITER": "1", "BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM": "0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT": "1", "BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT": "1.0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_COST": "0", "BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "7.0",
    "BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM": "0",
}
X138_GAP_ENV = {
    "BIOHUB_READMIT_RADIUS_UM": "4", "BIOHUB_READMIT_MIN_SCORE": "0.965",
    "BIOHUB_GAPFILL_MAX_GAP": "3", "BIOHUB_GAPFILL_MIN_SCORE": "0.5", "BIOHUB_GAPFILL_STEP_UM": "5.0",
    "BIOHUB_GAPFILL_PEAK_RADIUS_UM": "3.5", "BIOHUB_GAPFILL_EXCLUDE_UM": "2.0",
    "BIOHUB_GAPFILL_ALLOW_SYNTHETIC": "0", "BIOHUB_GAPFILL_CONTEXT": "1", "BIOHUB_GAPFILL_MAX_ADDED_FRAC": "0.03",
    "BIOHUB_LOWDET_THRESHOLD": "0.3",
}
ALL_KNOBS = set(X138_FLOW_ENV) | set(X138_GAP_ENV) | {"BIOHUB_CACHE_DIR", "BIOHUB_X138_FAST_RELINK", "BIOHUB_ILP_TIMEOUT_S"}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply(text, pairs):
    for name, old, new in pairs:
        assert text.count(old) == 1, (name, text.count(old))
        text = text.replace(old, new, 1)
    return text


def _base_text():
    skip_without_base()
    nb = json.loads(BASE_NB.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 1
    return "".join(cells[0]["source"])


@pytest.fixture(scope="module")
def texts():
    skip_without_base()
    rp, x1 = _load("relink_prob_946"), _load("x138_port_946")
    base = _base_text()
    return _apply(base, rp.PAIRS), _apply(base, x1.PAIRS), x1


def test_off_build_is_pure_insertion(texts):
    import difflib
    n05, off, _ = texts
    d = [l for l in difflib.unified_diff(n05.splitlines(), off.splitlines(), n=0, lineterm="")
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    removed = [l for l in d if l.startswith("-")]
    env = [l for l in d if l.startswith("+os.environ[")]
    assert removed == [] and env == []
    compile(off, "off", "exec")


# ------------------------------------------------------------------------------------------ namespace from the text
_FUNCS = ("node_point", "edge_distance_um", "_next_node_id", "_position_um", "motion_relink_edges",
          "_rp_load", "_rp_motion_relink")


def _region(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


@contextlib.contextmanager
def _env(values):
    saved = {k: os.environ.get(k) for k in ALL_KNOBS | set(values)}
    for k in ALL_KNOBS:
        os.environ.pop(k, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _namespace(text, env, rp_dir=None):
    ns = {"np": np, "math": math, "os": os, "json": json, "time": time, "Path": Path, "cKDTree": cKDTree,
          "linear_sum_assignment": linear_sum_assignment, "VOXEL_SCALE_UM": VOXEL, "OUTPUT_MOTION_RELINK": True,
          "MOTION_RELINK_TIGHT_UM": 6.0, "MOTION_RELINK_RELAXED_UM": 10.0, "MOTION_RELINK_VELOCITY_WEIGHT": 0.5,
          "MOTION_RELINK_LEARNED_BONUS": 1.0, "MOTION_RELINK_MAX_FRAME_NODES": 2600,
          "refine_synthetic_midpoint": lambda dataset, t, p, cache, stats: p}
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    src = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS:
            src.append("".join(lines[node.lineno - 1:node.end_lineno]))
    with _env(env):
        exec(_region(text, "_RP_W = float(", "\ndef _rp_load("), ns)
        exec("\n\n".join(src), ns)
        if "# ==== x138 public-lineage port" in text:
            exec(_region(text, "# ==== x138 public-lineage port", "# ==== end x138 public-lineage port"), ns)
    if rp_dir is not None:
        ns["_RP_DIR"] = Path(rp_dir)
    return ns


def _stats():
    return {k: 0 for k in ("motion_relink_edges", "motion_relink_tight_edges", "motion_relink_relaxed_edges",
                           "motion_relink_frames", "motion_relink_skipped_large_frame")}


def _cloud(seed, n=120, frames=6, drift=(0.0, 3.0, 1.0), noise=1.2, spacing=9.0):
    """nodes_by_id (voxel coordinates) of n cells moving with drift µm/frame plus noise; ids by frame."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, spacing * n ** (1 / 3) * 3, size=(n, 3))
    nodes, nid, truth = {}, 0, set()
    prev_ids = None
    for t in range(frames):
        ids = []
        order = rng.permutation(n)
        for k in order:
            p = pos[k]
            nodes[nid] = {"node_id": nid, "t": t, "z": p[0] / VOXEL[0], "y": p[1] / VOXEL[1], "x": p[2] / VOXEL[2]}
            ids.append((k, nid))
            nid += 1
        cur = dict(ids)
        if prev_ids is not None:
            truth |= {(prev_ids[k], cur[k]) for k in range(n)}
        prev_ids = cur
        pos = pos + np.asarray(drift) + rng.normal(0, noise, size=pos.shape)
    return nodes, truth


def _rp_table(nodes, seed):
    rng = np.random.default_rng(seed)
    by_t = {}
    for nid, n in nodes.items():
        by_t.setdefault(n["t"], []).append(nid)
    table = {}
    for t, ids in by_t.items():
        for s in ids:
            for g in by_t.get(t + 1, []):
                if rng.random() < 0.05:
                    table[(s, g)] = float(rng.random())
    return table


def _edges(out):
    return sorted((int(e["source_id"]), int(e["target_id"])) for e in out)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_relink_equals_n05_relink(texts, seed):
    n05_text, off_text, _ = texts
    nodes, _truth = _cloud(seed)
    table = _rp_table(nodes, seed)
    learned = {k: v for k, v in list(table.items())[:40]}
    ref_ns = _namespace(n05_text, {})
    fast_ns = _namespace(off_text, {"BIOHUB_X138_FAST_RELINK": "1"})
    assert fast_ns["_X138_FAST_RELINK"] and not fast_ns["_X138_FLOW_ON"]
    for ns in (ref_ns, fast_ns):
        ns["_RP_W"], ns["_RP_TABLE"] = 32.0, dict(table)
    a = ref_ns["motion_relink_edges"](nodes, _stats(), learned)
    b = fast_ns["motion_relink_edges"](nodes, _stats(), learned)
    assert _edges(a) == _edges(b) and len(a) > 0
    # knob off: the entry never leaves the N05 code path
    off_ns = _namespace(off_text, {})
    off_ns["_RP_W"], off_ns["_RP_TABLE"] = 32.0, dict(table)
    assert not off_ns["_X138_ANY_ON"]
    assert _edges(off_ns["motion_relink_edges"](nodes, _stats(), learned)) == _edges(a)


def test_flow_seed_builds_field_and_follows_drift(texts):
    _, off_text, _ = texts
    nodes, truth = _cloud(7, n=150, frames=6, drift=(0.0, 5.0, 4.5), noise=0.6, spacing=8.0)
    ns = _namespace(off_text, X138_FLOW_ENV)
    ns["_RP_W"] = 0.0
    assert ns["_X138_FLOW_ON"]
    st = _stats()
    out = ns["motion_relink_edges"](nodes, st, {})
    got = set(_edges(out))
    assert st.get("motion_relink_flow_frames", 0) >= 4 and st.get("motion_relink_flow_predicted", 0) > 0
    ref = set(_edges(_namespace(texts[0], {})["motion_relink_edges"](nodes, _stats(), {})))
    assert len(got & truth) >= len(ref & truth)   # a coherent drift is what the field is for
    assert len(got & truth) / len(truth) > 0.9


def _track_nodes():
    """voxel coordinates. Track A: t0..4 then a 2-frame hole, track B: t7..10 at the same place.
    Track C ends at t2 with a discarded high-score peak at t3. 10 complete filler tracks keep the 3% budget >= 2."""
    nodes, edges, nid = {}, [], 0

    def add(t, z, y, x):
        nonlocal nid
        nodes[nid] = {"node_id": nid, "t": t, "z": float(z), "y": float(y), "x": float(x)}
        nid += 1
        return nid - 1

    def chain(ts, z, y, x):
        ids = [add(t, z, y, x + 2 * t) for t in ts]
        edges.extend({"source_id": a, "target_id": b, "edge_prob": 0.9, "distance_um": 1.0} for a, b in zip(ids, ids[1:]))
        return ids

    a = chain(range(0, 5), 10, 200, 200)
    b = chain(range(7, 11), 10, 200, 200)
    c = chain(range(0, 3), 10, 600, 600)
    for k in range(10):
        chain(range(0, 11), 20, 1000 + 80 * k, 1000)
    return nodes, edges, a, b, c


def _dump(tmp, nodes):
    low, score = [], []
    for t in (5, 6):                                   # the hole in A/B: sub-threshold peaks on the line
        low.append((t, 10, 200, 200 + 2 * t + 1)); score.append(0.6)
    low.append((3, 10, 600, 607)); score.append(0.98)  # C's discarded detection, ~1.2 µm from its predicted spot
    for n in nodes.values():                           # every node's own peak (excluded by GAPFILL_EXCLUDE_UM)
        low.append((n["t"], n["z"], n["y"], n["x"])); score.append(0.99)
    np.savez_compressed(Path(tmp) / "vid.npz", coords=np.zeros((0, 4), np.int16),
                        low_coords=np.asarray(low, np.int16), low_score=np.asarray(score, np.float32))


def test_readmit_and_gapfill_on_synthetic_dump(texts, tmp_path):
    _, off_text, _ = texts
    nodes, edges, a, b, c = _track_nodes()
    _dump(tmp_path, nodes)
    env = {**X138_GAP_ENV, "BIOHUB_CACHE_DIR": str(tmp_path)}   # load_low_detections reads the dir at call time
    ns = _namespace(off_text, env)
    assert ns["_X138_READMIT_ON"] and ns["_X138_GAPFILL_ON"] and not ns["_X138_FLOW_ON"]
    st = {}
    n0 = len(nodes)
    with _env(env):
        out = ns["readmit_discarded_detections"](dict(nodes), edges, st, dataset="vid")
    assert st["readmitted_nodes"] == 1 and len(out) == n0 + 1
    fresh = [v for v in out.values() if v.get("readmitted")]
    assert fresh[0]["t"] == 3 and int(fresh[0]["x"]) == 607

    st = {}
    with _env(env):
        nodes2, edges2 = ns["fill_gaps_from_low_detections"](dict(nodes), list(edges), st, dataset="vid", frame_cache={})
    assert st["gapfill_pairs_g2"] == 1 and st["gapfill_added_nodes"] == 2 and st["gapfill_added_edges"] == 3
    new = [e for e in edges2 if e.get("gap_filled")]
    assert new[0]["source_id"] == a[-1] and new[-1]["target_id"] == b[0]


def test_relink_once_drops_fresh_ids_from_probability_table(texts, tmp_path, capsys):
    _, off_text, _ = texts
    nodes, _edges_unused, *_ = _track_nodes()
    fresh_id = max(nodes) + 1
    nodes[fresh_id] = {"node_id": fresh_id, "t": 3, "z": 10.0, "y": 600.0, "x": 607.0, "readmitted": 1}
    src = np.array([fresh_id, 0, 1], np.int32)
    tgt = np.array([0, 1, fresh_id], np.int32)
    np.savez(tmp_path / "vid.npz", src=src, tgt=tgt, prob=np.array([0.9, 0.8, 0.7], np.float32), gate_um=np.float32(15))
    ns = _namespace(off_text, {**X138_GAP_ENV, "BIOHUB_CACHE_DIR": str(tmp_path)}, rp_dir=tmp_path)
    ns["_RP_W"] = 32.0
    out = ns["_x138_relink_once"](nodes, _stats(), {}, "vid")
    assert ns["_RP_TABLE"] == {} and isinstance(out, list)
    assert "pairs=3 dropped_fresh=2" in capsys.readouterr().out


def test_runtime_script_pairs_on_vendor_script():
    if not PREDICT_SCRIPT.is_file():
        pytest.skip("organizer predict script not present (set ORGANIZER_REPO)")
    x1 = _load("x138_port_946")   # the pairs are module data; this test does not need the base notebook
    s = _apply(PREDICT_SCRIPT.read_text(), x1.LOWDET_PAIRS + x1.ILP_PAIRS)
    compile(s, str(PREDICT_SCRIPT), "exec")
    assert s.count("_LOWDET.append(") == 1 and s.count("timeout=_ilp_timeout") == 1
    # the notebook block that applies them must parse and embed both pair lists
    blk = x1._RUNTIME_BLOCK
    compile(blk, "runtime_block", "exec")
    assert "x138_lowdet_peaks" in blk and "x138_ilp_timeout" in blk
