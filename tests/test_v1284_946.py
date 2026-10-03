"""kaggle/v1284_946.py + build_r946.py V1284 — CPU checks (no Kaggle, no GPU).

1. V1284 unset == V1284=0 (byte-identical build), no V1284 trace in the output. An invalid value, or a build without
   the x138 port, fails.
2. V1284=1: the notebook only gains one block, **inserted** at the end of the x138 port runtime block; the metadata is
   identical apart from id/title and the head dataset at the end of dataset_sources.
3. The module is x138's text verbatim (sha256) plus a stats wrapper, and the wrapper returns the same coordinates as
   x138's refine. Trilinear index_features equals the native gather at integer coordinates.
4. The 4 runtime pairs match the organizer predict script (if present) and the as-run script (env V1284_ASRUN_SCRIPT,
   if present) once each, and applied after the x138 low-detection dump they keep the dump. In the reverse order the
   dump anchor occurs 0 times (x138's silent failure).
5. The notebook block run on a fake /kaggle/input: the right head -> patch + marker; sha mismatch, missing file or
   applying twice -> RuntimeError.
Build tests need the public base notebook, torch tests need torch, and script tests need the organizer repository (or
V1284_ASRUN_SCRIPT); each skips without its input.
"""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import PREDICT_SCRIPT, skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
sys.path.insert(0, str(LANE))
import v1284_946 as v1  # noqa: E402

X138_MODULE_SHA256 = "34a2b4653f35ae73b5a7bdb46a095c812da5533c875cd1cdc4db83e4c3fd303c"
BUILD_ENV = {
    "TERTIARY_SEED_DATASET": "impala9397/ctg-seedC4-snap",
    "NOTEBOOK_PATCH": "kaggle/x138_port_946.py",
    "FAST_IO": "1",
    "OVERRIDES": json.dumps({"BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed"}),
}


def _build(tmp: Path, name: str, **extra) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if k not in ("V1284", "FAST_IO", "BIOHUB_CSV_FLOAT")}
    env.update(BUILD_ENV, KERNEL_SLUG="ctg-v1284-test")
    env.update(extra)
    dst = tmp / name
    subprocess.run([sys.executable, str(LANE / "build_r946.py"), str(dst)], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return dst


def _code(d: Path) -> str:
    nb = json.loads((d / "notebook.ipynb").read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    skip_without_base()
    tmp = tmp_path_factory.mktemp("v1284")
    return {"none": _build(tmp, "none"), "zero": _build(tmp, "zero", V1284="0"), "on": _build(tmp, "on", V1284="1")}


# ---------------------------------------------------------------- build
def test_knob_off_is_byte_identical(builds):
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (builds["none"] / f).read_bytes() == (builds["zero"] / f).read_bytes()
    assert "V1284" not in _code(builds["none"]) and "v1284" not in _code(builds["none"])
    meta = json.loads((builds["none"] / "kernel-metadata.json").read_text())
    assert v1.DATASET not in meta["dataset_sources"]


def test_invalid_knob_value_fails(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad", V1284="yes")


def test_needs_x138_port_runtime_block(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as e:
        _build(tmp_path, "nox138", V1284="1", NOTEBOOK_PATCH="kaggle/relink_prob_946.py",
               OVERRIDES=json.dumps({"BIOHUB_RELINK_PROB_W": "32"}))
    assert "V1284" in e.value.stderr


def test_knob_on_is_pure_insertion_after_x138_runtime(builds):
    off, on = _code(builds["none"]), _code(builds["on"])
    d = [l for l in difflib.unified_diff(off.splitlines(), on.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++"))]
    assert [l for l in d if l.startswith("@@")].__len__() == 1
    assert not [l for l in d if l.startswith("-")]
    added = "\n".join(l[1:] for l in d if l.startswith("+"))
    assert added.strip() == v1.block().strip()
    # the block sits right after the end of the x138 port runtime block, after the tertiary block, before the
    # inference launch
    i_blk = on.index("# === lane-r: V1284")
    assert on.index("TERTIARY PATCH APPLIED") < on.index("X138 SCRIPT PATCH APPLIED") < i_blk
    assert on.index(v1.X138_RUNTIME_END) + len(v1.X138_RUNTIME_END) == i_blk - 1  # the block starts with a single '\n'
    assert i_blk < on.index("predict_cmd = [")
    compile(on, "on", "exec")
    m_off = json.loads((builds["none"] / "kernel-metadata.json").read_text())
    m_on = json.loads((builds["on"] / "kernel-metadata.json").read_text())
    assert m_on["dataset_sources"] == m_off["dataset_sources"] + [v1.DATASET]
    assert {k: v for k, v in m_on.items() if k != "dataset_sources"} == {k: v for k, v in m_off.items() if k != "dataset_sources"}
    # the CSV keeps integers (the float CSV scored -0.011 on the public LB)
    assert "max(0, int(round(float(node['z']))))" in on and "CSV_FLOAT" not in on


# ---------------------------------------------------------------- module
def test_module_is_x138_verbatim_plus_stats():
    assert hashlib.sha256(v1.X138_MODULE_SRC.encode()).hexdigest() == X138_MODULE_SHA256
    assert v1.MODULE_SRC.startswith(v1.X138_MODULE_SRC)
    compile(v1.MODULE_SRC, "v1284_coordinate_refinement.py", "exec")


@pytest.fixture()
def refmod(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    (tmp_path / "v1284_coordinate_refinement.py").write_text(v1.MODULE_SRC)
    torch.manual_seed(0)
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.normal_(head[2].weight, std=0.5)
    torch.save({"state_dict": head.state_dict(), "mean": torch.zeros(224), "scale": torch.ones(224)}, tmp_path / "h.pt")
    monkeypatch.setenv("V1284_MODE", "candidate")
    monkeypatch.setenv("V1284_HEAD", str(tmp_path / "h.pt"))
    spec = importlib.util.spec_from_file_location("v1284_coordinate_refinement", tmp_path / "v1284_coordinate_refinement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, torch


def test_stats_wrapper_returns_x138_coordinates(refmod, capsys):
    mod, torch = refmod
    feat = torch.randn(1, 32, 6, 20, 20)
    arr = np.array([[3, 0, 0, 0], [3, 2, 7, 9], [3, 5, 19, 19], [3, 3, 10, 4]], dtype=np.int16)
    ds = Path("/x/vid_a.zarr")
    with torch.no_grad():  # predict_video runs under @torch.no_grad()
        got = mod.refine(ds, 3, arr, feat)
        ref = mod._x138_refine(ds, 3, arr, feat)
    assert got.dtype == np.float32 and np.array_equal(got, ref)
    assert not np.array_equal(got[:, 1:], arr[:, 1:].astype(np.float32))  # the head really moves them
    assert (got[:, 0] == arr[:, 0]).all() and len(got) == len(arr)       # t and count unchanged
    assert got.min() >= 0 and (got[:, 1:] <= np.array(feat.shape[-3:]) - 1).all()
    empty = np.empty((0, 4), np.int16)
    assert mod.refine(ds, 4, empty, feat) is empty
    mod._v1284_report()
    out = capsys.readouterr().out
    assert "V1284 STATS ds=vid_a frames=1 nodes=4 max_um=" in out
    mx = float(out.split("max_um=")[1].split()[0])
    assert 0 < mx <= 2.00001


def test_trilinear_matches_native_gather_on_integer_coords(refmod):
    mod, torch = refmod
    maps = torch.randn(1, 32, 6, 20, 20)
    q = torch.tensor([[[0, 0, 0], [5, 19, 19], [2, 7, 9], [3, 10, 4]]], dtype=torch.float32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    tri = mod.index_features(None, maps, q, mask)
    z, y, x = q[0, :, 0].long(), q[0, :, 1].long(), q[0, :, 2].long()
    native = maps[0, :, z, y, x].T
    assert torch.allclose(tri[0], native, atol=1e-6)
    # a fractional coordinate gives the value between the two neighbouring voxels
    half = mod.index_features(None, maps, torch.tensor([[[2.0, 7.0, 9.5]]]), torch.ones(1, 1, dtype=torch.bool))
    assert torch.allclose(half[0, 0], 0.5 * (maps[0, :, 2, 7, 9] + maps[0, :, 2, 7, 10]), atol=1e-6)


# ---------------------------------------------------------------- runtime pairs on real script text
def _scripts():
    out = []
    if PREDICT_SCRIPT.is_file():
        out.append(PREDICT_SCRIPT)
    asrun = os.environ.get("V1284_ASRUN_SCRIPT")
    if asrun and Path(asrun).is_file():
        out.append(Path(asrun))
    return out


@pytest.mark.parametrize("script", _scripts() or [None])
def test_script_pairs_once_each_and_after_lowdet(script):
    if script is None:
        pytest.skip("no predict script text (set ORGANIZER_REPO or V1284_ASRUN_SCRIPT)")
    x1 = _load("x138_port_946")
    s = script.read_text()
    plain = v1.apply_script(s)
    compile(plain, str(script), "exec")
    assert "coords = coords.astype(np.int16)" not in plain
    lowdet = s
    for name, old, new in x1.LOWDET_PAIRS + x1.ILP_PAIRS:
        assert lowdet.count(old) == 1, name
        lowdet = lowdet.replace(old, new, 1)
    both = v1.apply_script(lowdet)
    compile(both, str(script), "exec")
    assert both.count("_LOWDET.append(") == 1 and both.count("arr = _v1284_refine(") == 1
    # refine comes after the lowdet dump, right before coord_offset
    assert both.index("_LOWDET.append(") < both.index("arr = _v1284_refine(") < both.index("coord_offset[t] = (global_node_count")
    # in the reverse order the x138 port's 'x138_lowdet_peaks' anchor disappears (the cause of x138's silent failure)
    assert plain.count(x1.LOWDET_PAIRS[1][1]) == 0


# ---------------------------------------------------------------- notebook block on a fake /kaggle/input
def _run_block(tmp: Path, script_text: str, head_bytes: bytes | None, sha: str | None = None, runs: int = 1):
    inp = tmp / "input"
    ps = tmp / "repo" / "scripts" / "predict_unet_transformer.py"
    ps.parent.mkdir(parents=True, exist_ok=True)
    ps.write_text(script_text)
    if head_bytes is not None:
        hp = inp / "datasets" / "anvithpothula" / "biohub-v1284-head-s075" / v1.HEAD_FILE
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_bytes(head_bytes)
    real_sha = hashlib.sha256(head_bytes or b"").hexdigest()
    blk = v1.block().replace("/kaggle/input", str(inp)).replace(v1.HEAD_SHA256, sha or real_sha)
    g = {"Path": Path, "os": os, "_ps": ps, "_sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    old_env = {k: os.environ.get(k) for k in ("V1284_MODE", "V1284_HEAD")}
    try:
        for _ in range(runs):
            exec(compile(blk, "v1284_block", "exec"), g)
    finally:
        for k, val in old_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
    return ps


MINI_SCRIPT = (
    "import numpy as np\nimport tracksdata as td\n\n"
    "def predict_video(ds_path, unet_out):\n"
    "    for f_idx, t in enumerate([0]):\n"
    "        if True:\n"
    "            if True:\n"
    "                arr = np.zeros((0, 4), np.int16)\n"
    "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n"
    "    coords = coords.astype(np.int16)\n"
    "    return coords\n\n"
    "def predict(weights_path, device):\n"
    "    model, window_size, downsample = load_model(weights_path, device)\n"
)


def test_block_patches_script_and_writes_module(tmp_path, capsys):
    ps = _run_block(tmp_path, MINI_SCRIPT, b"head")
    s = ps.read_text()
    assert s == v1.apply_script(MINI_SCRIPT)
    assert (ps.parent / "v1284_coordinate_refinement.py").read_text() == v1.MODULE_SRC
    out = capsys.readouterr().out
    assert "V1284 HEAD:" in out and "V1284 head patched AFTER the readmit dump patch; mode = candidate (4 anchors)" in out


@pytest.mark.parametrize("case", ["bad_sha", "missing", "twice"])
def test_block_fails_loudly(tmp_path, case):
    with pytest.raises(RuntimeError):
        if case == "bad_sha":
            _run_block(tmp_path, MINI_SCRIPT, b"head", sha="0" * 64)
        elif case == "missing":
            _run_block(tmp_path, MINI_SCRIPT, None)
        else:
            _run_block(tmp_path, MINI_SCRIPT, b"head", runs=2)


# ================================================================ V1284=late (count-neutral)
@pytest.fixture(scope="module")
def late_build(tmp_path_factory):
    skip_without_base()
    return _build(tmp_path_factory.mktemp("v1284late"), "late", V1284="late")


def test_late_changes_only_block_and_csv_node_line(builds, late_build):
    off, late = _code(builds["none"]), _code(late_build)
    d = [l for l in difflib.unified_diff(off.splitlines(), late.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++", "@@"))]
    removed = [l[1:] for l in d if l.startswith("-")]
    assert removed == [v1._CSV_NODE_OLD.splitlines()[-1]]  # the only changed existing line: the CSV node writerow
    added = "\n".join(l[1:] for l in d if l.startswith("+"))
    for part in (v1.late_block(), v1.late_func()):
        for line in part.strip().splitlines():
            assert line in added
    old_lines = set(v1._CSV_NODE_OLD.splitlines())
    assert all(l in added for l in v1._CSV_NODE_NEW.splitlines() if l not in old_lines)
    compile(late, "late", "exec")
    # none of the graph-changing V1284=1 code is present
    assert "UNetNodeTransformer._index_features = _v1284_index" not in late
    assert "keep_float" not in late and "Preserve refined geometry" not in late
    m_off = json.loads((builds["none"] / "kernel-metadata.json").read_text())
    m_late = json.loads((late_build / "kernel-metadata.json").read_text())
    assert m_late["dataset_sources"] == m_off["dataset_sources"] + [v1.DATASET]


def test_v1284_eq_1_unchanged_by_late_mode(builds):
    on = _code(builds["on"])
    assert v1.block() in on and "V1284_LATE" not in on and "_v1284_late_coords" not in on


@pytest.mark.parametrize("script", _scripts() or [None])
def test_late_script_pairs_keep_int16_graph(script):
    if script is None:
        pytest.skip("no predict script text (set ORGANIZER_REPO or V1284_ASRUN_SCRIPT)")
    x1 = _load("x138_port_946")
    s = script.read_text()
    for base in (s, _apply_pairs(s, x1.LOWDET_PAIRS + x1.ILP_PAIRS)):
        late = v1.apply_script_late(base)
        compile(late, str(script), "exec")
        assert late.count("coords = coords.astype(np.int16)") == 1
        assert "_v1284_index" not in late and "arr = _v1284_refine(" not in late
        # every original line is kept (insertion only)
        assert all(line in late.splitlines() for line in base.splitlines())


def _apply_pairs(text, pairs):
    for name, old, new in pairs:
        assert text.count(old) == 1, name
        text = text.replace(old, new, 1)
    return text


def _linefit_ns(text):
    """Runnable namespace with only linefit_smooth_output_graph and its constants, from the built notebook text."""
    import ast
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "linefit_smooth_output_graph")
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("OUTPUT_LINEFIT_")]
    ns = {"np": np, "os": os, "Path": Path}
    exec(compile(ast.Module(body=consts + [fn], type_ignores=[]), "linefit", "exec"), ns)
    return ns


def _late_ns(tmp_path, late_text):
    ns = _linefit_ns(late_text)
    exec(compile(v1.late_func().replace(v1.LATE_DIR, str(tmp_path)), "late_func", "exec"), ns)
    return ns


def _track(n=6, t0=0, z=10.0, y=40.0, x=40.0, first_id=0):
    nodes = {first_id + k: {"node_id": first_id + k, "t": t0 + k, "z": z, "y": y + 2.0 * k, "x": x - 1.0 * k} for k in range(n)}
    edges = [{"source_id": first_id + k, "target_id": first_id + k + 1} for k in range(n - 1)]
    return nodes, edges


def test_late_coords_equal_linefit_of_refined_positions(tmp_path, late_build, capsys):
    """CSV coordinates = linefit run from the refined centres. Non-detection nodes and the graph stay unchanged."""
    import copy
    ns = _late_ns(tmp_path, _code(late_build))
    rng = np.random.default_rng(0)
    nodes, edges = _track(6)
    C = np.array([[n["t"], n["z"], n["y"], n["x"]] for n in nodes.values()], np.int16)
    d = rng.uniform(-0.4, 0.4, size=(6, 3)) * np.array([1.0, 4.0, 4.0])
    R = (C[:, 1:] + d).astype(np.float32)
    np.savez(tmp_path / "vid.npz", coords=C, refined=R)
    raw = {i: (n["t"], n["z"], n["y"], n["x"]) for i, n in nodes.items()}
    # as the pipeline did: linefit on the raw coordinates -> final
    final = ns["linefit_smooth_output_graph"](copy.deepcopy(nodes), edges, __import__("collections").defaultdict(int))
    # add one midpoint node (not a detection), with an id outside the range
    final[99] = {"node_id": 99, "t": 2, "z": 30.5, "y": 10.0, "x": 10.0}
    out = ns["_v1284_late_coords"]("vid", final, edges, raw)
    assert set(out) == set(range(6))
    # expected: linefit starting from the refined centres
    ref_nodes = {i: {"node_id": i, "t": nodes[i]["t"], "z": float(R[i, 0]), "y": float(R[i, 1]), "x": float(R[i, 2])} for i in nodes}
    want = ns["linefit_smooth_output_graph"](ref_nodes, edges, __import__("collections").defaultdict(int))
    for i in range(6):
        assert out[i] == pytest.approx((want[i]["z"], want[i]["y"], want[i]["x"]), abs=1e-4)
    assert "csv_nodes=7 refined=6 not_detection=1" in capsys.readouterr().out


def test_late_coords_skip_moved_or_reused_ids(tmp_path, late_build, capsys):
    ns = _late_ns(tmp_path, _code(late_build))
    nodes, edges = _track(4)
    C = np.array([[n["t"], n["z"], n["y"], n["x"]] for n in nodes.values()], np.int16)
    np.savez(tmp_path / "vid.npz", coords=C, refined=(C[:, 1:] + 0.3).astype(np.float32))
    raw = {i: (n["t"], n["z"], n["y"], n["x"]) for i, n in nodes.items()}
    raw[3] = (3, 99.0, 99.0, 99.0)  # the geff coordinates differ from the capture row -> not treated as a detection
    out = ns["_v1284_late_coords"]("vid", nodes, edges, raw)
    assert set(out) == {0, 1, 2}
    assert "refined=3 not_detection=1" in capsys.readouterr().out


def test_late_coords_fails_loudly(tmp_path, late_build):
    ns = _late_ns(tmp_path, _code(late_build))
    nodes, edges = _track(2)
    raw = {i: (n["t"], n["z"], n["y"], n["x"]) for i, n in nodes.items()}
    with pytest.raises(RuntimeError):   # no side file
        ns["_v1284_late_coords"]("missing", nodes, edges, raw)
    C = np.array([[n["t"], n["z"], n["y"], n["x"]] for n in nodes.values()], np.int16)
    np.savez(tmp_path / "vid.npz", coords=C, refined=C[:, 1:].astype(np.float32))
    with pytest.raises(RuntimeError):   # most nodes do not match a detection row -> the node_id assumption is broken
        ns["_v1284_late_coords"]("vid", nodes, edges, {0: (0, 0.0, 0.0, 0.0)})
    np.savez(tmp_path / "far.npz", coords=C, refined=(C[:, 1:] + np.array([4.0, 0, 0])).astype(np.float32))
    with pytest.raises(RuntimeError):   # displacement above 2 um
        ns["_v1284_late_coords"]("far", nodes, edges, raw)


# ================================================================ own head: V1284_HEAD_DATASET / V1284_HEAD_SHA256
OWN_DS = "impala9397/ctg-coordhead-test"
OWN_SHA = "ab" * 32


def test_own_head_knob_default_is_byte_identical(tmp_path, late_build):
    """Without the knob the late build is byte-identical to before (function defaults == the public head)."""
    assert v1.late_block() == v1.late_block(v1.DATASET, v1.HEAD_SHA256)
    assert v1.block() == v1.block(v1.DATASET, v1.HEAD_SHA256)
    again = _build(tmp_path, "late-again", V1284="late")
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (again / f).read_bytes() == (late_build / f).read_bytes()


def test_own_head_knob_swaps_only_dataset_and_sha(tmp_path, late_build):
    own = _build(tmp_path, "late-own", V1284="late", V1284_HEAD_DATASET=OWN_DS, V1284_HEAD_SHA256=OWN_SHA)
    a, b = _code(late_build), _code(own)
    assert b == a.replace(v1.DATASET.split("/")[1], OWN_DS.split("/")[1]).replace(
        v1.DATASET.split("/")[0], OWN_DS.split("/")[0]).replace(v1.HEAD_SHA256, OWN_SHA)
    assert v1.DATASET not in b and v1.HEAD_SHA256 not in b and OWN_SHA in b
    m_a = json.loads((late_build / "kernel-metadata.json").read_text())
    m_b = json.loads((own / "kernel-metadata.json").read_text())
    assert m_b["dataset_sources"] == m_a["dataset_sources"][:-1] + [OWN_DS]
    assert {k: v for k, v in m_b.items() if k != "dataset_sources"} == {k: v for k, v in m_a.items() if k != "dataset_sources"}


@pytest.mark.parametrize("extra", [
    {"V1284_HEAD_DATASET": OWN_DS},                                             # no sha
    {"V1284_HEAD_SHA256": OWN_SHA},                                             # no dataset
    {"V1284_HEAD_DATASET": "noslash", "V1284_HEAD_SHA256": OWN_SHA},
    {"V1284_HEAD_DATASET": OWN_DS, "V1284_HEAD_SHA256": "xyz"},
    {"V1284_HEAD_DATASET": OWN_DS, "V1284_HEAD_SHA256": OWN_SHA, "V1284": "0"},  # head given while the knob is off
])
def test_own_head_knob_bad_values_fail(tmp_path, extra):
    extra = dict(extra)
    extra.setdefault("V1284", "late")
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad-own", **extra)


def test_own_head_block_runs_on_fake_input(tmp_path, capsys):
    """The late block finds and verifies the head by our dataset path and sha (it does not look at the public slug)."""
    head = b"own-head-bytes"
    sha = hashlib.sha256(head).hexdigest()
    inp = tmp_path / "input"
    hp = inp / "datasets" / "impala9397" / "ctg-coordhead-test" / v1.HEAD_FILE
    hp.parent.mkdir(parents=True)
    hp.write_bytes(head)
    ps = tmp_path / "repo" / "scripts" / "predict_unet_transformer.py"
    ps.parent.mkdir(parents=True)
    ps.write_text(MINI_SCRIPT)
    blk = v1.late_block(OWN_DS, sha).replace("/kaggle/input", str(inp))
    g = {"Path": Path, "os": os, "_ps": ps, "_sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    old = {k: os.environ.get(k) for k in ("V1284_MODE", "V1284_HEAD")}
    try:
        exec(compile(blk, "late_block", "exec"), g)
        assert os.environ["V1284_HEAD"] == str(hp)
    finally:
        for k, val in old.items():
            os.environ.pop(k, None) if val is None else os.environ.__setitem__(k, val)
    assert f"sha256={sha}" in capsys.readouterr().out


# ================================================================ head average: V1284_HEAD2_DATASET / V1284_HEAD2_SHA256 (V1284=1)
HEAD2_DS = "impala9397/ctg-coordhead-test2"
HEAD2_SHA = "cd" * 32


def test_head2_knob_default_is_byte_identical(tmp_path, builds):
    """Without the knob the V1284=1 build is byte-identical to before (head2=None is the default)."""
    assert v1.block() == v1.block(v1.DATASET, v1.HEAD_SHA256, None)
    again = _build(tmp_path, "on-again", V1284="1")
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (again / f).read_bytes() == (builds["on"] / f).read_bytes()
    assert "V1284_HEAD2" not in _code(builds["on"]) and "HEADAVG" not in _code(builds["on"])


def test_head2_knob_is_pure_insertion_after_head1_env(tmp_path, builds):
    avg = _build(tmp_path, "avg", V1284="1", V1284_HEAD2_DATASET=HEAD2_DS, V1284_HEAD2_SHA256=HEAD2_SHA)
    on, b = _code(builds["on"]), _code(avg)
    d = [l for l in difflib.unified_diff(on.splitlines(), b.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++"))]
    assert sum(l.startswith("@@") for l in d) == 1 and not [l for l in d if l.startswith("-")]
    added = "\n".join(l[1:] for l in d if l.startswith("+"))
    assert added.strip() == v1.head2_section(HEAD2_DS, HEAD2_SHA).strip()
    assert b.index(v1._HEAD2_AFTER) < b.index("V1284 HEAD2:") < b.index("_v1_src = _ps.read_text()\n_v1_lowdet_on")
    compile(b, "avg", "exec")
    m_on = json.loads((builds["on"] / "kernel-metadata.json").read_text())
    m_b = json.loads((avg / "kernel-metadata.json").read_text())
    assert m_b["dataset_sources"] == m_on["dataset_sources"] + [HEAD2_DS]
    assert {k: v for k, v in m_b.items() if k != "dataset_sources"} == {k: v for k, v in m_on.items() if k != "dataset_sources"}


@pytest.mark.parametrize("extra", [
    {"V1284_HEAD2_DATASET": HEAD2_DS},                                                  # no sha
    {"V1284_HEAD2_SHA256": HEAD2_SHA},                                                  # no dataset
    {"V1284_HEAD2_DATASET": HEAD2_DS, "V1284_HEAD2_SHA256": HEAD2_SHA, "V1284": "late"},  # not allowed with late
    {"V1284_HEAD2_DATASET": HEAD2_DS, "V1284_HEAD2_SHA256": HEAD2_SHA, "V1284": "0"},
    {"V1284_HEAD2_DATASET": v1.DATASET, "V1284_HEAD2_SHA256": v1.HEAD_SHA256},          # same dataset as head1
    {"V1284_HEAD2_DATASET": HEAD2_DS, "V1284_HEAD2_SHA256": "xyz"},
])
def test_head2_knob_bad_values_fail(tmp_path, extra):
    extra = dict(extra)
    extra.setdefault("V1284", "1")
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad-head2", **extra)


def _save_head(torch, path, seed, std, mean=0.0, scale=1.0):
    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.normal_(head[2].weight, std=std)
    torch.save({"state_dict": head.state_dict(), "mean": torch.full((224,), mean), "scale": torch.full((224,), scale)}, path)
    return head


def _avg_module(tmp_path, monkeypatch, head1, head2):
    monkeypatch.setenv("V1284_MODE", "candidate")
    monkeypatch.setenv("V1284_HEAD", str(head1))
    monkeypatch.setenv("V1284_HEAD2", str(head2))
    src = tmp_path / "v1284_coordinate_refinement.py"
    src.write_text(v1.MODULE_SRC + v1.HEADAVG_APPENDIX)
    spec = importlib.util.spec_from_file_location("v1284_coordinate_refinement", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_headavg_identical_heads_equal_single_head(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    _save_head(torch, tmp_path / "a.pt", 0, 0.5, mean=0.1, scale=2.0)
    mod = _avg_module(tmp_path, monkeypatch, tmp_path / "a.pt", tmp_path / "a.pt")
    feat = torch.randn(1, 32, 6, 20, 20)
    arr = np.array([[3, 0, 0, 0], [3, 2, 7, 9], [3, 5, 19, 19], [3, 3, 10, 4]], dtype=np.int16)
    ds = Path("/x/vid_a.zarr")
    with torch.no_grad():
        got = mod.refine(ds, 3, arr, feat)            # stats wrapper → head average
        single = mod._x138_single_refine(ds, 3, arr, feat)
    assert mod._x138_refine is mod._headavg_refine
    assert got.dtype == np.float32 and np.array_equal(got, single)   # bit-identical
    assert not np.array_equal(got[:, 1:], arr[:, 1:].astype(np.float32))
    mod._headavg_report()
    out = capsys.readouterr().out
    assert "V1284 HEADAVG ds=vid_a frames=1 nodes=4 avg_mean_um=" in out and "head_diff_mean_um=0.0000" in out


def test_headavg_is_mean_of_bounded_displacements_and_bounded(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    # large weights -> each head's bounded displacement is close to 2 µm (in different directions), each with its own
    # mean/scale normalisation
    h1 = _save_head(torch, tmp_path / "a.pt", 1, 50.0, mean=0.2, scale=1.5)
    h2 = _save_head(torch, tmp_path / "b.pt", 2, 50.0, mean=-0.3, scale=0.7)
    mod = _avg_module(tmp_path, monkeypatch, tmp_path / "a.pt", tmp_path / "b.pt")
    torch.manual_seed(5)
    feat = torch.randn(1, 32, 8, 30, 30)
    rng = np.random.default_rng(0)
    arr = np.concatenate([np.full((200, 1), 4), rng.integers(2, [6, 28, 28], size=(200, 3))], axis=1).astype(np.int16)
    ds = Path("/x/vid_b.zarr")
    with torch.no_grad():
        got = mod.refine(ds, 4, arr, feat)
        x = mod.sample_features(feat, arr).float()
        d1 = mod.bounded(h1.eval(), (x - 0.2) / 1.5).numpy()
        d2 = mod.bounded(h2.eval(), (x + 0.3) / 0.7).numpy()
    um = (got[:, 1:] - arr[:, 1:].astype(np.float32)) * mod.SPACING
    assert np.allclose(um, 0.5 * (d1 + d2), atol=1e-4)
    n1, n2, na = (np.linalg.norm(v, axis=1) for v in (d1, d2, um))
    assert n1.max() <= 2.00001 and n2.max() <= 2.00001 and n1.min() > 1.8   # each head is near the bound
    assert na.max() <= 2.00001 and (got[:, 0] == arr[:, 0]).all() and len(got) == len(arr)
    assert na.mean() < 0.5 * (n1.mean() + n2.mean())                          # different directions -> a shorter mean
    empty = np.empty((0, 4), np.int16)
    assert mod.refine(ds, 5, empty, feat) is empty
    mod._headavg_report(); mod._v1284_report()
    out = capsys.readouterr().out
    assert "V1284 HEADAVG ds=vid_b frames=1 nodes=200 " in out and "V1284 STATS ds=vid_b frames=1 nodes=200 " in out
    assert float(out.split("avg_max_um=")[1].split()[0]) <= 2.0001


def _run_avg_block(tmp, head1: bytes, head2: bytes | None, sha2: str | None = None, same_file=False):
    inp = tmp / "input"
    ps = tmp / "repo" / "scripts" / "predict_unet_transformer.py"
    ps.parent.mkdir(parents=True, exist_ok=True)
    ps.write_text(MINI_SCRIPT)
    hp1 = inp / "datasets" / "anvithpothula" / "biohub-v1284-head-s075" / v1.HEAD_FILE
    hp1.parent.mkdir(parents=True, exist_ok=True)
    hp1.write_bytes(head1)
    if head2 is not None:
        hp2 = inp / "datasets" / "impala9397" / "ctg-coordhead-test2" / v1.HEAD_FILE
        hp2.parent.mkdir(parents=True, exist_ok=True)
        if same_file:
            hp2.symlink_to(hp1)
        else:
            hp2.write_bytes(head2)
    s1 = hashlib.sha256(head1).hexdigest()
    s2 = sha2 or hashlib.sha256(head2 or b"").hexdigest()
    blk = v1.block(v1.DATASET, s1, (HEAD2_DS, s2)).replace("/kaggle/input", str(inp))
    g = {"Path": Path, "os": os, "_ps": ps, "_sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    old = {k: os.environ.get(k) for k in ("V1284_MODE", "V1284_HEAD", "V1284_HEAD2")}
    try:
        exec(compile(blk, "v1284_avg_block", "exec"), g)
        env = {k: os.environ.get(k) for k in old}
    finally:
        for k, val in old.items():
            os.environ.pop(k, None) if val is None else os.environ.__setitem__(k, val)
    return ps, env, hp1


def test_head2_block_runs_on_fake_input(tmp_path, capsys):
    ps, env, hp1 = _run_avg_block(tmp_path, b"head-one", b"head-two")
    assert ps.read_text() == v1.apply_script(MINI_SCRIPT)
    assert (ps.parent / "v1284_coordinate_refinement.py").read_text() == v1.MODULE_SRC + v1.HEADAVG_APPENDIX
    assert env["V1284_HEAD"] == str(hp1) and env["V1284_HEAD2"].endswith("ctg-coordhead-test2/" + v1.HEAD_FILE)
    out = capsys.readouterr().out
    assert f"sha256={hashlib.sha256(b'head-one').hexdigest()}" in out
    assert f"V1284 HEAD2: {env['V1284_HEAD2']} sha256={hashlib.sha256(b'head-two').hexdigest()}" in out
    assert "V1284 HEADAVG module appended" in out and "mode = candidate (4 anchors)" in out


@pytest.mark.parametrize("case", ["bad_sha2", "missing2", "same_file"])
def test_head2_block_fails_loudly(tmp_path, case):
    with pytest.raises(RuntimeError):
        if case == "bad_sha2":
            _run_avg_block(tmp_path, b"head-one", b"head-two", sha2="0" * 64)
        elif case == "missing2":
            _run_avg_block(tmp_path, b"head-one", None, sha2="0" * 64)
        else:
            _run_avg_block(tmp_path, b"head-one", b"head-one", same_file=True)


# ================================================================ head average, head2 = seed list: V1284_HEAD2_SHA256S (V1284=1)
HEAD2S_DS = "impala9397/ctg-coordhead-test-seeds"
HEAD2S_SHAS = ["e1" * 32, "e2" * 32, "e3" * 32]


def test_head2s_knob_is_the_head2_section_swapped(tmp_path, builds):
    """The list build == the single-head2 build with the head2 section swapped for the list section; relative to the
    V1284=1 build without the knob it is an insertion only."""
    avg = _build(tmp_path, "avg1", V1284="1", V1284_HEAD2_DATASET=HEAD2_DS, V1284_HEAD2_SHA256=HEAD2_SHA)
    ens = _build(tmp_path, "ens", V1284="1", V1284_HEAD2_DATASET=HEAD2S_DS, V1284_HEAD2_SHA256S=",".join(HEAD2S_SHAS))
    a, b, on = _code(avg), _code(ens), _code(builds["on"])
    sec1, secs = v1.head2_section(HEAD2_DS, HEAD2_SHA), v1.head2s_section(HEAD2S_DS, HEAD2S_SHAS)
    assert a.count(sec1) == 1 and b.count(secs) == 1
    assert b == a.replace(sec1, secs)
    assert b == on.replace(v1._HEAD2_AFTER, v1._HEAD2_AFTER + secs, 1)
    assert "HEADAVG_CACHE" not in b and "_x138_refine = _headens_refine" in b and "V1284_HEAD2S" in b
    compile(b, "ens", "exec")
    m_on = json.loads((builds["on"] / "kernel-metadata.json").read_text())
    m_b = json.loads((ens / "kernel-metadata.json").read_text())
    assert m_b["dataset_sources"] == m_on["dataset_sources"] + [HEAD2S_DS]
    assert {k: v for k, v in m_b.items() if k != "dataset_sources"} == {k: v for k, v in m_on.items() if k != "dataset_sources"}
    # a list with spaces gives the same build
    ens_sp = _build(tmp_path, "ens-sp", V1284="1", V1284_HEAD2_DATASET=HEAD2S_DS, V1284_HEAD2_SHA256S=" , ".join(HEAD2S_SHAS))
    assert (ens_sp / "notebook.ipynb").read_bytes() == (ens / "notebook.ipynb").read_bytes()


@pytest.mark.parametrize("extra", [
    {"V1284_HEAD2_SHA256S": ",".join(HEAD2S_SHAS)},                                            # no dataset
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": ",".join(HEAD2S_SHAS), "V1284_HEAD2_SHA256": HEAD2_SHA},  # both
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": ",".join(HEAD2S_SHAS), "V1284": "late"},
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": ",".join(HEAD2S_SHAS), "V1284": "0"},
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": HEAD2S_SHAS[0] + ",xyz"},         # sha format
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": HEAD2S_SHAS[0] + ",," + HEAD2S_SHAS[1]},  # empty entry
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": ",".join([HEAD2S_SHAS[0]] * 2)},  # duplicate
    {"V1284_HEAD2_DATASET": v1.DATASET, "V1284_HEAD2_SHA256S": ",".join(HEAD2S_SHAS)},          # same dataset as head1
    {"V1284_HEAD2_DATASET": HEAD2S_DS, "V1284_HEAD2_SHA256S": HEAD2S_SHAS[0] + "," + v1.HEAD_SHA256},  # head1 sha
])
def test_head2s_knob_bad_values_fail(tmp_path, extra):
    extra = dict(extra)
    extra.setdefault("V1284", "1")
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad-head2s", **extra)


def _ens_module(tmp_path, monkeypatch, head1, heads2):
    monkeypatch.setenv("V1284_MODE", "candidate")
    monkeypatch.setenv("V1284_HEAD", str(head1))
    monkeypatch.setenv("V1284_HEAD2S", os.pathsep.join(str(p) for p in heads2))
    src = tmp_path / "v1284_coordinate_refinement_ens.py"
    src.write_text(v1.MODULE_SRC + v1.HEADENS_APPENDIX)
    spec = importlib.util.spec_from_file_location("v1284_coordinate_refinement", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("std", [0.5, 50.0])   # 50: each head is near the 2 µm bound (the bound really acts)
def test_headens_one_head2_equals_headavg_bitwise(tmp_path, monkeypatch, std):
    """One file in the list == the current headavg (same head1 and head2): bit-identical coordinates, boundary clip
    included."""
    torch = pytest.importorskip("torch")
    _save_head(torch, tmp_path / "a.pt", 1, std, mean=0.2, scale=1.5)
    _save_head(torch, tmp_path / "b.pt", 2, std, mean=-0.3, scale=0.7)
    avg = _avg_module(tmp_path, monkeypatch, tmp_path / "a.pt", tmp_path / "b.pt")
    ens = _ens_module(tmp_path, monkeypatch, tmp_path / "a.pt", [tmp_path / "b.pt"])
    torch.manual_seed(7)
    feat = torch.randn(1, 32, 8, 30, 30)
    rng = np.random.default_rng(1)
    arr = np.concatenate([np.full((300, 1), 4), rng.integers(0, [8, 30, 30], size=(300, 3))], axis=1).astype(np.int16)
    arr[:4, 1:] = [[0, 0, 0], [7, 29, 29], [0, 29, 0], [7, 0, 29]]   # corners: the clip applies
    ds = Path("/x/vid_c.zarr")
    with torch.no_grad():
        got = ens.refine(ds, 4, arr, feat)
        ref = avg.refine(ds, 4, arr, feat)
    assert ens._x138_refine is ens._headens_refine and avg._x138_refine is avg._headavg_refine
    assert got.dtype == ref.dtype == np.float32 and np.array_equal(got, ref)
    assert not np.array_equal(got[:, 1:], arr[:, 1:].astype(np.float32))


def test_headens_is_half_head1_plus_half_seed_mean_each_bounded(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    h1 = _save_head(torch, tmp_path / "h1.pt", 11, 50.0, mean=0.2, scale=1.5)
    specs = [(21, -0.3, 0.7), (22, 0.1, 1.1), (23, 0.0, 1.0), (24, 0.4, 2.0), (25, -0.1, 0.9)]
    hs = [(_save_head(torch, tmp_path / f"s{k}.pt", seed, 50.0, mean=m, scale=sc), m, sc) for k, (seed, m, sc) in enumerate(specs)]
    mod = _ens_module(tmp_path, monkeypatch, tmp_path / "h1.pt", [tmp_path / f"s{k}.pt" for k in range(5)])
    torch.manual_seed(5)
    feat = torch.randn(1, 32, 8, 30, 30)
    rng = np.random.default_rng(0)
    arr = np.concatenate([np.full((200, 1), 4), rng.integers(2, [6, 28, 28], size=(200, 3))], axis=1).astype(np.int16)
    with torch.no_grad():
        got = mod.refine(Path("/x/vid_b.zarr"), 4, arr, feat)
        mod.refine(Path("/x/vid_d.zarr"), 0, arr[:50], feat)
        x = mod.sample_features(feat, arr).float()
        d1 = mod.bounded(h1.eval(), (x - 0.2) / 1.5).numpy()
        dk = [mod.bounded(h.eval(), (x - m) / sc).numpy() for h, m, sc in hs]
        pre = sum(h((x - m) / sc) for h, m, sc in hs) / 5        # comparison: mean first, then bound (not used)
        d2_prebound = mod.bounded(lambda z: z, pre).numpy()
    um = (got[:, 1:] - arr[:, 1:].astype(np.float32)) * mod.SPACING
    want = 0.5 * (d1 + np.mean(dk, axis=0))
    assert np.allclose(um, want, atol=1e-4)
    assert not np.allclose(um, 0.5 * (d1 + d2_prebound), atol=1e-3)      # every head is bounded on its own
    nk = [np.linalg.norm(d, axis=1) for d in dk]
    assert max(n.max() for n in nk) <= 2.00001 and min(n.min() for n in nk) > 1.8   # each head is near the bound
    na = np.linalg.norm(um, axis=1)
    assert na.max() <= 2.00001 and (got[:, 0] == arr[:, 0]).all() and len(got) == len(arr)
    assert na.mean() < np.linalg.norm(d1, axis=1).mean()                   # different directions -> a shorter mean
    empty = np.empty((0, 4), np.int16)
    assert mod.refine(Path("/x/vid_b.zarr"), 5, empty, feat) is empty
    mod._headens_report(); mod._v1284_report()
    out = capsys.readouterr().out
    assert "V1284 HEADENS ds=vid_b frames=1 nodes=200 n_head2=5 " in out and "V1284 HEADENS ds=vid_d frames=1 nodes=50 n_head2=5 " in out
    assert "V1284 STATS ds=vid_b frames=1 nodes=200 " in out
    assert float(out.split("avg_max_um=")[1].split()[0]) <= 2.0001
    assert float(out.split("head2_seed_spread_um=")[1].split()[0]) > 0


def _run_ens_block(tmp, head1: bytes, files: dict, shas=None, head1_sha=None):
    """head1 under datasets/<owner>/<slug>/, the list dataset under input/<slug>/ as on Kaggle."""
    inp = tmp / "input"
    ps = tmp / "repo" / "scripts" / "predict_unet_transformer.py"
    ps.parent.mkdir(parents=True, exist_ok=True)
    ps.write_text(MINI_SCRIPT)
    hp1 = inp / "datasets" / "anvithpothula" / "biohub-v1284-head-s075" / v1.HEAD_FILE
    hp1.parent.mkdir(parents=True, exist_ok=True)
    hp1.write_bytes(head1)
    d2 = inp / HEAD2S_DS.split("/")[1]
    for name, data in files.items():
        d2.mkdir(parents=True, exist_ok=True)
        (d2 / name).write_bytes(data)
    if shas is None:
        shas = [hashlib.sha256(files[n]).hexdigest() for n in sorted(files)]
    blk = v1.block(v1.DATASET, head1_sha or hashlib.sha256(head1).hexdigest(), head2s=(HEAD2S_DS, shas))
    blk = blk.replace("/kaggle/input", str(inp))
    g = {"Path": Path, "os": os, "_ps": ps, "_sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    old = {k: os.environ.get(k) for k in ("V1284_MODE", "V1284_HEAD", "V1284_HEAD2", "V1284_HEAD2S")}
    try:
        exec(compile(blk, "v1284_ens_block", "exec"), g)
        env = {k: os.environ.get(k) for k in old}
    finally:
        for k, val in old.items():
            os.environ.pop(k, None) if val is None else os.environ.__setitem__(k, val)
    return ps, env, hp1, d2


def test_head2s_block_runs_on_fake_input(tmp_path, capsys):
    files = {"own_s2.pt": b"two", "own_s0.pt": b"zero", "own_s1.pt": b"one"}
    ps, env, hp1, d2 = _run_ens_block(tmp_path, b"head-one", files)
    assert ps.read_text() == v1.apply_script(MINI_SCRIPT)
    assert (ps.parent / "v1284_coordinate_refinement.py").read_text() == v1.MODULE_SRC + v1.HEADENS_APPENDIX
    assert env["V1284_HEAD"] == str(hp1) and env["V1284_HEAD2"] is None
    assert env["V1284_HEAD2S"] == os.pathsep.join(str(d2 / n) for n in ("own_s0.pt", "own_s1.pt", "own_s2.pt"))  # sorted
    out = capsys.readouterr().out
    for n in ("own_s0.pt", "own_s1.pt", "own_s2.pt"):
        assert f"V1284 HEAD2: {d2 / n} sha256={hashlib.sha256(files[n]).hexdigest()}" in out
    assert "V1284 HEADENS module appended" in out and "n_head2 = 3" in out and "mode = candidate (4 anchors)" in out
    assert "HEADAVG" not in out


@pytest.mark.parametrize("case", ["bad_sha", "missing_file", "extra_file", "order", "no_dataset"])
def test_head2s_block_fails_loudly(tmp_path, case):
    files = {"a.pt": b"A", "b.pt": b"B", "c.pt": b"C"}
    shas = [hashlib.sha256(files[n]).hexdigest() for n in sorted(files)]
    with pytest.raises(RuntimeError):
        if case == "bad_sha":
            _run_ens_block(tmp_path, b"head-one", files, shas=shas[:2] + ["0" * 64])
        elif case == "missing_file":
            _run_ens_block(tmp_path, b"head-one", {"a.pt": b"A", "b.pt": b"B"}, shas=shas)
        elif case == "extra_file":
            _run_ens_block(tmp_path, b"head-one", {**files, "d.pt": b"D"}, shas=shas)
        elif case == "order":
            _run_ens_block(tmp_path, b"head-one", files, shas=[shas[1], shas[0], shas[2]])
        else:
            _run_ens_block(tmp_path, b"head-one", {}, shas=shas)
