"""kaggle/csvfloat_946.py + build_r946.py BIOHUB_CSV_FLOAT — CPU checks (no Kaggle, no GPU).

1. BIOHUB_CSV_FLOAT unset == "0" (byte-identical build), no CSV_FLOAT trace in the output. An invalid value fails the
   build.
2. "1" (on top of FAST_IO=1, as deployed): the only removed line is the node writer line, the added lines are the new
   writer and the marker, and the metadata is identical.
3. Running the new writer through the same csv.DictWriter as the notebook and reading it back with the organizer's
   csv_to_geffs gives Float64 z/y/x with values round(v, 2) and negatives as 0.0. Integer id columns are unchanged.
The build tests need the public base notebook and test 3 needs the organizer repository; each skips without it.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ORGANIZER_REPO, skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
sys.path.insert(0, str(LANE))
import csvfloat_946 as cf  # noqa: E402

BUILD_ENV = {   # CSV_FLOAT does not depend on the notebook patch; x138_port_946 is the shipped chain head
    "TERTIARY_SEED_DATASET": "impala9397/ctg-seedC4-snap",
    "NOTEBOOK_PATCH": "kaggle/x138_port_946.py",
    "OVERRIDES": json.dumps({"BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed"}),
    "FAST_IO": "1",
}
COLS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _build(tmp: Path, name: str, **extra) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if k not in ("BIOHUB_CSV_FLOAT", "FAST_IO")}
    env.update(BUILD_ENV, KERNEL_SLUG="ctg-csvfloat-test", **extra)
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
    tmp = tmp_path_factory.mktemp("csvf")
    return {"none": _build(tmp, "none"), "zero": _build(tmp, "zero", BIOHUB_CSV_FLOAT="0"),
            "on": _build(tmp, "on", BIOHUB_CSV_FLOAT="1")}


# ---------------------------------------------------------------- build
def test_knob_off_is_byte_identical(builds):
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (builds["none"] / f).read_bytes() == (builds["zero"] / f).read_bytes()
    assert "CSV_FLOAT" not in _code(builds["none"]) and cf.WRITER_OLD in _code(builds["none"])


def test_invalid_knob_value_fails(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad", BIOHUB_CSV_FLOAT="yes")


def test_knob_on_changes_only_intended_lines(builds):
    off, on = _code(builds["none"]), _code(builds["on"])
    d = [l for l in difflib.unified_diff(off.splitlines(), on.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++", "@@"))]
    removed = [l[1:] for l in d if l.startswith("-")]
    added = [l[1:] for l in d if l.startswith("+")]
    assert len(removed) == 1 and cf.WRITER_OLD in removed[0]
    assert removed[0].replace(cf.WRITER_OLD, cf.WRITER_NEW) in added
    assert "print('CSV_FLOAT z/y/x written as round(v, 2)', flush = True)" in added and len(added) == 2
    assert on.count(cf.WRITER_OLD) == 0 and on.count(cf.WRITER_NEW) == 1
    assert on.index("CSV_FLOAT z/y/x") > on.index(cf.WRITER_NEW)  # the marker comes after the whole CSV is written
    compile(on, "on", "exec")
    m_off = json.loads((builds["none"] / "kernel-metadata.json").read_text())
    m_on = json.loads((builds["on"] / "kernel-metadata.json").read_text())
    assert m_off == m_on


def test_apply_requires_exact_anchor():
    with pytest.raises(SystemExit):
        cf.apply("no writer here")
    with pytest.raises(SystemExit):
        cf.apply((cf.WRITER_OLD + "\n") * 2 + cf.MARKER_OLD)


# ---------------------------------------------------------------- writer semantics
def _writer_row(expr_fields: str, node: dict) -> dict:
    """eval the notebook writer's z/y/x dict fragment verbatim (only the node variable is passed)."""
    return eval("{" + expr_fields + "}", {"max": max, "int": int, "round": round, "float": float}, {"node": node})


def _csv_text(nodes: dict, fields: str) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    w.writeheader()
    rid = 0
    for nid in sorted(nodes):
        n = nodes[nid]
        w.writerow({"id": rid, "dataset": "v", "row_type": "node", "node_id": nid, "t": n["t"],
                    **_writer_row(fields, n), "source_id": -1, "target_id": -1})
        rid += 1
    w.writerow({"id": rid, "dataset": "v", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": min(nodes), "target_id": max(nodes)})
    return buf.getvalue()


def test_written_values_round_trip_through_vendor_reader(tmp_path):
    pl = pytest.importorskip("polars")
    if not (ORGANIZER_REPO / "scripts" / "csv_to_geffs.py").is_file():
        pytest.skip("organizer repository not present (set ORGANIZER_REPO)")
    pytest.importorskip("tracksdata")   # the organizer reader's own dependency (not in this repo's dependency groups)
    sys.path[:0] = [str(ORGANIZER_REPO / "src"), str(ORGANIZER_REPO / "scripts")]
    from csv_to_geffs import build_graph_from_rows
    nodes = {1: dict(t=0, z=0.8666666, y=80.0, x=60.5333333), 2: dict(t=1, z=-0.48, y=253.0666, x=-0.5333),
             3: dict(t=1, z=12.5, y=7.004999, x=100.125)}
    new = _csv_text(nodes, cf.WRITER_NEW)
    old = _csv_text(nodes, cf.WRITER_OLD)
    (tmp_path / "new.csv").write_text(new)
    df = pl.read_csv(tmp_path / "new.csv")
    assert all(df[c].dtype == pl.Float64 for c in ("z", "y", "x"))
    assert all(df[c].dtype == pl.Int64 for c in ("id", "node_id", "t", "source_id", "target_id"))
    nr = df.filter(pl.col("row_type") == "node").sort("node_id")
    exp = {k: [max(0.0, round(nodes[i][k], 2)) for i in sorted(nodes)] for k in "zyx"}
    assert {k: nr[k].to_list() for k in "zyx"} == exp
    assert min(min(v) for v in exp.values()) == 0.0  # clamp kept -> passes the notebook guard that rejects negatives
    g = build_graph_from_rows(nr, df.filter(pl.col("row_type") == "edge"))
    got = g.node_attrs(attr_keys=["z", "y", "x"]).sort("z")
    assert sorted(got["z"].to_list()) == sorted(exp["z"]) and g.num_edges() == 1
    # the original writer keeps integers (the knob-off CSV text does not depend on this patch)
    assert "0,v,node,1,0,1,80,61,-1,-1" in old
