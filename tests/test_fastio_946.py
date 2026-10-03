"""kaggle/fastio_946.py + build_r946.py FAST_IO — CPU checks (no Kaggle, no GPU).

1. FAST_IO unset == FAST_IO=0 (byte-identical build), no FASTIO trace in the output. An invalid value fails the build.
2. FAST_IO=1: the only removed lines are the 7 intended ones, every added line is inside fastio_946's new text, and
   the metadata is identical apart from id/title.
3. On a fake /kaggle/input tree the original code and the patched code **pick the same file** (secondary manifest,
   tertiary seed, DeepCenter checkpoint). When the direct path exists the patch does not walk; when it is missing or
   fails, the patch walks in the same order as the original.
Every test except the pair-list check builds a kernel and needs the public base notebook (they skip without it).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
sys.path.insert(0, str(LANE))
import fastio_946 as fio  # noqa: E402

SEC_SHA = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
SEC_SLUG = "biohub-temporal-unet3d-seed314159-v1"
DC_SLUG = "biohub-deepcenter-unet3d-center-prior-v1"
BUILD_ENV = {   # FAST_IO does not depend on the notebook patch; x138_port_946 is the shipped chain head
    "TERTIARY_SEED_DATASET": "impala9397/ctg-seedC4-snap",
    "NOTEBOOK_PATCH": "kaggle/x138_port_946.py",
    "OVERRIDES": json.dumps({"BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed"}),
}
EXPECTED_REMOVED = [
    "    if input_root.exists():",
    "        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))",
    "    for manifest_path in candidates:",
    '_t_candidates += list(Path("/kaggle/input").rglob("edge_predictor_ep100.pth"))',
    "_t_path = next((c for c in _t_candidates if c.is_file()), None)",
    "    if input_root.exists():",
    "    for checkpoint_path in _dc_checkpoint_candidates():",
]


def _build(tmp: Path, name: str, **extra) -> Path:
    skip_without_base()
    env = {k: v for k, v in os.environ.items() if k not in ("FAST_IO",)}
    env.update(BUILD_ENV, KERNEL_SLUG="ctg-fastio-test", **extra)
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
    tmp = tmp_path_factory.mktemp("fio")
    return {"none": _build(tmp, "none"), "zero": _build(tmp, "zero", FAST_IO="0"), "on": _build(tmp, "on", FAST_IO="1")}


# ---------------------------------------------------------------- build
def test_knob_off_is_byte_identical(builds):
    for f in ("notebook.ipynb", "kernel-metadata.json"):
        assert (builds["none"] / f).read_bytes() == (builds["zero"] / f).read_bytes()
    assert "FASTIO" not in _code(builds["none"]) and "_fastio" not in _code(builds["none"])


def test_invalid_knob_value_fails(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _build(tmp_path, "bad", FAST_IO="yes")


def test_knob_on_changes_only_intended_lines(builds):
    off, on = _code(builds["none"]), _code(builds["on"])
    d = [l for l in difflib.unified_diff(off.splitlines(), on.splitlines(), n=0, lineterm="")
         if not l.startswith(("---", "+++", "@@"))]
    removed = [l[1:] for l in d if l.startswith("-")]
    added = [l[1:] for l in d if l.startswith("+")]
    assert removed == EXPECTED_REMOVED
    new_text = "".join(new for _, _, new in fio.pairs_for(off))
    assert all(l in new_text for l in added), [l for l in added if l not in new_text]
    assert on.count("FASTIO ") >= 3 and on.count("print(f'FASTIO") + on.count('print(f"FASTIO') == 3
    compile(on, "on", "exec")
    m_off = json.loads((builds["none"] / "kernel-metadata.json").read_text())
    m_on = json.loads((builds["on"] / "kernel-metadata.json").read_text())
    assert m_off == m_on  # built with the same slug: datasets, GPU and private flag all identical


def test_without_tertiary_only_static_pairs():
    assert [n for n, _, _ in fio.pairs_for("no tertiary here")] == [n for n, _, _ in fio.STATIC_PAIRS]


# ---------------------------------------------------------------- fake /kaggle/input
def _fake_input(root: Path, *, sec="std", ter="lower", dc="std", noise=300) -> Path:
    """sec/ter/dc: std (direct path), lower (lower-case slug mount), odd (non-standard location, found only by
    the walk), none."""
    comp = root / "competitions" / "biohub-cell-tracking-during-development" / "train"
    for i in range(noise):  # noise files, like zarr chunks
        p = comp / f"v{i % 7}.zarr" / "0" / str(i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    # a manifest with a different sha is always present (decoy)
    decoy = root / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1" / "ARTIFACT_MANIFEST.json"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(json.dumps({"model": {"weight_sha256": "0" * 64}}))
    sec_dir = {"std": root / "datasets" / "pilkwang" / SEC_SLUG, "odd": root / "zz-mount" / "seed" / SEC_SLUG}.get(sec)
    if sec_dir:
        sec_dir.mkdir(parents=True, exist_ok=True)
        (sec_dir / "ARTIFACT_MANIFEST.json").write_text(json.dumps({"model": {"weight_sha256": SEC_SHA}}))
    ter_dir = {"lower": root / "datasets" / "impala9397" / "ctg-seedc4-snap",
               "std": root / "datasets" / "impala9397" / "ctg-seedC4-snap",
               "odd": root / "aa-other" / "snap"}.get(ter)
    if ter_dir:
        ter_dir.mkdir(parents=True, exist_ok=True)
        (ter_dir / "edge_predictor_ep100.pth").write_bytes(b"w")
        (ter_dir / "config.json").write_text("{}")
    dc_dir = {"std": root / "datasets" / "pilkwang" / DC_SLUG, "odd": root / "q-mount" / "dc"}.get(dc)
    if dc_dir:
        p = dc_dir / "weights" / "full_frame_center" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"good")
    return root


def _seg(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


def _run(src: str, root: Path, ns: dict | None = None) -> dict:
    g = {"Path": Path, "json": json, "os": os, "time": time, "__name__": "seg"}
    g.update(ns or {})
    exec(compile(src.replace("/kaggle/input", str(root)), "seg", "exec"), g)
    return g


def _pick_secondary(text, root):
    src = _seg(text, "_secondary_manifest_explicit = Path(", "SECONDARY_WEIGHTS_ROOT = ")
    try:
        g = _run(src, root)
    except FileNotFoundError:
        return None, None
    return g["SECONDARY_ARTIFACTS"], g.get("_FASTIO_SECONDARY_WALKED", [None])[0]


@pytest.mark.parametrize("layout", ["std", "odd", "none"])
def test_secondary_resolves_same_as_original(builds, tmp_path, monkeypatch, capsys, layout):
    monkeypatch.delenv("BIOHUB_SECONDARY_ARTIFACT_MANIFEST", raising=False)
    root = _fake_input(tmp_path / "input", sec=layout)
    off, walked_off = _pick_secondary(_code(builds["none"]), root)
    on, walked_on = _pick_secondary(_code(builds["on"]), root)
    assert off == on
    if layout == "std":
        assert on == root / "datasets" / "pilkwang" / SEC_SLUG and walked_on == 0
        assert re.search(r"FASTIO secondary manifest_dir=\S+ walked=0 elapsed_s=\d", capsys.readouterr().out)
    elif layout == "odd":
        assert on == root / "zz-mount" / "seed" / SEC_SLUG and walked_on == 1
    else:
        assert on is None


def _pick_tertiary(text, root):
    g = _run(_seg(text, "_t_candidates = [", "if _t_path is None:"), root)
    return g["_t_path"], g.get("_fastio_t_walked", [None])[0]


@pytest.mark.parametrize("layout", ["lower", "std", "odd", "none"])
def test_tertiary_resolves_same_as_original(builds, tmp_path, capsys, layout):
    root = _fake_input(tmp_path / "input", ter=layout)
    off, _ = _pick_tertiary(_code(builds["none"]), root)
    on, walked = _pick_tertiary(_code(builds["on"]), root)
    # on a case-insensitive file system (the macOS default) the original direct path may already match, so compare
    # whether both point to the same file
    assert (off is None) == (on is None)
    if on is not None:
        assert off.resolve() == on.resolve() or os.path.samefile(off, on)
    if layout in ("lower", "std"):
        assert walked == 0
        assert "FASTIO tertiary path=" in capsys.readouterr().out
    elif layout == "odd":
        assert walked == 1
    else:
        assert on is None and walked == 1


def _dc_ns(text, root):
    consts = "\n".join(l for l in text.splitlines()
                       if l.startswith(("DEEPCENTER_MANIFEST_DEFAULT =", "DEEPCENTER_CHECKPOINT_DEFAULT =",
                                        "DEEPCENTER_RELATIVE =")))
    body = _seg(text, "# Extract candidate DeepCenter checkpoint paths from an artifact manifest",
                "try:\n    import torch\n\nexcept Exception as _dc_torch_error:\n")
    return _run(consts + "\n" + body, root)


def _dc_load(candidates):
    """Mimic the loader: skip missing paths, succeed when the content is b'good'. Returns (chosen path, paths tried)."""
    tried = []
    for p in candidates:
        if not p.exists():
            continue
        tried.append(p)
        if p.read_bytes() == b"good":
            return p, tried
    return None, tried


@pytest.mark.parametrize("case", ["explicit_good", "explicit_missing", "explicit_bad", "odd_only", "none"])
def test_deepcenter_candidates_same_choice_and_order(builds, tmp_path, monkeypatch, case):
    root = _fake_input(tmp_path / "input", dc={"odd_only": "odd", "none": "none"}.get(case, "std"))
    std_best = root / "datasets" / "pilkwang" / DC_SLUG / "weights" / "full_frame_center" / "best.pt"
    if case == "explicit_bad":  # the direct path exists but fails to load -> the walk must find the good one elsewhere
        std_best.write_bytes(b"bad")
        odd = root / "q-mount" / "dc" / "weights" / "full_frame_center" / "best.pt"
        odd.parent.mkdir(parents=True, exist_ok=True)
        odd.write_bytes(b"good")
    for k in ("BIOHUB_DEEPCENTER_MANIFEST", "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT", "BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT",
              "BIOHUB_DEEPCENTER_RELATIVE"):
        monkeypatch.delenv(k, raising=False)
    explicit = std_best if case != "explicit_missing" else root / "nowhere" / "best.pt"
    monkeypatch.setenv("BIOHUB_DEEPCENTER_CHECKPOINT", str(explicit))

    g_off = _dc_ns(_code(builds["none"]), root)
    g_on = _dc_ns(_code(builds["on"]), root)
    full = g_off["_dc_checkpoint_candidates"]()
    assert list(g_on["_fastio_dc_candidates"]()) == full  # consumed to the end, it equals the original list
    assert g_on["_dc_checkpoint_candidates"]() == full  # the skip flag is back in its original state

    g_on["_FASTIO_DC_WALKED"][0] = 0
    pick_off, tried_off = _dc_load(full)
    pick_on, tried_on = _dc_load(g_on["_fastio_dc_candidates"]())
    assert pick_on == pick_off and tried_on == tried_off
    walked = g_on["_FASTIO_DC_WALKED"][0]
    if case == "explicit_good":
        assert pick_on == std_best and walked == 0
    elif case == "explicit_missing":
        assert pick_on == std_best and walked == 0  # the direct path from preferred_dirs finds it
    elif case == "explicit_bad":
        assert pick_on == root / "q-mount" / "dc" / "weights" / "full_frame_center" / "best.pt" and walked == 1
    elif case == "odd_only":
        assert pick_on is not None and "q-mount" in str(pick_on) and walked == 1
    else:
        assert pick_on is None and walked == 1


def test_deepcenter_marker_at_call_site(builds):
    on = _code(builds["on"])
    i = on.index("DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()")
    assert "FASTIO deepcenter path=" in on[i:i + 600]
    assert "for checkpoint_path in _fastio_dc_candidates():" in on
    # the error-message path still uses the original list, walk included
    assert "checked = '\\n'.join((str(p) for p in _dc_checkpoint_candidates()[:80]))" in on
