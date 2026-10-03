"""coordhead/train_coordhead.py on synthetic data. CPU only, no competition data.

- the ground-truth reader understands the organizer's geff layout (zarr v3 group, axis scales in the attributes);
- pair matching is optimal 1:1 within 7 µm per frame and keeps only pairs within the radius;
- the MLP learns a bounded displacement and is saved in the exact layout the kernel loads (drop-in for V1284).
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "kaggle"))
_spec = importlib.util.spec_from_file_location("train_coordhead", ROOT / "coordhead" / "train_coordhead.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

SCALE = np.array([1.625, 0.40625, 0.40625])


def _write_geff(path, t, zyx):
    zarr = pytest.importorskip("zarr")
    g = zarr.open_group(str(path), mode="w")
    axes = [{"name": "t", "type": "time", "scale": 1.0}] + [
        {"name": a, "type": "space", "scale": float(s)} for a, s in zip("zyx", SCALE)]
    g.attrs["geff"] = {"geff_version": "1.1", "directed": True, "axes": axes}
    g.create_array("nodes/ids", data=np.arange(len(t), dtype=np.uint64))
    g.create_array("nodes/props/t/values", data=np.asarray(t, dtype=np.int64))
    for k, a in enumerate("zyx"):
        g.create_array(f"nodes/props/{a}/values", data=np.asarray(zyx[:, k], dtype=np.int64))


def test_read_geff_nodes_scales_voxels_to_um(tmp_path):
    t = np.array([0, 0, 1])
    zyx = np.array([[10, 100, 120], [20, 40, 60], [11, 101, 121]])
    _write_geff(tmp_path / "x.geff", t, zyx)
    G = tc.read_geff_nodes(tmp_path / "x.geff")
    assert np.array_equal(G["t"], t)
    assert np.allclose(G["sc"], SCALE)
    assert np.allclose(G["um"], zyx * SCALE)


def test_match_pairs_is_one_to_one_within_7um_and_radius_filtered():
    # frame 0: two GT cells; three detections — two near GT (1.0 and 3.0 µm off), one far away
    # frame 1: one GT cell whose only detection is 5.5 µm away (matched within 7 µm, but outside a 4 µm radius)
    gt_um = np.array([[16.25, 40.0, 40.0], [16.25, 60.0, 40.0], [16.25, 40.0, 40.0]])
    G = {"t": np.array([0, 0, 1]), "um": gt_um, "sc": SCALE}
    det_um = np.array([[16.25, 41.0, 40.0], [16.25, 63.0, 40.0], [16.25, 90.0, 90.0], [16.25, 45.5, 40.0]])
    C = np.c_[[0, 0, 0, 1], det_um / SCALE]
    idx, target, dist, n_gt, n_m7 = tc.match_pairs(C, G, radius=4.0)
    assert n_gt == 3 and n_m7 == 3
    assert idx.tolist() == [0, 1]
    assert np.allclose(dist, [1.0, 3.0])
    assert np.allclose(target, gt_um[:2] - det_um[:2])


def test_match_pairs_prefers_the_globally_optimal_assignment():
    # two GT cells 2 µm apart, two detections between them: greedy nearest-first would cross the pairs
    gt_um = np.array([[10.0, 20.0, 20.0], [10.0, 22.0, 20.0]])
    det_um = np.array([[10.0, 21.2, 20.0], [10.0, 23.5, 20.0]])
    G = {"t": np.array([0, 0]), "um": gt_um, "sc": SCALE}
    idx, target, dist, _, _ = tc.match_pairs(np.c_[[0, 0], det_um / SCALE], G, radius=4.0)
    order = np.argsort(idx)
    assert idx[order].tolist() == [0, 1]
    assert np.allclose(dist[order], [1.2, 1.5], atol=1e-5)  # total 2.7 µm, not 0.8 + 3.5 µm


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 224)).astype(np.float32)
    W = rng.normal(scale=0.05, size=(224, 3))
    Y = X @ W
    Y = Y / np.maximum(1.0, np.linalg.norm(Y, axis=1, keepdims=True) / 1.5)  # keep targets inside the 2 µm bound
    return X, Y


def test_mlp_learns_a_bounded_displacement():
    pytest.importorskip("torch")
    X, Y = _synthetic()
    model = tc.fit_mlp(X[:2500], Y[:2500], hidden=32, epochs=15, seed=0)
    P = tc.predict(model, X[2500:])
    err = np.linalg.norm(P - Y[2500:], axis=1).mean()
    base = np.linalg.norm(Y[2500:], axis=1).mean()
    assert err < 0.6 * base
    assert np.linalg.norm(P, axis=1).max() <= 2.0 + 1e-5


def test_fit_writes_a_drop_in_v1284_head(tmp_path):
    torch = pytest.importorskip("torch")
    X, Y = _synthetic(n=600, seed=1)
    pairs = tmp_path / "pairs.npz"
    np.savez(pairs, feats=X, target=Y, movie=np.zeros(len(X), int), is_val=np.array([False]), stems=np.array(["6bba_00000001"]))

    class A:  # argparse stand-in
        pass
    a = A()
    a.pairs, a.out, a.hidden, a.epochs, a.loss, a.seed, a.balanced = str(pairs), str(tmp_path / "head.pt"), 32, 2, "l2", 0, False
    tc.cmd_fit(a)
    saved = torch.load(a.out, map_location="cpu", weights_only=True)
    assert set(saved) == {"state_dict", "mean", "scale"}
    assert tuple(saved["state_dict"]["0.weight"].shape) == (32, 224)
    assert tuple(saved["state_dict"]["2.weight"].shape) == (3, 32)
    ns = tc._module_ns()  # the kernel's own module text: its make_head must accept the saved weights
    head = ns["make_head"]()
    head.load_state_dict(saved["state_dict"])
