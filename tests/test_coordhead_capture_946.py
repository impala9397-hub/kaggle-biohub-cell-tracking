"""kaggle/coordhead_capture_946.py — CPU checks (no Kaggle, no GPU).

1. The recipe part of the capture build equals the iter2 build: up to the cut ('Prediction completed') the two differ
   only in three places (the block, the stems, the train dir).
2. The 4 runtime pairs match the organizer predict script once each and compile. The edge-loop skip is switched on by
   env only.
3. Save code: row-count and grid x downsample checks, float32/float16 choice, npz content.
Test 1 needs the public base notebook, test 2 the organizer repository and test 3 torch; each skips without it.
The video stems are synthetic placeholders (the build never opens them).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from conftest import PREDICT_SCRIPT, skip_without_base

ROOT = Path(__file__).resolve().parent.parent
LANE = ROOT / "kaggle"
sys.path.insert(0, str(LANE))
import coordhead_capture_946 as cap  # noqa: E402
import v1284_946 as v1  # noqa: E402

STEMS = ["44b6_00000001", "6bba_00000002"]
F32_STEMS = ["6bba_00000002"]


def _code(d: Path) -> str:
    nb = json.loads((d / "notebook.ipynb").read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    skip_without_base()
    tmp = tmp_path_factory.mktemp("cap")
    (tmp / "stems.txt").write_text("".join(s + "\n" for s in STEMS))
    (tmp / "f32.txt").write_text("".join(s + "\n" for s in F32_STEMS))
    # KAGGLE_OWNER is dropped so the kernel id below is the builder's default owner
    env = {k: v for k, v in os.environ.items() if k not in ("V1284", "FAST_IO", "BIOHUB_CSV_FLOAT", "KAGGLE_OWNER")}
    subprocess.run([sys.executable, str(LANE / "coordhead_capture_946.py"), str(tmp / "cap"), "ctg-cap-test",
                    str(tmp / "stems.txt"), "--f32", str(tmp / "f32.txt")], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    env.update(cap.ITER2_ENV, KERNEL_SLUG="ctg-cap-test")
    subprocess.run([sys.executable, str(LANE / "build_r946.py"), str(tmp / "ref")], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)
    return tmp / "cap", tmp / "ref"


def test_capture_is_iter2_until_prediction_end(built):
    c, r = _code(built[0]), _code(built[1])
    r_head = r[: r.index(cap.END_ANCHOR) + len(cap.END_ANCHOR)]
    undo = (c.replace(v1.X138_RUNTIME_END + cap.block(F32_STEMS), v1.X138_RUNTIME_END)
             .replace(cap._TEST_DIR_NEW, cap._TEST_DIR_OLD)
             .replace(cap._STEMS_TMPL % {"stems_json": json.dumps(STEMS)}, cap._STEMS_OLD))
    assert undo == r_head + cap._TAIL
    assert "submission.csv" not in cap._TAIL and "solver" not in cap._TAIL
    compile(c, "cap", "exec")
    m = json.loads((built[0] / "kernel-metadata.json").read_text())
    assert m["id"] == "impala9397/ctg-cap-test" and m["is_private"] is True and m["enable_gpu"] is True


def test_script_pairs_on_vendor_script():
    if not PREDICT_SCRIPT.is_file():
        pytest.skip("organizer predict script not present (set ORGANIZER_REPO)")
    s = PREDICT_SCRIPT.read_text()
    p = cap.apply_script(s)
    compile(p, "ps", "exec")
    assert p.count("_v1c_sample(unet_out[:, f_idx], arr)") == 1
    assert "os.environ.get('V1C_SKIP_EDGES', '0') == '1'" in p
    assert all(line in p.splitlines() for line in s.splitlines() if "for f_idx in range(W - 1):" not in line)


@pytest.mark.parametrize("f32", [True, False])
def test_save_block(tmp_path, f32, capsys):
    torch = pytest.importorskip("torch")
    ns = {}
    exec(v1.X138_MODULE_SRC, ns)
    feat = torch.randn(1, 32, 6, 20, 20)
    V = []
    for t in range(3):
        arr = np.array([[t, 1, 2, 3], [t, 5, 19, 19]], np.int16) if t != 1 else np.empty((0, 4), np.int16)
        V.append((arr.copy(), ns["sample_features"](feat, arr).float().cpu().numpy()))
    coords = np.concatenate([e[0] for e in V]).astype(np.float32)
    ds_arr = np.array([1, 4, 4], np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    os.environ["V1C_OUT"] = str(tmp_path)
    os.environ["V1C_F32_STEMS"] = json.dumps(["vid"] if f32 else [])
    g = dict(np=np, json=json, os=os, Path=Path, _V1C=V, coords=coords, ds_arr=ds_arr, ds_path=Path("/x/vid.zarr"))
    exec(textwrap.dedent(cap._SAVE_NEW.replace(cap._SAVE_OLD, "")), g)
    z = np.load(tmp_path / "vid.npz")
    assert np.array_equal(z["coords"], coords) and z["grid"].shape == (4, 4)
    assert z["feats"].dtype == (np.float32 if f32 else np.float16) and z["feats"].shape == (4, 224)
    assert "V1C CAPTURE dataset=vid detections=4" in capsys.readouterr().out
    bad = dict(g, _V1C=V[:1])
    with pytest.raises(RuntimeError):
        exec(textwrap.dedent(cap._SAVE_NEW.replace(cap._SAVE_OLD, "")), bad)
