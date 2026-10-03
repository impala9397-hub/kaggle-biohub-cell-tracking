"""coordhead_capture_946.py — build a Kaggle kernel that CAPTURES training features for our own coordinate head.

What: build_r946 is run with the iter2 recipe (the x138flow overrides with BIOHUB_MOTION_RELINK_FLOW_ITER=2, FAST_IO=1),
then the notebook is changed so that it
  (1) reads competition `train/` videos from a fixed stem list instead of `test/` (TEST_DIR -> train, test_stems -> list);
  (2) for every first-seen detection of the predict script (_ps; fused primary + secondary + tertiary, same 8-view TTA)
      stores `sample_features(unet_out[:, f_idx], arr)` of the x138 V1284 module (224 = 7 offsets x 32 channels of the
      primary UNet feature map) and writes one `/kaggle/working/coordhead_capture/<stem>.npz` per video:
        coords (N,4) int16  = the full-resolution coordinates that go into the graph (right after the int16 cast at
                              the end of predict_video)
        grid   (N,4) int16  = downsampled grid coordinates (the `arr` that x138's refine receives)
        feats  (N,224)      = float32 (stems in the --f32 list, used for the validation gate) or float16 (training)
      Row i = detection i = this pipeline's geff node_id (the same assumption as V1284=late).
  (3) skips the edge-prediction loop after detection (V1C_SKIP_EDGES=1; detection and features do not depend on it) and
      cuts the ILP and all post-processing (the notebook ends at the 'Prediction completed' line, followed only by a
      capture check and a summary).
Build-time checks: every anchor exactly once. Runtime checks: script anchors once, an npz for every stem, row count =
detection count, grid x downsample == coords. Every failure is loud (RuntimeError).

Usage:
  python kaggle/coordhead_capture_946.py <dst-dir> <slug> <stems-file> [--f32 <stems-file>]
    <stems-file>: whitespace-separated stems of the train videos this kernel processes (split [i::2] over 2 GPUs)
    --f32: stems whose features are saved as float32 (the validation videos)
The capture then feeds coordhead/train_coordhead.py (pairs -> CV -> fit).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import v1284_946 as v1  # noqa: E402

OUT_DIR = "/kaggle/working/coordhead_capture"

# iter2 recipe = the x138flow overrides (kaggle/recipes.json) with FLOW_ITER=2, plus FAST_IO
ITER2_OVERRIDES = {
    "BIOHUB_DIV_DIP_MODE": "4", "BIOHUB_DIV_COMBO_MAX": "-2.6659", "BIOHUB_DIV_DIV_MISSING": "-9.0",
    "BIOHUB_DIV_Z_DIP_MU": "0.6187965957674368", "BIOHUB_DIV_Z_DIP_SD": "0.24116301719876107",
    "BIOHUB_DIV_Z_SYM_MU": "1.1339288035330837", "BIOHUB_DIV_Z_SYM_SD": "0.5528905816566464",
    "BIOHUB_DIV_Z_DIV_MU": "1.239899766683567", "BIOHUB_DIV_Z_DIV_SD": "2.1790329837769318",
    "BIOHUB_RELINK_PROB_W": "32", "BIOHUB_MOTION_RELINK_FLOW_MODE": "seed", "BIOHUB_MOTION_RELINK_FLOW_K": "12",
    "BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM": "40.0", "BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM": "1.5",
    "BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES": "4", "BIOHUB_MOTION_RELINK_FLOW_GATE": "1",
    "BIOHUB_MOTION_RELINK_FLOW_ITER": "2", "BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM": "0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT": "1", "BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT": "1.0",
    "BIOHUB_MOTION_RELINK_FLOW_RAW_COST": "0", "BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM": "7.0",
    "BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM": "0",
}
ITER2_ENV = {
    "TERTIARY_SEED_DATASET": "impala9397/ctg-seedC4-snap",
    "NOTEBOOK_PATCH": "kaggle/x138_port_946.py",
    "FAST_IO": "1",
    "OVERRIDES": json.dumps(ITER2_OVERRIDES),
}

# ---- runtime pairs on _ps ----
_IMPORT_OLD = "import tracksdata as td\n"
_IMPORT_NEW = ("import tracksdata as td\n"
               "from v1284_coordinate_refinement import sample_features as _v1c_sample\n"
               "_V1C: list = []\n")
_CAP_OLD = "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n"
_CAP_NEW = ("                _V1C.append((arr.copy(), _v1c_sample(unet_out[:, f_idx], arr).float().cpu().numpy()))\n"
            + _CAP_OLD)
_SAVE_OLD = "    coords = coords.astype(np.int16)\n"
_SAVE_NEW = (_SAVE_OLD +
             "    _v1c_grid = np.concatenate([e[0] for e in _V1C]).astype(np.int16) if _V1C else np.empty((0, 4), np.int16)\n"
             "    _v1c_feat = np.concatenate([e[1] for e in _V1C]).astype(np.float32) if _V1C else np.empty((0, 224), np.float32)\n"
             "    _V1C.clear()\n"
             "    if len(_v1c_grid) != len(coords) or len(_v1c_feat) != len(coords) or _v1c_feat.shape[1:] != (224,):\n"
             "        raise RuntimeError(f'V1C: {len(_v1c_grid)}/{len(_v1c_feat)} captured rows vs {len(coords)} detections for {ds_path.stem}')\n"
             "    _v1c_full = _v1c_grid.astype(np.float32)\n"
             "    _v1c_full[:, 1:] *= ds_arr\n"
             "    if not np.array_equal(_v1c_full.astype(np.int16), coords):\n"
             "        raise RuntimeError(f'V1C: grid x downsample != coords for {ds_path.stem}')\n"
             "    if not np.isfinite(_v1c_feat).all():\n"
             "        raise RuntimeError(f'V1C: non-finite features for {ds_path.stem}')\n"
             "    _v1c_f32 = ds_path.stem in set(json.loads(os.environ.get('V1C_F32_STEMS', '[]')))\n"
             "    _v1c_dir = Path(os.environ['V1C_OUT'])\n"
             "    _v1c_dir.mkdir(parents = True, exist_ok = True)\n"
             "    np.savez(_v1c_dir / f'{ds_path.stem}.npz', coords = coords, grid = _v1c_grid,\n"
             "             feats = _v1c_feat if _v1c_f32 else _v1c_feat.astype(np.float16), downsample = ds_arr)\n"
             "    print(f'V1C CAPTURE dataset={ds_path.stem} detections={len(coords)} f32={_v1c_f32} '\n"
             "          f'feat_abs_mean={float(np.abs(_v1c_feat).mean()):.5f}', flush = True)\n")
_EDGE_OLD = ("        for f_idx in range(W - 1):\n"
             "            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]\n")
_EDGE_NEW = ("        for f_idx in (() if os.environ.get('V1C_SKIP_EDGES', '0') == '1' else range(W - 1)):\n"
             "            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]\n")
SCRIPT_PAIRS = [
    ("v1c_import", _IMPORT_OLD, _IMPORT_NEW),
    ("v1c_capture", _CAP_OLD, _CAP_NEW),
    ("v1c_save", _SAVE_OLD, _SAVE_NEW),
    ("v1c_skip_edges", _EDGE_OLD, _EDGE_NEW),
]

_BLOCK_TMPL = '''
# === lane-r: coordinate-head feature CAPTURE — kaggle/coordhead_capture_946.py (train videos, no ILP/post) ===
(_ps.parent / 'v1284_coordinate_refinement.py').write_text(%(module_json)s)
os.environ['V1C_OUT'] = '%(out_dir)s'
os.environ['V1C_F32_STEMS'] = %(f32_json)s
os.environ['V1C_SKIP_EDGES'] = '1'
_v1c_src = _ps.read_text()
for _v1c_name, _v1c_old, _v1c_new in %(pairs_json)s:
    _v1c_n = _v1c_src.count(_v1c_old)
    if _v1c_n != 1:
        raise RuntimeError(f"V1C script anchor '{_v1c_name}' expected exactly 1 occurrence, found {_v1c_n}")
    _v1c_src = _v1c_src.replace(_v1c_old, _v1c_new, 1)
compile(_v1c_src, str(_ps), 'exec')
_ps.write_text(_v1c_src)
for _v1c_must in ('_v1c_sample(unet_out[:, f_idx], arr)', "print(f'V1C CAPTURE", "os.environ.get('V1C_SKIP_EDGES'"):
    if _ps.read_text().count(_v1c_must) != 1:
        raise RuntimeError('V1C patch did not persist marker: ' + _v1c_must)
print('V1C CAPTURE PATCH APPLIED (4 anchors) out=', os.environ['V1C_OUT'], flush=True)
'''

_TEST_DIR_OLD = "TEST_DIR = COMP_DIR / 'test'\n"
_TEST_DIR_NEW = "TEST_DIR = COMP_DIR / 'train'  # lane-r coordhead capture: competition train videos\n"
_STEMS_OLD = "test_stems = list_test_stems()\n"
_STEMS_TMPL = '''_V1C_STEMS = %(stems_json)s
_v1c_missing = [s for s in _V1C_STEMS if not (TEST_DIR / f'{s}.zarr').is_dir()]
if _v1c_missing:
    raise FileNotFoundError(f'V1C: train zarr missing: {_v1c_missing}')
test_stems = list(_V1C_STEMS)
'''
END_ANCHOR = "print(f'Prediction completed in {predict_seconds / 60:.2f} minutes')\n"
_TAIL = '''
# === lane-r: coordhead capture — stop here (no ILP / post-processing / submission) ===
import numpy as _v1c_np
import shutil as _v1c_shutil
_v1c_out = Path(os.environ['V1C_OUT'])
_v1c_rows = 0
for _v1c_s in test_stems:
    _v1c_f = _v1c_out / f'{_v1c_s}.npz'
    if not _v1c_f.is_file():
        raise RuntimeError(f'V1C: capture missing for {_v1c_s}')
    with _v1c_np.load(_v1c_f) as _v1c_z:
        _v1c_rows += len(_v1c_z['coords'])
_v1c_bytes = sum(p.stat().st_size for p in _v1c_out.glob('*.npz'))
print(f'V1C DONE videos={len(test_stems)} detections={_v1c_rows} bytes={_v1c_bytes} predict_min={predict_seconds / 60:.2f}', flush=True)
for _v1c_junk in (REPO_DIR, WORKING_DIR / 'secondary_seed_weights'):
    if Path(_v1c_junk).exists():
        _v1c_shutil.rmtree(_v1c_junk, ignore_errors=True)
'''


def block(f32_stems: list[str]) -> str:
    return _BLOCK_TMPL % {
        "module_json": json.dumps(v1.X138_MODULE_SRC, ensure_ascii=False),
        "out_dir": OUT_DIR,
        "f32_json": repr(json.dumps(sorted(f32_stems))),
        "pairs_json": json.dumps([list(p) for p in SCRIPT_PAIRS], ensure_ascii=False),
    }


def apply_script(script: str) -> str:
    for name, old, new in SCRIPT_PAIRS:
        n = script.count(old)
        if n != 1:
            raise RuntimeError(f"V1C script anchor '{name}' expected exactly 1 occurrence, found {n}")
        script = script.replace(old, new, 1)
    return script


def apply(text: str, stems: list[str], f32_stems: list[str]) -> str:
    if not stems or len(set(stems)) != len(stems):
        raise SystemExit("V1C: stems must be a non-empty list without duplicates")
    for name, old, new in (("v1c_x138_end", v1.X138_RUNTIME_END, v1.X138_RUNTIME_END + block(f32_stems)),
                           ("v1c_train_dir", _TEST_DIR_OLD, _TEST_DIR_NEW),
                           ("v1c_stems", _STEMS_OLD, _STEMS_TMPL % {"stems_json": json.dumps(list(stems))})):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"V1C notebook anchor '{name}' expected exactly 1 occurrence, found {n}")
        text = text.replace(old, new, 1)
    n = text.count(END_ANCHOR)
    if n != 1:
        raise SystemExit(f"V1C end anchor expected 1, found {n}")
    text = text[: text.index(END_ANCHOR) + len(END_ANCHOR)] + _TAIL
    compile(text, "notebook_cell", "exec")
    return text


def _read_stems(p: str) -> list[str]:
    return Path(p).read_text().split()


def main() -> None:
    args = sys.argv[1:]
    f32: list[str] = []
    if "--f32" in args:
        i = args.index("--f32")
        f32 = _read_stems(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 3:
        raise SystemExit(__doc__)
    dst, slug, stems_file = Path(args[0]).resolve(), args[1], args[2]
    stems = _read_stems(stems_file)
    env = {k: v for k, v in os.environ.items() if k not in ("V1284", "BIOHUB_CSV_FLOAT", "SCRIPT_PATCH")}
    env.update(ITER2_ENV, KERNEL_SLUG=slug)
    subprocess.run([sys.executable, str(HERE / "build_r946.py"), str(dst)], cwd=ROOT, env=env, check=True)
    nb_path = dst / "notebook.ipynb"
    nb = json.loads(nb_path.read_text())
    cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    if len(cells) != 1:
        raise SystemExit("expected one code cell")
    ci = cells[0]
    s = apply("".join(nb["cells"][ci]["source"]), stems, [x for x in f32 if x in stems])
    nb["cells"][ci]["source"] = s.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb))
    owner = os.environ.get("KAGGLE_OWNER", "impala9397")
    print(f"coordhead capture: {len(stems)} train videos ({sum(x in stems for x in f32)} float32) -> {owner}/{slug}")


if __name__ == "__main__":
    main()
