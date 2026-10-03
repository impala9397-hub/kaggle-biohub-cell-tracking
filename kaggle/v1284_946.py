"""v1284_946.py — build-time patch that adds x138's V1284 coordinate-regression head to the build_r946 kernel.

build_r946.py applies `apply(text)` to the notebook cell and adds `anvithpothula/biohub-v1284-head-s075` to
dataset_sources only when the build env sets `V1284=1` (`V1284=late` calls `apply_late` instead: the count-neutral
variant at the end of this file). With `V1284` unset or "0" this file is not even imported → the build is byte-identical.

Source and license: X138_MODULE_SRC and the four script edits below are taken verbatim from the public notebook
`anvithpothula/biohub-x138` (now titled "Biohub 0.953 LB | ORIGINAL"), released under Apache-2.0; see NOTICE.
The head weights (`anvithpothula/biohub-v1284-head-s075`) are CC0. Everything else in this file is ours (MIT).

What (public `anvithpothula/biohub-x138` notebook, lines 1603–1640, pulled 2026-09-22, ported as is):
  At the first-seen detection centres arr (t,z,y,x; downsampled voxels) of each frame, sample the primary UNet features
  (edge features averaged over test-time augmentation, TTA) as the centre vector plus 6 directional differences
  (7 × 32 = 224 dims), feed them to an MLP 224→32→3, and move each centre by the displacement bounded to ‖Δ‖ ≤ 2 µm.
  At runtime, write `v1284_coordinate_refinement.py` next to the predict script (_ps) and apply 4 pairs to _ps:
    1. v1284_import      after `import tracksdata as td`, import refine / index_features
    2. v1284_refine      before `coord_offset[t] = (...)`, insert `arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])`
    3. v1284_keep_float  remove `coords = coords.astype(np.int16)` at the end of predict_video → float coordinates flow
                         into the ILP, geff and post-processing
    4. v1284_trilinear   after `load_model(weights_path, ...)`, set `UNetNodeTransformer._index_features = _v1284_index`
                         (class-wide: the primary, secondary and tertiary edge-feature lookups become trilinear; with
                         integer coordinates this equals the original gather)
  The submission CSV still holds `int(round())` integers (BIOHUB_CSV_FLOAT is not used; it scored −0.011 on the public
  leaderboard).

Order (the silent failure noted in x138's comments): the x138 port's low-detection dump patch ('x138_lowdet_peaks') also
  anchors on the `coord_offset[t] = ...` line, and that anchor disappears if V1284 runs first. So the block is inserted
  right after the last print of the x138 port runtime block → it applies to the as-run text **after** tertiary (after the
  TTA anchor), relink_prob, node_select and x138 lowdet/ILP have all been applied to _ps. In builds with lowdet on, it
  checks that `_LOWDET.append(` is in the script both before and after the V1284 patch.
All failures are loud: anchor count != 1, head file missing or matched more than once, sha256 mismatch, marker not
  persisted → RuntimeError (no try/except).

Addition (not in x138; output unchanged): a wrapper around refine collects, per dataset, the detection count and the
  max/mean displacement (µm, with the same SPACING as x138's check) and prints
  `V1284 STATS ds=<stem> frames=<n> nodes=<n> max_um=<..> mean_um=<..>` when the predict process exits.
Weights: `v1284_head.pt`, 33,913 B, sha256 625a0d93…da00c (CC0, downloaded 2026-09-24; loaded on CPU: keys
  state_dict/mean/scale; 0.weight (32,224), 2.weight (3,32), mean/scale (224,)).
Head average (build V1284_HEAD2_DATASET/V1284_HEAD2_SHA256; V1284=1 only; off by default → byte-identical): mounts a
  second head after checking its sha256 and appends HEADAVG_APPENDIX to the end of the module (file append; existing
  lines unchanged). Each detection moves by 0.5 * (bounded(head1, (x−mean1)/scale1) + bounded(head2, (x−mean2)/scale2))
  µm, converted to voxels by dividing by SPACING — each vector is ≤ 2 µm, so their mean is ≤ 2 µm as well. With the
  same head twice, the result is bit-identical to x138 refine.
  Markers: `V1284 HEAD2: <path> sha256=<..>`, `V1284 HEADAVG module appended`, and `V1284 HEADAVG ds=...` per video.
Head average with second head = seed mean (build V1284_HEAD2_DATASET + V1284_HEAD2_SHA256S, a comma-separated list;
  either this or V1284_HEAD2_SHA256): reads every `*.pt` in the dataset in file-name order; their sha256 list must equal
  the knob, order included (a different count or any differing sha → RuntimeError). Appends HEADENS_APPENDIX instead of
  HEADAVG_APPENDIX to the end of the module: each detection moves by 0.5 * bounded(head1) + 0.5 * mean_k bounded(head2_k)
  µm — every head is bounded on its own (≤ 2 µm), so the mean is ≤ 2 µm as well. A one-element list is bit-identical
  to HEADAVG (tests/test_v1284_946.py). The single-file path and the knob-off output stay byte-identical.
  Markers: `V1284 HEAD2: <path> sha256=<..>` per head2, `V1284 HEADENS module appended`, and `V1284 HEADENS ds=...`
  per video.
Markers (stdout): `V1284 HEAD: <path> sha256=<..>`,
  `V1284 head patched AFTER the readmit dump patch; mode = candidate (4 anchors)`, `V1284 STATS ...` (per dataset).
"""
from __future__ import annotations

import json

DATASET = "anvithpothula/biohub-v1284-head-s075"
HEAD_FILE = "v1284_head.pt"
HEAD_SHA256 = "625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c"
HEAD_BYTES = 33913

# x138's v1284_coordinate_refinement.py, unchanged to the last character
# (sha256 34a2b465…d303c, checked in tests/test_v1284_946.py).
X138_MODULE_SRC = r'''"""Frozen-feature coordinate regression at first-seen fused detector centers."""
import os
from pathlib import Path
import numpy as np
import torch

SPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)
OFFSETS = ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))
_CACHE = None


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def sample_features(feature, arr):
    xyz = torch.as_tensor(arr[:, 1:], device=feature.device, dtype=torch.long)
    blocks = []
    for offset in OFFSETS:
        loc = xyz + torch.tensor(offset, device=feature.device)
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size-1)
        blocks.append(feature[0, :, loc[:,0], loc[:,1], loc[:,2]].T)
    # Directional differences plus the central representation.
    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)


def index_features(self, maps, coords, mask):
    """Trilinear lookup; integer coordinates reproduce native gather exactly."""
    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)
    for batch in range(len(maps)):
        n = int(mask[batch].sum())
        if not n:
            continue
        q = coords[batch, :n].clone()
        for axis, size in enumerate(maps.shape[-3:]):
            q[:,axis].clamp_(0, size-1)
        low = q.floor().long()
        frac = q-low
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):
                    shift = torch.tensor([z,y,x], device=maps.device)
                    loc = low+shift
                    for axis, size in enumerate(maps.shape[-3:]):
                        loc[:,axis].clamp_(0,size-1)
                    weight = torch.where(shift.bool(), frac, 1-frac).prod(dim=1)
                    out[batch,:n] += maps[batch,:,loc[:,0],loc[:,1],loc[:,2]].T * weight[:,None]
    return out


def refine(ds_path, t, arr, feature):
    global _CACHE
    mode = os.environ['V1284_MODE']
    if not len(arr):
        return arr
    if mode == 'zero':
        return arr.astype(np.float32)
    x = sample_features(feature, arr).float()
    if mode == 'capture':
        folder = Path(os.environ['V1284_CAPTURE']) / ds_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f'{int(t):04d}.npz', coords=arr, features=x.cpu().numpy())
        return arr
    if _CACHE is None:
        saved = torch.load(os.environ['V1284_HEAD'], map_location='cpu', weights_only=True)
        head = make_head().to(feature.device)
        head.load_state_dict(saved['state_dict']); head.eval()
        _CACHE = (head, saved['mean'].to(feature.device), saved['scale'].to(feature.device))
    head, mean, scale = _CACHE
    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING
    result = arr.astype(np.float32).copy()
    result[:,1:] += shift
    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)
    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:
        raise RuntimeError('invalid V1284 displacement')
    return result
'''

# Our addition: collects displacement statistics only (the return value is x138 refine's, unchanged).
STATS_APPENDIX = r'''

# ---- lane-r (kaggle/v1284_946.py): displacement stats only; the returned coordinates are x138's refine output ----
import atexit as _v1284_atexit
_x138_refine = refine
_V1284_STATS = {}


def refine(ds_path, t, arr, feature):
    out = _x138_refine(ds_path, t, arr, feature)
    if len(arr) and os.environ.get('V1284_MODE') == 'candidate':
        d = np.linalg.norm((np.asarray(out, dtype=np.float32)[:, 1:] - arr[:, 1:].astype(np.float32)) * SPACING, axis=1)
        st = _V1284_STATS.setdefault(ds_path.stem, [0, 0.0, 0.0, 0])
        st[0] += int(len(d))
        st[1] = max(st[1], float(d.max()))
        st[2] += float(d.sum())
        st[3] += 1
    return out


def _v1284_report():
    for stem, (n, mx, sm, frames) in sorted(_V1284_STATS.items()):
        print(f'V1284 STATS ds={stem} frames={frames} nodes={n} max_um={mx:.4f} mean_um={sm / max(n, 1):.4f}', flush=True)


_v1284_atexit.register(_v1284_report)
'''

MODULE_SRC = X138_MODULE_SRC + STATS_APPENDIX

# ---- runtime pairs on _ps (the 4 replaces of x138 notebook lines 1631–1638: same anchors, same new text) ----
_IMPORT_OLD = "import tracksdata as td\n"
_IMPORT_NEW = ("import tracksdata as td\n"
               "from v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index\n")
_REFINE_OLD = "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n"
_REFINE_NEW = ("                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n"
               "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n")
_INT16_OLD = "    coords = coords.astype(np.int16)\n"
_INT16_NEW = "    # Preserve refined geometry through association and graph output.\n"
_LOAD_OLD = "    model, window_size, downsample = load_model(weights_path, device)\n"
_LOAD_NEW = ("    model, window_size, downsample = load_model(weights_path, device)\n"
             "    UNetNodeTransformer._index_features = _v1284_index\n")
SCRIPT_PAIRS = [
    ("v1284_import", _IMPORT_OLD, _IMPORT_NEW),
    ("v1284_refine", _REFINE_OLD, _REFINE_NEW),
    ("v1284_keep_float", _INT16_OLD, _INT16_NEW),
    ("v1284_trilinear", _LOAD_OLD, _LOAD_NEW),
]

# ---- notebook insertion point: the last print of the x138 port runtime block (end of x138_port_946._RUNTIME_BLOCK) ----
X138_RUNTIME_END = "          f'ilp_timeout_s={_x1_ilp_timeout}', flush=True)\n"

_BLOCK_TMPL = '''
# === lane-r: V1284 coordinate-regression head — kaggle/v1284_946.py (build V1284=1, runtime patch on _ps) ===
# Placed after the x138 port's low-detection dump patch for the same reason as in x138: both anchor on the `coord_offset[t] = ...` line.
_v1_candidates = [Path('/kaggle/input/%(slug)s/%(fname)s'), Path('/kaggle/input/datasets/%(owner)s/%(slug)s/%(fname)s')]
_v1_head = next((c for c in _v1_candidates if c.is_file()), None)
if _v1_head is None:
    _v1_found = sorted(Path('/kaggle/input').rglob('%(slug)s/%(fname)s'))
    if len(_v1_found) != 1:
        raise RuntimeError(('V1284 head mount mismatch', [str(p) for p in _v1_found]))
    _v1_head = _v1_found[0]
_v1_sha = _sha256_file(_v1_head)
if _v1_sha != '%(sha)s':
    raise RuntimeError(f'V1284 head sha256 mismatch: {_v1_sha} at {_v1_head}')
print('V1284 HEAD:', _v1_head, 'sha256=' + _v1_sha, 'bytes=', _v1_head.stat().st_size, flush=True)
(_ps.parent / 'v1284_coordinate_refinement.py').write_text(%(module_json)s)
os.environ['V1284_MODE'] = 'candidate'
os.environ['V1284_HEAD'] = str(_v1_head)
_v1_src = _ps.read_text()
_v1_lowdet_on = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0) > 0 and bool(os.environ.get('BIOHUB_CACHE_DIR', '').strip())
if _v1_lowdet_on and '_LOWDET.append(' not in _v1_src:
    raise RuntimeError('V1284: low-detection dump is on but not yet in the script — V1284 must run after it')
for _v1_name, _v1_old, _v1_new in %(pairs_json)s:
    _v1_n = _v1_src.count(_v1_old)
    if _v1_n != 1:
        raise RuntimeError(f"V1284 script anchor '{_v1_name}' expected exactly 1 occurrence, found {_v1_n}")
    _v1_src = _v1_src.replace(_v1_old, _v1_new, 1)
compile(_v1_src, str(_ps), 'exec')
_ps.write_text(_v1_src)
_v1_src = _ps.read_text()
for _v1_must in ('from v1284_coordinate_refinement import', 'arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])',
                 'UNetNodeTransformer._index_features = _v1284_index'):
    if _v1_src.count(_v1_must) != 1:
        raise RuntimeError('V1284 patch did not persist marker: ' + _v1_must)
if 'coords = coords.astype(np.int16)' in _v1_src:
    raise RuntimeError('V1284: int16 coordinate cast still present')
if _v1_lowdet_on and _v1_src.count('_LOWDET.append(') != 1:
    raise RuntimeError('V1284: low-detection dump lost after V1284 patch')
print('V1284 head patched AFTER the readmit dump patch; mode =', os.environ['V1284_MODE'],
      '(4 anchors) lowdet_in_script=', '_LOWDET.append(' in _v1_src, flush=True)
'''


# ---- head average (V1284_HEAD2_*): appended after the stats wrapper. The wrapper's refine looks up `_x138_refine`
#      at call time, so rebinding that name to the averaging refine makes V1284 STATS measure the averaged
#      displacement. The coordinate update is the same formula as x138 refine; only the bounded displacement (µm)
#      is the mean of the two heads.
HEADAVG_APPENDIX = r"""

# ---- lane-r (kaggle/v1284_946.py): head average — mean of two heads' bounded displacements (um) ----
_x138_single_refine = _x138_refine
_HEADAVG_CACHE = None
_HEADAVG_STATS = {}


def _headavg_load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    head = make_head().to(device)
    head.load_state_dict(saved['state_dict']); head.eval()
    return head, saved['mean'].to(device), saved['scale'].to(device)


def _headavg_refine(ds_path, t, arr, feature):
    global _HEADAVG_CACHE
    if os.environ['V1284_MODE'] != 'candidate' or not len(arr):
        return _x138_single_refine(ds_path, t, arr, feature)
    x = sample_features(feature, arr).float()
    if _HEADAVG_CACHE is None:
        _HEADAVG_CACHE = (_headavg_load(os.environ['V1284_HEAD'], feature.device),
                          _headavg_load(os.environ['V1284_HEAD2'], feature.device))
    (h1, m1, s1), (h2, m2, s2) = _HEADAVG_CACHE
    d1 = bounded(h1, (x-m1)/s1)
    d2 = bounded(h2, (x-m2)/s2)
    um = (0.5 * (d1 + d2)).cpu().numpy()
    shift = um / SPACING
    result = arr.astype(np.float32).copy()
    result[:,1:] += shift
    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)
    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:
        raise RuntimeError('invalid V1284 head-average displacement')
    n1 = np.linalg.norm(d1.cpu().numpy(), axis=1)
    n2 = np.linalg.norm(d2.cpu().numpy(), axis=1)
    na = np.linalg.norm(um, axis=1)
    nd = np.linalg.norm((d1 - d2).cpu().numpy(), axis=1)
    st = _HEADAVG_STATS.setdefault(ds_path.stem, [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
    st[0] += 1; st[1] += int(len(na)); st[2] += float(na.sum()); st[3] = max(st[3], float(na.max()))
    st[4] += float(n1.sum()); st[5] += float(n2.sum()); st[6] += float(nd.sum())
    return result


_x138_refine = _headavg_refine


def _headavg_report():
    for stem, (frames, n, sa, mx, s1, s2, sd) in sorted(_HEADAVG_STATS.items()):
        k = max(n, 1)
        print(f'V1284 HEADAVG ds={stem} frames={frames} nodes={n} avg_mean_um={sa / k:.4f} avg_max_um={mx:.4f} '
              f'head1_mean_um={s1 / k:.4f} head2_mean_um={s2 / k:.4f} head_diff_mean_um={sd / k:.4f} (head outputs before border clip)', flush=True)


_v1284_atexit.register(_headavg_report)
"""

# Second-head section, inserted inside the block right after the `os.environ['V1284_HEAD'] = ...` line
# (only when V1284_HEAD2_* is set).
_HEAD2_AFTER = "os.environ['V1284_HEAD'] = str(_v1_head)\n"
_HEAD2_TMPL = """# --- lane-r: V1284 head average — second head (build V1284_HEAD2_DATASET) ---
_v1_candidates2 = [Path('/kaggle/input/%(slug)s/%(fname)s'), Path('/kaggle/input/datasets/%(owner)s/%(slug)s/%(fname)s')]
_v1_head2 = next((c for c in _v1_candidates2 if c.is_file()), None)
if _v1_head2 is None:
    _v1_found2 = sorted(Path('/kaggle/input').rglob('%(slug)s/%(fname)s'))
    if len(_v1_found2) != 1:
        raise RuntimeError(('V1284 head2 mount mismatch', [str(p) for p in _v1_found2]))
    _v1_head2 = _v1_found2[0]
_v1_sha2 = _sha256_file(_v1_head2)
if _v1_sha2 != '%(sha)s':
    raise RuntimeError(f'V1284 head2 sha256 mismatch: {_v1_sha2} at {_v1_head2}')
if _v1_head2.resolve() == _v1_head.resolve():
    raise RuntimeError(f'V1284 head2 is the same file as head1: {_v1_head2}')
print('V1284 HEAD2:', _v1_head2, 'sha256=' + _v1_sha2, 'bytes=', _v1_head2.stat().st_size, flush=True)
os.environ['V1284_HEAD2'] = str(_v1_head2)
with (_ps.parent / 'v1284_coordinate_refinement.py').open('a') as _v1_f:
    _v1_f.write(%(appendix_json)s)
_v1_mod = (_ps.parent / 'v1284_coordinate_refinement.py').read_text()
compile(_v1_mod, 'v1284_coordinate_refinement.py', 'exec')
if _v1_mod.count('_x138_refine = _headavg_refine') != 1:
    raise RuntimeError('V1284 head average appendix did not persist')
print('V1284 HEADAVG module appended: displacement = mean(bounded(head1), bounded(head2)) um; head1 =',
      os.environ['V1284_HEAD'], 'head2 =', os.environ['V1284_HEAD2'], flush=True)
"""


def head2_section(dataset: str, sha: str) -> str:
    owner, slug = _check_head(dataset, sha)[0].split("/", 1)
    return _HEAD2_TMPL % {"owner": owner, "slug": slug, "fname": HEAD_FILE, "sha": sha,
                          "appendix_json": json.dumps(HEADAVG_APPENDIX, ensure_ascii=False)}


# ---- head average, second head = mean of several same-recipe seeds (V1284_HEAD2_SHA256S); appended instead of
#      HEADAVG_APPENDIX. head2 = mean_k bounded(head2_k): summed in file-name order, then divided by the count
#      (one file: d/1 = d, bit-identical to HEADAVG).
HEADENS_APPENDIX = r"""

# ---- lane-r (kaggle/v1284_946.py): head average, second head = mean of same-recipe seeds ----
# displacement um = 0.5 * (bounded(head1) + mean_k bounded(head2_k)); every head is bounded on its own (<= 2 um each).
_x138_single_refine = _x138_refine
_HEADENS_CACHE = None
_HEADENS_STATS = {}


def _headens_load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    head = make_head().to(device)
    head.load_state_dict(saved['state_dict']); head.eval()
    return head, saved['mean'].to(device), saved['scale'].to(device)


def _headens_refine(ds_path, t, arr, feature):
    global _HEADENS_CACHE
    if os.environ['V1284_MODE'] != 'candidate' or not len(arr):
        return _x138_single_refine(ds_path, t, arr, feature)
    x = sample_features(feature, arr).float()
    if _HEADENS_CACHE is None:
        _HEADENS_CACHE = (_headens_load(os.environ['V1284_HEAD'], feature.device),
                          [_headens_load(p, feature.device) for p in os.environ['V1284_HEAD2S'].split(os.pathsep)])
    (h1, m1, s1), heads2 = _HEADENS_CACHE
    d1 = bounded(h1, (x-m1)/s1)
    d2k = [bounded(h, (x-m)/s) for h, m, s in heads2]
    d2 = d2k[0]
    for d in d2k[1:]:
        d2 = d2 + d
    d2 = d2 / len(d2k)
    um = (0.5 * (d1 + d2)).cpu().numpy()
    shift = um / SPACING
    result = arr.astype(np.float32).copy()
    result[:,1:] += shift
    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)
    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:
        raise RuntimeError('invalid V1284 head-ensemble displacement')
    n1 = np.linalg.norm(d1.cpu().numpy(), axis=1)
    n2 = np.linalg.norm(d2.cpu().numpy(), axis=1)
    na = np.linalg.norm(um, axis=1)
    nd = np.linalg.norm((d1 - d2).cpu().numpy(), axis=1)
    nk = np.mean([np.linalg.norm(d.cpu().numpy(), axis=1) for d in d2k], axis=0)
    sp = np.mean([np.linalg.norm((d - d2).cpu().numpy(), axis=1) for d in d2k], axis=0)
    st = _HEADENS_STATS.setdefault(ds_path.stem, [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    st[0] += 1; st[1] += int(len(na)); st[2] += float(na.sum()); st[3] = max(st[3], float(na.max()))
    st[4] += float(n1.sum()); st[5] += float(n2.sum()); st[6] += float(nk.sum()); st[7] += float(sp.sum())
    st[8] += float(nd.sum())
    return result


_x138_refine = _headens_refine


def _headens_report():
    n_head2 = len(os.environ.get('V1284_HEAD2S', '').split(os.pathsep))
    for stem, (frames, n, sa, mx, s1, s2, sk, ssp, sd) in sorted(_HEADENS_STATS.items()):
        k = max(n, 1)
        print(f'V1284 HEADENS ds={stem} frames={frames} nodes={n} n_head2={n_head2} avg_mean_um={sa / k:.4f} '
              f'avg_max_um={mx:.4f} head1_mean_um={s1 / k:.4f} head2_mean_um={s2 / k:.4f} '
              f'head2_seed_mean_um={sk / k:.4f} head2_seed_spread_um={ssp / k:.4f} head_diff_mean_um={sd / k:.4f} '
              '(head outputs before border clip)', flush=True)


_v1284_atexit.register(_headens_report)
"""

# List section, inserted inside the block at the same place as the single-head2 section (after _HEAD2_AFTER;
# only when V1284_HEAD2_SHA256S is set).
_HEAD2S_TMPL = """# --- lane-r: V1284 head average — second head = mean of %(n)d seeds (build V1284_HEAD2_DATASET + V1284_HEAD2_SHA256S) ---
_v1_dirs2 = [Path('/kaggle/input/%(slug)s'), Path('/kaggle/input/datasets/%(owner)s/%(slug)s')]
_v1_dir2 = next((c for c in _v1_dirs2 if c.is_dir()), None)
if _v1_dir2 is None:
    _v1_found2 = sorted({p.parent for p in Path('/kaggle/input').rglob('%(slug)s/*.pt')})
    if len(_v1_found2) != 1:
        raise RuntimeError(('V1284 head2 list mount mismatch', [str(p) for p in _v1_found2]))
    _v1_dir2 = _v1_found2[0]
_v1_heads2 = sorted(_v1_dir2.glob('*.pt'))
_v1_shas2 = [_sha256_file(p) for p in _v1_heads2]
if _v1_shas2 != %(shas_json)s:
    raise RuntimeError(f'V1284 head2 list sha256 mismatch in {_v1_dir2}: files {[p.name for p in _v1_heads2]} sha256 {_v1_shas2}')
for _v1_p, _v1_s in zip(_v1_heads2, _v1_shas2):
    if _v1_p.resolve() == _v1_head.resolve():
        raise RuntimeError(f'V1284 head2 list contains head1: {_v1_p}')
    print('V1284 HEAD2:', _v1_p, 'sha256=' + _v1_s, 'bytes=', _v1_p.stat().st_size, flush=True)
os.environ['V1284_HEAD2S'] = os.pathsep.join(str(p) for p in _v1_heads2)
with (_ps.parent / 'v1284_coordinate_refinement.py').open('a') as _v1_f:
    _v1_f.write(%(appendix_json)s)
_v1_mod = (_ps.parent / 'v1284_coordinate_refinement.py').read_text()
compile(_v1_mod, 'v1284_coordinate_refinement.py', 'exec')
if _v1_mod.count('_x138_refine = _headens_refine') != 1 or '_x138_refine = _headavg_refine' in _v1_mod:
    raise RuntimeError('V1284 head-ensemble appendix did not persist')
print('V1284 HEADENS module appended: displacement = 0.5 * bounded(head1) + 0.5 * mean_k bounded(head2_k) um; n_head2 =',
      len(_v1_heads2), 'head1 =', os.environ['V1284_HEAD'], 'head2 =', os.environ['V1284_HEAD2S'], flush=True)
"""


def head2s_section(dataset: str, shas) -> str:
    shas = list(shas)
    if not shas:
        raise SystemExit("V1284 head2 list (V1284_HEAD2_SHA256S) is empty")
    for s in shas:
        _check_head(dataset, s)
    if len(set(shas)) != len(shas):
        raise SystemExit(f"V1284 head2 list contains the same sha256 twice: {shas}")
    owner, slug = dataset.split("/", 1)
    return _HEAD2S_TMPL % {"owner": owner, "slug": slug, "n": len(shas), "shas_json": json.dumps(shas),
                           "appendix_json": json.dumps(HEADENS_APPENDIX, ensure_ascii=False)}

def _check_head(dataset: str, sha: str) -> tuple[str, str]:
    """Validate a head's dataset ('owner/slug') and sha256 (64 lowercase hex) and return them.

    Build knobs V1284_HEAD_DATASET / V1284_HEAD_SHA256 (build_r946) switch to a private dataset holding a head file we
    trained in the same format (`v1284_head.pt`, keys state_dict/mean/scale). With the defaults the output is
    byte-identical."""
    if dataset.count("/") != 1 or not all(dataset.split("/")) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"V1284 head: dataset must be 'owner/slug' and sha256 must be 64 hex chars (got {dataset!r}, {sha!r})")
    return dataset, sha


def block(dataset: str = DATASET, sha: str = HEAD_SHA256, head2: tuple[str, str] | None = None,
          head2s: tuple[str, list[str]] | None = None) -> str:
    owner, slug = _check_head(dataset, sha)[0].split("/", 1)
    out = _BLOCK_TMPL % {
        "owner": owner, "slug": slug, "fname": HEAD_FILE, "sha": sha,
        "module_json": json.dumps(MODULE_SRC, ensure_ascii=False),
        "pairs_json": json.dumps([list(p) for p in SCRIPT_PAIRS], ensure_ascii=False),
    }
    if head2s is not None:   # second head = mean of the seed list (cannot be combined with a single-file head2)
        if head2 is not None:
            raise SystemExit("V1284 head2: the single file (V1284_HEAD2_SHA256) and the list (V1284_HEAD2_SHA256S) cannot be used together")
        if head2s[0] == dataset:
            raise SystemExit(f"V1284 head2 dataset is the same as head1: {dataset}")
        if sha in head2s[1]:
            raise SystemExit(f"V1284 head2 list contains the head1 sha256: {sha}")
        if out.count(_HEAD2_AFTER) != 1:
            raise SystemExit("V1284 head2 insertion anchor must occur exactly once")
        return out.replace(_HEAD2_AFTER, _HEAD2_AFTER + head2s_section(*head2s), 1)
    if head2 is None:
        return out
    if head2[0] == dataset:
        raise SystemExit(f"V1284 head2 dataset is the same as head1: {dataset}")
    if out.count(_HEAD2_AFTER) != 1:
        raise SystemExit("V1284 head2 insertion anchor must occur exactly once")
    return out.replace(_HEAD2_AFTER, _HEAD2_AFTER + head2_section(*head2), 1)


def apply_script(script: str) -> str:
    """Apply the 4 pairs to the predict script text with the runtime rule (each anchor exactly once).

    For tests and local smoke runs."""
    for name, old, new in SCRIPT_PAIRS:
        n = script.count(old)
        if n != 1:
            raise RuntimeError(f"V1284 script anchor '{name}' expected exactly 1 occurrence, found {n}")
        script = script.replace(old, new, 1)
    return script


def apply(text: str, dataset: str = DATASET, sha: str = HEAD_SHA256,
          head2: tuple[str, str] | None = None, head2s: tuple[str, list[str]] | None = None) -> tuple[str, list[str]]:
    """Insert the V1284 block at the end of the x138 port runtime block in the notebook cell text.

    The end anchor must occur exactly once (else SystemExit).
    head2=(dataset, sha): use the mean of the two heads' bounded displacements (default None → byte-identical to before).
    head2s=(dataset, [sha, ...]): the second head is the mean bounded displacement of every *.pt in that dataset
    (file-name order; the sha order must match)."""
    n = text.count(X138_RUNTIME_END)
    if n != 1:
        raise SystemExit(f"V1284: x138 port runtime block end anchor found {n} times — expected exactly 1 "
                         "(NOTEBOOK_PATCH must contain the x138_port_946 chain)")
    text = text.replace(X138_RUNTIME_END, X138_RUNTIME_END + block(dataset, sha, head2, head2s), 1)
    return text, [name for name, _, _ in SCRIPT_PAIRS]


# ============================================================================================ V1284=late (count-neutral)
# The graph pipeline stays byte-identical to the fastio-ref reference build (int16 coordinates, native feature lookup,
# same ILP/relink/division). The predict script computes x138 refine **on the side** for every first-seen detection
# (arr is unchanged) and, at the end of predict_video, writes
#   /tmp/biohub_v1284late/<stem>.npz  coords=(N,4) int16 at full resolution (= exactly the coordinates fed to the graph),
#                                     refined=(N,3) float32 at full resolution.
# Row i is detection i, and in this pipeline geff node_id == detection row index (relink_prob's probability table makes
# the same assumption and uses (src, tgt) row indices as node ids). Only when submission.csv is written, a node of
# detection origin — node_id < N and (t, z, y, x) exactly equal to coords[node_id] — gets its z/y/x from the refined
# coordinates (see _v1284_late_coords) and is rounded with int(round()) as before. All other nodes (gap midpoints,
# moved nodes) keep their original coordinates. The node and edge sets and their order do not change.
LATE_DIR = "/tmp/biohub_v1284late"
LATE_SCRIPT_PAIRS = [
    ("v1284late_import", _IMPORT_OLD,
     "import tracksdata as td\n"
     "from v1284_coordinate_refinement import refine as _v1284_refine\n"
     "_V1L_REF: list = []\n"),
    ("v1284late_capture", _REFINE_OLD,
     "                _V1L_REF.append(_v1284_refine(ds_path, t, arr, unet_out[:, f_idx]))\n" + _REFINE_OLD),
    ("v1284late_save", _INT16_OLD,
     _INT16_OLD +
     "    _v1l_ref = np.concatenate(_V1L_REF).astype(np.float32) if _V1L_REF else np.empty((0, 4), np.float32)\n"
     "    _V1L_REF.clear()\n"
     "    if len(_v1l_ref) != len(coords):\n"
     "        raise RuntimeError(f'V1284_LATE: {len(_v1l_ref)} refined rows vs {len(coords)} detections for {ds_path.stem}')\n"
     "    _v1l_ref[:, 1:] *= ds_arr\n"
     f"    _v1l_dir = Path('{LATE_DIR}')\n"
     "    _v1l_dir.mkdir(parents = True, exist_ok = True)\n"
     "    np.savez(_v1l_dir / f'{ds_path.stem}.npz', coords = coords, refined = _v1l_ref[:, 1:].astype(np.float32))\n"
     "    print(f'V1284_LATE CAPTURE dataset={ds_path.stem} detections={len(coords)}', flush = True)\n"),
]

_CSV_NODE_OLD = (
    "        for node_id in sorted(nodes_by_id):\n"
    "            node = nodes_by_id[node_id]\n"
    "            writer.writerow({'id': row_id, 'dataset': dataset, 'row_type': 'node', 'node_id': int(node['node_id']), "
    "'t': int(node['t']), 'z': max(0, int(round(float(node['z'])))), 'y': max(0, int(round(float(node['y'])))), "
    "'x': max(0, int(round(float(node['x'])))), 'source_id': -1, 'target_id': -1})\n")
_CSV_NODE_NEW = (
    "        _v1l_zyx = _v1284_late_coords(dataset, nodes_by_id, edges, _v1l_raw)\n"
    "        for node_id in sorted(nodes_by_id):\n"
    "            node = nodes_by_id[node_id]\n"
    "            _v1l_z, _v1l_y, _v1l_x = _v1l_zyx.get(node_id, (node['z'], node['y'], node['x']))\n"
    "            writer.writerow({'id': row_id, 'dataset': dataset, 'row_type': 'node', 'node_id': int(node['node_id']), "
    "'t': int(node['t']), 'z': max(0, int(round(float(_v1l_z)))), 'y': max(0, int(round(float(_v1l_y)))), "
    "'x': max(0, int(round(float(_v1l_x)))), 'source_id': -1, 'target_id': -1})\n")
# Snapshot of node coordinates right after reading geff (before post-processing): filter_output_graph's linefit
# modifies the node dicts in place.
_RAW_OLD = "        raw_node_count = len(nodes_by_id)\n"
_RAW_NEW = ("        _v1l_raw = {int(_nid): (int(_n['t']), float(_n['z']), float(_n['y']), float(_n['x'])) "
            "for _nid, _n in nodes_by_id.items()}\n" + _RAW_OLD)
LATE_NOTEBOOK_PAIRS = [("v1284late_raw_snapshot", _RAW_OLD, _RAW_NEW), ("v1284late_csv", _CSV_NODE_OLD, _CSV_NODE_NEW)]

_LATE_FUNC = '''

_V1L_TOTALS = {'csv_nodes': 0, 'refined': 0, 'int_changed': 0, 'not_detection': 0, 'sum_um': 0.0, 'max_um': 0.0,
               'sum_raw_um': 0.0, 'max_raw_um': 0.0}


def _v1284_late_coords(dataset, nodes_by_id, edges, raw):
    \"\"\"{node_id: (z, y, x)} for detection-origin CSV nodes; every other node keeps its coordinates.

    detection-origin = node_id is detection row i of the predict capture (this pipeline writes node_id == row) and the
    node's geff coordinates before post-processing equal coords[i] exactly, same t. The refinement delta
    d_i = refined_i - coords_i goes through the pipeline's own linefit smoothing on the final graph (linear in positions,
    topology-gated only), so the CSV coordinate is what linefit would have produced from the refined centre:
    final_i + L(d)_i. Nodes that are not detection-origin get no shift.
    \"\"\"
    import collections as _v1l_col
    path = Path('%(late_dir)s') / f'{dataset}.npz'
    if not path.is_file():
        raise RuntimeError(f'V1284_LATE: refined coordinates were not captured: {path} missing')
    z = np.load(path)
    C = z['coords'].astype(np.float64)
    R = z['refined'].astype(np.float64)
    if C.ndim != 2 or C.shape[1] != 4 or R.shape != (len(C), 3) or not np.isfinite(R).all():
        raise RuntimeError(f'V1284_LATE: bad capture {path}: coords {C.shape} refined {R.shape}')
    sc = np.array([1.625, 0.40625, 0.40625])
    det, not_det = {}, 0
    for nid, node in nodes_by_id.items():
        i = int(nid)
        r = raw.get(i)
        if r is None or not 0 <= i < len(C) or int(node['t']) != r[0]:
            not_det += 1
            continue
        if (r[0], r[1], r[2], r[3]) != (int(C[i, 0]), C[i, 1], C[i, 2], C[i, 3]):
            not_det += 1
            continue
        det[i] = R[i] - C[i, 1:]
    if not det or len(det) < 0.5 * len(nodes_by_id):
        raise RuntimeError(f'V1284_LATE: only {len(det)} of {len(nodes_by_id)} nodes matched a detection row in {dataset} '
                           '(node_id == detection row assumption broken?)')
    raw_um = np.array([float(np.linalg.norm(d * sc)) for d in det.values()])
    if raw_um.max() > 2.00001:
        raise RuntimeError(f'V1284_LATE: raw displacement {raw_um.max():.4f} um > 2 um in {dataset}')
    delta_nodes = {nid: {'node_id': nid, 't': int(node['t']),
                         'z': float(det[nid][0]) if nid in det else 0.0,
                         'y': float(det[nid][1]) if nid in det else 0.0,
                         'x': float(det[nid][2]) if nid in det else 0.0} for nid, node in nodes_by_id.items()}
    smoothed = linefit_smooth_output_graph(delta_nodes, edges, _v1l_col.defaultdict(int))
    out, changed, fin_um = {}, 0, []
    for nid in det:
        n = nodes_by_id[nid]
        s = smoothed[nid]
        d = np.array([float(s['z']), float(s['y']), float(s['x'])])
        old = (float(n['z']), float(n['y']), float(n['x']))
        out[nid] = (old[0] + d[0], old[1] + d[1], old[2] + d[2])
        fin_um.append(float(np.linalg.norm(d * sc)))
        if tuple(max(0, int(round(v))) for v in out[nid]) != tuple(max(0, int(round(v))) for v in old):
            changed += 1
    fin_um = np.array(fin_um)
    if not np.isfinite(fin_um).all() or fin_um.max() > 4.0:
        raise RuntimeError(f'V1284_LATE: smoothed displacement {fin_um.max():.4f} um out of range in {dataset}')
    T = _V1L_TOTALS
    T['csv_nodes'] += len(nodes_by_id); T['refined'] += len(out); T['int_changed'] += changed; T['not_detection'] += not_det
    T['sum_um'] += float(fin_um.sum()); T['max_um'] = max(T['max_um'], float(fin_um.max()))
    T['sum_raw_um'] += float(raw_um.sum()); T['max_raw_um'] = max(T['max_raw_um'], float(raw_um.max()))
    print(f'V1284_LATE CSV ds={dataset} csv_nodes={len(nodes_by_id)} refined={len(out)} not_detection={not_det} '
          f'int_changed={changed} ({changed / len(nodes_by_id):.4f}) csv_shift_mean_um={fin_um.mean():.4f} '
          f'csv_shift_max_um={fin_um.max():.4f} raw_mean_um={raw_um.mean():.4f} raw_max_um={raw_um.max():.4f} | total '
          f"csv_nodes={T['csv_nodes']} refined={T['refined']} not_detection={T['not_detection']} int_changed={T['int_changed']} "
          f"({T['int_changed'] / max(T['csv_nodes'], 1):.4f}) csv_shift_mean_um={T['sum_um'] / max(T['refined'], 1):.4f} "
          f"csv_shift_max_um={T['max_um']:.4f} raw_mean_um={T['sum_raw_um'] / max(T['refined'], 1):.4f} "
          f"raw_max_um={T['max_raw_um']:.4f}", flush = True)
    return out
'''

_LATE_BLOCK_TMPL = '''
# === lane-r: V1284 LATE (count-neutral) — kaggle/v1284_946.py (build V1284=late, runtime patch on _ps) ===
# Graph untouched (int16, native lookup); refined coordinates go only to a side file → submission.csv changes only detection-origin node coordinates.
_v1_candidates = [Path('/kaggle/input/%(slug)s/%(fname)s'), Path('/kaggle/input/datasets/%(owner)s/%(slug)s/%(fname)s')]
_v1_head = next((c for c in _v1_candidates if c.is_file()), None)
if _v1_head is None:
    _v1_found = sorted(Path('/kaggle/input').rglob('%(slug)s/%(fname)s'))
    if len(_v1_found) != 1:
        raise RuntimeError(('V1284 head mount mismatch', [str(p) for p in _v1_found]))
    _v1_head = _v1_found[0]
_v1_sha = _sha256_file(_v1_head)
if _v1_sha != '%(sha)s':
    raise RuntimeError(f'V1284 head sha256 mismatch: {_v1_sha} at {_v1_head}')
print('V1284 HEAD:', _v1_head, 'sha256=' + _v1_sha, 'bytes=', _v1_head.stat().st_size, flush=True)
(_ps.parent / 'v1284_coordinate_refinement.py').write_text(%(module_json)s)
os.environ['V1284_MODE'] = 'candidate'
os.environ['V1284_HEAD'] = str(_v1_head)
_v1_src = _ps.read_text()
_v1_lowdet_on = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0) > 0 and bool(os.environ.get('BIOHUB_CACHE_DIR', '').strip())
if _v1_lowdet_on and '_LOWDET.append(' not in _v1_src:
    raise RuntimeError('V1284: low-detection dump is on but not yet in the script — V1284 must run after it')
for _v1_name, _v1_old, _v1_new in %(pairs_json)s:
    _v1_n = _v1_src.count(_v1_old)
    if _v1_n != 1:
        raise RuntimeError(f"V1284 script anchor '{_v1_name}' expected exactly 1 occurrence, found {_v1_n}")
    _v1_src = _v1_src.replace(_v1_old, _v1_new, 1)
compile(_v1_src, str(_ps), 'exec')
_ps.write_text(_v1_src)
_v1_src = _ps.read_text()
for _v1_must in ('from v1284_coordinate_refinement import refine as _v1284_refine\\n',
                 '_V1L_REF.append(_v1284_refine(ds_path, t, arr, unet_out[:, f_idx]))',
                 'coords = coords.astype(np.int16)', "print(f'V1284_LATE CAPTURE"):
    if _v1_src.count(_v1_must) != 1:
        raise RuntimeError('V1284_LATE patch did not persist marker: ' + _v1_must)
for _v1_never in ('_v1284_index', 'arr = _v1284_refine('):
    if _v1_never in _v1_src:
        raise RuntimeError('V1284_LATE: graph-changing V1284 code present: ' + _v1_never)
if _v1_lowdet_on and _v1_src.count('_LOWDET.append(') != 1:
    raise RuntimeError('V1284: low-detection dump lost after V1284 patch')
print('V1284_LATE head patched AFTER the readmit dump patch; mode =', os.environ['V1284_MODE'],
      '(3 anchors, graph untouched) lowdet_in_script=', '_LOWDET.append(' in _v1_src, flush=True)
'''


def late_block(dataset: str = DATASET, sha: str = HEAD_SHA256) -> str:
    owner, slug = _check_head(dataset, sha)[0].split("/", 1)
    return _LATE_BLOCK_TMPL % {
        "owner": owner, "slug": slug, "fname": HEAD_FILE, "sha": sha,
        "module_json": json.dumps(MODULE_SRC, ensure_ascii=False),
        "pairs_json": json.dumps([list(p) for p in LATE_SCRIPT_PAIRS], ensure_ascii=False),
    }


def late_func() -> str:
    return _LATE_FUNC % {"late_dir": LATE_DIR}


def apply_script_late(script: str) -> str:
    for name, old, new in LATE_SCRIPT_PAIRS:
        n = script.count(old)
        if n != 1:
            raise RuntimeError(f"V1284 script anchor '{name}' expected exactly 1 occurrence, found {n}")
        script = script.replace(old, new, 1)
    return script


def apply_late(text: str, dataset: str = DATASET, sha: str = HEAD_SHA256) -> tuple[str, list[str]]:
    """V1284=late: runtime block (at the end of the x138 port runtime block) + 2 notebook anchors (raw coordinate
    snapshot, CSV node writer) + lookup function."""
    n = text.count(X138_RUNTIME_END)
    if n != 1:
        raise SystemExit(f"V1284: x138 port runtime block end anchor found {n} times — expected exactly 1 "
                         "(NOTEBOOK_PATCH must contain the x138_port_946 chain)")
    text = text.replace(X138_RUNTIME_END, X138_RUNTIME_END + late_block(dataset, sha) + late_func(), 1)
    for name, old, new in LATE_NOTEBOOK_PAIRS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"V1284 late notebook anchor '{name}' expected exactly 1 occurrence, found {n}")
        text = text.replace(old, new, 1)
    return text, [name for name, _, _ in LATE_SCRIPT_PAIRS + LATE_NOTEBOOK_PAIRS]
