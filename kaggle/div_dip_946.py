"""div_dip_946.py — filter or rank division candidates by image evidence: the intensity dip between the two nuclei
(NOTEBOOK_PATCH; 4 anchors for the dip, plus 3 for the opt-in t+1 two-blob veto below).

Labels: S57 and S86 are earlier submission recipes. Background on the pipeline: docs/method.md.

dip = (minimum over the middle of the intensity profile) / (mean of its two ends), where the profile is measured in
frame t+1 along the segment existing daughter -> candidate daughter (10th-percentile background subtracted).
True sisters are two separate nuclear blobs, so the middle goes dark (low dip); a duplicate detection of the same
nucleus does not.
Offline measurement on 151 GT divisions: AUC 0.820 [0.779, 0.858]; the current ranking key (distance) scores 0.379.
Only raw pixels are used, so there is no leakage from a trained model memorizing its training data. The numerical
definition is the same as dip_between in our offline division-ranking probe.

Knobs (default off = identical behavior to S57)
  BIOHUB_DIV_DIP_MODE=0|1|2|3|4      0 off / 1 dip as ranking key only / 2 dip as gate only / 3 both
                                      4 combined score c = z(dip) + z(sym) - z(div) for both the gate (c <= COMBO_MAX)
                                        and the ranking; c replaces the divergence and symmetry gates
                                        (151 GT divisions: AUC 0.849, dip alone 0.817)
  BIOHUB_DIV_DIP_MAX=<float>          gate: only candidates with dip <= this value pass (default 0.30)
  BIOHUB_DIV_DIP_REPLACE_GATES=0|1    in gate modes, 1 replaces the divergence and symmetry gates with dip,
                                      0 adds dip to the existing gates (default 1)
  BIOHUB_DIV_DIP_RANK_GEO_W=<float>   ranking key = dip + w x (parent_dist + 0.15 x sister_dist) (default 0 = dip only)
  BIOHUB_DIV_COMBO_MAX=<float>        mode-4 gate threshold (default 0.0)
  BIOHUB_DIV_Z_{DIP,SYM,DIV}_{MU,SD}  mode-4 normalization constants (defaults: the values measured offline on the
                                      151 GT divisions, set at build time)
  BIOHUB_DIV_DIV_MISSING=<float>      value used when divergence cannot be measured because a grandchild is missing
                                      (default -9.0 = strong penalty)
Markers: DIV_DIP_CONFIG (once, when the helper is defined) / run_stats: safe_division_dip_checked, _rejected, _missing

t+1 two-blob veto (2026-09-17, opt-in; the measurement is defined as pair_feats in our offline val40 veto study)
  After the combo gate passes and before the DeepCenter rejection: around the two daughters in frame t+1 (existing
  daughter, candidate daughter), smooth with σ 0.8 µm, subtract the 10th-percentile background of the context box,
  find the peak within ±1.5 µm of each node, and compute
    peak_ratio = lower peak / higher peak,
    dip_low    = valley along the segment between the two peaks (isotropic 0.25 µm sampling) / lower peak
  BIOHUB_DIV_T1_PEAK_RATIO_MIN=<float>   reject the candidate if peak_ratio is below this value (empty = off)
  BIOHUB_DIV_T1_DIP_LOW_MAX=<float>      reject the candidate if dip_low is above this value (empty = off)
  With both knobs empty (the default) the code path and the output are identical to S86. A candidate whose values
  cannot be measured (missing frame, etc.) is never rejected.
  Markers: DIV_T1_CONFIG (once when the helper is defined, plus 'rejected=N of M' per video)
  run_stats: safe_division_t1_checked, _rejected, _missing
  divlog (only when the val40 instrumentation patch is present): stage = 't1_twoblob_reject'
"""

_HELPER = '''
# ==== division dip evidence (kaggle/div_dip_946.py) =============================
_DIP_MODE = int(os.environ.get('BIOHUB_DIV_DIP_MODE', '0'))
_DIP_MAX = float(os.environ.get('BIOHUB_DIV_DIP_MAX', '0.30'))
_DIP_REPLACE = os.environ.get('BIOHUB_DIV_DIP_REPLACE_GATES', '1') != '0'
_DIP_RANK_GEO_W = float(os.environ.get('BIOHUB_DIV_DIP_RANK_GEO_W', '0.0'))
_DIP_FRAME_KEEP = 16
_COMBO_MAX = float(os.environ.get('BIOHUB_DIV_COMBO_MAX', '0.0'))
_Z = {'DIP': (float(os.environ.get('BIOHUB_DIV_Z_DIP_MU', '0.0')), float(os.environ.get('BIOHUB_DIV_Z_DIP_SD', '1.0'))),
      'SYM': (float(os.environ.get('BIOHUB_DIV_Z_SYM_MU', '0.0')), float(os.environ.get('BIOHUB_DIV_Z_SYM_SD', '1.0'))),
      'DIV': (float(os.environ.get('BIOHUB_DIV_Z_DIV_MU', '0.0')), float(os.environ.get('BIOHUB_DIV_Z_DIV_SD', '1.0')))}
_DIV_MISSING = float(os.environ.get('BIOHUB_DIV_DIV_MISSING', '-9.0'))
print(f'DIV_DIP_CONFIG mode={_DIP_MODE} max={_DIP_MAX} replace_gates={int(_DIP_REPLACE)} rank_geo_w={_DIP_RANK_GEO_W} '
      f'combo_max={_COMBO_MAX} z={_Z} div_missing={_DIV_MISSING}', flush = True)


def _div_combo_value(dip, child_dist, parent_dist, sister_dist, existing_child_id, candidate_id, out_by_source, nodes_by_id, t):
    """c = z(dip) + z(sym) - z(div). Lower = more likely a true division. None if dip is missing."""
    if dip is None:
        return None
    sym = abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6)
    div = _DIV_MISSING
    c1_succ = out_by_source.get(existing_child_id, [])
    q_succ = out_by_source.get(candidate_id, [])

    if len(c1_succ) == 1 and len(q_succ) == 1:
        g1 = nodes_by_id.get(int(c1_succ[0]['target_id']))
        g2 = nodes_by_id.get(int(q_succ[0]['target_id']))

        if g1 is not None and g2 is not None and int(g1['t']) == t + 2 and int(g2['t']) == t + 2:
            div = edge_distance_um(g1, g2) - sister_dist
    return ((dip - _Z['DIP'][0]) / _Z['DIP'][1] + (sym - _Z['SYM'][0]) / _Z['SYM'][1]
            - (div - _Z['DIV'][0]) / _Z['DIV'][1])


def _div_dip(dataset, t1, node_a, node_b, frame_cache, stats):
    """Intensity dip between two nodes in frame t1. None if it cannot be computed."""
    stats['safe_division_dip_checked'] = stats.get('safe_division_dip_checked', 0) + 1

    try:
        vol = read_test_frame(dataset, int(t1), frame_cache)
    except Exception:
        stats['safe_division_dip_missing'] = stats.get('safe_division_dip_missing', 0) + 1
        return None

    while len(frame_cache) > _DIP_FRAME_KEEP:
        frame_cache.pop(next(iter(frame_cache)))
    a = np.array([float(node_a['z']), float(node_a['y']), float(node_a['x'])])
    b = np.array([float(node_b['z']), float(node_b['y']), float(node_b['x'])])
    Z, Y, X = vol.shape
    n = 11
    prof = []

    for f in np.linspace(0.0, 1.0, n):
        p = a + (b - a) * f
        z, y, x = (int(round(v)) for v in p)
        box = vol[max(0, z - 1):min(Z, z + 2), max(0, y - 2):min(Y, y + 3), max(0, x - 2):min(X, x + 3)]
        prof.append(float(box.mean()) if box.size else float('nan'))
    prof = np.array(prof)
    z, y, x = (int(round(v)) for v in (a + b) / 2.0)
    ctx = vol[max(0, z - 4):min(Z, z + 5), max(0, y - 20):min(Y, y + 21), max(0, x - 20):min(X, x + 21)]
    bg = float(np.percentile(ctx, 10)) if ctx.size else 0.0
    prof = np.clip(prof - bg, 0, None)
    ends = (prof[0] + prof[-1]) / 2.0

    if not np.isfinite(ends) or ends <= 0:
        stats['safe_division_dip_missing'] = stats.get('safe_division_dip_missing', 0) + 1
        return None
    return float(prof[3:n - 3].min() / ends)
# ==== end division dip evidence ========================================================

'''

_DEF_OLD = "def add_safe_divisions_postlink("
_DEF_NEW = _HELPER + _DEF_OLD

_GATE_OLD = "                if SAFE_DIV_REQUIRE_DIVERGENCE:\n"
_GATE_NEW = ("                _dip_v = None\n"
             "                _combo_v = None\n"
             "\n"
             "                if _DIP_MODE in (2, 3, 4):\n"
             "                    _dip_v = _div_dip(dataset, int(candidate['t']), existing_child, candidate, frame_cache, stats)\n"
             "\n"
             "                    if _DIP_MODE == 4:\n"
             "                        _combo_v = _div_combo_value(_dip_v, child_dist, parent_dist, sister_dist, existing_child_id, candidate_id, out_by_source, nodes_by_id, t)\n"
             "                        _dip_reject = _combo_v is None or _combo_v > _COMBO_MAX\n"
             "                    else:\n"
             "                        _dip_reject = _dip_v is None or _dip_v > _DIP_MAX\n"
             "\n"
             "                    if _dip_reject:\n"
             "                        stats['safe_division_dip_rejected'] = stats.get('safe_division_dip_rejected', 0) + 1\n"
             "                        continue\n"
             "\n"
             "                if SAFE_DIV_REQUIRE_DIVERGENCE and not (_DIP_MODE == 4 or (_DIP_MODE in (2, 3) and _DIP_REPLACE)):\n")

_SYM_OLD = "                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:\n"
_SYM_NEW = "                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0 and not (_DIP_MODE == 4 or (_DIP_MODE in (2, 3) and _DIP_REPLACE)):\n"

_RANK_OLD = "                score = parent_dist + 0.15 * sister_dist\n"
_RANK_NEW = ("                score = parent_dist + 0.15 * sister_dist\n"
             "\n"
             "                if _DIP_MODE == 4:\n"
             "                    score = _combo_v if _combo_v is not None else 99.0\n"
             "\n"
             "                elif _DIP_MODE in (1, 3):\n"
             "                    if _dip_v is None:\n"
             "                        _dip_v = _div_dip(dataset, int(candidate['t']), existing_child, candidate, frame_cache, stats)\n"
             "                    score = (_dip_v if _dip_v is not None else 9.0) + _DIP_RANK_GEO_W * score\n")

# ---------------------------------------------------------------------------------------------- t+1 two-blob veto (opt-in)
# pure measurement (numpy + scipy.ndimage only; VOXEL_SCALE_UM passed in) — the SAME text runs in the notebook and, via
# load_t1_features(), offline, so the kernel-side values can be checked against the offline val40 veto measurement.
_T1_CORE = '''
def _div_t1_features(vol, a_zyx, b_zyx, scale, sigma_um = 0.8, pad_xy_um = 8.0, pad_z_um = 5.0, peak_r_um = 1.5, step_um = 0.25):
    """t+1 two-blob evidence between two voxel positions: peak_ratio (low/high peak, peaks searched within peak_r_um of each
    node) and dip_low (valley of the isotropic line profile between the two peaks / the lower peak). None if not measurable."""
    from scipy import ndimage
    sc = np.asarray(scale, dtype = np.float64)
    pa = np.asarray(a_zyx, dtype = np.float64)
    pb = np.asarray(b_zyx, dtype = np.float64)
    P = np.stack([pa, pb])
    pad = np.array([pad_z_um, pad_xy_um, pad_xy_um]) / sc
    lo = np.maximum(np.floor(P.min(0) - pad).astype(int), 0)
    hi = np.minimum(np.ceil(P.max(0) + pad).astype(int) + 1, np.array(vol.shape))
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)

    if crop.size == 0:
        return None
    bg = float(np.percentile(crop, 10))
    sm = np.clip(ndimage.gaussian_filter(crop, sigma = tuple(sigma_um / sc)) - bg, 0, None)
    qa, qb = pa - lo, pb - lo
    idx = np.indices(sm.shape)

    def _peak(p):
        d2 = sum(((idx[k] - p[k]) * sc[k]) ** 2 for k in range(3))
        m = d2 <= peak_r_um * peak_r_um

        if not m.any():
            return None
        i = np.unravel_index(np.argmax(np.where(m, sm, -np.inf)), sm.shape)
        return float(sm[i])
    peak_a, peak_b = _peak(qa), _peak(qb)

    if peak_a is None or peak_b is None or min(peak_a, peak_b) <= 0:
        return None
    low, high = min(peak_a, peak_b), max(peak_a, peak_b)
    L = float(np.linalg.norm((qb - qa) * sc))
    n = max(int(L / step_um) + 1, 7)
    f = np.linspace(0, 1, n)
    pts = qa[None, :] + (qb - qa)[None, :] * f[:, None]
    prof = ndimage.map_coordinates(sm, pts.T, order = 1, mode = 'nearest')
    s = np.linspace(0, L, n)
    wa, wb = s <= peak_r_um, s >= L - peak_r_um
    ka, kb = int(np.argmax(np.where(wa, prof, -np.inf))), int(np.argmax(np.where(wb, prof, -np.inf)))
    pk_a, pk_b = float(prof[ka]), float(prof[kb])

    if kb > ka and min(pk_a, pk_b) > 0:
        dip_low = float(prof[ka:kb + 1].min()) / min(pk_a, pk_b)
    else:
        dip_low = 1.0
    return {'peak_ratio': low / high, 'dip_low': dip_low, 'peak_low': low, 'bg': bg}
'''

_T1_HELPER = '''
# ==== t+1 two-blob evidence (kaggle/div_dip_946.py, opt-in: BIOHUB_DIV_T1_PEAK_RATIO_MIN / BIOHUB_DIV_T1_DIP_LOW_MAX) ====
_T1_RATIO_MIN = os.environ.get('BIOHUB_DIV_T1_PEAK_RATIO_MIN', '').strip()
_T1_DIP_LOW_MAX = os.environ.get('BIOHUB_DIV_T1_DIP_LOW_MAX', '').strip()
_T1_RATIO_MIN = float(_T1_RATIO_MIN) if _T1_RATIO_MIN else None
_T1_DIP_LOW_MAX = float(_T1_DIP_LOW_MAX) if _T1_DIP_LOW_MAX else None
_T1_ENABLED = _T1_RATIO_MIN is not None or _T1_DIP_LOW_MAX is not None

if _T1_ENABLED:
    print(f'DIV_T1_CONFIG ratio_min={_T1_RATIO_MIN} dip_low_max={_T1_DIP_LOW_MAX}', flush = True)
''' + _T1_CORE + '''

def _div_t1_twoblob(dataset, t1, node_a, node_b, frame_cache, stats):
    """t+1 two-blob features between the existing child and the candidate; None if not measurable."""
    stats['safe_division_t1_checked'] = stats.get('safe_division_t1_checked', 0) + 1

    try:
        vol = read_test_frame(dataset, int(t1), frame_cache)
    except Exception:
        stats['safe_division_t1_missing'] = stats.get('safe_division_t1_missing', 0) + 1
        return None

    while len(frame_cache) > _DIP_FRAME_KEEP:
        frame_cache.pop(next(iter(frame_cache)))
    a = (float(node_a['z']), float(node_a['y']), float(node_a['x']))
    b = (float(node_b['z']), float(node_b['y']), float(node_b['x']))
    feats = _div_t1_features(vol, a, b, VOXEL_SCALE_UM)

    if feats is None:
        stats['safe_division_t1_missing'] = stats.get('safe_division_t1_missing', 0) + 1
    return feats


def _div_t1_reject(feats):
    """True when an enabled knob is violated. A missing measurement never vetoes."""
    if feats is None:
        return False

    if _T1_RATIO_MIN is not None and feats['peak_ratio'] < _T1_RATIO_MIN:
        return True

    if _T1_DIP_LOW_MAX is not None and feats['dip_low'] > _T1_DIP_LOW_MAX:
        return True
    return False
# ==== end t+1 two-blob evidence ==========================================================

'''

# gate: right after the combo/dip rejection (`continue`), before the divergence block, geometric_candidates and the DeepCenter veto
_T1_GATE_OLD = ("                        continue\n"
                "\n"
                "                if SAFE_DIV_REQUIRE_DIVERGENCE and not (_DIP_MODE == 4 or (_DIP_MODE in (2, 3) and _DIP_REPLACE)):\n")
_T1_GATE_NEW = ("                        continue\n"
                "\n"
                "                if _T1_ENABLED:\n"
                "                    _t1_feats = _div_t1_twoblob(dataset, int(candidate['t']), existing_child, candidate, frame_cache, stats)\n"
                "\n"
                "                    if _div_t1_reject(_t1_feats):\n"
                "                        stats['safe_division_t1_rejected'] = stats.get('safe_division_t1_rejected', 0) + 1\n"
                "                        _t1_bykey = globals().get('_V40_BYKEY')\n"
                "\n"
                "                        if _t1_bykey is not None:\n"
                "                            _t1_bykey.get((int(source_id), int(candidate_id)), {})['stage'] = 't1_twoblob_reject'\n"
                "                        continue\n"
                "\n"
                "                if SAFE_DIV_REQUIRE_DIVERGENCE and not (_DIP_MODE == 4 or (_DIP_MODE in (2, 3) and _DIP_REPLACE)):\n")

# per-video log line (filter_output_graph, right after the safe-division stage)
_T1_LOG_OLD = "    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset = dataset, deepcenter_bundle = deepcenter_bundle, frame_cache = repair_frame_cache, deepcenter_cache = deepcenter_heatmap_cache)\n"
_T1_LOG_NEW = (_T1_LOG_OLD +
               "\n"
               "    if _T1_ENABLED:\n"
               "        print(f\"DIV_T1_CONFIG ratio_min={_T1_RATIO_MIN} dip_low_max={_T1_DIP_LOW_MAX} rejected={stats.get('safe_division_t1_rejected', 0)} of {stats.get('safe_division_t1_checked', 0)} ds={dataset}\", flush = True)\n")


def load_t1_features():
    """offline: the notebook's _div_t1_features (same text) → callable(vol, a_zyx, b_zyx, scale)."""
    import numpy as np
    ns = {"np": np}
    exec(_T1_CORE, ns)
    return ns["_div_t1_features"]


PAIRS = [("dip_helper", _DEF_OLD, _DEF_NEW),
         ("dip_gate", _GATE_OLD, _GATE_NEW),
         ("dip_sym", _SYM_OLD, _SYM_NEW),
         ("dip_rank", _RANK_OLD, _RANK_NEW),
         ("t1_helper", _DEF_OLD, _T1_HELPER + _DEF_OLD),      # applied after dip_helper → lands between the dip helper and the def
         ("t1_gate", _T1_GATE_OLD, _T1_GATE_NEW),
         ("t1_log", _T1_LOG_OLD, _T1_LOG_NEW)]
