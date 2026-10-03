"""node_select_946.py — remove final-graph nodes whose detector logits are weak and whose neighborhood is empty
(NOTEBOOK_PATCH, default off).

Labels: S57 and S86 are earlier submission recipes; N01-N03 are the candidates that introduced the knobs below.
Background on the pipeline: docs/method.md.

Evidence (offline replay on our 40-video validation split, val40, 2026-09-16):
  On the val40 replica of the S57 post-processing (C0), the rule `logit_min<6 & logit_max<8 & density10<=1` removes
  13.5 % of the nodes, and P(matched-to-GT) among them is 0.05 % (break-even 0.10 %). Re-scored with the vendored
  competition metric: +0.0056 (44b6 +0.0051 / 6bba +0.0057; a rule chosen on one embryo transfers to the other).
  Almost all of the gain comes from the N_pred correction factor of the adjusted edge Jaccard (1.0055 -> 1.0133);
  edge Jaccard changes by -0.0017 (TP edges -78, FP edges -43).

Feature definitions (exactly as in the offline probe)
  logit_pk   = max over the 3x3x3 window (clipped at the map border) around the node's grid position
               (t, rint(z/1), rint(y/4), rint(x/4)) in the logit map (T, Z, Y/4, X/4) of the PUBLIC primary detector
               (pilkwang)
  logit_seed = the same, on the map of the secondary detector (seed314159)
  logit_min  = min(logit_pk, logit_seed)   logit_max = max(logit_pk, logit_seed)
               ** min/max across the two detectors (not min/max within the 3x3x3 window) **
  density10  = number of other nodes within 10 µm in the same frame (µm scale z 1.625 / xy 0.40625,
               cKDTree.query_ball_point - 1)
  node set   = the final graph after the short-track filter (synthetic nodes included; nodes left isolated by the
               removal are kept)

This port (source=primary+secondary)
  In the kernel, detection runs in a subprocess (the base notebook's scripts/predict_unet_transformer.py) and the
  notebook only receives the output geff. Therefore
  (1) a runtime script patch saves primary_det and secondary_det (the raw secondary map, before its alignment to the
      primary) as float16 (Z, Y', X') arrays to /tmp/biohub_nodesel/<dataset>/, inside the retention-guard block (the
      first window in which each frame appears = the window used for detection), and
  (2) the notebook's filter_output_graph, after linefit and right before FINAL, computes the features above with the
      same formulas as the probe and removes the matching nodes.
  Differences from the probe: the kernel's primary/secondary maps are 8-view D4 TTA averages (the probe cache holds
  4-view flip averages), and it is unknown whether a frame's window position (for t>0, the second frame of the
  window) matches the cache. The threshold scale is left unchanged.

Knobs (default off = identical behavior to S86)
  BIOHUB_NODE_SELECT=0|1              1 turns the rule on
  BIOHUB_NODE_SELECT_LOGIT_MIN=6.0    min(logit_pk, logit_seed) < this value
  BIOHUB_NODE_SELECT_LOGIT_MAX=8.0    max(logit_pk, logit_seed) < this value
  BIOHUB_NODE_SELECT_DENSITY_UM=10.0  density radius (µm)
  BIOHUB_NODE_SELECT_DENSITY_MAX=1    number of other nodes in the same frame <= this value
  BIOHUB_NODE_SELECT_CAP=0            (N02, 2026-09-16) if >0, rank each video's rule candidates by logit_min ascending
                                      (ties: logit_max ascending, then node id) and remove at most
                                      floor(cap × N_video_final_nodes) of them. Same ranking and rounding down as the
                                      per-video cap variant V2 (cap_per_video) of the offline adaptive probe.
                                      0 = off = identical to N01 (val40, V2 c=0.20: n_drop 103178, max per-video drop 20.0 %)
  BIOHUB_DIV_FORK_PRUNE=0|1           (N03, 2026-09-16) if 1, also apply the fork-prune rule (D3), as a **union** with
                                      the node-select mask. If the two sisters of a fork lie within
                                      BIOHUB_DIV_FORK_PRUNE_UM and the pipeline's own safe-division gate (the acceptance
                                      test of add_safe_divisions_postlink; its thresholds are read from the
                                      BIOHUB_SAFE_DIV_* variables below) does not accept the pair, remove the weaker
                                      sister (lower logit_max; on a tie, the larger node id).
                                      A verbatim port of D3 + resolve_pairs + gate_accept from the offline
                                      de-duplication probe.
                                      0 = off = identical to N01/N02 (no extra log lines, no extra stats keys).
  BIOHUB_DIV_FORK_PRUNE_UM=7.0        sister-distance threshold X (µm)
                                      The gate thresholds BIOHUB_SAFE_DIV_MAX_UM (9.0), _EXISTING_CHILD_MAX_UM (10.0),
                                      _SISTER_MAX_UM (14.0), _SISTER_SYMMETRY_TAU (0.6) and _DIVERGE_UM (2.25) are read
                                      from the environment (exactly the values S86 sets).
  Evidence (offline de-duplication probe on val40 C0, 2026-09-16): D3 X=7 on top of N01 gives +0.00837 on the worse
  held-out embryo (parameters chosen on one embryo, scored on the other; N01 alone +0.00510). It touches 430 nodes,
  loses 0.04 % of the TP edges, leaves edge Jaccard essentially unchanged (-0.0001), and the whole gain comes from
  division_jaccard (+0.0325). **The division denominator is only 47 events, so this evidence is structurally weak.**
  The two masks must be computed on the **same (pre-drop) graph** and then united to match the probe; that is why
  the knob was added to this file (same hook, same function) rather than to a separate module. N02's cap knob was
  added to this file the same way.

Markers: NODE_SELECT_CONFIG (once per dataset), NODE_SELECT SCRIPT PATCH APPLIED (once)
         NODE_SELECT_CAP cap=.. n_video=.. n_candidates=.. n_dropped=.. (once per dataset when cap>0)
         DIV_FORK_PRUNE_CONFIG .. / DIV_FORK_PRUNE forks=.. failing_gate=.. pruned_nodes=.. (once each per dataset
         when the knob is on)
run_stats: node_select_dropped_nodes, node_select_dropped_edges, node_select_frac
           (only with fork prune on) fork_prune_forks, fork_prune_sibling_pairs, fork_prune_failing_gate,
           fork_prune_dropped_nodes, fork_prune_dropped_new
The anchors are notebook cell text after div_dip_946 has been applied (its PAIRS are prepended unchanged).
"""
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("div_dip_946", _Path(__file__).with_name("div_dip_946.py"))
_dip = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dip)

NS_MAP_DIR = "/tmp/biohub_nodesel"

# ---------------------------------------------------------------------------------------------- notebook helper
HELPER = '''# ==== node selection rule (kaggle/node_select_946.py) ===============================
_NS_ON = os.environ.get('BIOHUB_NODE_SELECT', '0') == '1'
_NS_LOGIT_MIN = float(os.environ.get('BIOHUB_NODE_SELECT_LOGIT_MIN', '6.0'))
_NS_LOGIT_MAX = float(os.environ.get('BIOHUB_NODE_SELECT_LOGIT_MAX', '8.0'))
_NS_DENSITY_UM = float(os.environ.get('BIOHUB_NODE_SELECT_DENSITY_UM', '10.0'))
_NS_DENSITY_MAX = int(os.environ.get('BIOHUB_NODE_SELECT_DENSITY_MAX', '1'))
_NS_CAP = float(os.environ.get('BIOHUB_NODE_SELECT_CAP', '0'))
_FP_ON = os.environ.get('BIOHUB_DIV_FORK_PRUNE', '0') == '1'
_FP_UM = float(os.environ.get('BIOHUB_DIV_FORK_PRUNE_UM', '7.0'))
_FP_PARENT_ADDED_UM = float(os.environ.get('BIOHUB_SAFE_DIV_MAX_UM', '9.0'))
_FP_PARENT_EXIST_UM = float(os.environ.get('BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM', '10.0'))
_FP_SISTER_UM = float(os.environ.get('BIOHUB_SAFE_DIV_SISTER_MAX_UM', '14.0'))
_FP_TAU = float(os.environ.get('BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU', '0.6'))
_FP_DIVERGE_UM = float(os.environ.get('BIOHUB_SAFE_DIV_DIVERGE_UM', '2.25'))
_NS_MAP_DIR = Path('%(map_dir)s')
_NS_SOURCE = 'primary+secondary'


def _ns_logit_max27(vol, zi, yi, xi):
    """max over the 3x3x3 window (clipped at the map border) — same formula as _sample_logits in the offline val40 probe."""
    Z, Y, X = vol.shape
    best = np.full(len(zi), -np.inf, np.float32)

    for dz in (-1, 0, 1):
        zz = np.clip(zi + dz, 0, Z - 1)

        for dy in (-1, 0, 1):
            yy = np.clip(yi + dy, 0, Y - 1)

            for dx in (-1, 0, 1):
                xx = np.clip(xi + dx, 0, X - 1)
                best = np.maximum(best, vol[zz, yy, xx])
    return best


def _ns_select_mask(t, z, y, x, maps, downsample, logit_min, logit_max, density_um, density_max):
    """maps: {t: (primary (Z, Y', X'), secondary (Z, Y', X'))} float32 logit. -> (drop, logit_min, logit_max, density)"""
    n = len(t)
    lpk = np.full(n, np.nan, np.float32)
    lsd = np.full(n, np.nan, np.float32)
    dens = np.zeros(n, int)
    um = np.stack([z * VOXEL_SCALE_UM[0], y * VOXEL_SCALE_UM[1], x * VOXEL_SCALE_UM[2]], 1)

    for tt in np.unique(t):
        ii = np.flatnonzero(t == tt)
        p_map, s_map = maps[int(tt)]
        Z, Y, X = p_map.shape
        zi = np.clip(np.rint(z[ii] / downsample[0]).astype(int), 0, Z - 1)
        yi = np.clip(np.rint(y[ii] / downsample[1]).astype(int), 0, Y - 1)
        xi = np.clip(np.rint(x[ii] / downsample[2]).astype(int), 0, X - 1)
        lpk[ii] = _ns_logit_max27(p_map, zi, yi, xi)
        lsd[ii] = _ns_logit_max27(s_map, zi, yi, xi)

        if len(ii) >= 2:
            tree = cKDTree(um[ii])
            dens[ii] = tree.query_ball_point(um[ii], density_um, return_length = True) - 1
    lmin = np.minimum(lpk, lsd)
    lmax = np.maximum(lpk, lsd)
    drop = (lmin < logit_min) & (lmax < logit_max) & (dens <= density_max)
    return (drop, lmin, lmax, dens)


def _ns_cap_drop(drop, lmin, lmax, ids, cap):
    """per-video cap (same formula as V2 cap_per_video in the offline adaptive val40 probe): rank candidates by logit_min ascending
    (ties: logit_max ascending, then node id ascending) and drop at most floor(cap * n_video) of them. -> (capped_drop, k)"""
    n_video = len(drop)
    k = int(np.floor(cap * n_video))
    cand = np.flatnonzero(drop)

    if len(cand) <= k:
        return (drop.copy(), k)
    order = np.lexsort((ids[cand], lmax[cand], lmin[cand]))
    out = np.zeros_like(drop)
    out[cand[order[:k]]] = True
    return (out, k)


def _ns_load_maps(dataset, frames):
    map_dir = _NS_MAP_DIR / str(dataset)
    meta_path = map_dir / 'meta.json'

    if not meta_path.is_file():
        raise RuntimeError(f'BIOHUB_NODE_SELECT is on but the logit maps were not captured: {meta_path} missing')
    meta = json.loads(meta_path.read_text())
    maps = {}

    for tt in frames:
        p_path = map_dir / f'p_{int(tt):04d}.npy'
        s_path = map_dir / f's_{int(tt):04d}.npy'

        if not (p_path.is_file() and s_path.is_file()):
            raise RuntimeError(f'BIOHUB_NODE_SELECT is on but frame {int(tt)} of {dataset} has no captured logit map')
        maps[int(tt)] = (np.load(p_path).astype(np.float32), np.load(s_path).astype(np.float32))
    return (meta, maps)


def _ns_gate_accept(p, c1, c2, um, t, ch0, odeg):
    """the pipeline's own safe-division acceptance test (fork p -> c1, c2). True = plausible division.
    Same formula as gate_accept in the offline val40 de-duplication probe."""
    d1 = float(np.linalg.norm(um[p] - um[c1]))
    d2 = float(np.linalg.norm(um[p] - um[c2]))
    sis = float(np.linalg.norm(um[c1] - um[c2]))
    lo = min(d1, d2)
    hi = max(d1, d2)

    if lo > _FP_PARENT_ADDED_UM or hi > _FP_PARENT_EXIST_UM or sis > _FP_SISTER_UM:
        return False

    if abs(d1 - d2) / max((d1 + d2) / 2.0, 1e-6) > _FP_TAU:
        return False

    if odeg[c1] != 1 or odeg[c2] != 1:
        return False
    g1 = int(ch0[c1])
    g2 = int(ch0[c2])

    if g1 < 0 or g2 < 0 or t[g1] != t[c1] + 1 or t[g2] != t[c2] + 1:
        return False
    return float(np.linalg.norm(um[g1] - um[g2])) - sis >= _FP_DIVERGE_UM


def _ns_resolve_pairs(pairs, rank, n):
    """greedy NMS: scan by rank descending (ties: lower position = lower node id first); each survivor drops its partners
    that are still undecided. A dropped node drops nobody. Same formula as resolve_pairs in the offline val40 de-duplication probe."""
    if not pairs:
        return np.zeros(n, bool)
    nbr = {}

    for a, b in pairs:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    state = np.zeros(n, np.int8)

    for i in sorted(nbr, key = lambda i: (-rank[i], i)):
        if state[i]:
            continue
        state[i] = 1

        for j in nbr[i]:
            if state[j] == 0:
                state[j] = 2
    return state == 2


def _ns_fork_mask(ids, t, um, rank, edges, radius_um):
    """D3 (offline val40 de-duplication probe): if the two sisters of a fork lie within radius_um and the safe-division gate
    does not accept the pair, drop the weaker one. ids is ascending, so position order = node id order. -> (drop, diag)"""
    n = len(ids)
    pos = {}

    for i in range(n):
        pos[int(ids[i])] = i
    ch0 = np.full(n, -1, int)
    ch1 = np.full(n, -1, int)
    odeg = np.zeros(n, int)

    for edge in edges:
        a = pos.get(int(edge['source_id']), -1)
        b = pos.get(int(edge['target_id']), -1)

        if a < 0 or b < 0:
            continue
        odeg[a] += 1

        if ch0[a] < 0:
            ch0[a] = b
        elif ch1[a] < 0:
            ch1[a] = b
    forks = np.flatnonzero(odeg == 2)
    pairs = []
    n_pairs = 0

    for p in forks:
        c1 = int(ch0[p])
        c2 = int(ch1[p])

        if c1 < 0 or c2 < 0 or t[c1] != t[c2]:
            continue

        if float(np.linalg.norm(um[c1] - um[c2])) > radius_um:
            continue
        n_pairs += 1

        if not _ns_gate_accept(int(p), c1, c2, um, t, ch0, odeg):
            pairs.append((c1, c2))
    drop = _ns_resolve_pairs(pairs, rank, n)
    diag = {'forks': int(len(forks)), 'pairs': n_pairs, 'failing_gate': len(pairs),
            'out_deg_ge3': int((odeg >= 3).sum()), 'dropped': int(drop.sum())}
    return (drop, diag)


def node_select_final_graph(nodes_by_id, edges, stats, dataset):
    """Remove the nodes that match the rule, and the edges touching them, from the final graph (no relink, ids kept)."""
    print(f'NODE_SELECT_CONFIG on={int(_NS_ON)} logit_min={_NS_LOGIT_MIN} logit_max={_NS_LOGIT_MAX} '
          f'density_um={_NS_DENSITY_UM} density_max={_NS_DENSITY_MAX} source={_NS_SOURCE} dataset={dataset}', flush = True)
    stats['node_select_dropped_nodes'] = 0
    stats['node_select_dropped_edges'] = 0
    stats['node_select_frac'] = 0.0

    if (not _NS_ON and not _FP_ON) or not nodes_by_id:
        return (nodes_by_id, edges)
    ids = np.array(sorted(nodes_by_id), dtype = int)
    t = np.array([int(nodes_by_id[i]['t']) for i in ids], dtype = int)
    z = np.array([float(nodes_by_id[i]['z']) for i in ids], dtype = float)
    y = np.array([float(nodes_by_id[i]['y']) for i in ids], dtype = float)
    x = np.array([float(nodes_by_id[i]['x']) for i in ids], dtype = float)
    meta, maps = _ns_load_maps(dataset, np.unique(t))
    drop, lmin, lmax, dens = _ns_select_mask(t, z, y, x, maps, meta['downsample'], _NS_LOGIT_MIN, _NS_LOGIT_MAX, _NS_DENSITY_UM, _NS_DENSITY_MAX)

    if not _NS_ON:
        drop = np.zeros(len(ids), bool)

    if _NS_ON and _NS_CAP > 0:
        n_cand = int(drop.sum())
        drop, cap_k = _ns_cap_drop(drop, lmin, lmax, ids, _NS_CAP)
        print(f'NODE_SELECT_CAP cap={_NS_CAP:.2f} n_video={len(ids)} n_candidates={n_cand} n_dropped={int(drop.sum())} '
              f'k={cap_k} dataset={dataset}', flush = True)
    if _FP_ON:
        _fp_um = np.stack([z * VOXEL_SCALE_UM[0], y * VOXEL_SCALE_UM[1], x * VOXEL_SCALE_UM[2]], 1)
        fork_drop, fp_diag = _ns_fork_mask(ids, t, _fp_um, lmax, edges, _FP_UM)
        stats['fork_prune_forks'] = fp_diag['forks']
        stats['fork_prune_sibling_pairs'] = fp_diag['pairs']
        stats['fork_prune_failing_gate'] = fp_diag['failing_gate']
        stats['fork_prune_dropped_nodes'] = fp_diag['dropped']
        stats['fork_prune_dropped_new'] = int((fork_drop & ~drop).sum())
        print(f'DIV_FORK_PRUNE_CONFIG on=1 radius_um={_FP_UM} parent_added<={_FP_PARENT_ADDED_UM} '
              f'parent_existing<={_FP_PARENT_EXIST_UM} sister<={_FP_SISTER_UM} tau<={_FP_TAU} '
              f'diverge>={_FP_DIVERGE_UM} dataset={dataset}', flush = True)
        print(f'DIV_FORK_PRUNE forks={fp_diag["forks"]} sibling_pairs={fp_diag["pairs"]} '
              f'failing_gate={fp_diag["failing_gate"]} pruned_nodes={fp_diag["dropped"]} '
              f'pruned_new={stats["fork_prune_dropped_new"]} out_deg_ge3={fp_diag["out_deg_ge3"]} '
              f'dataset={dataset}', flush = True)
        drop = drop | fork_drop
    drop_ids = set(int(i) for i in ids[drop])
    kept_edges = [edge for edge in edges if int(edge['source_id']) not in drop_ids and int(edge['target_id']) not in drop_ids]
    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id not in drop_ids}
    stats['node_select_dropped_nodes'] = len(drop_ids)
    stats['node_select_dropped_edges'] = len(edges) - len(kept_edges)
    stats['node_select_frac'] = len(drop_ids) / max(len(nodes_by_id), 1)
    print(f'[{dataset}] after node-select: {len(kept_nodes)} nodes, {len(kept_edges)} edges '
          f'(dropped_nodes = {len(drop_ids)}, dropped_edges = {len(edges) - len(kept_edges)}, frac = {stats["node_select_frac"]:.4f}, '
          f'cap = {_NS_CAP:.2f}, downsample = {meta["downsample"]}, map_shape = {meta["map_shape"]}, '
          f'logit_min<{_NS_LOGIT_MIN}: {int((lmin < _NS_LOGIT_MIN).sum())}, logit_max<{_NS_LOGIT_MAX}: {int((lmax < _NS_LOGIT_MAX).sum())}, '
          f'density<={_NS_DENSITY_MAX}: {int((dens <= _NS_DENSITY_MAX).sum())})', flush = True)
    return (kept_nodes, kept_edges)
# ==== end node selection rule ==============================================================

''' % {"map_dir": NS_MAP_DIR}

# ---------------------------------------------------------------------------------------------- script (runtime) pair
# Inside the retention-guard block's `if int(frame_indices[f]) not in seen_frames:` — the first window in which the frame
# appears, which is also the window used for detection.
_SCRIPT_OLD = ("                        if use_primary_detection:\n"
               "                            print('BIOHUB_RETENTION_GUARD ' + json.dumps(guard_record, sort_keys = True), flush = True)\n")
_SCRIPT_NEW = (_SCRIPT_OLD +
               "\n"
               "                        if os.environ.get('BIOHUB_NODE_SELECT', '0') == '1' or os.environ.get('BIOHUB_DIV_FORK_PRUNE', '0') == '1':\n"
               "                            _ns_dir = Path('%(map_dir)s') / ds_path.stem\n"
               "                            _ns_dir.mkdir(parents = True, exist_ok = True)\n"
               "                            _ns_t = int(frame_indices[f])\n"
               "                            _ns_p = primary_det.detach().float().cpu().numpy().reshape(tuple(primary_det.shape[-3:])).astype(np.float16)\n"
               "                            _ns_s = secondary_det.detach().float().cpu().numpy().reshape(tuple(secondary_det.shape[-3:])).astype(np.float16)\n"
               "                            np.save(_ns_dir / f'p_{_ns_t:04d}.npy', _ns_p)\n"
               "                            np.save(_ns_dir / f's_{_ns_t:04d}.npy', _ns_s)\n"
               "\n"
               "                            if not (_ns_dir / 'meta.json').is_file():\n"
               "                                (_ns_dir / 'meta.json').write_text(json.dumps({'downsample': [int(v) for v in downsample], 'map_shape': [int(v) for v in _ns_p.shape], 'window_frame': int(f)}))\n"
               "                                print(f'NODE_SELECT MAP CAPTURE dataset={ds_path.stem} downsample={[int(v) for v in downsample]} map_shape={list(_ns_p.shape)}', flush = True)\n"
               ) % {"map_dir": NS_MAP_DIR}
SCRIPT_PAIRS = [("node_select_capture", _SCRIPT_OLD, _SCRIPT_NEW)]

# Block the notebook applies to _ps at runtime. It is inserted after the TTA anchor; build_r946 inserts the tertiary
# block right after the same anchor, so this block actually runs **after** the tertiary patch (the anchors refer to
# that patched text).
_TTA_ANCHOR = "print('Edge-feature TTA patch installed and enabled')\n"
_RUNTIME_BLOCK = '''
# === lane-r: node-select logit map capture — kaggle/node_select_946.py (runtime patch on _ps) ===
_ns_pairs = %(pairs_json)s
_ns_src = _ps.read_text()
for _ns_name, _ns_old, _ns_new in _ns_pairs:
    _ns_n = _ns_src.count(_ns_old)
    if _ns_n != 1:
        raise RuntimeError(f"node-select script anchor '{_ns_name}' expected exactly 1 occurrence, found {_ns_n}")
    _ns_src = _ns_src.replace(_ns_old, _ns_new, 1)
compile(_ns_src, str(_ps), 'exec')
_ps.write_text(_ns_src)
print('NODE_SELECT SCRIPT PATCH APPLIED (1 anchor)', flush=True)
''' % {"pairs_json": _json.dumps([list(p) for p in SCRIPT_PAIRS], ensure_ascii=False)}

_DEF_OLD = "def filter_output_graph("
_DEF_NEW = HELPER + _DEF_OLD

_HOOK_OLD = "    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n"
_HOOK_NEW = _HOOK_OLD + "    nodes_by_id, edges = node_select_final_graph(nodes_by_id, edges, stats, dataset)\n"

PAIRS = list(_dip.PAIRS) + [("node_select_runtime", _TTA_ANCHOR, _TTA_ANCHOR + _RUNTIME_BLOCK),
                            ("node_select_helper", _DEF_OLD, _DEF_NEW),
                            ("node_select_hook", _HOOK_OLD, _HOOK_NEW)]
