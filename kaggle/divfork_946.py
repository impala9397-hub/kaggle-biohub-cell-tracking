"""divfork_946.py — metric-aware GT-free division-fork rules on the final graph (NOTEBOOK_PATCH, default off). 2026-09-27, Claude.

Where: one call `edges = _divfork_apply(nodes_by_id, edges, stats, dataset = dataset)` at the end of filter_output_graph
(after linefit + node_select, right before the FINAL print), helper defined next to the x138 port block. No node is added,
moved or removed; with BIOHUB_DIVFORK unset / '' the function returns its input list object unchanged (output identical).
Only consecutive (t -> t+1) edges are ever added, so the notebook's lineage guard is untouched.

Why (offline division anatomy, see docs/method.md): a predicted fork is an FP only when its node matches an annotated GT
cell (or its branches are cross-component / merged); forks on unannotated cells are free. Every FP fork on val40, TRAIN41
and the placeholder is a safe-division fork on an annotated, non-dividing cell whose added "daughter" is a neighbouring,
unannotated nucleus. Two physical tests separate those from true divisions:

Rules (BIOHUB_DIVFORK = comma list):
  prune   geometric. For a fork F(t) -> {c1, c2}, carry each child back one frame with the local flow (c - f). If any other
          node of frame t (not F) lies within PRUNE_ALT_UM (7.0 um) of a back-projected child, that child is the continuation
          of a neighbour, not a daughter: drop the safe-division edge (if exactly one child is safe-division flagged; else the
          child farther from F + f).
  chroma  chromatin conservation. A daughter carries about half the parent's chromatin. Background-subtracted intensity sums
          in a (2*RZ+1) x (2*RYX+1)^2 voxel box (2 / 8 -> about 6.5 um cube) around F (frame t) and each child (frame t+1);
          if the brighter child exceeds CHROMA_RMAX (1.1) x the parent, drop the edge as above. Reads frames with the
          notebook's read_test_frame (own 4-frame cache).
  ldf     late-daughter fork (evaluated offline; its candidate pool was 10^3 x too wide — not recommended).
  gf      gap fork (fires 0 times offline — not recommended).
Local flow = component-wise median displacement of the K nearest existing t -> t+1 edges (sources within FLOW_RADIUS_UM,
the query node itself excluded); fewer than FLOW_MIN samples -> zero.
marker: DIVFORK ds=<stem> rules=<..> prune=<n> chroma=<n> ldf=<n> gf=<n> forks_before=<n> forks_after=<n>
"""
import importlib.util as _ilu
from pathlib import Path as _Path

HELPER = r'''
# ---- metric-aware division forks (kaggle/divfork_946.py; BIOHUB_DIVFORK, default off) ----
_DIVFORK = tuple(r.strip() for r in os.environ.get('BIOHUB_DIVFORK', '').split(',') if r.strip())
_DF_P = {k: float(os.environ.get('BIOHUB_DIVFORK_' + k, v)) for k, v in (
    ('FLOW_K', '12'), ('FLOW_RADIUS_UM', '40'), ('FLOW_MIN', '4'),
    ('SIS_MIN', '4.0'), ('SIS_MAX', '14.0'),
    ('LDF_PD_MAX_UM', '12.0'), ('LDF_MID_MAX_UM', '4.0'), ('LDF_MIN_LEN', '3'),
    ('GF_PD_MAX_UM', '12.0'), ('GF_MID_MAX_UM', '4.0'), ('GF_MIN_LEN', '3'),
    ('PRUNE_ALT_UM', '7.0'), ('CHROMA_RMAX', '1.1'),
    ('CHROMA_RZ', '2'), ('CHROMA_RYX', '8'), ('CHROMA_BZ', '4'), ('CHROMA_BYX', '24'), ('CHROMA_FRAMES', '4'))}


def _divfork_integ(vol, node):
    """background-subtracted intensity sum in a (2*RZ+1) x (2*RYX+1)^2 voxel box (bg = 20th pct of a larger box)."""
    Z, Y, X = vol.shape
    z, y, x = (int(round(float(node[k]))) for k in ('z', 'y', 'x'))
    rz, ryx, bz, byx = (int(_DF_P[k]) for k in ('CHROMA_RZ', 'CHROMA_RYX', 'CHROMA_BZ', 'CHROMA_BYX'))
    box = vol[max(0, z - rz):min(Z, z + rz + 1), max(0, y - ryx):min(Y, y + ryx + 1), max(0, x - ryx):min(X, x + ryx + 1)]
    ctx = vol[max(0, z - bz):min(Z, z + bz + 1), max(0, y - byx):min(Y, y + byx + 1), max(0, x - byx):min(X, x + byx + 1)]
    if box.size == 0 or ctx.size == 0:
        return None
    bg = float(np.percentile(ctx, 20))
    return float(np.clip(box.astype(np.float64) - bg, 0, None).sum())


def _divfork_apply(nodes_by_id, edges, stats, dataset = None):
    """GT-free fork additions on the final graph (see kaggle/divfork_946.py). Returns edges (+ added)."""
    if not _DIVFORK or not edges:
        return edges
    P = _DF_P
    succ, pred = {}, {}
    sd_edges = set()
    for e in edges:
        s, t_ = int(e['source_id']), int(e['target_id'])
        succ.setdefault(s, []).append(t_)
        pred.setdefault(t_, []).append(s)
        if e.get('safe_division') or e.get('divfork'):
            sd_edges.add((s, t_))
    pos = {nid: _position_um(n) for nid, n in nodes_by_id.items()}
    tof = {nid: int(n['t']) for nid, n in nodes_by_id.items()}
    by_t = {}
    for nid in sorted(nodes_by_id):
        by_t.setdefault(tof[nid], []).append(nid)
    flow_idx = {}

    def flow_at(t, p, self_id = None):
        if t not in flow_idx:
            src = [u for u in by_t.get(t, []) if len(succ.get(u, ())) >= 1]
            if src:
                d = np.stack([pos[succ[u][0]] - pos[u] for u in src])
                flow_idx[t] = (cKDTree(np.stack([pos[u] for u in src])), src, d)
            else:
                flow_idx[t] = None
        fi = flow_idx[t]
        if fi is None:
            return np.zeros(3)
        tree, src, d = fi
        k = min(len(src), int(P['FLOW_K']) + 1)
        dist, idx = tree.query(p, k = k)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        keep = [j for dd, j in zip(dist, idx) if dd <= P['FLOW_RADIUS_UM'] and src[int(j)] != self_id][:int(P['FLOW_K'])]
        if len(keep) < int(P['FLOW_MIN']):
            return np.zeros(3)
        return np.median(d[np.array(keep, dtype = int)], axis = 0)

    def fwd_len(n, cap):
        k = 1
        while k < cap and len(succ.get(n, ())) == 1:
            n = succ[n][0]
            k += 1
        return k

    def back_len(n, cap):
        k = 1
        while k < cap and len(pred.get(n, ())) == 1:
            n = pred[n][0]
            k += 1
        return k

    starts_by_t = {}
    for nid in sorted(nodes_by_id):
        if nid not in pred:
            starts_by_t.setdefault(tof[nid], []).append(nid)
    start_tree = {t: (cKDTree(np.stack([pos[n] for n in ids])), ids) for t, ids in starts_by_t.items()}
    forks_before = sum(1 for s in succ if len(succ[s]) >= 2)
    added, used_src, used_tgt = [], set(), set()
    n_ldf = n_gf = n_prune = n_chroma = 0
    removed = set()

    if 'prune' in _DIVFORK:
        frame_tree = {}
        for f_id in sorted(s for s in succ if len(succ[s]) == 2):
            t = tof[f_id]
            if t not in frame_tree:
                ids = by_t.get(t, [])
                frame_tree[t] = (cKDTree(np.stack([pos[n] for n in ids])), ids)
            tree, ids = frame_tree[t]
            fl = flow_at(t, pos[f_id], f_id)
            alt = []
            for c in succ[f_id]:
                dd, jj = tree.query(pos[c] - fl, k = min(3, len(ids)))
                alt.append(min((float(x) for x, j in zip(np.atleast_1d(dd), np.atleast_1d(jj)) if ids[int(j)] != f_id),
                               default = float('inf')))
            if min(alt) >= P['PRUNE_ALT_UM']:
                continue
            kids = succ[f_id]
            sd_kids = [c for c in kids if (f_id, c) in sd_edges]
            drop = sd_kids[0] if len(sd_kids) == 1 else max(kids, key = lambda c: (float(np.linalg.norm(pos[c] - pos[f_id] - fl)), c))
            removed.add((f_id, drop))
            n_prune += 1

    if 'chroma' in _DIVFORK and dataset is not None:
        fcache = {}

        def frame(t):
            if t not in fcache:
                while len(fcache) >= int(P['CHROMA_FRAMES']):
                    fcache.pop(next(iter(fcache)))
                fcache[t] = read_test_frame(dataset, int(t), {})
            return fcache[t]
        for f_id in sorted((s for s in succ if len(succ[s]) == 2), key = lambda n: (tof[n], n)):
            kids = succ[f_id]
            if any((f_id, c) in removed for c in kids):
                continue
            try:
                i_f = _divfork_integ(frame(tof[f_id]), nodes_by_id[f_id])
                i_c = [_divfork_integ(frame(tof[c]), nodes_by_id[c]) for c in kids]
            except Exception:
                stats['divfork_chroma_failed'] = stats.get('divfork_chroma_failed', 0) + 1
                continue
            if i_f is None or i_f <= 0 or any(v is None for v in i_c) or max(i_c) / i_f <= P['CHROMA_RMAX']:
                continue
            fl = flow_at(tof[f_id], pos[f_id], f_id)
            sd_kids = [c for c in kids if (f_id, c) in sd_edges]
            drop = sd_kids[0] if len(sd_kids) == 1 else max(kids, key = lambda c: (float(np.linalg.norm(pos[c] - pos[f_id] - fl)), c))
            removed.add((f_id, drop))
            n_chroma += 1

    if 'gf' in _DIVFORK:
        props = []
        for e_id in sorted(nodes_by_id):
            if e_id in succ or e_id not in pred or back_len(e_id, int(P['GF_MIN_LEN'])) < int(P['GF_MIN_LEN']):
                continue
            t = tof[e_id]
            st = start_tree.get(t + 1)
            if st is None:
                continue
            q1 = pos[e_id] + flow_at(t, pos[e_id], e_id)
            near = sorted(st[1][int(j)] for j in st[0].query_ball_point(q1, P['GF_PD_MAX_UM']))
            near = [n for n in near if fwd_len(n, int(P['GF_MIN_LEN'])) >= int(P['GF_MIN_LEN'])]
            for i in range(len(near)):
                for j in range(i + 1, len(near)):
                    a, b = near[i], near[j]
                    sis = float(np.linalg.norm(pos[a] - pos[b]))
                    if not (P['SIS_MIN'] <= sis <= P['SIS_MAX']):
                        continue
                    mid = float(np.linalg.norm((pos[a] + pos[b]) / 2.0 - q1))
                    if mid <= P['GF_MID_MAX_UM']:
                        props.append((mid, e_id, a, b))
        for mid, e_id, a, b in sorted(props):
            if e_id in used_src or a in used_tgt or b in used_tgt:
                continue
            for c in (a, b):
                added.append({'source_id': e_id, 'target_id': c, 'edge_prob': None,
                              'distance_um': float(np.linalg.norm(pos[c] - pos[e_id])), 'divfork': 'gf'})
                used_tgt.add(c)
            used_src.add(e_id)
            n_gf += 1

    if 'ldf' in _DIVFORK:
        props = []
        all_tree = {}
        for s_id in sorted(nodes_by_id):
            if s_id in pred or s_id in used_tgt or fwd_len(s_id, int(P['LDF_MIN_LEN'])) < int(P['LDF_MIN_LEN']):
                continue
            t2 = tof[s_id]
            if (t2 - 1) not in all_tree:
                ids = by_t.get(t2 - 1, [])
                all_tree[t2 - 1] = (cKDTree(np.stack([pos[n] for n in ids])), ids) if ids else None
            at = all_tree[t2 - 1]
            if at is None:
                continue
            for a_id in sorted(at[1][int(j)] for j in at[0].query_ball_point(pos[s_id], P['LDF_PD_MAX_UM'])):
                if len(succ.get(a_id, ())) != 1 or len(pred.get(a_id, ())) != 1:
                    continue
                p_id = pred[a_id][0]
                a2 = succ[a_id][0]
                if len(succ.get(p_id, ())) != 1 or tof[p_id] != t2 - 2 or tof[a2] != t2:
                    continue
                sis = float(np.linalg.norm(pos[a2] - pos[s_id]))
                if not (P['SIS_MIN'] <= sis <= P['SIS_MAX']):
                    continue
                f1 = flow_at(t2 - 2, pos[p_id], p_id)
                q1 = pos[p_id] + f1
                q2 = q1 + flow_at(t2 - 1, q1, a_id)
                mid = float(np.linalg.norm((pos[a2] + pos[s_id]) / 2.0 - q2))
                if mid <= P['LDF_MID_MAX_UM']:
                    props.append((mid, a_id, s_id))
        for mid, a_id, s_id in sorted(props):
            if a_id in used_src or s_id in used_tgt:
                continue
            added.append({'source_id': a_id, 'target_id': s_id, 'edge_prob': None,
                          'distance_um': float(np.linalg.norm(pos[s_id] - pos[a_id])), 'divfork': 'ldf'})
            used_src.add(a_id)
            used_tgt.add(s_id)
            n_ldf += 1

    out = edges
    if removed:
        out = [e for e in out if (int(e['source_id']), int(e['target_id'])) not in removed]
    if added:
        out = [*out, *added]
    forks_after = forks_before + n_ldf + n_gf - n_prune - n_chroma
    stats['divfork_chroma'] = stats.get('divfork_chroma', 0) + n_chroma
    stats['divfork_ldf'] = stats.get('divfork_ldf', 0) + n_ldf
    stats['divfork_gf'] = stats.get('divfork_gf', 0) + n_gf
    stats['divfork_prune'] = stats.get('divfork_prune', 0) + n_prune
    print(f"DIVFORK ds={dataset} rules={','.join(_DIVFORK)} prune={n_prune} chroma={n_chroma} ldf={n_ldf} gf={n_gf} "
          f"forks_before={forks_before} forks_after={forks_after}", flush = True)
    return out
'''

_x1_spec = _ilu.spec_from_file_location("x138_port_946", _Path(__file__).with_name("x138_port_946.py"))
_x1 = _ilu.module_from_spec(_x1_spec)
_x1_spec.loader.exec_module(_x1)

_END = "# ==== end x138 public-lineage port ="
_CALL_OLD = "    print(f'[{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges')\n"
_CALL_NEW = ("    edges = _divfork_apply(nodes_by_id, edges, stats, dataset = dataset)\n" + _CALL_OLD)

DIVFORK_PAIRS = [("divfork_helper", _END, HELPER + "\n" + _END),
                 ("divfork_call", _CALL_OLD, _CALL_NEW)]

PAIRS = list(_x1.PAIRS) + DIVFORK_PAIRS


def offline_ns(env=None):
    """exec HELPER into a namespace with the notebook's geometry helpers (for the offline replay harness and tests).

    Frames are read from DIVFORK_OFFLINE_DATA (default <repo>/data/train, competition data, not included)."""
    import os
    import numpy as np
    from scipy.spatial import cKDTree
    vox = (1.625, 0.40625, 0.40625)

    def _position_um(node):
        return np.array([float(node['z']) * vox[0], float(node['y']) * vox[1], float(node['x']) * vox[2]], dtype=np.float64)

    def read_test_frame(dataset, t, frame_cache):
        import zarr
        root = os.environ.get("DIVFORK_OFFLINE_DATA", str(_Path(__file__).resolve().parents[1] / "data/train"))
        return np.asarray(zarr.open_group(str(_Path(root) / f"{dataset}.zarr"), mode="r")["0"][t])

    ns = {"os": os, "np": np, "cKDTree": cKDTree, "_position_um": _position_um, "VOXEL_SCALE_UM": vox,
          "read_test_frame": read_test_frame}
    saved = {k: os.environ.get(k) for k in list(os.environ) if k.startswith("BIOHUB_DIVFORK")}
    for k in saved:
        os.environ.pop(k)
    try:
        os.environ.update(env or {})
        exec(compile(HELPER, "divfork_helper", "exec"), ns)
    finally:
        for k in list(os.environ):
            if k.startswith("BIOHUB_DIVFORK"):
                os.environ.pop(k)
        os.environ.update({k: v for k, v in saved.items() if v is not None})
    return ns
