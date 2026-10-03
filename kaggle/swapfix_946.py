"""swapfix_946.py — node-count-neutral relink repair right before linefit smoothing (NOTEBOOK_PATCH, default off).

What: after every repair stage (gap-close, gap2, safe divisions, short-track filter) and before linefit smoothing, each
frame pair t→t+1 is re-assigned among the *existing* one-to-one edges (source out-degree 1, target in-degree 1, not a
division edge). Every such source keeps exactly one out-edge; only which target it points to may change. Node set,
edge count, divisions and track starts/ends outside the re-assigned set are untouched, so the node-count factor and
the division term cannot move through this step. linefit then smooths along the repaired topology.

Cost of source x → target y (µm units, GT-free):
    w_flow · |(y − x) − f(x)|                   f = median displacement of the k nearest other one-to-one edges of the
                                                  same frame (the x138 flow field, rebuilt on the final graph)
  + w_vel  · |(y − x) − v_in(x)|                v_in = mean step of x's own track over the last `win` frames (if any)
  + w_vel  · |(y − x) − v_out(y)|               v_out = mean step of y's own track over the next `win` frames (if any)
  + w_p    · (−log max(p(x,y), p_floor))        p = the captured transformer relink probability (relink_prob table);
                                                  pairs touching a node the post-processing synthesised use p_syn
and the current edge gets a `stay` bonus (a change must win by more than `stay`). mode "hung" solves each frame with
Hungarian over those sources × their targets (+ free track-start targets when free_targets); mode "pair" only swaps
two edges at a time (greedy by gain, each edge at most once per frame). `only_ambiguous` restricts the re-assignment to
sources whose best alternative is within `stay + margin` of their current edge.

Knobs (env, read in the notebook): BIOHUB_SWAPFIX=1 turns it on; BIOHUB_SWAPFIX_PARAMS='{"w_p": 4, ...}' (JSON, keys
= repair() keyword arguments). Off (unset / "0"): the only added lines are inert definitions and one `if` → the output
is byte-identical to the base build.
marker: `SWAPFIX ds=<stem> changed=<n> frames=<n> t=<s>` per dataset, only when on.
"""
from __future__ import annotations

FUNC_SRC = r'''
def _swf_repair(nodes_by_id, edges, table, real_ids, scale, mode="hung", gate_um=14.0, w_flow=1.0, w_vel=0.0,
                win=2, w_p=4.0, p_floor=0.01, p_syn=0.3, stay=1.0, k=12, radius_um=40.0, exclude_um=1.5,
                min_samples=4, free_targets=False, only_ambiguous=False, margin=0.0, max_changes_frac=1.0,
                free_min_p=0.0, join_min_p=0.0, join_resid_um=5.0):
    """Re-assign one-to-one edges per frame pair; returns (new_edges, info). Deterministic, GT-free."""
    import math as _m
    import numpy as _np
    from scipy.optimize import linear_sum_assignment as _lsa
    from scipy.spatial import cKDTree as _KD

    scale = _np.asarray(scale, dtype=_np.float64)
    pos = {int(i): _np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=_np.float64) * scale
           for i, n in nodes_by_id.items()}
    tt = {int(i): int(n["t"]) for i, n in nodes_by_id.items()}
    out_e, in_e = {}, {}
    for e in edges:
        out_e.setdefault(int(e["source_id"]), []).append(e)
        in_e.setdefault(int(e["target_id"]), []).append(e)

    def simple(e):
        s, g = int(e["source_id"]), int(e["target_id"])
        return (len(out_e.get(s, ())) == 1 and len(in_e.get(g, ())) == 1 and not e.get("safe_division")
                and tt.get(g) == tt.get(s, -9) + 1)

    # current successor / predecessor along one-to-one chains (updated as frames are repaired)
    succ, pred = {}, {}
    for e in edges:
        if simple(e):
            succ[int(e["source_id"])] = int(e["target_id"])
            pred[int(e["target_id"])] = int(e["source_id"])

    def v_in(x):
        steps, cur = [], x
        for _ in range(win):
            p_ = pred.get(cur)
            if p_ is None:
                break
            steps.append(pos[cur] - pos[p_])
            cur = p_
        return _np.mean(steps, axis=0) if steps else None

    def v_out(y):
        steps, cur = [], y
        for _ in range(win):
            s_ = succ.get(cur)
            if s_ is None:
                break
            steps.append(pos[s_] - pos[cur])
            cur = s_
        return _np.mean(steps, axis=0) if steps else None

    def prob(x, y):
        if x not in real_ids or y not in real_ids:
            return p_syn
        return table.get((x, y), 0.0)

    by_t = {}
    for i in pos:
        by_t.setdefault(tt[i], []).append(i)
    changed, frames, n_amb = 0, 0, 0
    info = {"swf_changed": 0, "swf_frames": 0, "swf_sources": 0, "swf_ambiguous": 0, "swf_free_used": 0, "swf_joined": 0}
    n_out, n_in = {}, {}
    for e in edges:
        n_out[int(e["source_id"])] = n_out.get(int(e["source_id"]), 0) + 1
        n_in[int(e["target_id"])] = n_in.get(int(e["target_id"]), 0) + 1
    edge_of = {int(e["source_id"]): e for e in edges if simple(e)}
    for t in sorted(by_t):
        src = sorted(s for s in by_t[t] if s in edge_of and s in succ)
        if not src:
            continue
        cur_tg = [succ[s] for s in src]
        tg = list(cur_tg)
        if free_targets:
            taken = {int(e["target_id"]) for e in edges}
            tg += sorted(i for i in by_t.get(t + 1, []) if i not in taken)
        # flow field from this frame's one-to-one edges
        fs = _np.stack([pos[s] for s in src])
        fd = _np.stack([pos[succ[s]] - pos[s] for s in src])
        tree = _KD(fs) if len(src) >= min_samples else None
        kq = min(k + 1, len(src))
        pred_disp = _np.zeros_like(fs)
        for a, s in enumerate(src):
            if tree is not None:
                d_, ix = tree.query(fs[a], k=kq, distance_upper_bound=radius_um)
                d_, ix = _np.atleast_1d(d_), _np.atleast_1d(ix)
                keep = _np.isfinite(d_) & (d_ >= exclude_um)
                ix = ix[keep][:k]
                if ix.size:
                    pred_disp[a] = _np.median(fd[ix], axis=0)
        tpos = _np.stack([pos[g] for g in tg])
        ttree = _KD(tpos)
        big = 1e6
        C = _np.full((len(src), len(tg)), big)
        vins = [v_in(s) if w_vel > 0 else None for s in src]
        vouts = [v_out(g) if w_vel > 0 else None for g in tg]
        def cost(a, b):
            g = tg[b]
            step = tpos[b] - fs[a]
            c = w_flow * float(_np.linalg.norm(step - pred_disp[a]))
            if w_vel > 0:
                if vins[a] is not None:
                    c += w_vel * float(_np.linalg.norm(step - vins[a]))
                if vouts[b] is not None:
                    c += w_vel * float(_np.linalg.norm(step - vouts[b]))
            if w_p > 0:
                c += w_p * -_m.log(max(prob(src[a], g), p_floor))
            return c

        for a in range(len(src)):
            for b in ttree.query_ball_point(fs[a], r=gate_um):
                if b >= len(src) and free_min_p > 0.0 and prob(src[a], tg[b]) < free_min_p:
                    continue
                C[a, b] = cost(a, b)
            C[a, a] = cost(a, a) - stay          # current target sits at column a (always feasible)
        if only_ambiguous:
            amb = []
            for a in range(len(src)):
                row = C[a].copy(); cur = row[a]; row[a] = big
                if row.min() < cur + margin:
                    amb.append(a)
            n_amb += len(amb)
            if not amb:
                frames += 1
                continue
            cols = sorted(set(amb) | {b for a in amb for b in _np.flatnonzero(C[a] < big)})
            # columns outside `amb` rows whose owner is not ambiguous stay fixed: restrict to amb rows and the
            # columns reachable from them, then keep the owners of those columns in the problem too
            rows = sorted(set(amb) | {c for c in cols if c < len(src)})
            cols = sorted(set(cols) | set(rows))
            sub = C[_np.ix_(rows, cols)]
            r_i, c_i = _lsa(sub)
            assign = {rows[r]: cols[c] for r, c in zip(r_i, c_i)}
        elif mode != "pair":
            r_i, c_i = _lsa(C)
            assign = dict(zip(r_i.tolist(), c_i.tolist()))
        if mode == "pair":
            assign = {}
            gains = []
            for a in range(len(src)):
                for b in _np.flatnonzero(C[a, :len(src)] < big):
                    if b <= a or C[b, a] >= big:
                        continue
                    gain = (C[a, a] + C[b, b]) - (C[a, b] + C[b, a])
                    if gain > 0:
                        gains.append((-gain, a, int(b)))
            used = set()
            for _g, a, b in sorted(gains):
                if a in used or b in used:
                    continue
                used |= {a, b}
                assign[a] = b
                assign[b] = a
        moves = [(a, b) for a, b in assign.items() if b != a and C[a, b] < big]
        if max_changes_frac < 1.0 and len(moves) > max_changes_frac * len(src):
            moves = []
        frames += 1
        for a, b in moves:
            s, g_old, g_new = src[a], cur_tg[a], tg[b]
            e = edge_of[s]
            e["target_id"] = g_new
            e["distance_um"] = float(_np.linalg.norm(pos[g_new] - pos[s]))
            e["swapfix"] = 1
            if b >= len(src):
                info["swf_free_used"] += 1
        # bookkeeping for the next frames (pred of moved targets, in_e for later `simple` checks is not needed:
        # later frames only look at succ/pred)
        for a, b in moves:
            s, g_old = src[a], cur_tg[a]
            if pred.get(g_old) == s:
                del pred[g_old]
        for a, b in moves:
            s, g_new = src[a], tg[b]
            succ[s] = g_new
            pred[g_new] = s
        for a, b in moves:
            n_in[cur_tg[a]] = n_in.get(cur_tg[a], 0) - 1
            n_in[tg[b]] = n_in.get(tg[b], 0) + 1
        changed += len(moves)
        info["swf_sources"] += len(src)
        if join_min_p > 0.0:
            # track end (t) -> track start (t+1): adds an edge (node set unchanged), only when the transformer is
            # confident (p >= join_min_p) and the target sits within join_resid_um of the flow-predicted position
            ends = sorted(i for i in by_t[t] if n_out.get(i, 0) == 0)
            starts = sorted(i for i in by_t.get(t + 1, []) if n_in.get(i, 0) == 0)
            if ends and starts and tree is not None:
                spos = _np.stack([pos[g] for g in starts])
                J = _np.full((len(ends), len(starts)), big)
                stree = _KD(spos)
                for a, x in enumerate(ends):
                    d_, ix = tree.query(pos[x], k=kq, distance_upper_bound=radius_um)
                    d_, ix = _np.atleast_1d(d_), _np.atleast_1d(ix)
                    ix = ix[_np.isfinite(d_)][:k]
                    fl = _np.median(fd[ix], axis=0) if ix.size else _np.zeros(3)
                    for b in stree.query_ball_point(pos[x], r=gate_um):
                        pj = prob(x, starts[b])
                        if pj < join_min_p:
                            continue
                        r = float(_np.linalg.norm(spos[b] - pos[x] - fl))
                        if r <= join_resid_um:
                            J[a, b] = w_flow * r + w_p * -_m.log(max(pj, p_floor))
                r_i, c_i = _lsa(J)
                for a, b in zip(r_i.tolist(), c_i.tolist()):
                    if J[a, b] >= big:
                        continue
                    x, g = ends[a], starts[b]
                    edges.append({"source_id": x, "target_id": g, "edge_prob": prob(x, g),
                                  "distance_um": float(_np.linalg.norm(pos[g] - pos[x])), "swapfix_join": 1})
                    n_out[x] = 1
                    n_in[g] = 1
                    succ[x] = g
                    pred[g] = x
                    info["swf_joined"] += 1
    info["swf_changed"] = changed
    info["swf_frames"] = frames
    info["swf_ambiguous"] = n_amb
    return edges, info
'''

_ns: dict = {}
exec(FUNC_SRC, _ns)


def repair(nodes_by_id, edges, table, real_ids, scale, **params):
    """offline entry (used by our val40 replay harness, not included): same function text as the notebook helper."""
    return _ns["_swf_repair"](nodes_by_id, edges, table, real_ids, scale, **params)


# ---------------------------------------------------------------------------------------------- notebook pairs
# chain: x138_port_946.PAIRS (which already carries relink_prob_946.PAIRS) + the three pairs below, applied in order.
import importlib.util as _ilu  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_spec = _ilu.spec_from_file_location("x138_port_946", _Path(__file__).with_name("x138_port_946.py"))
_x138 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_x138)

HELPER = ("# ==== swapfix: one-to-one edge re-assignment before linefit (kaggle/swapfix_946.py; default off) ====\n"
          "import json as _swf_json\n"
          "_SWF_ON = os.environ.get('BIOHUB_SWAPFIX', '0').strip() == '1'\n"
          "_SWF_PARAMS = _swf_json.loads(os.environ.get('BIOHUB_SWAPFIX_PARAMS', '') or '{}')\n"
          "_SWF_REAL = {}\n"
          + FUNC_SRC +
          "\n\ndef _swf_apply(nodes_by_id, edges, stats, dataset):\n"
          "    _t0 = time.time()\n"
          "    table, _n_pairs, _gate = _rp_load(dataset)\n"
          "    edges, info = _swf_repair(nodes_by_id, edges, table, _SWF_REAL.get(dataset, set()), VOXEL_SCALE_UM, **_SWF_PARAMS)\n"
          "    stats.update(info)\n"
          "    stats['swf_s'] = round(time.time() - _t0, 1)\n"
          "    print(f\"SWAPFIX ds={dataset} changed={info['swf_changed']} sources={info['swf_sources']} \"\n"
          "          f\"free_used={info['swf_free_used']} frames={info['swf_frames']} pairs={_n_pairs} \"\n"
          "          f\"real={len(_SWF_REAL.get(dataset, ()))} edges={len(edges)} t={stats['swf_s']}\", flush = True)\n"
          "    return edges\n"
          "# ==== end swapfix ========================================================================================\n\n")

_DEF_OLD = "def filter_output_graph("
_ENTRY_OLD = "    edges: list[dict[str, object]] = []\n\n    for edge in raw_edges:\n"
_ENTRY_NEW = ("    if _SWF_ON:\n"
              "        _SWF_REAL[dataset] = {int(_i) for _i in nodes_by_id}\n"
              + _ENTRY_OLD)
_CALL_OLD = "    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n"
_CALL_NEW = ("    if _SWF_ON:\n"
             "        edges = _swf_apply(nodes_by_id, edges, stats, dataset)\n"
             + _CALL_OLD)

SWF_PAIRS = [("swapfix_helper", _DEF_OLD, HELPER + _DEF_OLD),
             ("swapfix_real_ids", _ENTRY_OLD, _ENTRY_NEW),
             ("swapfix_call", _CALL_OLD, _CALL_NEW)]


# ---------------------------------------------------------------------------------------------- family (b) flow knobs
# Knob-gated edits inside the x138 flow relink (_x138_flow_motion_relink_edges). All default 0 = inert.
#   BIOHUB_FLOWX_VEL_A=<a>     sources with a relinked predecessor predict source + (1−a)·flow + a·own last step
#   BIOHUB_FLOWX_BIDIR_W=<w>   + w·|(source − target) − b(target)|, b = backward field (target → source displacement of the
#                              same seed / refined matches, sampled around the target) — forward/backward consistency
#   BIOHUB_FLOWX_RESID_UM=<r>  refinement rounds (FLOW_ITER ≥ 2) keep every previous-round match whose flow residual
#                              is ≤ r and re-assign only the rest (second iteration for residual-large nodes only)
FLX_KNOBS = ("_FLX_VEL_A = float(os.environ.get('BIOHUB_FLOWX_VEL_A', '0') or 0)\n"
             "_FLX_BIDIR_W = float(os.environ.get('BIOHUB_FLOWX_BIDIR_W', '0') or 0)\n"
             "_FLX_RESID_UM = float(os.environ.get('BIOHUB_FLOWX_RESID_UM', '0') or 0)\n")
_F_VEL_OLD = "                predicted_arr[i] = source_pos + flow_step\n"
_F_VEL_NEW = (_F_VEL_OLD +
              "                if _FLX_VEL_A > 0.0 and prev_pos is not None:\n"
              "                    predicted_arr[i] = source_pos + (1.0 - _FLX_VEL_A) * flow_step + _FLX_VEL_A * (source_pos - prev_pos)\n")
_F_BPRE_OLD = "        target_tree = cKDTree(target_arr)\n"
_F_BPRE_NEW = ("        _flx_bpred = None\n"
               "        if _FLX_BIDIR_W > 0.0 and flow is not None and _flx_bflow[0] is not None:\n"
               "            _flx_bpred = [_flx_bflow[0](_tp, flow_exclude_um) for _tp in target_arr]\n"
               + _F_BPRE_OLD)
_F_BCOST_OLD = ("                if _RP_W != 0.0:  # relink_prob_946 bonus (N05 relink kept)\n"
                "                    cost[i, j] -= _RP_W * _RP_TABLE.get((source_id, target_id), 0.0)\n")
_F_BCOST_NEW = (_F_BCOST_OLD +
                "                if _flx_bpred is not None and _flx_bpred[j] is not None:\n"
                "                    cost[i, j] += _FLX_BIDIR_W * float(np.linalg.norm((source_pos - target_pos) - _flx_bpred[j]))\n")
_F_LOOP_OLD = "    times = sorted(ids_by_t)\n    previous_flow = None\n"
_F_LOOP_NEW = ("    _flx_bflow = [None]\n\n"
               "    def _flx_bfield(matches):\n"
               "        return _flow_predictor([position_um[m[1]] for m in matches],\n"
               "                               [position_um[m[0]] - position_um[m[1]] for m in matches])\n\n"
               + _F_LOOP_OLD)
_F_SEED_OLD = "            seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)\n"
_F_SEED_NEW = "            _flx_bflow[0] = None\n" + _F_SEED_OLD
_F_FIELD_OLD = "            flow = _field_from(seed)\n"
_F_FIELD_NEW = (_F_FIELD_OLD +
                "            if _FLX_BIDIR_W > 0.0:\n"
                "                _flx_bflow[0] = _flx_bfield(seed)\n")
_F_REF_OLD = "                flow = refined\n"
_F_REF_NEW = (_F_REF_OLD +
              "                if _FLX_BIDIR_W > 0.0:\n"
              "                    _flx_bflow[0] = _flx_bfield([(s, g) for s, g, _r, _m, _n, _p in frame_matches])\n")
_F_RES_OLD = ("            unmatched_sources = set(source_ids)\n"
              "            unmatched_targets = set(target_ids)\n"
              "            frame_matches = []\n")
_F_RES_NEW = ("            _flx_keep = [m for m in frame_matches if round_index > 0 and _FLX_RESID_UM > 0.0 and m[3] <= _FLX_RESID_UM]\n"
              "            unmatched_sources = set(source_ids) - {m[0] for m in _flx_keep}\n"
              "            unmatched_targets = set(target_ids) - {m[1] for m in _flx_keep}\n"
              "            frame_matches = list(_flx_keep)\n")
FLX_PAIRS = [("flowx_knobs", _DEF_OLD, FLX_KNOBS + "\n" + _DEF_OLD),
             ("flowx_vel", _F_VEL_OLD, _F_VEL_NEW),
             ("flowx_bpred", _F_BPRE_OLD, _F_BPRE_NEW),
             ("flowx_bcost", _F_BCOST_OLD, _F_BCOST_NEW),
             ("flowx_loop", _F_LOOP_OLD, _F_LOOP_NEW),
             ("flowx_seed", _F_SEED_OLD, _F_SEED_NEW),
             ("flowx_field", _F_FIELD_OLD, _F_FIELD_NEW),
             ("flowx_refine", _F_REF_OLD, _F_REF_NEW),
             ("flowx_resid", _F_RES_OLD, _F_RES_NEW)]
# FLX_PAIRS stay OUT of the kernel chain: on the val40 replay every family-(b) knob was flat or failed the per-embryo
# gate. They remain here for the offline replay harness (`--patch module`; not included in this repository).
PAIRS = list(_x138.PAIRS) + SWF_PAIRS
