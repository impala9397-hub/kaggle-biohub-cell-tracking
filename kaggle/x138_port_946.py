"""x138_port_946.py — ports onto N05 the three scoring mechanisms that the public notebook `anvithpothula/biohub-x138`
(LB 0.953) added on top of `amanatar/biohub-geometric-fusion` (0.948) (NOTEBOOK_PATCH, all off by default).

N05 = build_r946 + tertiary seedC4 + NOTEBOOK_PATCH relink_prob_946 (→ node_select_946 → div_dip_946) + div_dip mode 4 +
BIOHUB_RELINK_PROB_W=32. This file prepends relink_prob_946.PAIRS unchanged and appends the pairs below. With every knob
empty, the code path, output and stats are identical to N05 (every added line is inert, gated by a knob).

Ported (x138's env names are kept; the values map 1:1)
  (1) Low-threshold detection dump — runtime script patch (predict_unet_transformer.py)
      BIOHUB_LOWDET_THRESHOLD=<p>  if >0, runs _detect_cells_pooled once more per detection frame at threshold p and
                                   collects (t,z,y,x) + sigmoid scores. x138 = 0.3
      BIOHUB_CACHE_DIR=<dir>       one <dir>/<stem>.npz per dataset (coords, low_coords, low_score).
                                   x138 = /kaggle/working/edge_cache; here /tmp/biohub_x138_lowdet (keeps the output
                                   folder clean; no effect on results)
      x138's edge-candidate cache (BIOHUB_CACHE_EDGE_THRESHOLD=1.0) is not ported: its condition `probs > 1.0` is never
      true for softmax/sigmoid probabilities (x138 also writes only empty arrays) — a provable no-op.
  (2) Readmit — notebook, right after the first motion re-link in filter_output_graph (same place as in x138)
      BIOHUB_READMIT_RADIUS_UM=<um> turns it on if >0 (x138 = 4). BIOHUB_READMIT_MIN_SCORE (x138 = 0.965)
      A detection peak that the ILP discarded (score >= MIN_SCORE, farther than GAPFILL_EXCLUDE_UM from existing nodes)
      and that lies within RADIUS of an open track end/start is restored as a node, and the re-link runs once more. As
      in x138, the pool is built by load_low_detections, so this works only with BIOHUB_GAPFILL_MAX_GAP >= 1.
      Difference (our relink kept): the second re-link also uses relink_prob's learned-probability bonus (w=32).
      _rp_motion_relink runs an extra diagnostic w=0 reference pass, so the second re-link runs only once, through
      _x138_relink_once, without that reference pass. Restored nodes get new ids from _next_node_id, and such an id can
      coincide with the index of a detection the ILP discarded, so probability-table entries containing that id are
      dropped (as in x138, new nodes carry no learned probability).
  (3) Gap fill — notebook, right after recover_strict_gap2 in filter_output_graph, before safe-division (same place as
      in x138)
      BIOHUB_GAPFILL_MAX_GAP=<g>  turns it on if >=1 (x138 = 3). _MIN_SCORE 0.5, _STEP_UM 5.0, _PEAK_RADIUS_UM 3.5,
      _EXCLUDE_UM 2.0, _ALLOW_SYNTHETIC 0, _CONTEXT 1, _MAX_ADDED_FRAC 0.03
      Bridges a track end (t) to a track start (t+g+1) with a chain of low-threshold peaks (Hungarian, shorter gaps
      first, added nodes capped at 3 % of the node count).
  (4) Flow relink — notebook; at the entry of motion_relink_edges, branches to x138's flow version
      BIOHUB_MOTION_RELINK_FLOW_MODE=seed|prev (x138 = seed; off/empty = the original N05 function)
      _K 12, _RADIUS_UM 40, _EXCLUDE_UM 1.5, _MIN_SAMPLES 4, _GATE 1, _ITER 1, _SEED_GATE_UM 0, _RAW_ADMIT 1,
      _Z_WEIGHT 1.0, _RAW_COST 0, _TIGHT_UM 7.0, _RELAXED_UM 0
      Per frame, confident tight seed matches build a neighbour displacement field (median of k-NN displacements), and
      the sources are assigned again against the positions that field predicts. x138's function body verbatim + the
      one-line `cost -= w·p` of relink_prob (our relink kept).
      x138's BIOHUB_MOTION_RELINK_TIGHT_UM=5.5 is not ported (already tested on the LB: 0.953 vs 0.954; excluded as
      instructed). So the seed gate (SEED_GATE_UM=0 → MOTION_RELINK_TIGHT_UM) is N05's 6.0, not x138's 5.5.
  (6) Fast relink — BIOHUB_X138_FAST_RELINK=1 (our knob; x138 does not have it)
      Even with flow off, routes motion_relink_edges to x138's version of the function (flow off, RAW_COST default
      0.05). That version gathers candidate pairs with a ball query on a target cKDTree (N05 runs a Python double loop
      over all source×target pairs per frame); the admitted pair set (raw <= gate), the cost matrix and the Hungarian
      input are the same as N05's, so the assignment is the same — only the run time drops.
      tests/test_x138_port_946.py checks edge identity with the N05 function on synthetic data; in a commit run, the
      check is that the first re-link's RELINK_PROB_CHANGED line (edges_w0/edges_w/added/removed) matches the N05
      commit log.
      Added to offset the cost of readmit's extra re-link (12 h limit on the hidden run).
  (5) ILP timeout + stage timing — runtime script patch
      BIOHUB_ILP_TIMEOUT_S=<s> if >0, gives SCIP a timeout (x138 = 1200; the incumbent is returned at the limit) and
      prints the ILP time and the detection+edges time per dataset. Below the limit the output does not change.

Not ported
  The V1284 coordinate-regression head (its dataset `biohub-v1284-head-s075` was private at the time, HTTP 403, so the
  weights could not be obtained; ported later in v1284_946.py), MOTION_RELINK_TIGHT_UM 5.5, REPAIR_DEADLINE_S 27000 +
  fallback (with N05's measured 9–10 h run time it would switch off repair for every dataset after 7.5 h),
  FRAME_CACHE_MAX_FRAMES, skipping the /kaggle/input rglob (time only), removal of validator/PPSWEEP (N05 never had
  them).

Markers: X138 SCRIPT PATCH APPLIED (once, only when on)
         LOWDET <stem>: N peaks (per dataset)
         [<stem>] ILP <s>s (ILP knob)
         X138_PORT ds=.. (per dataset, when any notebook knob is on)
run_stats (only when on): motion_relink_flow_frames, motion_relink_flow_predicted, readmitted_nodes, gapfill_pool_peaks,
         gapfill_pool_excluded, gapfill_candidates, gapfill_pairs_g1..3, gapfill_peak_nodes, gapfill_added_nodes,
         gapfill_added_edges, gapfill_budget_hit, x138_relink_s, x138_readmit_s, x138_gapfill_s
Provenance: the public notebook code (Apache-2.0; untrusted data) was only read, never executed on our machine. The
function bodies are verbatim from the x138 version pulled on 2026-09-22. Ported parts keep the Apache-2.0 license;
see NOTICE. The rest of this file is ours (MIT).
"""
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("relink_prob_946", _Path(__file__).with_name("relink_prob_946.py"))
_rp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rp)

LOWDET_DIR_DEFAULT = "/tmp/biohub_x138_lowdet"

# ---------------------------------------------------------------------------------------------- notebook helper
# x138 cell 2 (knob constants) + cell 5 (functions), verbatim; the flow relink is x138's motion_relink_edges renamed,
# plus relink_prob's bonus line right after its cost line.
_KNOBS = '''
MOTION_RELINK_FLOW_MODE = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_MODE", "off").strip().lower()
MOTION_RELINK_FLOW_K = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_K", "8"))
MOTION_RELINK_FLOW_RADIUS_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM", "25.0"))
MOTION_RELINK_FLOW_EXCLUDE_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM", "1.5"))
MOTION_RELINK_FLOW_MIN_SAMPLES = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES", "4"))
MOTION_RELINK_FLOW_GATE = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_GATE", "0").strip() == "1"
MOTION_RELINK_FLOW_ITER = int(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_ITER", "1"))
MOTION_RELINK_FLOW_SEED_GATE_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM", "0"))
MOTION_RELINK_FLOW_RAW_ADMIT = os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT", "1").strip() != "0"
MOTION_RELINK_FLOW_Z_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT", "1.0"))
MOTION_RELINK_FLOW_RAW_COST = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RAW_COST", "0.05"))
MOTION_RELINK_FLOW_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM", "0"))
MOTION_RELINK_FLOW_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM", "0"))
READMIT_RADIUS_UM = float(os.environ.get("BIOHUB_READMIT_RADIUS_UM", "0"))
READMIT_MIN_SCORE = float(os.environ.get("BIOHUB_READMIT_MIN_SCORE", "0.965"))
GAPFILL_MAX_GAP = int(os.environ.get("BIOHUB_GAPFILL_MAX_GAP", "0"))
GAPFILL_MIN_SCORE = float(os.environ.get("BIOHUB_GAPFILL_MIN_SCORE", "0.5"))
GAPFILL_STEP_UM = float(os.environ.get("BIOHUB_GAPFILL_STEP_UM", "5.0"))
GAPFILL_PEAK_RADIUS_UM = float(os.environ.get("BIOHUB_GAPFILL_PEAK_RADIUS_UM", "3.5"))
GAPFILL_EXCLUDE_UM = float(os.environ.get("BIOHUB_GAPFILL_EXCLUDE_UM", "2.0"))
GAPFILL_ALLOW_SYNTHETIC = int(os.environ.get("BIOHUB_GAPFILL_ALLOW_SYNTHETIC", "0"))
GAPFILL_CONTEXT = os.environ.get("BIOHUB_GAPFILL_CONTEXT", "1") != "0"
GAPFILL_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAPFILL_MAX_ADDED_FRAC", "0.03"))
_X138_FAST_RELINK = os.environ.get('BIOHUB_X138_FAST_RELINK', '0').strip() == '1'
_X138_FLOW_ON = MOTION_RELINK_FLOW_MODE in ('seed', 'prev')
_X138_READMIT_ON = READMIT_RADIUS_UM > 0
_X138_GAPFILL_ON = GAPFILL_MAX_GAP >= 1
_X138_ANY_ON = _X138_FLOW_ON or _X138_READMIT_ON or _X138_GAPFILL_ON or _X138_FAST_RELINK
'''

_FUNCS = r'''def _gapfill_bump(stats: dict[str, int], key: str, n: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + n


def load_low_detections(
    nodes_by_id: dict[int, dict[str, object]],
    dataset: str | None,
    stats: dict[str, int],
) -> dict[int, dict[str, np.ndarray]] | None:
    """Per-frame pool of the detector's sub-threshold peaks from the prediction cell's dump.

    Peaks below GAPFILL_MIN_SCORE and peaks within GAPFILL_EXCLUDE_UM of a node
    of their frame (the node set's own peaks among them) are dropped. None when
    the dump is missing or unreadable, and the filler then does nothing.
    """
    cache_dir = os.environ.get("BIOHUB_CACHE_DIR", "").strip()
    if GAPFILL_MAX_GAP < 1 or not cache_dir or not dataset:
        return None
    cache_path = Path(cache_dir) / f"{dataset}.npz"
    if not cache_path.exists():
        print(f"  [{dataset}] no low-detection dump at {cache_path}; gap filler idle")
        return None
    try:
        with np.load(cache_path) as cz:
            if "low_coords" not in cz.files:
                print(f"  [{dataset}] dump has no low_coords; gap filler idle")
                return None
            low = np.asarray(cz["low_coords"], dtype=np.float64).reshape(-1, 4)
            score = np.asarray(cz["low_score"], dtype=np.float64).reshape(-1)
    except Exception as exc:
        print(f"  [{dataset}] low-detection dump unreadable ({type(exc).__name__}: {exc}); gap filler idle")
        return None
    return build_low_detection_pool(nodes_by_id, low, score, stats, dataset)


def build_low_detection_pool(
    nodes_by_id: dict[int, dict[str, object]],
    low: np.ndarray,
    score: np.ndarray,
    stats: dict[str, int],
    dataset: str | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    keep = score >= GAPFILL_MIN_SCORE
    low, score = low[keep], score[keep]
    scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
    node_um_by_t: dict[int, list] = {}
    for node in nodes_by_id.values():
        node_um_by_t.setdefault(int(node["t"]), []).append(np.array(node_point(node), dtype=np.float64) * scale)
    pool: dict[int, dict[str, np.ndarray]] = {}
    excluded = 0
    for t in np.unique(low[:, 0]).astype(int).tolist() if len(low) else []:
        sel = low[:, 0] == t
        vox = low[sel, 1:]
        um = vox * scale
        sc = score[sel]
        existing = node_um_by_t.get(t)
        if existing:
            d, _ = cKDTree(np.stack(existing)).query(um, k=1)
            free = d > GAPFILL_EXCLUDE_UM
            excluded += int((~free).sum())
            vox, um, sc = vox[free], um[free], sc[free]
        if len(vox):
            pool[t] = {"vox": vox, "um": um, "score": sc}
    n_free = int(sum(len(p["vox"]) for p in pool.values()))
    _gapfill_bump(stats, "gapfill_pool_peaks", n_free)
    _gapfill_bump(stats, "gapfill_pool_excluded", excluded)
    if dataset is not None:
        print(f"  [{dataset}] low-detection pool: {len(low)} peaks >= {GAPFILL_MIN_SCORE}, "
              f"{excluded} on existing nodes, {n_free} free")
    return pool


def readmit_discarded_detections(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> dict[int, dict[str, object]]:
    """Add detector peaks the ILP discarded that sit next to an open track end or start.

    A free peak (load_low_detections: not within GAPFILL_EXCLUDE_UM of a node)
    scoring at least READMIT_MIN_SCORE comes back when a node at t-1 with no
    outgoing edge, or one at t+1 with no incoming edge, is within
    READMIT_RADIUS_UM. ``edges`` are the first re-link's; the caller re-links
    afterwards, which places the new nodes or leaves them isolated for the
    prune. Non-fatal: any failure leaves the node set as it was.
    """
    if READMIT_RADIUS_UM <= 0:
        return nodes_by_id
    try:
        pool = load_low_detections(nodes_by_id, dataset, stats)
        if not pool:
            return nodes_by_id
        has_out = {int(e["source_id"]) for e in edges}
        has_in = {int(e["target_id"]) for e in edges}
        scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
        anchors: dict[int, list] = {}
        for nid, node in nodes_by_id.items():
            t = int(node["t"])
            um = np.array(node_point(node), dtype=np.float64) * scale
            if nid not in has_out:
                anchors.setdefault(t + 1, []).append(um)
            if nid not in has_in:
                anchors.setdefault(t - 1, []).append(um)
        next_id = _next_node_id(nodes_by_id)
        added = 0
        for t in sorted(pool):
            near = anchors.get(int(t))
            if not near:
                continue
            peaks = pool[t]
            d, _ = cKDTree(np.stack(near)).query(peaks["um"], k=1)
            keep = (d <= READMIT_RADIUS_UM) & (peaks["score"] >= READMIT_MIN_SCORE)
            for vox in peaks["vox"][keep]:
                nodes_by_id[next_id] = {"node_id": next_id, "t": int(t), "z": float(vox[0]), "y": float(vox[1]),
                                        "x": float(vox[2]), "readmitted": 1}
                next_id += 1
                added += 1
        _gapfill_bump(stats, "readmitted_nodes", added)
        if dataset is not None:
            print(f"  [{dataset}] readmitted {added} discarded detections within {READMIT_RADIUS_UM} um of an open end/start")
    except Exception as exc:
        print(f"  [{dataset}] readmit skipped (non-fatal): {type(exc).__name__}: {exc}")
    return nodes_by_id


def fill_gaps_from_low_detections(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    pool: dict[int, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Bridge a track end at t to a track start at t+g+1 through sub-threshold peaks.

    Runs after the single-frame closer and gap2, on what they left open. For
    each candidate pair (span within GAPFILL_STEP_UM per frame, gap2's
    direction test when a neighbour exists) the straight line from end to
    start is sampled at each missing frame and the nearest free peak within
    GAPFILL_PEAK_RADIUS_UM of the sample is taken; a bridge needs a peak at
    every frame except at most GAPFILL_ALLOW_SYNTHETIC of them, which get an
    interpolated node refined against the frame like the closer's midpoints.
    The pairs of one (t, g) are assigned by Hungarian on span/(g+1) plus the
    mean peak deviation, shorter gaps first. Added nodes are capped at
    GAPFILL_MAX_ADDED_FRAC of the node set.
    """
    if GAPFILL_MAX_GAP < 1 or not edges or not nodes_by_id:
        return nodes_by_id, edges
    if pool is None:
        pool = load_low_detections(nodes_by_id, dataset, stats)
    if not pool:
        return nodes_by_id, edges
    scale = np.array(VOXEL_SCALE_UM, dtype=np.float64)
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
        incoming.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    pos = {nid: np.array(node_point(node), dtype=np.float64) * scale for nid, node in nodes_by_id.items()}
    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for nid, node in nodes_by_id.items():
        t = int(node["t"])
        if nid not in outgoing:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in incoming:
            starts_by_t.setdefault(t, []).append(nid)
    trees = {t: cKDTree(p["um"]) for t, p in pool.items()}
    used_peak = {t: np.zeros(len(p["um"]), dtype=bool) for t, p in pool.items()}
    budget = int(round(len(nodes_by_id) * GAPFILL_MAX_ADDED_FRAC))
    frame_cache = frame_cache if frame_cache is not None else {}
    next_id = _next_node_id(nodes_by_id)
    used_end: set[int] = set()
    used_start: set[int] = set()
    added_nodes = 0
    new_edges: list[dict[str, object]] = []

    def context_ok(end_id: int, start_id: int) -> bool:
        if not GAPFILL_CONTEXT:
            return True
        step = pos[start_id] - pos[end_id]
        sn = float(np.linalg.norm(step))
        if sn <= 0.01:
            return True
        prev = incoming.get(end_id)
        if prev:
            other = pos[end_id] - pos[prev[0]]
            on = float(np.linalg.norm(other))
            if on > 0.01 and float(np.dot(other, step)) / (on * sn) <= -0.25:
                return False
        nxt = outgoing.get(start_id)
        if nxt:
            other = pos[nxt[0]] - pos[start_id]
            on = float(np.linalg.norm(other))
            if on > 0.01 and float(np.dot(other, step)) / (on * sn) <= -0.25:
                return False
        return True

    def chain_for(end_id: int, start_id: int, g: int):
        span = pos[start_id] - pos[end_id]
        t0 = int(nodes_by_id[end_id]["t"])
        items, dev, synthetic = [], [], 0
        for k in range(1, g + 1):
            q = pos[end_id] + span * (k / (g + 1))
            tk = t0 + k
            tree = trees.get(tk)
            j = None
            if tree is not None:
                cand = [c for c in tree.query_ball_point(q, r=GAPFILL_PEAK_RADIUS_UM) if not used_peak[tk][c]]
                if cand:
                    dists = np.linalg.norm(pool[tk]["um"][cand] - q, axis=1)
                    best = int(np.argmin(dists))
                    j = cand[best]
                    dev.append(float(dists[best]))
            if j is None:
                synthetic += 1
                if synthetic > GAPFILL_ALLOW_SYNTHETIC:
                    return None
                dev.append(GAPFILL_PEAK_RADIUS_UM)
            items.append((tk, j, q))
        cost = float(np.linalg.norm(span)) / (g + 1) + (sum(dev) / len(dev) if dev else 0.0)
        return cost, items

    for g in range(1, GAPFILL_MAX_GAP + 1):
        gate = GAPFILL_STEP_UM * (g + 1)
        for t in sorted(ends_by_t):
            if added_nodes + g > budget:
                _gapfill_bump(stats, "gapfill_budget_hit")
                break
            ends = [e for e in ends_by_t[t] if e not in used_end]
            starts = [s for s in starts_by_t.get(t + g + 1, []) if s not in used_start]
            if not ends or not starts:
                continue
            end_pts = np.stack([pos[e] for e in ends])
            start_pts = np.stack([pos[s] for s in starts])
            start_tree = cKDTree(start_pts)
            cost = np.full((len(ends), len(starts)), np.inf)
            n_chains = 0
            for i, js in enumerate(start_tree.query_ball_point(end_pts, r=gate)):
                for j in js:
                    if not context_ok(ends[i], starts[j]):
                        continue
                    chain = chain_for(ends[i], starts[j], g)
                    if chain is None:
                        continue
                    cost[i, j] = chain[0]
                    n_chains += 1
            if n_chains == 0:
                continue
            _gapfill_bump(stats, "gapfill_candidates", n_chains)
            finite = np.isfinite(cost)
            big = float(np.max(cost[finite])) * 1000.0 + 1.0
            row_ind, col_ind = linear_sum_assignment(np.where(finite, cost, big))
            picks = sorted((float(cost[i, j]), int(i), int(j)) for i, j in zip(row_ind, col_ind) if finite[i, j])
            for _, i, j in picks:
                if added_nodes + g > budget:
                    _gapfill_bump(stats, "gapfill_budget_hit")
                    break
                chain = chain_for(ends[i], starts[j], g)  # an earlier pick may have taken a peak
                if chain is None:
                    continue
                prev = ends[i]
                for tk, pk, q in chain[1]:
                    nid = next_id
                    next_id += 1
                    if pk is not None:
                        vox = pool[tk]["vox"][pk]
                        used_peak[tk][pk] = True
                        node = {"node_id": nid, "t": tk, "z": float(vox[0]), "y": float(vox[1]), "x": float(vox[2]), "gapfill_peak": 1}
                        _gapfill_bump(stats, "gapfill_peak_nodes")
                    else:
                        p = q / scale
                        refined = refine_synthetic_midpoint(dataset, tk, (float(p[0]), float(p[1]), float(p[2])), frame_cache, stats)
                        node = {"node_id": nid, "t": tk, "z": float(refined[0]), "y": float(refined[1]), "x": float(refined[2]), "gap_synthetic": 1}
                        _gapfill_bump(stats, "gapfill_synthetic_nodes")
                    nodes_by_id[nid] = node
                    new_edges.append({
                        "source_id": prev, "target_id": nid, "edge_prob": None,
                        "distance_um": edge_distance_um(nodes_by_id[prev], node), "gap_filled": 1,
                    })
                    prev = nid
                    added_nodes += 1
                new_edges.append({
                    "source_id": prev, "target_id": starts[j], "edge_prob": None,
                    "distance_um": edge_distance_um(nodes_by_id[prev], nodes_by_id[starts[j]]), "gap_filled": 1,
                })
                used_end.add(ends[i])
                used_start.add(starts[j])
                _gapfill_bump(stats, f"gapfill_pairs_g{g}")
    _gapfill_bump(stats, "gapfill_added_nodes", added_nodes)
    _gapfill_bump(stats, "gapfill_added_edges", len(new_edges))
    if dataset is not None and new_edges:
        print(f"  [{dataset}] gap filler: +{added_nodes} nodes, +{len(new_edges)} edges")
    return nodes_by_id, [*edges, *new_edges]
'''

_FLOW = r'''def _x138_flow_motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def _flow_predictor(flow_src, flow_disp):
        """Median displacement of the nearest flow samples, or None.

        Cells move with their neighbours, so a displacement field sampled
        from confident links predicts a source's next position better than
        its own last step (experiments/motion_model_audit.py).
        """
        if len(flow_src) < MOTION_RELINK_FLOW_MIN_SAMPLES:
            return None
        flow_src = np.asarray(flow_src, dtype=np.float64)
        flow_disp = np.asarray(flow_disp, dtype=np.float64)
        tree = cKDTree(flow_src)
        k_query = min(MOTION_RELINK_FLOW_K + 1, len(flow_src))

        def predict(source_pos, exclude_um):
            dist, idx = tree.query(source_pos, k=k_query, distance_upper_bound=MOTION_RELINK_FLOW_RADIUS_UM)
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            keep = np.isfinite(dist) & (dist >= exclude_um)
            idx = idx[keep][:MOTION_RELINK_FLOW_K]
            if idx.size == 0:
                return None
            return np.median(flow_disp[idx], axis=0)

        return predict

    # The flow residual is weighted along z when asked; the raw distance never is.
    flow_weight = np.array([MOTION_RELINK_FLOW_Z_WEIGHT, 1.0, 1.0], dtype=np.float64)
    flow_anisotropic = MOTION_RELINK_FLOW_Z_WEIGHT != 1.0

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
        flow=None,
        flow_exclude_um: float = 0.0,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        target_arr = np.stack([position_um[target_id] for target_id in target_ids])
        source_arr = np.stack([position_um[source_id] for source_id in source_ids])
        predicted_arr = np.empty_like(source_arr)
        flow_hit = np.zeros(len(source_ids), dtype=bool)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            flow_step = flow(source_pos, flow_exclude_um) if flow is not None else None
            if flow_step is not None:
                predicted_arr[i] = source_pos + flow_step
                flow_hit[i] = True
                stats["motion_relink_flow_predicted"] = stats.get("motion_relink_flow_predicted", 0) + 1
            elif prev_pos is None:
                predicted_arr[i] = source_pos
            else:
                predicted_arr[i] = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
        target_tree = cKDTree(target_arr)
        gate_radius = gate_um * (1.0 + 1e-9) + 1e-9
        gate_neighbours = target_tree.query_ball_point(source_arr, r=gate_radius)
        # With a flow prior, a pair is also admitted when the target sits
        # within the gate of the *predicted* position: a cell moving 8 um with
        # its neighbours can then compete in the tight pass instead of waiting
        # for the relaxed one where slower cells have already taken its target.
        # With RAW_ADMIT off, that is the only admission for flow-predicted
        # sources; sources without a field sample keep the raw gate.
        flow_gated = flow is not None and MOTION_RELINK_FLOW_GATE
        if flow_gated:
            predicted_radius = gate_radius / MOTION_RELINK_FLOW_Z_WEIGHT if flow_anisotropic else gate_radius
            around_predicted = target_tree.query_ball_point(predicted_arr, r=predicted_radius)
            gate_neighbours = [
                sorted(set(near_source) | set(near_predicted))
                for near_source, near_predicted in zip(gate_neighbours, around_predicted)
            ]
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            predicted = predicted_arr[i]
            raw_admits = MOTION_RELINK_FLOW_RAW_ADMIT or not (flow_gated and flow_hit[i])
            for j in sorted(gate_neighbours[i]):
                target_id = target_ids[j]
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if flow_anisotropic:
                    motion = float(np.linalg.norm((target_pos - predicted) * flow_weight))
                else:
                    motion = float(np.linalg.norm(target_pos - predicted))
                if not ((raw_admits and raw <= gate_um) or (flow_gated and motion <= gate_um)):
                    continue
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + MOTION_RELINK_FLOW_RAW_COST * raw - MOTION_RELINK_LEARNED_BONUS * prob
                if _RP_W != 0.0:  # relink_prob_946 bonus (N05 relink kept)
                    cost[i, j] -= _RP_W * _RP_TABLE.get((source_id, target_id), 0.0)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    def _field_from(matches):
        return _flow_predictor(
            [position_um[m[0]] for m in matches],
            [position_um[m[1]] - position_um[m[0]] for m in matches],
        )

    seed_gate_um = MOTION_RELINK_FLOW_SEED_GATE_UM if MOTION_RELINK_FLOW_SEED_GATE_UM > 0 else MOTION_RELINK_TIGHT_UM
    flow_tight_um = MOTION_RELINK_FLOW_TIGHT_UM if MOTION_RELINK_FLOW_TIGHT_UM > 0 else MOTION_RELINK_TIGHT_UM
    flow_relaxed_um = MOTION_RELINK_FLOW_RELAXED_UM if MOTION_RELINK_FLOW_RELAXED_UM > 0 else MOTION_RELINK_RELAXED_UM

    times = sorted(ids_by_t)
    previous_flow = None
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        flow = None
        flow_exclude_um = 0.0
        if MOTION_RELINK_FLOW_MODE == "prev":
            flow = previous_flow
        elif MOTION_RELINK_FLOW_MODE == "seed":
            # Confident tight matches first, then everything is assigned
            # against the field they define. A source's own seed match is
            # excluded from its sample so a wrong seed cannot vote for itself.
            seed = assign_pass(source_ids, target_ids, seed_gate_um, previous_flow)
            flow = _field_from(seed)
            flow_exclude_um = MOTION_RELINK_FLOW_EXCLUDE_UM
            if flow is not None:
                stats["motion_relink_flow_frames"] = stats.get("motion_relink_flow_frames", 0) + 1
        rounds = MOTION_RELINK_FLOW_ITER if MOTION_RELINK_FLOW_MODE != "off" else 1
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for round_index in range(max(1, rounds)):
            if round_index > 0:
                # Refine: the field from the previous round's final matches,
                # then assign every source again against it.
                refined = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])
                if refined is None:
                    break
                flow = refined
                flow_exclude_um = MOTION_RELINK_FLOW_EXCLUDE_UM
            unmatched_sources = set(source_ids)
            unmatched_targets = set(target_ids)
            frame_matches = []
            passes = (("tight", flow_tight_um), ("relaxed", flow_relaxed_um)) if flow is not None else (
                ("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM))
            for pass_name, gate_um in passes:
                pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
                pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
                matches = assign_pass(pass_sources, pass_targets, gate_um, flow, flow_exclude_um)
                for source_id, target_id, raw, motion, prob in matches:
                    if source_id not in unmatched_sources or target_id not in unmatched_targets:
                        continue
                    unmatched_sources.remove(source_id)
                    unmatched_targets.remove(target_id)
                    frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            if pass_name == "tight":
                stats["motion_relink_tight_edges"] += 1
            else:
                stats["motion_relink_relaxed_edges"] += 1
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        if MOTION_RELINK_FLOW_MODE != "off":
            previous_flow = _field_from([(s, g) for s, g, _r, _m, _n, _p in frame_matches])
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges
'''

_RELINK_ONCE = '''

def _x138_relink_once(nodes_by_id, stats, learned_edge_probs, dataset):
    """re-link after readmit: relink_prob bonus included, WITHOUT relink_prob's diagnostic w=0 reference pass.
    Probability-table entries whose src/tgt id is a readmitted node are dropped: _next_node_id can hand out an id that
    equals the index of a detection the ILP discarded, and a new node must not inherit that detection's probabilities."""
    global _RP_TABLE

    if _RP_W == 0.0:
        return motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    table, n_pairs, _gate = _rp_load(dataset)
    fresh = {int(nid) for nid, node in nodes_by_id.items() if node.get('readmitted')}
    _RP_TABLE = {k: v for k, v in table.items() if k[0] not in fresh and k[1] not in fresh} if fresh else table
    print(f'X138_RELINK_ONCE dataset={dataset} w={_RP_W:g} pairs={n_pairs} dropped_fresh={n_pairs - len(_RP_TABLE)} '
          f'nodes={len(nodes_by_id)}', flush = True)

    try:
        return motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    finally:
        _RP_TABLE = {}
'''

HELPER = ("# ==== x138 public-lineage port (kaggle/x138_port_946.py; knobs default off) ==========\n"
          + _KNOBS + "\n\n" + _FUNCS + "\n\n" + _FLOW + _RELINK_ONCE
          + "# ==== end x138 public-lineage port =======================================================\n\n")

_DEF_OLD = "def motion_relink_edges("

_SIG = ("def motion_relink_edges(nodes_by_id: dict[int, dict[str, object]], stats: dict[str, int], "
        "learned_edge_probs: dict[tuple[int, int], float] | None = None) -> list[dict[str, object]]:\n")
_ENTRY_OLD = _SIG + "    if not OUTPUT_MOTION_RELINK or not nodes_by_id:\n"
_ENTRY_NEW = (_SIG +
              "    if _X138_FLOW_ON or _X138_FAST_RELINK:\n"
              "        return _x138_flow_motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n"
              "\n"
              "    if not OUTPUT_MOTION_RELINK or not nodes_by_id:\n")

# (2) readmit + relink timing, right after the first re-link (the call site relink_prob_946 already rewrote)
_CALL_OLD = "        motion_edges = _rp_motion_relink(nodes_by_id, stats, learned_edge_probs, dataset)\n"
_CALL_NEW = ("        _x1_t0 = time.time()\n" + _CALL_OLD +
             "\n"
             "        if _X138_ANY_ON:\n"
             "            stats['x138_relink_s'] = round(time.time() - _x1_t0, 1)\n"
             "\n"
             "        if _X138_READMIT_ON and motion_edges:\n"
             "            _x1_t0 = time.time()\n"
             "            _x1_before = len(nodes_by_id)\n"
             "            nodes_by_id = readmit_discarded_detections(nodes_by_id, motion_edges, stats, dataset = dataset)\n"
             "\n"
             "            if len(nodes_by_id) > _x1_before:\n"
             "                motion_edges = _x138_relink_once(nodes_by_id, stats, learned_edge_probs, dataset) or motion_edges\n"
             "            stats['x138_readmit_s'] = round(time.time() - _x1_t0, 1)\n")

# (3) gap fill right after gap2, before the safe-division repair
_GAP_OLD = "    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset = dataset)\n"
_GAP_NEW = (_GAP_OLD +
            "\n"
            "    if _X138_GAPFILL_ON:\n"
            "        _x1_t0 = time.time()\n"
            "        nodes_by_id, edges = fill_gaps_from_low_detections(nodes_by_id, edges, stats, dataset = dataset, frame_cache = repair_frame_cache)\n"
            "        stats['x138_gapfill_s'] = round(time.time() - _x1_t0, 1)\n"
            "        print(f'[{dataset}] after x138 low-detection gap filler: {len(nodes_by_id)} nodes, {len(edges)} edges', flush = True)\n")

# per-dataset marker, right before FINAL
_MARK_OLD = "    nodes_by_id, edges = node_select_final_graph(nodes_by_id, edges, stats, dataset)\n"
_MARK_NEW = (_MARK_OLD +
             "\n"
             "    if _X138_ANY_ON:\n"
             "        print(f\"X138_PORT ds={dataset} flow={MOTION_RELINK_FLOW_MODE if _X138_FLOW_ON else 'off'} \"\n"
             "              f\"flow_frames={stats.get('motion_relink_flow_frames', 0)} flow_predicted={stats.get('motion_relink_flow_predicted', 0)} \"\n"
             "              f\"readmitted={stats.get('readmitted_nodes', 0)} pool_peaks={stats.get('gapfill_pool_peaks', 0)} \"\n"
             "              f\"gapfill_candidates={stats.get('gapfill_candidates', 0)} gapfill_pairs=\"\n"
             "              f\"{stats.get('gapfill_pairs_g1', 0)}/{stats.get('gapfill_pairs_g2', 0)}/{stats.get('gapfill_pairs_g3', 0)} \"\n"
             "              f\"gapfill_nodes={stats.get('gapfill_added_nodes', 0)} gapfill_edges={stats.get('gapfill_added_edges', 0)} \"\n"
             "              f\"budget_hit={stats.get('gapfill_budget_hit', 0)} t_relink={stats.get('x138_relink_s', 0)} \"\n"
             "              f\"t_readmit={stats.get('x138_readmit_s', 0)} t_gapfill={stats.get('x138_gapfill_s', 0)} \"\n"
             "              f\"final_nodes={len(nodes_by_id)} final_edges={len(edges)}\", flush = True)\n")

NB_PAIRS = [("x138_helper", _DEF_OLD, HELPER + _DEF_OLD),
            ("x138_flow_entry", _ENTRY_OLD, _ENTRY_NEW),
            ("x138_readmit", _CALL_OLD, _CALL_NEW),
            ("x138_gapfill", _GAP_OLD, _GAP_NEW),
            ("x138_marker", _MARK_OLD, _MARK_NEW)]

# ---------------------------------------------------------------------------------------------- script (runtime) pairs
# anchors = the N05 as-run scripts/predict_unet_transformer.py (tertiary + relink_prob + node_select applied),
# each checked to occur exactly once in the N05 commit-run output tracking_repo (2026-09-22).
_LD_GLOBALS_OLD = "@torch.no_grad()\ndef predict_video("
_LD_GLOBALS_NEW = ("_LOWDET_THRESHOLD = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0)\n"
                   "_LOWDET_DIR = os.environ.get('BIOHUB_CACHE_DIR', '').strip()\n"
                   "_LOWDET: list = []\n"
                   "\n"
                   "\n" + _LD_GLOBALS_OLD)
# x138 'lowdet peaks' pair, verbatim
_LD_PEAKS_OLD = ("                arr = _detect_cells_pooled(\n"
                 "                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n"
                 "                )\n"
                 "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n")
_LD_PEAKS_NEW = ("                arr = _detect_cells_pooled(\n"
                 "                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n"
                 "                )\n"
                 "                if _LOWDET_THRESHOLD > 0:\n"
                 "                    _low = _detect_cells_pooled(det_logits[f_idx][0], t, _LOWDET_THRESHOLD, pool_k)\n"
                 "                    if len(_low):\n"
                 "                        _lg = det_logits[f_idx][0][0]\n"
                 "                        _lz = torch.as_tensor(_low[:, 1:].astype(np.int64), device=_lg.device)\n"
                 "                        _lsc = torch.sigmoid(_lg[_lz[:, 0], _lz[:, 1], _lz[:, 2]]).float().cpu().numpy()\n"
                 "                        _LOWDET.append((_low.astype(np.float32), _lsc.astype(np.float32)))\n"
                 "                coord_offset[t] = (global_node_count, global_node_count + len(arr))\n")
# x138 'lowdet write' (without the always-empty edge cache), before build_graph in predict()
_LD_WRITE_OLD = "        graph = build_graph(coords, edges)\n"
_LD_WRITE_NEW = ("        if _LOWDET_DIR and _LOWDET_THRESHOLD > 0:\n"
                 "            _ld_dir = Path(_LOWDET_DIR)\n"
                 "            _ld_dir.mkdir(parents=True, exist_ok=True)\n"
                 "            if _LOWDET:\n"
                 "                _lc = np.concatenate([e[0] for e in _LOWDET]).astype(np.float32)\n"
                 "                _lc[:, 1:] *= np.array(downsample, dtype=np.float32)\n"
                 "                _lc = _lc.astype(np.int16)\n"
                 "                _lsc = np.concatenate([e[1] for e in _LOWDET]).astype(np.float32)\n"
                 "            else:\n"
                 "                _lc = np.empty((0, 4), np.int16)\n"
                 "                _lsc = np.empty(0, np.float32)\n"
                 "            _LOWDET.clear()\n"
                 "            np.savez_compressed(_ld_dir / f'{ds_path.stem}.npz', coords=coords, low_coords=_lc, low_score=_lsc)\n"
                 "            print(f'LOWDET {ds_path.stem}: {len(_lc)} peaks above {_LOWDET_THRESHOLD} nodes={len(coords)}', flush=True)\n"
                 + _LD_WRITE_OLD)
LOWDET_PAIRS = [("x138_lowdet_globals", _LD_GLOBALS_OLD, _LD_GLOBALS_NEW),
                ("x138_lowdet_peaks", _LD_PEAKS_OLD, _LD_PEAKS_NEW),
                ("x138_lowdet_write", _LD_WRITE_OLD, _LD_WRITE_NEW)]

# x138 cell 4 ILP timeout + per-dataset timing, verbatim
_ILP_OLD = ('            solver = td.solvers.ILPSolver(\n'
            '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
            '                appearance_weight=cfg.ilp_appearance_weight,\n'
            '                disappearance_weight=cfg.ilp_disappearance_weight,\n'
            '                division_weight=cfg.ilp_division_weight,\n'
            '            )\n'
            '            with suppress_output():\n'
            '                graph = solver.solve(graph)\n')
_ILP_NEW = ('            import time as _ilp_time\n'
            '            _ilp_timeout = float(os.environ.get("BIOHUB_ILP_TIMEOUT_S", "0") or 0.0)\n'
            '            solver = td.solvers.ILPSolver(\n'
            '                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),\n'
            '                appearance_weight=cfg.ilp_appearance_weight,\n'
            '                disappearance_weight=cfg.ilp_disappearance_weight,\n'
            '                division_weight=cfg.ilp_division_weight,\n'
            '                timeout=_ilp_timeout if _ilp_timeout > 0 else None,\n'
            '            )\n'
            '            _ilp_t0 = _ilp_time.time()\n'
            '            _ilp_nodes, _ilp_edges = graph.num_nodes(), graph.num_edges()\n'
            '            with suppress_output():\n'
            '                graph = solver.solve(graph)\n'
            '            print(\n'
            '                f"[{name}] ILP {_ilp_time.time() - _ilp_t0:.1f}s"\n'
            '                f" | candidate nodes={_ilp_nodes} edges={_ilp_edges}"\n'
            '                f" | timeout={_ilp_timeout if _ilp_timeout > 0 else None}",\n'
            '                flush=True,\n'
            '            )\n')
_PV_OLD = "        coords, edges = predict_video(\n"
_PV_NEW = "        import time as _pv_time\n        _pv_t0 = _pv_time.time()\n" + _PV_OLD
_BG_OLD = "        graph = build_graph(coords, edges)\n"
_BG_NEW = (_BG_OLD +
           '        print(f"[{name}] detection+edges {_pv_time.time() - _pv_t0:.1f}s | detections={len(coords)}", flush=True)\n')
ILP_PAIRS = [("x138_ilp_timeout", _ILP_OLD, _ILP_NEW),
             ("x138_timing_start", _PV_OLD, _PV_NEW),
             ("x138_timing_print", _BG_OLD, _BG_NEW)]

# notebook block that applies the script pairs at runtime. It is appended after node_select_946's block, so it runs
# after tertiary + relink_prob + node_select have patched _ps, i.e. on exactly the N05 as-run script text.
_NS_PRINT = "print('NODE_SELECT SCRIPT PATCH APPLIED (1 anchor)', flush=True)\n"
_RUNTIME_BLOCK = '''
# === lane-r: x138 port — low-detection dump + ILP timeout (kaggle/x138_port_946.py, runtime patch on _ps) ===
_x1_lowdet_thr = float(os.environ.get('BIOHUB_LOWDET_THRESHOLD', '0') or 0)
_x1_cache_dir = os.environ.get('BIOHUB_CACHE_DIR', '').strip()
_x1_ilp_timeout = float(os.environ.get('BIOHUB_ILP_TIMEOUT_S', '0') or 0)
_x1_need_dump = (int(os.environ.get('BIOHUB_GAPFILL_MAX_GAP', '0')) >= 1 or float(os.environ.get('BIOHUB_READMIT_RADIUS_UM', '0')) > 0)
if _x1_need_dump and not (_x1_lowdet_thr > 0 and _x1_cache_dir):
    raise RuntimeError('x138 port: BIOHUB_GAPFILL_*/BIOHUB_READMIT_* are on but the low-detection dump is off '
                       '(set BIOHUB_LOWDET_THRESHOLD > 0 and BIOHUB_CACHE_DIR)')
if float(os.environ.get('BIOHUB_READMIT_RADIUS_UM', '0')) > 0 and int(os.environ.get('BIOHUB_GAPFILL_MAX_GAP', '0')) < 1:
    raise RuntimeError('x138 port: readmit reads its pool through load_low_detections, which is idle unless BIOHUB_GAPFILL_MAX_GAP >= 1')
_x1_pairs = []
if _x1_lowdet_thr > 0 and _x1_cache_dir:
    _x1_pairs += %(lowdet_json)s
if _x1_ilp_timeout > 0:
    _x1_pairs += %(ilp_json)s
if _x1_pairs:
    _x1_src = _ps.read_text()
    for _x1_name, _x1_old, _x1_new in _x1_pairs:
        _x1_n = _x1_src.count(_x1_old)
        if _x1_n != 1:
            raise RuntimeError(f"x138 script anchor '{_x1_name}' expected exactly 1 occurrence, found {_x1_n}")
        _x1_src = _x1_src.replace(_x1_old, _x1_new, 1)
    compile(_x1_src, str(_ps), 'exec')
    _ps.write_text(_x1_src)
    for _x1_must in (['_LOWDET_THRESHOLD', "print(f'LOWDET "] if _x1_lowdet_thr > 0 and _x1_cache_dir else []) + (['_ilp_timeout', 'detection+edges'] if _x1_ilp_timeout > 0 else []):
        if _x1_must not in _ps.read_text():
            raise RuntimeError('x138 script patch did not persist marker: ' + _x1_must)
    print(f'X138 SCRIPT PATCH APPLIED ({len(_x1_pairs)} anchors) lowdet_threshold={_x1_lowdet_thr} cache_dir={_x1_cache_dir} '
          f'ilp_timeout_s={_x1_ilp_timeout}', flush=True)
''' % {"lowdet_json": _json.dumps([list(p) for p in LOWDET_PAIRS], ensure_ascii=False),
       "ilp_json": _json.dumps([list(p) for p in ILP_PAIRS], ensure_ascii=False)}

X138_PAIRS = NB_PAIRS + [("x138_runtime", _NS_PRINT, _NS_PRINT + _RUNTIME_BLOCK)]

PAIRS = list(_rp.PAIRS) + X138_PAIRS
