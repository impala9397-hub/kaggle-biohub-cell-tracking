"""relink_prob_946.py — adds a learned link-probability bonus to the cost of every candidate pair in the final
re-link, motion_relink_edges (NOTEBOOK_PATCH, off by default). N04 = N01 node selection + this bonus (w = 32).

Evidence (offline probe on our 40-video validation split (val40, two embryos), 2026-09-16)
  The motion_relink_edges cost is cost = motion + 0.05·raw − MOTION_RELINK_LEARNED_BONUS·prob, but prob exists
  only on ILP-solution edges; every other candidate is scored by geometry alone. The probe `col_w32` subtracted
  w·(column-softmax of the learned edge logit) from the cost of **every** candidate pair inside the 10 µm relaxed
  gate (cost −= w·p_col, w = 32; the cardinality-first tight 6 µm → relaxed 10 µm structure unchanged). On the val40
  C0 replica (our offline replica of the S57 post-processing): col_w32 alone +0.00204 (per embryo +0.00230 and
  +0.00199), edge tp/fp/fn 21270/1042/973; col_w32+N01 +0.00731, worse cross-embryo held-out direction +0.00667
  (N01 alone +0.00510); division term unchanged.

This port (source = the kernel's own blended column-softmax)
  The kernel builds the link-probability matrix for each frame pair inside predict_video of the subprocess (the
  organizer's scripts/predict_unet_transformer.py) as `probs = torch.softmax(raw, dim=0)` (n_src × n_tgt, after 8-view
  feature TTA, the secondary low-margin blend and bidirectional fusion have all been applied), passes only the
  candidates above threshold to the ILP as edges, and solves it. Only the edge_prob of ILP-solution edges reaches the
  notebook, so no candidate matrix exists at re-link time. Therefore:
  (1) Runtime script patch: right after probs is computed, collect (src, tgt, prob) inside the capture gate
      (1.5 × BIOHUB_MOTION_RELINK_RELAXED_UM = 15 µm) and save them to /tmp/biohub_relinkprob/<dataset>.npz, one
      file per dataset. src/tgt are predict_video's global detection indices = build_graph's node ids = the geff
      node_id (the ILP GraphView and save_graph preserve ids; checked locally on 2026-09-16). The association head is
      not run again.
  (2) Notebook: right after the cost line of motion_relink_edges, add `cost[i, j] −= w · p(src, tgt)` (only in-gate
      candidates reach that line). The call site in filter_output_graph is wrapped in _rp_motion_relink; when w > 0 it
      runs one extra re-link with w = 0 as a reference and counts the **edges whose assignment changed** (this
      reference pass doubles the time of the re-link stage).
  Difference from the probe: the probe's p is the column-softmax of the primary model alone on the identity view
  (probe cache); the kernel's p is the blended 8-view column-softmax of the deployed path (median correlation
  r = 0.890 between the two on ILP edges). The thresholds and the scale of w are kept as in the probe.

Knob (off by default = same behaviour as N01: no extra env line, log line or stats key)
  BIOHUB_RELINK_PROB_W=<float>   0 = off; N04 uses 32. If nonzero, the script captures and the notebook adds the bonus.
Markers: RELINK_PROB SCRIPT PATCH APPLIED (once)
         RELINK_PROB CAPTURE dataset=.. pairs=.. (once per dataset, from the shard that processes it)
         RELINK_PROB_CONFIG w=32 source=.. dataset=.. (notebook, once per dataset)
         RELINK_PROB_CHANGED dataset=.. edges_w0=.. edges_w=.. added=.. removed=.. (once per dataset)
run_stats (only when on): relink_prob_pairs, relink_prob_edges_w0, relink_prob_changed_added,
         relink_prob_changed_removed
The anchors are the notebook cell text after node_select_946 (→ div_dip_946) has been applied (its PAIRS are
prepended unchanged).
"""
import importlib.util as _ilu
import json as _json
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("node_select_946", _Path(__file__).with_name("node_select_946.py"))
_ns = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ns)

RP_DIR = "/tmp/biohub_relinkprob"
RP_SOURCE = "kernel_blended_column_softmax_all_candidates"

# ---------------------------------------------------------------------------------------------- notebook helper
HELPER = '''# ==== relink learned-probability bonus (kaggle/relink_prob_946.py) ==================
_RP_W = float(os.environ.get('BIOHUB_RELINK_PROB_W', '0'))
_RP_DIR = Path('%(rp_dir)s')
_RP_SOURCE = '%(rp_source)s'
_RP_TABLE = {}


def _rp_load(dataset):
    """(src, tgt, prob) captured by predict_video -> {(src, tgt): prob}. Fails if missing (never silently falls back to 0)."""
    path = _RP_DIR / f'{dataset}.npz'

    if not path.is_file():
        raise RuntimeError(f'BIOHUB_RELINK_PROB_W is on but the candidate probabilities were not captured: {path} missing')
    z = np.load(path)
    src = z['src'].astype(np.int64)
    tgt = z['tgt'].astype(np.int64)
    prob = z['prob'].astype(np.float64)
    table = {}

    for k in range(len(src)):
        table[(int(src[k]), int(tgt[k]))] = float(prob[k])
    return (table, len(src), float(z['gate_um']))


def _rp_motion_relink(nodes_by_id, stats, learned_edge_probs, dataset):
    """w == 0: the original call. w != 0: load the probability table, run one extra w = 0 reference re-link, count edges whose assignment changed."""
    global _RP_W, _RP_TABLE

    if _RP_W == 0.0:
        return motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    _RP_TABLE, n_pairs, gate_um = _rp_load(dataset)
    print(f'RELINK_PROB_CONFIG w={_RP_W:g} source={_RP_SOURCE} pairs={n_pairs} capture_gate_um={gate_um:g} '
          f'relaxed_um={MOTION_RELINK_RELAXED_UM} tight_um={MOTION_RELINK_TIGHT_UM} ilp_bonus={MOTION_RELINK_LEARNED_BONUS} '
          f'dataset={dataset}', flush = True)
    _w = _RP_W
    _RP_W = 0.0

    try:
        ref = motion_relink_edges(nodes_by_id, dict(stats), learned_edge_probs)
    finally:
        _RP_W = _w
    out = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
    e0 = set((int(e['source_id']), int(e['target_id'])) for e in ref)
    e1 = set((int(e['source_id']), int(e['target_id'])) for e in out)
    s0 = set(s for s, _t in e0)
    s1 = set(s for s, _t in e1)
    stats['relink_prob_pairs'] = n_pairs
    stats['relink_prob_edges_w0'] = len(e0)
    stats['relink_prob_changed_added'] = len(e1 - e0)
    stats['relink_prob_changed_removed'] = len(e0 - e1)
    print(f'RELINK_PROB_CHANGED dataset={dataset} edges_w0={len(e0)} edges_w={len(e1)} added={len(e1 - e0)} '
          f'removed={len(e0 - e1)} sources_gained={len(s1 - s0)} sources_lost={len(s0 - s1)} pairs={n_pairs}', flush = True)
    _RP_TABLE = {}
    return out
# ==== end relink learned-probability bonus =================================================

''' % {"rp_dir": RP_DIR, "rp_source": RP_SOURCE}

_DEF_OLD = "def motion_relink_edges("
_DEF_NEW = HELPER + _DEF_OLD

_COST_OLD = "                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob\n"
_COST_NEW = (_COST_OLD +
             "\n"
             "                if _RP_W != 0.0:\n"
             "                    cost[i, j] -= _RP_W * _RP_TABLE.get((source_id, target_id), 0.0)\n")

_CALL_OLD = "        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)\n"
_CALL_NEW = "        motion_edges = _rp_motion_relink(nodes_by_id, stats, learned_edge_probs, dataset)\n"

NB_PAIRS = [("relink_prob_helper", _DEF_OLD, _DEF_NEW),
            ("relink_prob_cost", _COST_OLD, _COST_NEW),
            ("relink_prob_call", _CALL_OLD, _CALL_NEW)]

# ---------------------------------------------------------------------------------------------- script (runtime) pairs
# anchors = the scripts/predict_unet_transformer.py text after the tertiary patch (each checked to occur once in the
# S86 as-run script).
_INIT_OLD = "    all_edges: list[tuple[int, int, float, float]] = []\n"
_INIT_NEW = (_INIT_OLD +
             "    _rp_on = os.environ.get('BIOHUB_RELINK_PROB_W', '0') not in ('', '0')\n"
             "    _rp_gate = 1.5 * float(os.environ.get('BIOHUB_MOTION_RELINK_RELAXED_UM', '10.0'))\n"
             "    _rp_vox = np.array(voxel_size, dtype = np.float32)\n"
             "    _rp_src, _rp_tgt, _rp_p, _rp_frames = [], [], [], 0\n")

_PROBS_OLD = ("            raw = edge_logits_pair[0]\n"
              "            if cfg.edge_activation == \"softmax\":\n"
              "                probs = torch.softmax(raw, dim=0).cpu().numpy()\n"
              "            else:\n"
              "                probs = torch.sigmoid(raw).cpu().numpy()\n")
_PROBS_NEW = (_PROBS_OLD +
              "\n"
              "            if _rp_on:\n"
              "                _rp_ps = c_src[:, 1:].astype(np.float32) * _rp_vox\n"
              "                _rp_pt = c_tgt[:, 1:].astype(np.float32) * _rp_vox\n"
              "                _rp_d = np.sqrt(((_rp_ps[:, None, :] - _rp_pt[None, :, :]) ** 2).sum(axis = 2))\n"
              "                _rp_ii, _rp_jj = np.nonzero(_rp_d <= _rp_gate)\n"
              "                _rp_src.append(idx_src[_rp_ii].astype(np.int32))\n"
              "                _rp_tgt.append(idx_tgt[_rp_jj].astype(np.int32))\n"
              "                _rp_p.append(np.asarray(probs)[_rp_ii, _rp_jj].astype(np.float32))\n"
              "                _rp_frames += 1\n"
              "                del _rp_ps, _rp_pt, _rp_d, _rp_ii, _rp_jj\n")

_SAVE_OLD = "    return (coords, all_edges)\n"
_SAVE_NEW = ("    if _rp_on:\n"
             "        _rp_dir = Path('%(rp_dir)s')\n"
             "        _rp_dir.mkdir(parents = True, exist_ok = True)\n"
             "        _rp_S = np.concatenate(_rp_src) if _rp_src else np.zeros(0, np.int32)\n"
             "        _rp_T = np.concatenate(_rp_tgt) if _rp_tgt else np.zeros(0, np.int32)\n"
             "        _rp_P = np.concatenate(_rp_p) if _rp_p else np.zeros(0, np.float32)\n"
             "        np.savez(_rp_dir / f'{ds_path.stem}.npz', src = _rp_S, tgt = _rp_T, prob = _rp_P, gate_um = np.float32(_rp_gate))\n"
             "        print(f'RELINK_PROB CAPTURE dataset={ds_path.stem} pairs={len(_rp_S)} frame_pairs={_rp_frames} nodes={len(coords)} '\n"
             "              f'gate_um={_rp_gate:g} source=%(rp_source)s', flush = True)\n"
             ) % {"rp_dir": RP_DIR, "rp_source": RP_SOURCE} + _SAVE_OLD
SCRIPT_PAIRS = [("relink_prob_init", _INIT_OLD, _INIT_NEW),
                ("relink_prob_capture", _PROBS_OLD, _PROBS_NEW),
                ("relink_prob_save", _SAVE_OLD, _SAVE_NEW)]

# Notebook block that applies the script pairs to _ps at runtime. Same place as node_select_946's block (after the TTA
# anchor; build_r946 inserts the tertiary block right after the same anchor, so in practice it runs **after** the
# tertiary patch has been applied and **before** the node_select block; the two blocks' script anchors do not overlap).
_TTA_ANCHOR = "print('Edge-feature TTA patch installed and enabled')\n"
_RUNTIME_BLOCK = '''
# === lane-r: relink learned-probability capture — kaggle/relink_prob_946.py (runtime patch on _ps) ===
_rp_pairs = %(pairs_json)s
_rp_src_text = _ps.read_text()
for _rp_name, _rp_old, _rp_new in _rp_pairs:
    _rp_n = _rp_src_text.count(_rp_old)
    if _rp_n != 1:
        raise RuntimeError(f"relink-prob script anchor '{_rp_name}' expected exactly 1 occurrence, found {_rp_n}")
    _rp_src_text = _rp_src_text.replace(_rp_old, _rp_new, 1)
compile(_rp_src_text, str(_ps), 'exec')
_ps.write_text(_rp_src_text)
print('RELINK_PROB SCRIPT PATCH APPLIED (3 anchors)', flush=True)
''' % {"pairs_json": _json.dumps([list(p) for p in SCRIPT_PAIRS], ensure_ascii=False)}

PAIRS = list(_ns.PAIRS) + [("relink_prob_runtime", _TTA_ANCHOR, _TTA_ANCHOR + _RUNTIME_BLOCK)] + NB_PAIRS
