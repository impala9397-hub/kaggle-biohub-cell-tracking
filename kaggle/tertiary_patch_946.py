"""tertiary_patch_946.py — third-detector (tertiary) patch, matched to the formatting of the final
predict_unet_transformer.py in the public 0.946 kernel (reyhanksatria, edge-feature TTA lineage).

The logic is identical to our earlier tertiary patch for the previous base script; only 3 anchors (pre_init, det_blend,
call) were adapted to the 0.946 script's reformatting (one-line blend, `kw = v` spacing, blank lines). The other 7
anchors match unchanged. apply() requires every anchor to match exactly once.
"""
PAIRS = [
 [
  "signature",
  "    secondary_low_margin_max: float = 0.2,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:",
  "    secondary_low_margin_max: float = 0.2,\n    tertiary_model: UNetNodeTransformer | None = None,\n    tertiary_edge_weight: float = 0.0,\n    tertiary_detection_weight: float = 0.0,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:"
 ],
 [
  "pre_init",
  "        secondary_unet_out = None\n\n        if secondary_model is not None:",
  "        secondary_unet_out = None\n        tertiary_unet_out = None\n        tertiary_det_logits = None\n\n        if secondary_model is not None:"
 ],
 [
  "det_tta",
  "                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]",
  "                tertiary_det_logits = None\n                if tertiary_model is not None and (tertiary_detection_weight > 0.0 or tertiary_edge_weight > 0.0):\n                    if not globals().get(\"_TERTIARY_SEEN\"):\n                        globals()[\"_TERTIARY_SEEN\"] = True\n                        print(\"TERTIARY BLEND RUNNING  det_w =\", tertiary_detection_weight,\n                              \" edge_w =\", tertiary_edge_weight, flush=True)\n                if tertiary_model is not None and tertiary_detection_weight > 0.0:\n                    tertiary_unet_out, tertiary_det_logits = tertiary_model.encode(imgs)\n                    if cfg.det_tta:\n                        _tertiary_nv = 1\n                        for dims in [(-1,), (-2,), (-2, -1)]:\n                            _t_flip = imgs.flip(dims)\n                            _, _t_det_flip = tertiary_model.encode(_t_flip)\n                            for f in range(W):\n                                tertiary_det_logits[f] = tertiary_det_logits[f] + _t_det_flip[f].flip(dims)\n                            del _t_flip, _t_det_flip\n                            _tertiary_nv += 1\n                        for _k in (1, 3):\n                            _t_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                            _, _t_det_rot = tertiary_model.encode(_t_rot)\n                            for f in range(W):\n                                tertiary_det_logits[f] = tertiary_det_logits[f] + torch.rot90(\n                                    _t_det_rot[f], -_k, dims=(-2, -1))\n                            del _t_rot, _t_det_rot\n                            _tertiary_nv += 1\n                        _t_tr = imgs.transpose(-1, -2)\n                        _, _t_det_tr = tertiary_model.encode(_t_tr)\n                        for f in range(W):\n                            tertiary_det_logits[f] = tertiary_det_logits[f] + _t_det_tr[f].transpose(-1, -2)\n                        del _t_tr, _t_det_tr\n                        _tertiary_nv += 1\n                        _t_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n                        _, _t_det_at = tertiary_model.encode(_t_at)\n                        for f in range(W):\n                            tertiary_det_logits[f] = tertiary_det_logits[f] + torch.rot90(\n                                _t_det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n                        del _t_at, _t_det_at\n                        _tertiary_nv += 1\n                        for f in range(W):\n                            tertiary_det_logits[f] = tertiary_det_logits[f] / _tertiary_nv\n                elif tertiary_model is not None and tertiary_edge_weight > 0.0:\n                    tertiary_unet_out, _ = tertiary_model.encode(imgs)\n\n                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]"
 ],
 [
  "det_blend",
  "                    blended_det = (1.0 - secondary_detection_weight) * primary_det + secondary_detection_weight * secondary_det_aligned",
  "                    blended_det = (1.0 - secondary_detection_weight) * primary_det + secondary_detection_weight * secondary_det_aligned\n                    if tertiary_det_logits is not None:\n                        tertiary_det = tertiary_det_logits[f]\n                        tertiary_mean = tertiary_det.mean()\n                        tertiary_scale = tertiary_det.float().std(unbiased=False).clamp_min(1e-4)\n                        tertiary_ratio = (primary_scale / tertiary_scale).clamp(0.5, 2.0)\n                        tertiary_det_aligned = (\n                            (tertiary_det - tertiary_mean) * tertiary_ratio + primary_mean\n                        )\n                        blended_det = (\n                            (1.0 - tertiary_detection_weight) * blended_det\n                            + tertiary_detection_weight * tertiary_det_aligned\n                        )"
 ],
 [
  "det_clean",
  "            del secondary_det_logits\n\n        del imgs",
  "            del secondary_det_logits\n            if tertiary_det_logits is not None:\n                del tertiary_det_logits\n\n        del imgs"
 ],
 [
  "edge",
  "            raw = edge_logits_pair[0]",
  "            if tertiary_unet_out is not None and tertiary_edge_weight > 0.0:\n                _t_feat_src = tertiary_model._index_features(\n                    tertiary_unet_out[:, f_idx], p_coords_src, p_mask_src)\n                _t_feat_tgt = tertiary_model._index_features(\n                    tertiary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)\n                _t_logits = tertiary_model.predict_edges(\n                    _t_feat_src, _t_feat_tgt,\n                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt)\n                _c = edge_logits_pair.mean(dim=1, keepdim=True)\n                _s = edge_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)\n                _tc = _t_logits.mean(dim=1, keepdim=True)\n                _ts = _t_logits.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)\n                _t_aligned = (_t_logits - _tc) * (_s / _ts).clamp(0.5, 2.0) + _c\n                edge_logits_pair = (\n                    (1.0 - tertiary_edge_weight) * edge_logits_pair\n                    + tertiary_edge_weight * _t_aligned.to(edge_logits_pair.dtype)\n                )\n                del _t_feat_src, _t_feat_tgt, _t_logits, _t_aligned\n\n            raw = edge_logits_pair[0]"
 ],
 [
  "clean",
  "        if secondary_unet_out is not None:\n            del secondary_unet_out\n",
  "        if secondary_unet_out is not None:\n            del secondary_unet_out\n        if tertiary_unet_out is not None:\n            del tertiary_unet_out\n"
 ],
 [
  "init",
  "    secondary_model = None\n    secondary_weights_text",
  "    secondary_model = None\n    tertiary_model = None\n    tertiary_edge_weight = 0.0\n    tertiary_detection_weight = 0.0\n    secondary_weights_text"
 ],
 [
  "load",
  "    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:",
  "    # --- tertiary: read *outside* the secondary block. If there is no secondary or the detection blend is off,\n    #     fail here instead of being silently ignored (closes the 'silent no-op' path flagged by three reviewers).\n    tertiary_weights_text = os.environ.get(\"BIOHUB_TERTIARY_WEIGHTS\", \"\").strip()\n    tertiary_edge_weight = float(os.environ.get(\"BIOHUB_TERTIARY_EDGE_WEIGHT\", \"0\"))\n    tertiary_detection_weight = float(os.environ.get(\"BIOHUB_TERTIARY_DETECTION_WEIGHT\", \"0\"))\n    if tertiary_weights_text:\n        if not secondary_weights_text:\n            raise ValueError(\"BIOHUB_TERTIARY_WEIGHTS requires BIOHUB_SECONDARY_WEIGHTS (tertiary composes on the secondary blend)\")\n        if not 0.0 <= tertiary_edge_weight < 1.0:\n            raise ValueError(\"BIOHUB_TERTIARY_EDGE_WEIGHT must be in [0, 1)\")\n        if not 0.0 <= tertiary_detection_weight < 1.0:\n            raise ValueError(\"BIOHUB_TERTIARY_DETECTION_WEIGHT must be in [0, 1)\")\n        if (tertiary_detection_weight > 0.0 or tertiary_edge_weight > 0.0) and secondary_detection_weight <= 0.0:\n            raise ValueError(\"tertiary blend requires BIOHUB_SECONDARY_DETECTION_WEIGHT > 0 (the tertiary encode lives in the secondary detection path)\")\n        tertiary_model, _t_ws, _t_ds = load_model(Path(tertiary_weights_text), device)\n        if _t_ws != window_size or _t_ds != downsample:\n            raise ValueError(\n                \"Tertiary model has an incompatible inference grid: \"\n                f\"primary=(window={window_size}, downsample={downsample}), \"\n                f\"tertiary=(window={_t_ws}, downsample={_t_ds})\")\n        print(\n            f\"TERTIARY MODEL: {tertiary_weights_text} | \"\n            f\"edge weight={tertiary_edge_weight:.3f} | \"\n            f\"detection weight={tertiary_detection_weight:.3f}\",\n            flush=True)\n    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:"
 ],
 [
  "call",
  "                secondary_low_margin_max = secondary_low_margin_max)",
  "                secondary_low_margin_max = secondary_low_margin_max,\n                tertiary_model = tertiary_model,\n                tertiary_edge_weight = tertiary_edge_weight,\n                tertiary_detection_weight = tertiary_detection_weight)"
 ]
]

def apply(text: str) -> str:
    for name, old, new in PAIRS:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"tertiary anchor '{name}' expected exactly 1 occurrence, found {n}")
        text = text.replace(old, new, 1)
    for must in ("TERTIARY BLEND RUNNING", "TERTIARY MODEL:", "tertiary_detection_weight = tertiary_detection_weight"):
        if must not in text:
            raise RuntimeError(f"tertiary patch did not persist marker: {must}")
    return text
