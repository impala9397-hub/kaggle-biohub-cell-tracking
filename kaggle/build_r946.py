#!/usr/bin/env python3
"""build_r946.py — build a Kaggle kernel directory from the public 0.946 base notebook plus our patches.

The base is the public notebook "Biohub Cell Tracking: 0.946 LB" by Reyhan Ksatria (Apache-2.0; edge-feature TTA
lineage), fetched with `kaggle/fetch_base946.sh` into `kaggle/base946/notebook.ipynb`. This script never ships that
notebook; it reads it, checks its code-cell SHA-256, and writes `<dst>/notebook.ipynb` + `<dst>/kernel-metadata.json`.

On top of the base it can:
  (a) point the weight datasets at the original public pilkwang datasets (always on);
  (b) add our third detector ("tertiary") and runtime script patches;
  (c) override BIOHUB_* knobs, keeping the notebook's configuration guard consistent.

Usage:  KERNEL_SLUG=ctg-lane-r-base946 python kaggle/build_r946.py <dst-dir>
        (or, more conveniently, `python kaggle/build_recipe.py <recipe> <dst-dir>`)
env:
  KERNEL_SLUG            (required) kernel slug; the kernel id is <KAGGLE_OWNER>/<slug>
  KAGGLE_OWNER           (optional) Kaggle user that owns the kernel and the private datasets. Default impala9397.
  BASE946_DIR            (optional) directory holding the base notebook.ipynb. Default kaggle/base946.
  ALLOW_BASE_MISMATCH    (optional) "1" builds even if the base code cell does not match the version we used.
  TERTIARY_SEED_DATASET  (optional) e.g. impala9397/ctg-seedC4-snap — turns the tertiary detector on
  TERTIARY_SEED_FILE     (optional) default edge_predictor_ep100.pth
  TERTIARY_DET_W         (optional) default 0.20      TERTIARY_EDGE_W (optional) default 0
  OVERRIDES              (optional) JSON string, e.g. {"BIOHUB_DET_THRESHOLD": "0.97"}. Updates the env lines and the
                         notebook's configuration guard (_EXPECTED_NUMERIC / _EXPECTED_TEXT) together.
                         Overriding BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION also turns the post-run contract's
                         hard-coded literal (0.9, three places) into float(os.environ[...]) reads
                         (RETENTION_LITERAL_PAIRS).
  NOTEBOOK_PATCH         (optional) path to a python file defining `PAIRS = [(name, old, new), ...]`, applied to the
                         notebook cell text at BUILD time (each anchor must occur exactly once, else the build fails).
                         Applied before OVERRIDES, so a knob that the patch introduces via os.environ.get(...) can be
                         switched on by the same build's OVERRIDES. Example: kaggle/x138_port_946.py
  SCRIPT_PATCH           (optional) path to a python file defining `PAIRS`, applied at RUNTIME to the inference script
                         (_ps = scripts/predict_unet_transformer.py) by a block inserted right after the tertiary block
                         (or after the TTA anchor if there is no tertiary block). Same mechanics as TERTIARY_TMPL:
                         embedded json pairs, count == 1 per anchor, compile(), write,
                         stdout marker 'SCRIPT PATCH APPLIED: <module> (<n> anchors)'.
                         Anchors must match the script text AFTER the tertiary patch.
  FAST_IO                (optional) "1" applies fastio_946.py last: setup searches of /kaggle/input (secondary
                         manifest, tertiary seed, DeepCenter checkpoint) become direct-path-first with a lazy walk.
                         Unset or "0" leaves the output byte-identical.
  BIOHUB_CSV_FLOAT       (optional) "1" applies csvfloat_946.py after FAST_IO: submission.csv node z/y/x are written as
                         max(0.0, round(v, 2)) instead of max(0, int(round(v))) (+ stdout marker 'CSV_FLOAT').
                         Unset or "0" leaves the output byte-identical. (Scored -0.011 on the public LB; kept off.)
  V1284                  (optional) "1" applies v1284_946.py after FAST_IO: the V1284 coordinate-regression head of the
                         public x138 notebook (dataset anvithpothula/biohub-v1284-head-s075) is inserted at the end of
                         the x138 port runtime block and its dataset is added to dataset_sources. NOTEBOOK_PATCH must
                         contain the x138_port_946 chain (else the build fails).
                         "late" is the count-neutral variant: the graph (int16 coordinates, native lookup, ILP,
                         post-processing) is untouched; refined coordinates are kept in a side file and only the z/y/x
                         of detection-origin nodes in submission.csv change (still integer-rounded).
                         Unset or "0" leaves the output byte-identical.
  V1284_HEAD_DATASET     (optional, only with V1284=1 or late) replaces the head dataset, e.g. impala9397/ctg-coordhead-<tag>
  V1284_HEAD_SHA256      (required with V1284_HEAD_DATASET) sha256 of that dataset's v1284_head.pt, checked at runtime.
                         Without both, the public anvithpothula/biohub-v1284-head-s075 is used (byte-identical output).
                         Setting either one while V1284 is 0 fails the build.
  V1284_HEAD2_DATASET    (optional, only with V1284=1) second head dataset: the refine displacement becomes the mean of
                         the two heads' bounded displacements (µm).
  V1284_HEAD2_SHA256     (required with V1284_HEAD2_DATASET) sha256 of that v1284_head.pt, checked at runtime.
                         Without both, byte-identical. Setting either one while V1284 is not 1 fails the build.
  V1284_HEAD2_SHA256S    (instead of V1284_HEAD2_SHA256, comma list) head 2 = mean of the bounded displacements of every
                         *.pt in V1284_HEAD2_DATASET (file-name order). At runtime the sha256 of each file must match the
                         list, in order (else the kernel fails). Displacement = 0.5 * head1 + 0.5 * mean_k head2_k.
                         Unset leaves the output byte-identical.
With none of the optional env set, the output is byte-identical to a plain reproduction of the base notebook
(weights re-pointed to the pilkwang originals).
The base notebook has a single code cell; every patch must match its string anchors exactly once (or the stated count).
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("BASE946_DIR") or (HERE / "base946"))
META_TEMPLATE = HERE / "kernel-metadata.template.json"
OWNER = os.environ.get("KAGGLE_OWNER", "impala9397")

# The base notebook version we built on (pulled 2026-09-08, then "Biohub Cell Tracking: 0.946 LB"; the author later
# renamed the notebook to "...-0-947-lb" and published a newer version). SHA-256 of the single code cell's source text.
BASE946_KERNEL = "reyhanksatria/biohub-cell-tracking-0-946-lb"
BASE946_CODE_SHA256 = "5e940fc76d42f12dfa3e962a16509441a2c1ffea5be2b97b9d7a9e774ed72d10"

SLUG_MAP = {  # private re-uploads used by the base notebook -> the public pilkwang originals (the notebook checks checksums)
    "reyhanksatria/biohub-tracking-support-pack": "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "reyhanksatria/biohub-deepcenterunet3d-center-prior-v1": "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "reyhanksatria/biohub-temporalunet3d-seed-314159-v1": "pilkwang/biohub-temporal-unet3d-seed314159-v1",
}
PILKWANG_DATASETS = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]
# Robustness fix that is always applied (not an experimental variable).
# The public notebook sets BIOHUB_ALLOW_ARTIFACT_FALLBACK=1, which makes artifact_matches_target() return True
# unconditionally and lets find_artifacts_root() scan /kaggle/input **without sorting**.
# Any dataset holding weights/unet_transformer/split_0/edge_predictor_best.pth can then come first, so depending on the
# run the secondary (seed314159, sha 9bac2fa0) is taken as the primary (sha 12f6881e) and the kernel dies on a checksum error.
# Two of our kernels (ctg-lane-r-bonus20/bonus05) died this way while others with the same datasets survived.
# With 0, only candidates whose path contains the target slug are checked, so seed314159 drops out of the candidates.
# Runs that resolved correctly are unchanged — this only prevents a wrong resolution.
ROBUSTNESS = {"BIOHUB_ALLOW_ARTIFACT_FALLBACK": "0"}

# Post-run contract literals that change only when BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION is overridden.
# The notebook reads the threshold from env and records it as minimum_retention in retention_guard_*.jsonl,
# then, after the run (about 25 minutes later), re-checks that the value equals the literal 0.9 — changing only the env
# line makes the kernel die there. All three places become env reads (two checks + the report's configuration record).
RETENTION_KEY = "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"
_RET_EXPR = f"float(os.environ['{RETENTION_KEY}'])"
RETENTION_LITERAL_PAIRS = [
    ("contract",
     "    if float(_guard_record['minimum_retention']) != 0.9 or int(_guard_record['primary_candidates']) < 0",
     f"    if float(_guard_record['minimum_retention']) != {_RET_EXPR} or int(_guard_record['primary_candidates']) < 0"),
    ("decision",
     "float(_guard_record['retention']) < 0.9)",
     f"float(_guard_record['retention']) < {_RET_EXPR})"),
    ("report",
     "'configuration': {'minimum_candidate_retention': 0.9, ",
     f"'configuration': {{'minimum_candidate_retention': {_RET_EXPR}, "),
]

TTA_ANCHOR = "print('Edge-feature TTA patch installed and enabled')"
# Where new (previously unset) knobs are inserted. Must come before the configuration guard and before the pipeline
# reads the values (module-level constant assignments).
NEW_ENV_ANCHOR = "print('BIOHUB_PRESET:', BIOHUB_PRESET)"

TERTIARY_TMPL = '''

# === lane-r: third detector (tertiary) — build_r946.py TERTIARY_SEED_DATASET ===
# Our own seed, trained with the organizer recipe (epoch 100), blended sequentially on top of the secondary blend:
#   det = (1-w3)·[(1-w2)·p + w2·s2] + w3·t3   (t3 aligned to the primary mean/std, inside the retention guard's blended_det)
_t_candidates = [
    Path("/kaggle/input/datasets/{owner}/{slug}/{fname}"),
    Path("/kaggle/input/{slug}/{fname}"),
]
_t_candidates += list(Path("/kaggle/input").rglob("{fname}"))
_t_path = next((c for c in _t_candidates if c.is_file()), None)
if _t_path is None:
    raise FileNotFoundError("tertiary seed weights not found: {fname} (dataset {owner}/{slug})")
if not (_t_path.parent / "config.json").is_file():
    raise FileNotFoundError(f"tertiary config.json not found next to {{_t_path}}")
os.environ["BIOHUB_TERTIARY_WEIGHTS"] = str(_t_path)
os.environ["BIOHUB_TERTIARY_DETECTION_WEIGHT"] = "{det_w}"
os.environ["BIOHUB_TERTIARY_EDGE_WEIGHT"] = "{edge_w}"
print("TERTIARY SEED:", _t_path, "det_w={det_w} edge_w={edge_w}", flush=True)
print("TERTIARY SHA256:", _sha256_file(_t_path), flush=True)
_t_pairs = {pairs_json}
_t_s = _ps.read_text()
for _t_name, _t_old, _t_new in _t_pairs:
    _t_n = _t_s.count(_t_old)
    if _t_n != 1:
        raise RuntimeError(f"tertiary anchor '{{_t_name}}' expected exactly 1 occurrence, found {{_t_n}}")
    _t_s = _t_s.replace(_t_old, _t_new, 1)
compile(_t_s, str(_ps), "exec")
_ps.write_text(_t_s)
for _t_must in ("TERTIARY BLEND RUNNING", "TERTIARY MODEL:", "tertiary_detection_weight = tertiary_detection_weight"):
    if _t_must not in _ps.read_text():
        raise RuntimeError("tertiary patch did not persist marker: " + _t_must)
print("TERTIARY PATCH APPLIED  (10 anchors on the 0.946 edge-feature-TTA script)", flush=True)
'''

SCRIPT_PATCH_TMPL = '''

# === lane-r: runtime script patch — build_r946.py SCRIPT_PATCH ({modname}) ===
# Applies anchor pairs to the inference script (_ps). It comes after the tertiary block if there is one,
# so the anchors refer to the script text after the tertiary patch.
_sp_pairs = {pairs_json}
_sp_s = _ps.read_text()
for _sp_name, _sp_old, _sp_new in _sp_pairs:
    _sp_n = _sp_s.count(_sp_old)
    if _sp_n != 1:
        raise RuntimeError(f"script patch anchor '{{_sp_name}}' expected exactly 1 occurrence, found {{_sp_n}}")
    _sp_s = _sp_s.replace(_sp_old, _sp_new, 1)
compile(_sp_s, str(_ps), "exec")
_ps.write_text(_sp_s)
print("SCRIPT PATCH APPLIED: {modname} ({n} anchors)", flush=True)
'''


def _load_pairs(path_str: str, what: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Import <path>.py and return (module name, PAIRS = [(name, old, new), ...])."""
    import importlib.util
    p = Path(path_str)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        raise SystemExit(f"{what}: file not found — {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{what}: cannot import — {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = getattr(mod, "PAIRS", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise SystemExit(f"{what}: {p.name} has no non-empty PAIRS")
    pairs: list[tuple[str, str, str]] = []
    for i, item in enumerate(raw):
        if len(item) != 3 or not all(isinstance(x, str) for x in item):
            raise SystemExit(f"{what}: PAIRS[{i}] must be three strings (name, old, new)")
        name, old, new = item
        if not old or old == new:
            raise SystemExit(f"{what}: anchor '{name}' has an empty old text or old == new")
        pairs.append((name, old, new))
    return p.stem, pairs


def _check_base(code: str) -> None:
    """Refuse to build on a base notebook other than the version we used (anchors would not match, or worse, match
    a different program). ALLOW_BASE_MISMATCH=1 overrides."""
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if sha != BASE946_CODE_SHA256 and os.environ.get("ALLOW_BASE_MISMATCH") != "1":
        raise SystemExit(
            f"base notebook code cell sha256 {sha} != expected {BASE946_CODE_SHA256}.\n"
            f"Fetch the version we built on with kaggle/fetch_base946.sh, or set ALLOW_BASE_MISMATCH=1 to try anyway.")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dst = Path(sys.argv[1]).resolve()
    slug = os.environ.get("KERNEL_SLUG") or (_ for _ in ()).throw(SystemExit("KERNEL_SLUG is required"))
    t_ds = os.environ.get("TERTIARY_SEED_DATASET")
    t_file = os.environ.get("TERTIARY_SEED_FILE", "edge_predictor_ep100.pth")
    t_det = os.environ.get("TERTIARY_DET_W", "0.20")
    t_edge = os.environ.get("TERTIARY_EDGE_W", "0")
    overrides = json.loads(os.environ.get("OVERRIDES", "{}"))
    nb_patch = os.environ.get("NOTEBOOK_PATCH")
    sc_patch = os.environ.get("SCRIPT_PATCH")
    if os.environ.get("NO_ROBUSTNESS") != "1":
        for _k, _v in ROBUSTNESS.items():
            overrides.setdefault(_k, _v)

    base_nb = BASE / "notebook.ipynb"
    if not base_nb.is_file():
        raise SystemExit(f"base notebook not found: {base_nb}\n"
                         f"Run kaggle/fetch_base946.sh (needs the Kaggle CLI) or set BASE946_DIR.")
    nb = json.loads(base_nb.read_text())
    code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    if len(code_cells) != 1:
        raise SystemExit(f"base946 must have exactly one code cell (found {len(code_cells)})")
    ci = code_cells[0]
    s = "".join(nb["cells"][ci]["source"])
    _check_base(s)
    changes: list[str] = []

    # (a) weight paths -> pilkwang originals
    for old, new in SLUG_MAP.items():
        n = s.count(old)
        if n < 1:
            raise SystemExit(f"slug anchor not found: {old}")
        s = s.replace(old, new)
        changes.append(f"slug {old} -> {new} ({n}x)")
    if "reyhanksatria" in s:
        raise SystemExit("a reyhanksatria reference is left in the notebook")

    # (a2) build-time notebook patch — before OVERRIDES, so the loop below accepts a knob that the patch introduces
    #      via os.environ.get(...) as "read by the code" and can add a new env line for it.
    if nb_patch:
        nb_mod, nb_pairs = _load_pairs(nb_patch, "NOTEBOOK_PATCH")
        for name, old, new in nb_pairs:
            n = s.count(old)
            if n != 1:
                raise SystemExit(f"NOTEBOOK_PATCH anchor '{name}' expected exactly 1 occurrence, found {n}")
            s = s.replace(old, new, 1)
        changes.append(f"notebook patch {nb_mod} ({len(nb_pairs)} anchors, build-time)")

    # (a3) read SCRIPT_PATCH early. It is inserted in (b2), but the "is this knob read by the code" check in (c) must
    #      also see the os.environ.get(...) calls that the script patch adds — otherwise a knob that exists only on the
    #      script side could not be switched on in the same build.
    if sc_patch:
        sc_mod, sc_pairs = _load_pairs(sc_patch, "SCRIPT_PATCH")
    else:
        sc_mod, sc_pairs = None, []
    _knob_src = s + "".join(_new for _, _, _new in sc_pairs)

    # (c) knob overrides — every env line plus the configuration guard.
    #     Keys the notebook already sets: all those lines are rewritten.
    #     Keys the notebook does not set (read by the code but never switched on) get a new line after the anchor.
    #     The latter was the main target of our audit: the 0.946 recipe tunes several knobs of a feature and then
    #     leaves the one trigger that enables it at its default, so the whole feature is dead.
    for key, val in overrides.items():
        pat = re.compile(r"os\.environ\['" + re.escape(key) + r"'\]\s*=\s*'[^']*'")
        hits = pat.findall(s)
        if hits:
            s = pat.sub(f"os.environ['{key}'] = '{val}'", s)
        else:
            if not re.search(r"os\.environ\.get\(\s*['\"]" + re.escape(key) + r"['\"]", _knob_src):
                raise SystemExit(
                    f"override key {key} is neither set by the notebook nor read by the code — typo or nonexistent knob")
            if s.count(NEW_ENV_ANCHOR) != 1:
                raise SystemExit(f"new-env anchor found {s.count(NEW_ENV_ANCHOR)} times — must be exactly once")
            s = s.replace(NEW_ENV_ANCHOR,
                          f"os.environ['{key}'] = '{val}'\n" + NEW_ENV_ANCHOR, 1)
            changes.append(f"NEW knob {key} = {val} (not set by the notebook, read by the code)")
            continue
        # Update the configuration guards: the numeric guard (_EXPECTED_NUMERIC) and the text guard (_EXPECTED_TEXT).
        # The retention-guard threshold (BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION) lives in _EXPECTED_TEXT; an older
        # builder that only updated the numeric guard made such kernels die with "Configuration drift" at runtime.
        guarded = False
        m = re.search(r"^_EXPECTED_NUMERIC = \{.*\}$", s, flags=re.M)
        if m and f"'{key}'" in m.group(0):
            line = m.group(0)
            line2 = re.sub(r"'" + re.escape(key) + r"':\s*[-0-9.eE]+", f"'{key}': {float(val)}", line)
            s = s.replace(line, line2, 1); guarded = True
        m = re.search(r"^_EXPECTED_TEXT = \{.*\}$", s, flags=re.M)
        if m and f"'{key}'" in m.group(0):
            line = m.group(0)
            line2 = re.sub(r"'" + re.escape(key) + r"':\s*'[^']*'", f"'{key}': '{val}'", line)
            s = s.replace(line, line2, 1); guarded = True
        changes.append(f"override {key} = {val} ({len(hits)} env line(s){' + guard' if guarded else ''})")
        if key == RETENTION_KEY:
            for name, old, new in RETENTION_LITERAL_PAIRS:
                n = s.count(old)
                if n != 1:
                    raise SystemExit(f"retention literal anchor '{name}' expected exactly 1 occurrence, found {n}")
                s = s.replace(old, new, 1)
            changes.append(f"retention contract literals 0.9 -> {_RET_EXPR} ({len(RETENTION_LITERAL_PAIRS)} anchors)")

    # (b) tertiary
    tail_anchor = TTA_ANCHOR  # where runtime patch blocks are appended
    if t_ds:
        if s.count(TTA_ANCHOR) != 1:
            raise SystemExit(f"TTA anchor expected 1, found {s.count(TTA_ANCHOR)}")
        sys.path.insert(0, str(HERE))
        import tertiary_patch_946 as tp  # noqa: E402
        owner, dslug = t_ds.split("/", 1)
        block = TERTIARY_TMPL.format(owner=owner, slug=dslug, fname=t_file, det_w=t_det, edge_w=t_edge,
                                     pairs_json=json.dumps([list(p) for p in tp.PAIRS], ensure_ascii=False))
        s = s.replace(TTA_ANCHOR, TTA_ANCHOR + block, 1)
        changes.append(f"tertiary {t_ds}/{t_file} det_w={t_det} edge_w={t_edge} (10 anchors)")
        tail_anchor = TTA_ANCHOR + block

    # (b2) runtime script patch — right after the tertiary block (or after the TTA anchor if there is none)
    if sc_patch:
        if s.count(tail_anchor) != 1:
            raise SystemExit(f"script patch insertion anchor expected 1, found {s.count(tail_anchor)}")
        sc_block = SCRIPT_PATCH_TMPL.format(modname=sc_mod, n=len(sc_pairs),
                                            pairs_json=json.dumps([list(p) for p in sc_pairs], ensure_ascii=False))
        s = s.replace(tail_anchor, tail_anchor + sc_block, 1)
        changes.append(f"script patch {sc_mod} ({len(sc_pairs)} anchors, runtime on _ps)")

    # (d) FAST_IO=1 — setup searches of /kaggle/input become direct-path-first + lazy walk (fastio_946.py).
    #     Applied after every block insertion so it also catches the tertiary block's rglob. Unset or "0" does nothing
    #     (byte-identical).
    fast_io = os.environ.get("FAST_IO", "0")
    if fast_io not in ("0", "1"):
        raise SystemExit(f"FAST_IO must be 0 or 1 (got {fast_io!r})")
    if fast_io == "1":
        sys.path.insert(0, str(HERE))
        import fastio_946  # noqa: E402
        s, fio_names = fastio_946.apply(s)
        changes.append(f"FAST_IO direct-path-first input search ({len(fio_names)} anchors: {', '.join(fio_names)})")

    # (d2) V1284=1 — the V1284 coordinate-regression head of the x138 notebook (v1284_946.py). The block goes at the end
    #      of the x138 port runtime block, so it runs after tertiary, relink_prob, node_select and the x138 low-detection
    #      dump have patched _ps. The head dataset is added too. Unset or "0" does nothing (byte-identical).
    v1284 = os.environ.get("V1284", "0")
    if v1284 not in ("0", "1", "late"):
        raise SystemExit(f"V1284 must be one of 0, 1, late (got {v1284!r})")
    extra_datasets: list[str] = []
    head_ds = os.environ.get("V1284_HEAD_DATASET", "").strip()
    head_sha = os.environ.get("V1284_HEAD_SHA256", "").strip()
    if (head_ds or head_sha) and v1284 == "0":
        raise SystemExit("V1284_HEAD_DATASET/V1284_HEAD_SHA256 are only valid with V1284=1 or late")
    if bool(head_ds) != bool(head_sha):
        raise SystemExit("V1284_HEAD_DATASET and V1284_HEAD_SHA256 must be given together")
    head2_ds = os.environ.get("V1284_HEAD2_DATASET", "").strip()
    head2_sha = os.environ.get("V1284_HEAD2_SHA256", "").strip()
    head2_shas = os.environ.get("V1284_HEAD2_SHA256S", "").strip()
    if (head2_ds or head2_sha or head2_shas) and v1284 != "1":
        raise SystemExit("V1284_HEAD2_DATASET/V1284_HEAD2_SHA256(S) are only valid with V1284=1")
    if head2_sha and head2_shas:
        raise SystemExit("give either V1284_HEAD2_SHA256 (one file) or V1284_HEAD2_SHA256S (a list), not both")
    if bool(head2_ds) != bool(head2_sha or head2_shas):
        raise SystemExit("V1284_HEAD2_DATASET and V1284_HEAD2_SHA256(S) must be given together")
    if v1284 in ("1", "late"):
        sys.path.insert(0, str(HERE))
        import v1284_946  # noqa: E402
        head_kw = {"dataset": head_ds, "sha": head_sha} if head_ds else {}
        head_name = head_ds or v1284_946.DATASET
    if v1284 == "1":
        if head2_shas:
            head_kw["head2s"] = (head2_ds, [x.strip() for x in head2_shas.split(",")])
        elif head2_ds:
            head_kw["head2"] = (head2_ds, head2_sha)
        s, v1_names = v1284_946.apply(s, **head_kw)
        extra_datasets.append(head_name)
        changes.append(f"V1284 coordinate-regression head {head_name} ({len(v1_names)} runtime anchors: "
                       f"{', '.join(v1_names)}; after x138 lowdet dump)")
        if head2_shas:
            extra_datasets.append(head2_ds)
            changes.append(f"V1284 head average: 0.5 x {head_name} + 0.5 x mean of {len(head_kw['head2s'][1])} heads "
                           f"in {head2_ds} (bounded displacements)")
        elif head2_ds:
            extra_datasets.append(head2_ds)
            changes.append(f"V1284 head average: mean of {head_name} and {head2_ds} bounded displacements")
    elif v1284 == "late":
        s, v1_names = v1284_946.apply_late(s, **head_kw)
        extra_datasets.append(head_name)
        changes.append(f"V1284=late count-neutral head {head_name} — graph untouched, CSV node z/y/x only "
                       f"({len(v1_names)} anchors: {', '.join(v1_names)})")

    # (e) BIOHUB_CSV_FLOAT=1 — node z/y/x as round(v, 2) floats instead of integer rounding (csvfloat_946.py). Applied
    #     last. Unset or "0" does nothing (byte-identical).
    csv_float = os.environ.get("BIOHUB_CSV_FLOAT", "0")
    if csv_float not in ("0", "1"):
        raise SystemExit(f"BIOHUB_CSV_FLOAT must be 0 or 1 (got {csv_float!r})")
    if csv_float == "1":
        sys.path.insert(0, str(HERE))
        import csvfloat_946  # noqa: E402
        s, cf_names = csvfloat_946.apply(s)
        changes.append(f"BIOHUB_CSV_FLOAT node z/y/x as round(v, 2) ({len(cf_names)} anchors: {', '.join(cf_names)})")

    compile(s, "notebook_cell", "exec")  # syntax check of the notebook cell itself
    nb["cells"][ci]["source"] = s.splitlines(keepends=True)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb))
    meta = json.loads(META_TEMPLATE.read_text())
    meta["id"] = f"{OWNER}/{slug}"
    meta["title"] = slug
    meta["dataset_sources"] = PILKWANG_DATASETS + ([t_ds] if t_ds else []) + extra_datasets
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {dst.name}  ->  {OWNER}/{slug}")
    for c in changes:
        print("  -", c)
    print("  datasets:", meta["dataset_sources"])


if __name__ == "__main__":
    main()
