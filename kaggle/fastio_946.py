"""fastio_946.py — build-time patch: the setup's recursive `/kaggle/input` searches become "direct paths first, walk
only when none of them matches".

build_r946.py applies `pairs_for(text)` to the notebook cell only when the build env has `FAST_IO=1`, and only **after
every block (tertiary, script patches) has been inserted**. If `FAST_IO` is unset or "0", this file is not even
imported, so the build stays byte-identical.

Why (measured in N05-family commit runs): `/kaggle/input` contains the competition train zarr (millions of chunk
files), so a single walk takes tens to hundreds of seconds depending on Kaggle I/O. The N05 code runs the walk **to the
end first**, even though a direct path already matches.
  1. Secondary artifact: `candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))` builds the full list, and then
     the first candidate (the direct path) matches. flowdivsafe v1: 219.7 s; synthdet-fdm: 671 s.
  2. Tertiary seed (build_r946 TERTIARY_TMPL): `list(Path("/kaggle/input").rglob(fname))`. Both direct paths **always
     missed** because of the slug case (`ctg-seedC4-snap`, while Kaggle mounts `ctg-seedc4-snap`), so the walk result
     was used. 79–219 s.
  3. DeepCenter veto checkpoint: `sorted(input_root.glob('**/full_frame_center/**/{name}'))` × 3 names, i.e. three
     walks: 243–1436 s before the first video's repair (prediction time excluded). The first candidate (the direct
     path set by env) is the one that gets loaded.

How (identical output guaranteed):
  1 and 3 keep the candidate **order unchanged** and make only the walk part lazy. The original code takes the first
      success in order, so when a direct path succeeds no walk happens, and when the direct paths fail the walk results
      are tried next, in the original order. The chosen file is therefore always the original one (nothing fails where
      the original succeeded).
      DeepCenter: the dedup in `_dc_checkpoint_candidates()` (resolve key, first occurrence kept) is prefix-stable, so
      the list without the walk equals the head of the list with the walk. A generator yields that head first and the
      rest only if all of the head fails.
  2 adds the two lower-case-slug direct paths **before** the walk (the two original candidates stay in front). This is
      the only case where the order changes. The result equals the original when `/kaggle/input` holds exactly one file
      of that name (our dataset); checked against the path in the v1 log,
      `/kaggle/input/datasets/impala9397/ctg-seedc4-snap/edge_predictor_ep100.pth`, and by SHA.
      If the lower-case paths do not exist, it walks as before.
Markers (stdout): `FASTIO secondary …`, `FASTIO tertiary …`, `FASTIO deepcenter …`, each with the chosen path,
walked=0/1 and elapsed_s.
"""
from __future__ import annotations

import re

# ---- 1. secondary artifact ----
SEC_OLD = (
    "    input_root = Path('/kaggle/input')\n\n"
    "    if input_root.exists():\n"
    "        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))\n"
    "    seen = set()\n\n"
    "    for manifest_path in candidates:\n"
)
SEC_NEW = (
    "    input_root = Path('/kaggle/input')\n\n"
    "    # lane-r FASTIO: same order; walk only after every direct path has missed (lazy)\n"
    "    def _fastio_walk_manifests():\n"
    "        _FASTIO_SECONDARY_WALKED[0] = 1\n"
    "        if input_root.exists():\n"
    "            yield from input_root.rglob('ARTIFACT_MANIFEST.json')\n"
    "    seen = set()\n\n"
    "    for manifest_path in _fastio_itertools.chain(candidates, _fastio_walk_manifests()):\n"
)
SEC_PRE_OLD = "_secondary_expected_sha256 = "
SEC_PRE_NEW = (
    "import itertools as _fastio_itertools\n"
    "import time as _fastio_time\n"
    "_FASTIO_SECONDARY_WALKED = [0]\n"
    "_secondary_expected_sha256 = "
)
SEC_CALL_OLD = "SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()\n"
SEC_CALL_NEW = (
    "_fastio_t0 = _fastio_time.time()\n"
    "SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()\n"
    "print(f'FASTIO secondary manifest_dir={SECONDARY_ARTIFACTS} walked={_FASTIO_SECONDARY_WALKED[0]} '\n"
    "      f'elapsed_s={_fastio_time.time() - _fastio_t0:.2f}', flush = True)\n"
)

# ---- 3. DeepCenter checkpoint ----
DC_PRE_OLD = (
    "# Build the ordered list of DeepCenter checkpoint candidates\n"
    "def _dc_checkpoint_candidates() -> list[Path]:\n"
)
DC_PRE_NEW = (
    "# lane-r FASTIO: while this is True, _dc_checkpoint_candidates skips the /kaggle/input walk\n"
    "_FASTIO_DC_SKIP_WALK = False\n"
    "_FASTIO_DC_WALKED = [0]\n"
    "_FASTIO_DC_CAND_S = [0.0]\n\n"
    + DC_PRE_OLD
)
DC_WALK_OLD = (
    "    if input_root.exists():\n"
    "        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):\n"
    "            candidates.extend(sorted(input_root.glob(f'**/full_frame_center/**/{name}')))\n"
)
DC_WALK_NEW = (
    "    if input_root.exists() and not _FASTIO_DC_SKIP_WALK:\n"
    "        _FASTIO_DC_WALKED[0] += 1\n"
    "        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):\n"
    "            candidates.extend(sorted(input_root.glob(f'**/full_frame_center/**/{name}')))\n"
)
DC_GEN_OLD = "try:\n    import torch\n\nexcept Exception as _dc_torch_error:\n"
DC_GEN_NEW = (
    "# lane-r FASTIO: original candidate order, but the walk part is computed only after all earlier ones fail.\n"
    "# The dedup is prefix-stable, so the list without the walk == the head of the list with the walk.\n"
    "def _fastio_dc_candidates():\n"
    "    global _FASTIO_DC_SKIP_WALK\n"
    "    _t0 = time.time()\n"
    "    _FASTIO_DC_SKIP_WALK = True\n"
    "    try:\n"
    "        _known = _dc_checkpoint_candidates()\n"
    "    finally:\n"
    "        _FASTIO_DC_SKIP_WALK = False\n"
    "    _FASTIO_DC_CAND_S[0] = time.time() - _t0\n"
    "    yield from _known\n"
    "    _full = _dc_checkpoint_candidates()\n"
    "    yield from _full[len(_known):]\n\n"
    + DC_GEN_OLD
)
DC_LOOP_OLD = "    for checkpoint_path in _dc_checkpoint_candidates():\n        if not checkpoint_path.exists():\n"
DC_LOOP_NEW = "    for checkpoint_path in _fastio_dc_candidates():\n        if not checkpoint_path.exists():\n"
DC_CALL_OLD = "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()\n"
DC_CALL_NEW = (
    "_fastio_t0 = time.time()\n"
    "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()\n"
    "print(f\"FASTIO deepcenter path={DEEPCENTER_VETO_DETECTOR.get('path') if isinstance(DEEPCENTER_VETO_DETECTOR, dict) else None} \"\n"
    "      f'walked={_FASTIO_DC_WALKED[0]} candidates_s={_FASTIO_DC_CAND_S[0]:.2f} '\n"
    "      f'elapsed_s={time.time() - _fastio_t0:.2f}', flush = True)\n"
)

STATIC_PAIRS = [
    ("fastio_secondary_prelude", SEC_PRE_OLD, SEC_PRE_NEW),
    ("fastio_secondary_lazy_walk", SEC_OLD, SEC_NEW),
    ("fastio_secondary_marker", SEC_CALL_OLD, SEC_CALL_NEW),
    ("fastio_dc_prelude", DC_PRE_OLD, DC_PRE_NEW),
    ("fastio_dc_skip_walk", DC_WALK_OLD, DC_WALK_NEW),
    ("fastio_dc_lazy_candidates", DC_GEN_OLD, DC_GEN_NEW),
    ("fastio_dc_loader_loop", DC_LOOP_OLD, DC_LOOP_NEW),
    ("fastio_dc_marker", DC_CALL_OLD, DC_CALL_NEW),
]

# ---- 2. tertiary seed (build_r946 TERTIARY_TMPL, only when present) ----
_T_HEAD = re.compile(
    r'_t_candidates = \[\n'
    r'    Path\("/kaggle/input/datasets/(?P<owner>[^/"]+)/(?P<slug>[^/"]+)/(?P<fname>[^/"]+)"\),\n'
    r'    Path\("/kaggle/input/(?P=slug)/(?P=fname)"\),\n'
    r'\]\n')


def _tertiary_pair(text: str):
    m = list(_T_HEAD.finditer(text))
    if not m:
        return None
    if len(m) != 1:
        raise SystemExit(f"FAST_IO: tertiary candidate block expected exactly 1 occurrence, found {len(m)}")
    owner, slug, fname = m[0].group("owner"), m[0].group("slug"), m[0].group("fname")
    old = (f'_t_candidates += list(Path("/kaggle/input").rglob("{fname}"))\n'
           '_t_path = next((c for c in _t_candidates if c.is_file()), None)\n')
    lo = [f"/kaggle/input/datasets/{owner.lower()}/{slug.lower()}/{fname}", f"/kaggle/input/{slug.lower()}/{fname}"]
    new = (
        "# lane-r FASTIO: Kaggle mounts dataset slugs in lower case → add lower-case direct paths before the walk,\n"
        "# and walk only after every direct path has missed (lazy, original order).\n"
        "import itertools as _fastio_itertools\n"
        "import time as _fastio_time\n"
        "_fastio_t0 = _fastio_time.time()\n"
        "_fastio_t_walked = [0]\n"
        f"for _fastio_c in (Path({lo[0]!r}), Path({lo[1]!r})):\n"
        "    if _fastio_c not in _t_candidates:\n"
        "        _t_candidates.append(_fastio_c)\n"
        "def _fastio_walk_tertiary():\n"
        "    _fastio_t_walked[0] = 1\n"
        f'    yield from Path("/kaggle/input").rglob("{fname}")\n'
        "_t_path = next((c for c in _fastio_itertools.chain(_t_candidates, _fastio_walk_tertiary()) if c.is_file()), None)\n"
        "print(f'FASTIO tertiary path={_t_path} walked={_fastio_t_walked[0]} '\n"
        "      f'elapsed_s={_fastio_time.time() - _fastio_t0:.2f}', flush = True)\n"
    )
    if text.count(old) != 1:
        raise SystemExit(f"FAST_IO: tertiary rglob anchor expected exactly 1 occurrence, found {text.count(old)}")
    return ("fastio_tertiary_lazy_walk", old, new)


def pairs_for(text: str) -> list[tuple[str, str, str]]:
    """(name, old, new) pairs to apply to this notebook text. Without a tertiary block, only the 8 static pairs."""
    pairs = list(STATIC_PAIRS)
    tp = _tertiary_pair(text)
    if tp is not None:
        pairs.append(tp)
    return pairs


def apply(text: str) -> tuple[str, list[str]]:
    """Replace each anchor exactly once. Also return the names of the applied pairs."""
    names = []
    for name, old, new in pairs_for(text):
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"FAST_IO anchor '{name}' expected exactly 1 occurrence, found {n}")
        text = text.replace(old, new, 1)
        names.append(name)
    return text, names
