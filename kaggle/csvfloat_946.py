"""csvfloat_946.py — build-time patch: write node z/y/x in submission.csv as floats with two decimals instead of
rounded integers.

build_r946.py applies `apply(text)` to the notebook cell only when the build env has `BIOHUB_CSV_FLOAT=1`, and only
**after every block and FAST_IO have been applied**. If it is unset or "0", this file is not even imported, so the build
stays byte-identical.

Why (offline replay of the frozen N05 pipeline on all 40 videos of our validation split (val40, two embryos), scored
with the organizer metric unchanged): against the original writer `max(0, int(round(v)))`,
    float (= round(v, 2)): all40 +0.00293 (0.93324 → 0.93617); per embryo +0.00299 and +0.00291.
  A 4-video pilot gave −0.0021, but that came from a single node in one video crossing the 7 µm matching radius
  (7.008 → 6.954 µm).
  The organizer's csv_to_geffs casts z/y/x to pl.Float64, so the decimals reach the scorer unchanged.
Risk: the Kaggle Evaluation page describes z/y/x as "integer centroid coordinates in voxels". If the hosted scorer
  rejects floats, the submission fails (one submission slot lost); if it truncates them to integers, that equals floor,
  which gives −0.00101 on all40.
Outcome: on the public leaderboard this patch scored −0.011 against the same build with integer coordinates
  (0.956 → 0.945), so it stayed off.

What changes (each anchor exactly once):
  1. z/y/x in the node-row writer: `max(0, int(round(float(node['z']))))` → `max(0.0, round(float(node['z']), 2))`.
     The clamp is kept (2,722 negative coordinates, zero score effect; a later guard rejects negative coordinates).
  2. A stdout marker `CSV_FLOAT z/y/x written as round(v, 2)` right after the writer.
The -1 in edge rows and node_id/t/source/target stay integers.
"""
from __future__ import annotations

WRITER_OLD = ("'z': max(0, int(round(float(node['z'])))), "
              "'y': max(0, int(round(float(node['y'])))), "
              "'x': max(0, int(round(float(node['x']))))")
WRITER_NEW = ("'z': max(0.0, round(float(node['z']), 2)), "
              "'y': max(0.0, round(float(node['y']), 2)), "
              "'x': max(0.0, round(float(node['x']), 2))")
MARKER_OLD = "header = SUBMISSION_PATH.open().readline().strip().split(',')\n"
MARKER_NEW = ("print('CSV_FLOAT z/y/x written as round(v, 2)', flush = True)\n"
              + MARKER_OLD)

PAIRS = [
    ("csvfloat_node_writer", WRITER_OLD, WRITER_NEW),
    ("csvfloat_marker", MARKER_OLD, MARKER_NEW),
]


def apply(text: str) -> tuple[str, list[str]]:
    """Replace each anchor exactly once. Also return the names of the applied pairs."""
    names = []
    for name, old, new in PAIRS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"BIOHUB_CSV_FLOAT anchor '{name}' expected exactly 1 occurrence, found {n}")
        text = text.replace(old, new, 1)
        names.append(name)
    return text, names
