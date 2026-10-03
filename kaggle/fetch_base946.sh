#!/usr/bin/env bash
# fetch_base946.sh — pull the public base notebook this solution is built on and verify it.
#
# The base is "Biohub Cell Tracking: 0.946 LB" by Reyhan Ksatria (Apache-2.0):
#   https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-946-lb
# The author later renamed it to ...-0-947-lb and published a newer version (v5). We pulled it on 2026-09-08,
# when version 4 was the latest one. The SHA-256 of the code cell below is the ground truth: build_r946.py refuses
# to build on any other text unless ALLOW_BASE_MISMATCH=1.
#
# Needs the Kaggle CLI with credentials (https://github.com/Kaggle/kaggle-api). Writes kaggle/base946/notebook.ipynb
# (git-ignored; the notebook is not redistributed by this repository).
set -euo pipefail
cd "$(dirname "$0")"
REF="${BASE946_REF:-reyhanksatria/biohub-cell-tracking-0-947-lb/4}"
OUT="${BASE946_DIR:-base946}"
WANT="5e940fc76d42f12dfa3e962a16509441a2c1ffea5be2b97b9d7a9e774ed72d10"

mkdir -p "$OUT"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
kaggle kernels pull "$REF" -p "$tmp"
nb="$(ls "$tmp"/*.ipynb | head -n 1)"
cp "$nb" "$OUT/notebook.ipynb"

python3 - "$OUT/notebook.ipynb" "$WANT" <<'PY'
import hashlib, json, sys
nb = json.load(open(sys.argv[1]))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
src = "".join(code[0]["source"]) if len(code) == 1 else ""
sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
ok = sha == sys.argv[2]
print(f"base notebook code cell sha256 {sha}: {'OK' if ok else 'MISMATCH, expected ' + sys.argv[2]}")
sys.exit(0 if ok else 1)
PY
