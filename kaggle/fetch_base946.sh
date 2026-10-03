#!/usr/bin/env bash
# fetch_base946.sh — download the public base notebook this solution is built on and verify it.
#
#   kaggle/fetch_base946.sh                   download version 4 from Kaggle, verify it, install it
#   kaggle/fetch_base946.sh <notebook.ipynb>  verify and install a copy of version 4 you downloaded yourself
#
# The base is version 4 (2026-09-07) of "Biohub Cell Tracking: 0.946 LB" by Reyhan Ksatria (Apache-2.0):
#   https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb?scriptVersionId=348041532
# The author later renamed the notebook to "Biohub Cell Tracking: 0.947 LB" and published version 5, which is a
# different notebook. `kaggle kernels pull` returns only the latest version: a version-pinned pull
# (owner/slug/4) is refused with "403 Forbidden" (Kaggle CLI 2.2.4, checked 2026-10-03). So version 4 is downloaded
# from Kaggle's public download link for that script version, which needs no credentials.
#
# The SHA-256 of the notebook's single code cell is the ground truth. Nothing else is installed, and build_r946.py
# refuses to build on any other text unless ALLOW_BASE_MISMATCH=1.
#
# Writes $BASE946_DIR/notebook.ipynb (default kaggle/base946/, git-ignored; this repository does not redistribute the
# notebook). A relative BASE946_DIR is taken from the current directory, as in build_r946.py and the tests.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${BASE946_DIR:-$HERE/base946}"
PAGE="https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb?scriptVersionId=348041532"
URL="${BASE946_URL:-https://www.kaggle.com/kernels/scriptcontent/348041532/download}"
WANT="5e940fc76d42f12dfa3e962a16509441a2c1ffea5be2b97b9d7a9e774ed72d10"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [ "$#" -ge 1 ]; then
  cp "$1" "$tmp/notebook.ipynb"
elif ! curl -fsSL --retry 2 -o "$tmp/notebook.ipynb" "$URL"; then
  echo "download failed: $URL" >&2
  echo "Open $PAGE (version 4), download the notebook, then run: $0 <downloaded .ipynb>" >&2
  exit 1
fi

python3 - "$tmp/notebook.ipynb" "$WANT" <<'PY'
import hashlib, json, sys
try:
    nb = json.load(open(sys.argv[1], encoding="utf-8"))
except ValueError:
    sys.exit("not a notebook (JSON expected)")
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
src = "".join(code[0]["source"]) if len(code) == 1 else ""
sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
ok = sha == sys.argv[2]
print(f"base notebook code cell sha256 {sha}: {'OK' if ok else 'MISMATCH, expected ' + sys.argv[2]}")
sys.exit(0 if ok else 1)
PY

mkdir -p "$OUT"
cp "$tmp/notebook.ipynb" "$OUT/notebook.ipynb"
echo "installed $OUT/notebook.ipynb"
