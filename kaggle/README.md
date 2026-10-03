# kaggle/ — building submission kernels from a public notebook

Every kernel we submitted on the public 0.946 lineage was produced by `build_r946.py` from the **unmodified** public
notebook plus patch modules from this directory. Internally the lineage was called *lane R* (R for the rebase onto
the 0.946 notebook); the markers it writes into generated notebooks still say `lane-r`.

## Quick start

```bash
kaggle/fetch_base946.sh                                   # public base notebook, version 4 -> kaggle/base946/ (hash-checked)
python kaggle/build_recipe.py --list                      # milestone recipes, S56 ... final
python kaggle/build_recipe.py final-headavg-pmax8 out/final
python kaggle/build_recipe.py final-headavg-pmax8 --print-env   # the build_r946.py environment behind it
kaggle kernels push -p out/final                          # needs the owner's private weight datasets
```

`fetch_base946.sh` needs `curl` and `python3`, no Kaggle credentials. We built on version 4 of the base notebook;
the author has since published version 5, which is a different notebook, and `kaggle kernels pull` returns only the
latest version (a version-pinned pull such as `owner/slug/4` is refused with "403 Forbidden" by Kaggle CLI 2.2.4).
The script therefore downloads version 4 from Kaggle's public download link for that version and installs it only
if the SHA-256 of its code cell matches. If the download fails, download
[version 4](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb?scriptVersionId=348041532) by
hand and run `kaggle/fetch_base946.sh <downloaded .ipynb>`; the same check applies.

`KAGGLE_OWNER` (default `impala9397`) sets the kernel owner. `BASE946_DIR` points the fetch script, the builder and
the tests at another copy of the base notebook. The builder checks the SHA-256 of the base code cell and refuses any
other version (`ALLOW_BASE_MISMATCH=1` overrides).

## Verified: rebuilding the final kernels (2026-10-03)

These commands were run as written on 2026-10-03 (macOS, Kaggle CLI 2.2.4; the `kaggle kernels pull` line needs the
owner's account, because the kernels are private):

```bash
T=$(mktemp -d)
BASE946_DIR=$T/base946 kaggle/fetch_base946.sh
# base notebook code cell sha256 5e940fc76d42f12dfa3e962a16509441a2c1ffea5be2b97b9d7a9e774ed72d10: OK
BASE946_DIR=$T/base946 python kaggle/build_recipe.py final-headavg-pmax8 $T/build/headavg
kaggle kernels pull impala9397/ctg-lane-r-headavg-pmax8 -p $T/submitted/headavg -m
code() { jq -r '.cells[] | select(.cell_type == "code") | .source | if type == "array" then join("") else . end' "$1"; }
diff <(code $T/submitted/headavg/*.ipynb) <(code $T/build/headavg/notebook.ipynb)
```

The same was done for the second final selection (`final-headens5-pmax8` against
`impala9397/ctg-lane-r-headens5-pmax8`). Both kernels are at version 1 on Kaggle, the version that was scored.

| Check | `headavg-pmax8` | `headens5-pmax8` |
|---|---|---|
| Cells | markdown + code; markdown identical | same |
| Code cell length | 4,250 lines in both | 4,252 lines in both |
| Lines that differ | 37: 14 docstring lines and 12 comment lines translated to English, 11 comments whose only change is the path `kaggle/lane-r/` -> `kaggle/` (3 of these 23 comments sit inside embedded runtime-patch text) | the same 37 |
| Code compared without comments, docstrings and those paths (Python AST, embedded patch text included) | identical | identical |
| `kernel-metadata.json` | identical except the code file name, Kaggle's numeric id, and the dataset list, which Kaggle returns lower-cased and in its own order (the same six datasets) | same |
| Notebook-level metadata | the pulled copy has Kaggle's (nbformat 4.4, no papermill record); the build keeps the base notebook's | same |

So the repository rebuilds the submitted final kernels; the only differences are comments, docstrings, file paths
and ids.

## What a build does

```text
base946/notebook.ipynb (public, one code cell, ~2,900 lines)
  1. re-point the three weight datasets to the public pilkwang originals (the notebook verifies checksums)
  2. NOTEBOOK_PATCH   apply (name, old, new) pairs to the cell text; every old text must occur exactly once
  3. OVERRIDES        set BIOHUB_* env lines; update the notebook's configuration guard; add lines for knobs
                      the notebook reads but never sets (the build fails on a knob nobody reads)
  4. TERTIARY_*       insert the third-detector block (runtime patch of the predict script, 10 anchors)
  5. SCRIPT_PATCH     optional extra runtime patch of the predict script
  6. FAST_IO=1        direct-path-first input lookup (output identical)
  7. V1284=1|late     coordinate-regression head block, optional second head / head list
  8. BIOHUB_CSV_FLOAT optional float CSV writer (kept off)
  -> <dst>/notebook.ipynb, <dst>/kernel-metadata.json (private, GPU T4, internet off, dataset list)
```

Runtime patches are embedded in the notebook as JSON pairs and applied inside the kernel, after the notebook has
written the organizer's `predict_unet_transformer.py`: each anchor must match exactly once, the patched script must
`compile()`, and a marker is printed so the run log proves the patch ran.

## Modules

Notebook patches form a chain: each module's `PAIRS` starts with its predecessor's, so `NOTEBOOK_PATCH` names only
the last link.

```text
div_dip_946 -> node_select_946 -> relink_prob_946 -> x138_port_946 -> { swapfix_946 | divfork_946 }
```

| File | Role | Main knobs (all default off) | Final recipe |
|---|---|---|---|
| `build_r946.py` | the builder | see its docstring | |
| `build_recipe.py`, `recipes.json` | named build environments of submitted kernels | | |
| `tertiary_patch_946.py` | third detector blended into the detection logits | `TERTIARY_SEED_DATASET`, `TERTIARY_DET_W` | on (w 0.20) |
| `div_dip_946.py` | image-evidence division gate `z(dip) + z(sym) - z(div)`; optional t+1 two-blob veto | `BIOHUB_DIV_DIP_MODE=4`, `BIOHUB_DIV_COMBO_MAX` | on |
| `node_select_946.py` | node selection and fork pruning (link in the chain, off) | `BIOHUB_NODE_SELECT`, `BIOHUB_DIV_FORK_PRUNE` | off |
| `relink_prob_946.py` | learned-probability bonus in the final relink | `BIOHUB_RELINK_PROB_W=32` | on |
| `x138_port_946.py` | x138 flow relink, readmit, gap fill, low-detection dump, ILP timeout, fast relink | `BIOHUB_MOTION_RELINK_FLOW_*`, `BIOHUB_READMIT_*`, `BIOHUB_GAPFILL_*` | flow on |
| `v1284_946.py` | V1284 coordinate head (full or late), head average, head ensemble | `V1284`, `V1284_HEAD*_DATASET`, `V1284_HEAD2_SHA256(S)` | full + average |
| `fastio_946.py` | faster `/kaggle/input` lookups during setup | `FAST_IO=1` | on |
| `swapfix_946.py` | one-to-one edge re-assignment before smoothing | `BIOHUB_SWAPFIX`, `BIOHUB_SWAPFIX_PARAMS` | off |
| `divfork_946.py` | metric-aware fork pruning (geometric, chromatin) | `BIOHUB_DIVFORK` | off |
| `csvfloat_946.py` | float coordinates in `submission.csv` | `BIOHUB_CSV_FLOAT=1` | off |
| `coordhead_capture_946.py` | builds the kernel that captures head features on training videos | | |
| `fetch_base946.sh` | downloads version 4 of the public base notebook and verifies its hash | | |
| `kernel-metadata.template.json` | Kaggle kernel settings shared by all builds | | |

Code that was ported from the public x138 notebook (Apache-2.0) is marked in `x138_port_946.py` and `v1284_946.py`
and listed in [NOTICE](../NOTICE).

## Checks

- `tests/` builds kernels on CPU (with the base notebook present) and checks that an "off" build only adds inert
  lines, that each knob changes exactly the intended lines, that runtime anchors match the organizer's script once,
  and that the ported functions behave as expected on synthetic data. `tests/test_fetch_base946.py` checks that
  `fetch_base946.sh` pins the builder's hash and installs nothing else (run on local files, no network).
- When this repository was assembled from our working repository, every recipe here (13 milestones plus 7 variants)
  was built with both code bases. The notebooks matched line for line except translated comments and file paths,
  and the rebuilt final notebook sets the same 78 knob lines, in the same order, as the kernel we submitted.
