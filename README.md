# Biohub – Cell Tracking During Development: our solution (130th of 3,947)

Code and write-up of our entry to the Kaggle competition
[Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(June–September 2026).

**The problem.** Each test video is a 3D + time light-sheet recording of a developing zebrafish embryo with
fluorescent nuclei (100 frames of 64 x 256 x 256 voxels, 1.625 x 0.406 x 0.406 µm). A submission is a tracking graph:
one node per nucleus per frame and an edge from each nucleus to itself, or to its two daughters, in the next frame.
The score is the **adjusted edge Jaccard + 0.1 x division Jaccard**: predicted nodes are matched to the sparse
ground truth per frame within **7 µm**, an edge is correct when both ends match a ground-truth edge, and the edge
Jaccard is scaled by a node-count factor (`1 - 0.1 * (N_pred - N_total) / N_total`). The notebook must run on the
hidden test set within 12 hours on a GPU, offline.

## Result

| | Public LB (29 %) | Private LB (71 %, final) |
|---|---|---|
| Final submission (`headavg-pmax8`) | 0.96571 | **0.92649** |
| Rank | | **130 / 3,947 teams (top 3.3 %)** |
| Our best private score (own coordinate head, not selected) | 0.96071 | 0.93087 (about 75th) |

Medal: pending Kaggle finalization

Best public score: 0.96571 (the leaderboard truncated it to 0.965; public rank 53 at the deadline).

## Pipeline

We built on the strongest public notebook lineage (0.946 at the time), never editing it by hand: a builder applies
anchored, default-off patches to the unmodified notebook, so every submission is one traceable change
([how](kaggle/README.md)).

![Pipeline of the final submission](docs/figures/pipeline.svg)

## What we added on top of the public lineage

Each row is one submitted kernel that changed one thing relative to the row above.

| Change | Public LB change | Private LB change |
|---|---|---|
| *Start:* reproduction of the public 0.946 notebook on the original public weights (S56), score | *0.94489* | *0.91406* |
| **Third detector**: our own TemporalUNet3D (organizer recipe, epoch 100), blended into the detection logits at weight 0.20 | +0.0051 | +0.0015 |
| **Image-evidence division gate**: forks scored by the brightness gap between the daughters, symmetry and divergence, `z(dip) + z(sym) - z(div)` | +0.0032 | **+0.0070** |
| Looser division-gate threshold (1.5x pass volume) | +0.0009 | +0.0011 |
| **Learned-probability relink bonus**: the transformer's edge probability enters the cost of every relink candidate, not only the ILP's choice | +0.0010 | +0.0012 |
| **Flow-field relink**, ported from the public x138 notebook: match each cell against the position predicted by its neighbours' motion | +0.0019 | +0.0005 |
| **Coordinate head**: public V1284 head (MLP on UNet features, shift ≤ 2 µm), refined coordinates through the whole pipeline | +0.0047 | -0.0017 |
| **Average with our own head**, trained on 25x more matched pairs than the public one | +0.0021 | +0.0026 |
| **Safe-division gate 9 -> 8 µm**, checked on a pre-registered 33-video holdout | +0.0019 | +0.0002 |
| *Final:* `headavg-pmax8`, score | **0.96571** | **0.92649** |

Not in the final recipe: a second flow-refinement round (+0.0002 public, 0.0000 private) and our own head alone
(-0.0015 public, **+0.0070 private** against the public head).

![Public LB rejected our own head; private LB preferred it](docs/figures/lb_public_vs_private.svg)

## What did not work

| Idea | Public | Private | Why we think it failed |
|---|---|---|---|
| Drop weak, isolated nodes (val40 +0.0056) | -0.0257 | -0.0490 | The gain came from the node-count factor, which behaves differently on the hidden set |
| Float coordinates in `submission.csv` (val40 +0.0029) | -0.0113 | -0.0067 | The evaluation page asks for integers; the hosted scorer evidently treats floats differently from the organizer's local code |
| Detection retention guard off (-4.5 % nodes) | -0.0031 | -0.0010 | Node removal again |
| Other third detectors (three-frame window, 2x XY grid, synthetic pretraining) | -0.001 to -0.006 | -0.003 to +0.005 | Rejected on public; several were better on private |
| Division precision filters (plate orientation, opening angle, learned veto, two-peak test) | -0.003 to +0.001 | -0.009 to +0.001 | The forks they removed were still above break-even precision |
| Looser geometric division gates | -0.008 to -0.011 | -0.001 to +0.001 | Added mostly false divisions |
| x138 readmit + low-detection gap fill (+0.8 % nodes) | -0.0005 | +0.0003 | More nodes, little gain |
| One-to-one swap repair before smoothing | 0.0000 | +0.0001 | Identity switches are mostly two parallel tracks, not swaps |

Offline-only dead ends (learned family/path linkers, global ILP cost relaxation, GT-free duplicate removal) are
summarised in [docs/method.md](docs/method.md).

## Lessons

1. **A 29 % public leaderboard cannot judge 0.001-level decisions.** Offline, our own coordinate head beat the public
   one in both embryos (centre error 1.25 vs 1.53 µm). The public LB said -0.0015, so we kept the public head; the
   private LB said **+0.0070**, worth about 55 places. Across 88 single-change submissions the public and private
   changes had a rank correlation of 0.25.
2. **Node count is a lever whose sign you cannot see offline.** The metric rewards predicting fewer nodes, so
   node-removal rules look good on a local split; all of them lost on the leaderboard.
3. **Never change the submission format on a local scorer's word**: float coordinates, +0.0029 offline, -0.0113 on
   Kaggle.
4. **Division dials are noisy.** With few evaluated divisions, threshold curves read on the public LB had cliffs
   that the private set did not have.
5. **Pre-register holdout checks.** Our 33-video holdout ranked the four safe-division gates exactly as the private
   LB did; the public LB could not separate two of them.
6. **Measure the error budget first.** Oracles showed the missing score was in identity switches and divisions, not
   in node count; that is why we built coordinate refinement.

Details and evidence: [docs/lessons.md](docs/lessons.md). Full public vs private table: [docs/results.md](docs/results.md).
Every component explained: [docs/method.md](docs/method.md).

## Reproduction outline

The kernel needs the competition data and Kaggle datasets that are attached at run time; nothing large lives in
this repository.

1. **Base notebook.** `kaggle/fetch_base946.sh` pulls the public 0.946 notebook with the Kaggle CLI and checks the
   SHA-256 of its code cell (the builder refuses any other version).
2. **Build.** `python kaggle/build_recipe.py final-headavg-pmax8 out/final` writes `notebook.ipynb` and
   `kernel-metadata.json`. `python kaggle/build_recipe.py --list` shows every milestone recipe.
3. **Weights.** Public (CC0): pilkwang's support pack, DeepCenter and seed-314159 detectors, and the V1284 head.
   Ours are private Kaggle datasets because they were trained on competition data: the third detector
   (`impala9397/ctg-seedc4-snap`) and the coordinate heads (`impala9397/ctg-coordhead-own32-r4s0`,
   `impala9397/ctg-coordhead-own32-seeds5`). Their SHA-256 values are pinned in `kaggle/recipes.json`.
4. **Run.** `kaggle kernels push -p out/final`, then submit the finished version from the Kaggle UI (GPU T4,
   internet off).
5. **Own coordinate head.** `kaggle/coordhead_capture_946.py` builds a kernel that stores the head's input features
   for every detection on the training videos; `coordhead/train_coordhead.py pairs | cv | fit` matches them to the
   ground truth and trains the head.
6. **Tests.** `uv sync --all-groups && uv run --no-sync pytest`. Synthetic tests run anywhere; builder tests also need
   step 1 (`BASE946_DIR`), runtime-anchor tests a checkout of the organizer's baseline (`ORGANIZER_REPO`).
   Without them those tests skip with a reason.
7. **Figures.** `uv run --no-sync python scripts/make_figures.py` and `python scripts/make_pipeline_svg.py`.

## Repository map

```text
kaggle/        notebook builder, recipes and patch modules (see kaggle/README.md)
coordhead/     training of our own coordinate-regression head
docs/          method, results, lessons; figures; data/submissions.csv (all 158 submissions, public and private)
scripts/       figure generators
tests/         CPU tests (synthetic; builder tests skip without the public base notebook)
spec.md        design document of this repository
```

## Acknowledgements and licensing

- The competition hosts, Biohub, and the organizers' baseline
  ([royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition),
  BSD-3-Clause), whose architecture, training recipe and metric everything here builds on.
- The public notebooks we built on: Reyhan Ksatria's
  [0.946 notebook](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-946-lb) (our base),
  Anvith Pothula's [x138](https://www.kaggle.com/code/anvithpothula/biohub-0-953-lb-original) (flow relink, readmit,
  gap fill and the V1284 module, ported into our patches) and Aman Atar's
  [geometric fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion) (x138's base). All Apache-2.0.
- Public weights: pilkwang's support pack, DeepCenter and seed-314159 datasets, and Anvith Pothula's V1284 head
  (all CC0).
- Built solo, with AI coding agents (Claude Code, Codex) doing much of the implementation and offline analysis under
  my direction; every submission was reviewed and approved by me.

Our code is released under the [MIT License](LICENSE). Code ported from the public notebooks keeps its Apache-2.0
license; see [NOTICE](NOTICE) for what came from where and the license line we checked on each source page.
The competition data is not redistributed here.

Competition citation: Thibaut Goldsborough, Jordão Bragantini, Xiang Zhao, Gordon Leary, Teun Huijben,
Ilan da Silva Theodoro, Kyle Harrington, Chi-Li Chiu, Walter Reade, María Cruz, and Loïc A. Royer.
*Biohub - Cell Tracking During Development.* Kaggle, 2026.
