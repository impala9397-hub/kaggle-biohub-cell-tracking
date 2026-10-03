<!-- DRAFT of our Kaggle solution write-up. Not posted. Before posting, upload the figures as PNG
     (`uv run --no-sync python scripts/make_figures.py --png`; export pipeline.svg too) and replace the relative
     image paths. The GitHub repository has been public since 2026-10-03. -->

# 130th Place (Silver): One-Change Patches on a Public Notebook

*Subtitle: what held up on the private leaderboard, and what the public one missed about our own head*

Thank you to Biohub and the organizers for a well-specified tracking problem and a strong baseline, and to the
authors of the public notebooks and weights we built on (credited below).

## 1. Overview

| | Public (29 %) | Private (71 %) |
|---|---|---|
| Final selection (`headavg-pmax8`) | 0.96571 | **0.92649**: 130th of 3,947, silver |
| Our four own-head kernels (not selected) | 0.9596-0.9607 | 0.9307-0.9309: would have been 75th-76th |

We did not write a new tracker. We took the strongest public notebook lineage (0.946 public when we joined it) and
added one change per submission through a small build system. Seven of the eight changes in the final recipe also
improved the private score; the exception was the public coordinate head. The main lesson is about selection: our
own coordinate head was better in every offline test and +0.0070 on private, but read -0.0015 on public, and we
followed public.

![Pipeline of the final submission](figures/pipeline.svg)

## 2. What we built on

- **Organizer baseline**
  ([royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition),
  BSD-3-Clause): the TemporalUNet3D detector with a node-transformer edge predictor, the ILP and the metric.
- **Reyhan Ksatria,
  [Biohub Cell Tracking: 0.946 LB](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb?scriptVersionId=348041532)**
  (version 4), our base: two detectors with 8-view test-time augmentation (also applied to the features used for
  association), bidirectional edge fusion, the ILP, a two-pass motion relink, gap closing, a "safe-division" repair
  with a DeepCenter veto, and smoothing. Our reproduction scored 0.94489 public, 0.91406 private.
- **pilkwang's public weights**:
  [support pack](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1),
  [seed 314159](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1) and
  [DeepCenter](https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1) (CC0).
- **Anvith Pothula,
  [Biohub 0.953 LB | ORIGINAL](https://www.kaggle.com/code/anvithpothula/biohub-0-953-lb-original)** ("x138"): the
  flow-field relink we ported, and the
  [V1284 coordinate head](https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075) (CC0).
- **Aman Atar, [Biohub Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion)**, the base
  of x138.

![Detector/linker model (UNetNodeTransformer)](figures/arch_detector_linker.png)
*The organizer's detector/linker model, used by all three detectors in our blend. Drawn in the style of [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).*

The base notebook is a single 2,900-line cell. Instead of forking it, a builder applies anchored patches to the
unmodified notebook: each anchor must match exactly once and every addition sits behind a default-off knob. We
submitted 92 kernels this way, each one change against a named base.

## 3. Our additions and their measured effect

| Step (one submitted kernel each) | Public | Private |
|---|---|---|
| Reproduction of the public notebook (S56) | 0.94489 | 0.91406 |
| + Third detector: our own model (organizer recipe, epoch 100), blend weight 0.20 | +0.0051 | +0.0015 |
| + Image-evidence division gate | +0.0032 | **+0.0070** |
| + Looser gate threshold (1.5x pass volume) | +0.0009 | +0.0011 |
| + Learned-probability relink bonus | +0.0010 | +0.0012 |
| + Flow-field relink ported from x138 (one round, FLOW_ITER 1) | +0.0019 | +0.0005 |
| + Public V1284 coordinate head, full mode | +0.0047 | -0.0017 |
| + Average with our own head | +0.0021 | +0.0026 |
| + Safe-division gate 9 -> 8 µm (final) | +0.0019 | +0.0002 |
| Final | **0.96571** | **0.92649** |

- **Division gate.** The safe-division repair accepted forks by geometry alone. We score each candidate with image
  evidence: the brightness dip between the two daughters (two sister nuclei show a dark gap, one nucleus detected
  twice does not), the symmetry of the parent-daughter distances, and how far the daughters move apart one frame
  later, combined as `z(dip) + z(sym) - z(div)`. On the 151 training divisions the dip alone reaches AUC 0.82,
  against 0.38 for the distance key used before; the combined score reaches 0.85.
- **Relink bonus.** The final relink rebuilds almost every edge by distance; the transformer's probability entered
  only for the edge the ILP had chosen. We subtract `32 * p` from the cost of every candidate pair (+0.0020 offline,
  in both embryos).
- **Coordinate heads.** The V1284 head reads the UNet features at a detection and its six neighbours (224 values),
  applies an MLP 224-32-3 and moves the centre by at most 2 µm. We run it in full mode, so the refined coordinates
  flow through association, the ILP and the repairs. We trained the same architecture on 103,806 matched pairs from
  159 training videos, 25x the public head's data. The final kernel moves each centre by the mean of the two heads'
  bounded shifts.

![Coordinate head](figures/arch_coordinate_head.png)
*The coordinate head. The final submission averages the bounded shifts of the public V1284 head and our own head.*

![Change per step](figures/lb_steps.svg)

## 4. What didn't work

| Idea | Offline | Public | Private |
|---|---|---|---|
| Drop weak, isolated nodes (-11 % nodes) | +0.0056 | -0.0257 | -0.0490 |
| Float coordinates in `submission.csv` | +0.0029 | -0.0113 | -0.0067 |
| Division precision filters (orientation, opening angle, learned veto) | | -0.003 to +0.001 | -0.009 to +0.001 |
| Looser geometric division gates | | -0.008 to -0.011 | -0.001 to +0.001 |
| x138 readmit + gap fill (+0.8 % nodes) | | -0.0005 | +0.0003 |
| Second flow-relink round (FLOW_ITER 2; not in the final) | | +0.0002 | 0.0000 |

Node count was the trap. The metric multiplies the edge Jaccard by `1 - 0.1 * (N_pred - N_total) / N_total`, so
removing nodes looks good offline, and every node-removal rule we submitted lost. Float coordinates scored better
with the organizer's local code and worse on Kaggle; the evaluation page asks for integers. Offline-only dead ends
included duplicate removal without ground truth (an oracle was worth +0.028, the best rule scored -0.022), a cheaper
ILP division cost, and learned linkers (0-1 of 413 recoverable links recovered).

An oracle study on 40 held-out training videos set our priorities for the last week: perfect links between detected
cells would add +0.078 (mostly identity switches between two parallel tracks), perfect divisions +0.080, perfect
detection +0.052, and the true node count -0.005. So we built coordinate refinement instead of more node rules.

## 5. Public vs private: the own-head story

Offline, our head beat the public one in every test: centre error 1.25 vs 1.53 µm on 40 held-out videos (1.71
without a head), tracking score 0.9454 vs 0.9421, in both embryos and with three seeds. The public leaderboard
disagreed in three paired submissions:

| Base kernel | Public, own minus public head | Private, own minus public head |
|---|---|---|
| one flow round (as in the final) | -0.00146 | +0.00701 |
| two flow rounds | -0.00134 | +0.00703 |
| two flow rounds + swap repair | -0.00121 | +0.00706 |

We followed the public reading and hedged with the 50/50 head average. The own-head kernels scored 0.93068-0.93087
private; 74 teams finished above 0.93087, so the best of them would have placed 75th instead of 130th.

It was not a one-off. Over 88 single-change submissions, public and private changes had a Spearman correlation of
0.25, and their median disagreement (0.0017) exceeded the median public change (0.0015). Large effects agreed; the
0.001-0.003 decisions we spent most submissions on did not. A third detector trained on dense pseudo-labels, for
example, read -0.0056 public and +0.0115 private.

![Public vs private changes for 88 single-change submissions](figures/public_vs_private_deltas.svg)

What we would do differently: decide in advance which evidence wins. When a change is consistent offline (both
embryos, several seeds) and its public reading is within about 0.002, keep the offline verdict.

## 6. Pre-registered holdout checks

For the last decisions we ran the real final pipeline on 33 training videos never used for tuning (72 ground-truth
divisions) and wrote down the decision rule before looking at the result.

| Safe-division gate | Holdout | Public | Private |
|---|---|---|---|
| 9 µm | 0.97979 | 0.96381 | 0.92629 |
| **8 µm** | **0.98129** (P(gain) 0.83, video bootstrap) | 0.96571 | **0.92649** |
| 7.5 µm | 0.97883 | 0.96571 | 0.92573 |
| 7 µm | 0.97657 | 0.96178 | 0.92259 |

The holdout ranked the four gates exactly as the private leaderboard did; the public leaderboard could not separate
7.5 from 8 µm. A second check, pruning forks whose brighter child holds more than 1.1x the parent's chromatin,
gained +0.0013 on the holdout, read -0.0013 public and was +0.0002 private; we dropped it because of the public
reading. Both times the holdout pointed the same way as the private leaderboard.

## 7. Code

**Code:** https://github.com/impala9397-hub/kaggle-biohub-cell-tracking. It holds the builder, patch modules and
milestone recipes, the coordinate-head training script, CPU tests, figure scripts and all 158 of our submissions with
public and private scores.
`kaggle/fetch_base946.sh` downloads the base notebook and checks its hash, and
`python kaggle/build_recipe.py final-headavg-pmax8 out/final` rebuilds the final kernel; we checked that the rebuild
matches the submitted kernel except for comments, docstrings, file paths and ids. Our own weights were trained on
competition data and are not redistributed.

This was a solo entry, built with the help of AI coding agents (Claude Code and Codex) that wrote much of the code and
analysis under my direction.
