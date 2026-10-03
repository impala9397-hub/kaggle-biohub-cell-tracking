"""Train our own V1284-format coordinate-regression head: pairs -> by-movie cross-validation -> final fit. CPU only.

The head has exactly the layout of the public V1284 head of the x138 notebook (MLP 224 -> h -> 3, last layer
zero-initialised, output bounded to <= 2 µm), so the kernel can load it as a drop-in replacement
(kaggle/v1284_946.py, build env V1284_HEAD_DATASET / V1284_HEAD2_DATASET).

Inputs:
  * a feature capture: one <stem>.npz per training video, written on Kaggle by the capture kernel
    (kaggle/coordhead_capture_946.py). Per detection row: `coords` (t, z, y, x at full resolution, = geff node rows),
    `grid` (downsampled coordinates) and `feats` (the 224-d UNet feature vector the head reads at inference).
  * the organizer's ground truth for the same videos: <gt-dir>/<stem>.geff (competition train data, not included here).
  * a validation list (one stem per line). Those videos are never used for training; they are scored separately.

Commands:
  python coordhead/train_coordhead.py pairs --capture <dir> --gt-dir <train-dir> --val-list <file> --out pairs.npz [--radius 4]
      Per frame, GT nodes are matched to detections by optimal 1:1 assignment within 7 µm (the metric's gate); a pair
      is kept when its distance is <= radius µm. target = GT µm - detection µm (z, y, x), with the scale read from
      the geff (1.625 / 0.40625 / 0.40625 µm).
  python coordhead/train_coordhead.py cv --pairs pairs.npz --public <v1284_head.pt> --out <dir>
      5-fold by-movie CV on the non-validation movies (folds balanced per embryo). Centre error = mean ||target - pred||
      in µm. Rows: none / public V1284 (not held out: its training movies are unknown) / ridge (unbounded) / MLP h32 /
      h64 / variants. Every model is also scored on the validation pairs.
  python coordhead/train_coordhead.py fit --pairs pairs.npz --hidden 32 --out v1284_head.pt [--seed 0]
      Final fit on all non-validation pairs; saves {'state_dict', 'mean', 'scale'} (the x138 module layout).

Our result (159 training movies, 103,806 pairs; see docs/method.md): validation centre error 1.712 µm (none) ->
1.534 (public head) -> 1.248 (ours, h32).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

MATCH_UM = 7.0


def read_stems(path: str | Path) -> set[str]:
    return set(Path(path).read_text().split())


def read_geff_nodes(path: str | Path) -> dict:
    """Ground-truth nodes of one organizer .geff (zarr v3 group): frame t, position in µm and the axis scale."""
    import zarr
    g = zarr.open_group(str(path), mode="r")
    axes = {a["name"]: a for a in g.attrs["geff"]["axes"]}
    sc = np.array([float(axes[a]["scale"]) for a in ("z", "y", "x")], dtype=np.float64)
    t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
    zyx = np.stack([np.asarray(g[f"nodes/props/{a}/values"][:], dtype=np.float64) for a in ("z", "y", "x")], axis=1)
    return {"t": t, "um": zyx * sc, "sc": sc}


# ------------------------------------------------------------------------------------------------------------ pairs
def match_pairs(C: np.ndarray, G: dict, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Per-frame optimal 1:1 matching of GT nodes to detections within MATCH_UM; keep pairs <= radius.
    C: detections (N, 4) = t, z, y, x in voxels. Returns (detection index, target µm, distance µm, n_gt, n_match7)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    det_um = C[:, 1:].astype(np.float64) * G["sc"]
    out_i, out_tg, out_d, n_gt, n_m7 = [], [], [], 0, 0
    for t in np.unique(G["t"]):
        gsel = np.flatnonzero(G["t"] == t)
        dsel = np.flatnonzero(C[:, 0] == t)
        n_gt += len(gsel)
        if not len(dsel):
            continue
        tree = cKDTree(det_um[dsel])
        g_um = G["um"][gsel]
        big = 1e6
        cost = np.full((len(gsel), len(dsel)), big)
        for a, nbrs in enumerate(tree.query_ball_point(g_um, MATCH_UM)):
            for b in nbrs:
                cost[a, b] = np.linalg.norm(g_um[a] - det_um[dsel[b]])
        keep_r = np.flatnonzero((cost < big).any(1))
        keep_c = np.flatnonzero((cost < big).any(0))
        if not len(keep_r):
            continue
        sub = cost[np.ix_(keep_r, keep_c)]
        r, c = linear_sum_assignment(sub)
        for a, b in zip(r, c):
            d = sub[a, b]
            if d >= big:
                continue
            n_m7 += 1
            if d <= radius:
                i = dsel[keep_c[b]]
                out_i.append(i)
                out_tg.append(g_um[keep_r[a]] - det_um[i])
                out_d.append(d)
    return (np.array(out_i, np.int64), np.array(out_tg, np.float32).reshape(-1, 3), np.array(out_d, np.float32),
            n_gt, n_m7)


def pairs_one(args):
    path, gt_dir, radius = args
    import warnings
    warnings.filterwarnings("ignore")
    stem = Path(path).stem
    with np.load(path) as z:
        C, feats = z["coords"], z["feats"]
    G = read_geff_nodes(Path(gt_dir) / f"{stem}.geff")
    if not np.allclose(G["sc"], [1.625, 0.40625, 0.40625]):
        raise ValueError(f"{stem}: unexpected scale {G['sc']}")
    idx, target, dist, n_gt, n_m7 = match_pairs(C, G, radius)
    return dict(stem=stem, idx=idx, feats=np.asarray(feats[idx], np.float32), target=target, dist=dist,
                n_gt=n_gt, n_match7=n_m7, n_det=len(C))


def cmd_pairs(a):
    from multiprocessing import Pool
    files = sorted(Path(a.capture).glob("*.npz"))
    t0 = time.time()
    with Pool(a.procs) as pool:
        R = pool.map(pairs_one, [(str(f), a.gt_dir, a.radius) for f in files])
    val = read_stems(a.val_list)
    stems = np.array([r["stem"] for r in R])
    cnt = np.array([len(r["idx"]) for r in R])
    np.savez(a.out, stems=stems, counts=cnt,
             feats=np.concatenate([r["feats"] for r in R]).astype(np.float32),
             target=np.concatenate([r["target"] for r in R]), dist=np.concatenate([r["dist"] for r in R]),
             movie=np.repeat(np.arange(len(R)), cnt), is_val=np.array([s in val for s in stems]),
             n_gt=np.array([r["n_gt"] for r in R]), n_match7=np.array([r["n_match7"] for r in R]),
             n_det=np.array([r["n_det"] for r in R]), radius=np.float32(a.radius))
    tr = [r for r in R if r["stem"] not in val]
    va = [r for r in R if r["stem"] in val]
    for name, rr in (("train", tr), ("val", va)):
        if rr:
            n = sum(len(r["idx"]) for r in rr)
            d = np.concatenate([r["dist"] for r in rr]) if n else np.zeros(0)
            print(f"{name}: movies={len(rr)} pairs={n} gt_nodes={sum(r['n_gt'] for r in rr)} "
                  f"matched7={sum(r['n_match7'] for r in rr)} dist_mean={d.mean():.3f} median={np.median(d):.3f}")
    print(f"pairs done in {time.time() - t0:.0f}s -> {a.out}")


# ------------------------------------------------------------------------------------------------------------ models
def _torch():
    import torch
    torch.set_num_threads(8)
    return torch


def _module_ns():
    """The x138 coordinate-refinement module text (bounded(), make_head(), ...) exactly as the kernel runs it."""
    import v1284_946 as v1
    ns = {}
    exec(v1.X138_MODULE_SRC, ns)
    return ns


def make_head(hidden):
    torch = _torch()
    head = torch.nn.Sequential(torch.nn.Linear(224, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def fit_mlp(X, Y, hidden=32, epochs=40, lr=2e-3, wd=1e-4, batch=512, loss="l2", seed=0, w=None):
    torch = _torch()
    bounded = _module_ns()["bounded"]
    torch.manual_seed(seed)
    mean = X.mean(0)
    scale = X.std(0) + 1e-6
    Xs = torch.as_tensor((X - mean) / scale, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    Wt = torch.as_tensor(np.ones(len(X)) if w is None else w / np.mean(w), dtype=torch.float32)
    head = make_head(hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, batch):
            b = perm[k:k + batch]
            pred = bounded(head, Xs[b])
            diff = pred - Yt[b]
            if loss == "l2":        # mean Euclidean error (the reported centre error)
                l = (Wt[b] * torch.sqrt((diff ** 2).sum(1) + 1e-8)).mean()
            else:                   # mse
                l = (Wt[b] * (diff ** 2).sum(1)).mean()
            opt.zero_grad()
            l.backward()
            opt.step()
        sched.step()
    head.eval()
    return head, torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)


def predict(model, X):
    torch = _torch()
    bounded = _module_ns()["bounded"]
    head, mean, scale = model
    with torch.no_grad():
        return bounded(head, (torch.as_tensor(X, dtype=torch.float32) - mean) / scale).numpy()


def load_pt(path):
    torch = _torch()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    h = saved["state_dict"]["0.weight"].shape[0]
    head = make_head(h)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head, saved["mean"].float(), saved["scale"].float()


def fit_ridge(X, Y, lam=10.0):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Xs = np.c_[(X - mean) / scale, np.ones(len(X))]
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, Xs.T @ Y)
    return lambda Z: np.c_[(Z - mean) / scale, np.ones(len(Z))] @ W


def _bal(emb_rows):
    """Row weights so that each embryo carries the same total weight (6bba has ~6x the pairs of 44b6)."""
    w = np.ones(len(emb_rows))
    for e in np.unique(emb_rows):
        w[emb_rows == e] = len(emb_rows) / (len(np.unique(emb_rows)) * (emb_rows == e).sum())
    return w


def _err(Y, P):
    return np.linalg.norm(Y - P, axis=1)


def cmd_cv(a):
    D = np.load(a.pairs)
    X, Y, movie, is_val = D["feats"], D["target"].astype(np.float64), D["movie"], D["is_val"]
    stems = D["stems"]
    emb = np.array([s[:4] for s in stems])
    tr_m = [m for m in range(len(stems)) if not is_val[m]]
    rng = np.random.default_rng(0)
    fold_of = {}
    for e in ("44b6", "6bba"):
        ms = [m for m in tr_m if emb[m] == e]
        rng.shuffle(ms)
        for k, m in enumerate(ms):
            fold_of[m] = k % a.folds
    row_fold = np.array([fold_of.get(int(m), -1) for m in movie])
    val_rows = is_val[movie]
    tr_rows = ~val_rows
    public = load_pt(a.public)
    configs = {"mlp32_l2": dict(hidden=32, loss="l2"), "mlp64_l2": dict(hidden=64, loss="l2"),
               "mlp32_mse": dict(hidden=32, loss="mse"),
               "mlp32_l2_bal": dict(hidden=32, loss="l2", bal=True), "mlp64_l2_bal": dict(hidden=64, loss="l2", bal=True)}
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only.split(",")}
    if a.quick:
        configs = {"mlp32_l2": configs["mlp32_l2"]}
    P = {k: np.full((len(X), 3), np.nan) for k in ["ridge", *configs]}
    t0 = time.time()
    for f in range(a.folds):
        trn = tr_rows & (row_fold != f)
        tst = tr_rows & (row_fold == f)
        P["ridge"][tst] = fit_ridge(X[trn], Y[trn])(X[tst])
        for name, cfg in configs.items():
            m = fit_mlp(X[trn], Y[trn], epochs=a.epochs, w=_bal(emb[movie][trn]) if cfg.get("bal") else None,
                        **{k: v for k, v in cfg.items() if k != "bal"})
            P[name][tst] = predict(m, X[tst])
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    # validation rows: models fit on all training rows (as the final head would be)
    P["ridge"][val_rows] = fit_ridge(X[tr_rows], Y[tr_rows])(X[val_rows])
    for name, cfg in configs.items():
        m = fit_mlp(X[tr_rows], Y[tr_rows], epochs=a.epochs, w=_bal(emb[movie][tr_rows]) if cfg.get("bal") else None,
                    **{k: v for k, v in cfg.items() if k != "bal"})
        P[name][val_rows] = predict(m, X[val_rows])
    P["public"] = predict(public, X)
    P["none"] = np.zeros_like(Y)
    names = ["none", "public", "ridge", *configs]
    out = {}
    for split, rows in (("train_cv", tr_rows), ("val", val_rows)):
        for e in ("all", "44b6", "6bba"):
            sel = rows & ((emb[movie] == e) if e != "all" else True)
            if not sel.any():
                continue
            base = _err(Y[sel], P["none"][sel]).mean()
            r = {"pairs": int(sel.sum()), "movies": int(len(np.unique(movie[sel])))}
            for n in names:
                err = _err(Y[sel], P[n][sel])
                # per-movie improvement count vs none
                mv = movie[sel]
                per = [(_err(Y[sel][mv == m], P[n][sel][mv == m]).mean() < _err(Y[sel][mv == m], 0 * Y[sel][mv == m]).mean())
                       for m in np.unique(mv)]
                r[n] = {"err_um": float(err.mean()), "rel": float(err.mean() / base - 1), "movies_improved": int(sum(per))}
            out[f"{split}/{e}"] = r
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "cv.json").write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        print(k, f"pairs={r['pairs']} movies={r['movies']}",
              " | ".join(f"{n} {r[n]['err_um']:.4f} ({100 * r[n]['rel']:+.1f}%, {r[n]['movies_improved']}/{r['movies']})" for n in names))


def cmd_fit(a):
    torch = _torch()
    D = np.load(a.pairs)
    tr_rows = ~D["is_val"][D["movie"]]
    X, Y = D["feats"][tr_rows], D["target"][tr_rows].astype(np.float64)
    emb = np.array([s[:4] for s in D["stems"]])[D["movie"]][tr_rows]
    head, mean, scale = fit_mlp(X, Y, hidden=a.hidden, epochs=a.epochs, loss=a.loss, seed=a.seed,
                                w=_bal(emb) if a.balanced else None)
    torch.save({"state_dict": head.state_dict(), "mean": mean, "scale": scale}, a.out)
    print(f"saved {a.out}: pairs={len(X)} hidden={a.hidden} loss={a.loss} epochs={a.epochs} balanced={a.balanced}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pairs"); q.add_argument("--capture", required=True); q.add_argument("--out", required=True)
    q.add_argument("--gt-dir", required=True, help="directory with the organizer's <stem>.geff ground truth")
    q.add_argument("--val-list", required=True, help="file with validation stems (never trained on)")
    q.add_argument("--radius", type=float, default=4.0); q.add_argument("--procs", type=int, default=6)
    c = sub.add_parser("cv"); c.add_argument("--pairs", required=True); c.add_argument("--public", required=True)
    c.add_argument("--out", required=True); c.add_argument("--folds", type=int, default=5)
    c.add_argument("--epochs", type=int, default=40); c.add_argument("--quick", action="store_true")
    c.add_argument("--only", default="")
    f = sub.add_parser("fit"); f.add_argument("--pairs", required=True); f.add_argument("--out", required=True)
    f.add_argument("--hidden", type=int, default=32); f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--loss", default="l2"); f.add_argument("--seed", type=int, default=0)
    f.add_argument("--balanced", action="store_true")
    a = p.parse_args()
    {"pairs": cmd_pairs, "cv": cmd_cv, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
