"""Charts for README.md and docs/ — run:  uv run --with matplotlib --with numpy python scripts/make_figures.py

Inputs: docs/data/submissions.csv (Kaggle leaderboard scores of our submissions) and the offline error budget numbers
recorded below (val40 replay; see docs/method.md). Outputs: docs/figures/*.svg (text as paths, so they render the same
everywhere) and, with --png, PNG copies for a quick visual check.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
CSV = ROOT / "docs" / "data" / "submissions.csv"

# Palette: validated categorical slots (blue, orange, aqua) on a light surface; text in neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"     # public leaderboard
ORANGE = "#eb6834"   # private leaderboard
AQUA = "#1baf7a"     # our own coordinate head
GRAY = "#a8a7a1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "svg.fonttype": "path",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def scores() -> dict[str, tuple[float, float]]:
    out = {}
    for r in csv.DictReader(CSV.open()):
        if r["public"] and r["private"]:
            out[r["name"]] = (float(r["public"]), float(r["private"]))
    return out


def style(ax, xgrid=False, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt(v, nd=4):
    """Signed number with a true minus sign."""
    return f"{v:+.{nd}f}".replace("-", "\u2212")


def save(fig, name, png):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.25)
    if png:
        fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# The submitted chain from the public-notebook reproduction to the final selection (each step one change).
CHAIN = [
    ("S56", "S56\npublic 0.946\nreproduced", "Reproduce the public 0.946 notebook"),
    ("S57", "S57\n+ third\ndetector", "+ our third detector"),
    ("S83", "S83\n+ division\ngate", "+ image-evidence division gate"),
    ("S86", "S86\n+ looser\nthreshold", "+ looser division-gate threshold"),
    ("N05", "N05\n+ relink\nbonus", "+ learned relink bonus"),
    ("x138flow", "x138flow\n+ flow\nrelink", "+ flow-field relink (x138)"),
    ("flow-v1284", "flow-v1284\n+ public\nV1284 head", "+ public V1284 coordinate head"),
    ("flow-v1284-headavg", "headavg\n+ own-head\naverage", "+ average with our own head"),
    ("headavg-pmax8", "final\n+ 8 µm\ndivision gate", "+ division gate 8 µm (final)"),
]


def fig_milestones(S, png):
    names = [c[0] for c in CHAIN]
    p0, q0 = S["S56"]
    pub = np.array([S[n][0] - p0 for n in names])
    prv = np.array([S[n][1] - q0 for n in names])
    own_pub, own_prv = S["flow-ownhead"][0] - p0, S["flow-ownhead"][1] - q0
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    style(ax)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.plot(x, pub, color=BLUE, linewidth=2, marker="o", markersize=7, zorder=3, label="Public LB (29% of test data)")
    ax.plot(x, prv, color=ORANGE, linewidth=2, marker="o", markersize=7, zorder=3, label="Private LB (71%, final ranking)")
    i = names.index("flow-v1284")
    xb = i + 0.42  # the alternative we did not select: our own head instead of the public head, same base
    for y0, y1, c in ((pub[i - 1], own_pub, BLUE), (prv[i - 1], own_prv, ORANGE)):
        ax.plot([i - 1, xb], [y0, y1], color=c, linewidth=1.2, linestyle=(0, (3, 2)), zorder=2)
        ax.scatter([xb], [y1], s=64, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=4)
    ax.annotate("Hollow circles: our own head instead of the\npublic head (not selected). Against the public head:\n\u22120.0015 on public, +0.0070 on private.",
                xy=(xb, own_prv), xytext=(i + 0.55, 0.0028), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkB=6))
    ax.annotate(f"final  public {S['headavg-pmax8'][0]:.5f}", xy=(x[-1], pub[-1]), xytext=(x[-1] + 0.15, pub[-1]),
                color=INK, fontsize=9.5, va="center")
    ax.annotate(f"final  private {S['headavg-pmax8'][1]:.5f}", xy=(x[-1], prv[-1]), xytext=(x[-1] + 0.15, prv[-1]),
                color=INK, fontsize=9.5, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CHAIN], fontsize=8.8, color=INK2)
    ax.set_xlim(-0.4, len(x) - 1 + 1.75)
    ax.set_ylim(-0.0015, 0.0245)
    ax.set_ylabel("Score gain over the S56 reproduction")
    ax.set_title("Public LB rejected our own head; private LB preferred it", pad=26)
    ax.text(0, 1.035, f"Each point is one submitted kernel. S56 = public {p0:.5f}, private {q0:.5f}.",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    save(fig, "lb_public_vs_private", png)


def fig_steps(S, png):
    names = [c[0] for c in CHAIN]
    labels = [c[2] for c in CHAIN[1:]]
    dpub = [S[b][0] - S[a][0] for a, b in zip(names, names[1:])]
    dprv = [S[b][1] - S[a][1] for a, b in zip(names, names[1:])]
    y = np.arange(len(labels))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    style(ax, xgrid=True, ygrid=False)
    ax.barh(y + h / 2 + 0.02, dpub, height=h, color=BLUE, label="Public LB change", zorder=3)
    ax.barh(y - h / 2 - 0.02, dprv, height=h, color=ORANGE, label="Private LB change", zorder=3)
    for yy, v in zip(y, dpub):
        ax.text(v + (0.00012 if v >= 0 else -0.00012), yy + h / 2 + 0.02, fmt(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    for yy, v in zip(y, dprv):
        ax.text(v + (0.00012 if v >= 0 else -0.00012), yy - h / 2 - 0.02, fmt(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(-0.0032, 0.0086)
    ax.set_xlabel("Score change from the previous step")
    ax.set_title("The public head was a top public gain but the only step that lost on private", pad=26)
    tot_p = S[names[-1]][0] - S[names[0]][0]
    tot_q = S[names[-1]][1] - S[names[0]][1]
    ax.text(0, 1.035, f"Submitted chain S56 \u2192 final selection: public {tot_p:+.4f}, private {tot_q:+.4f}.",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    save(fig, "lb_steps", png)


# Offline error budget: oracle edits of the x138flow pipeline output on our 40-video validation split (val40),
# scored with the organizer metric. Baseline 0.93941. Each oracle fixes one error class and keeps the rest.
ORACLES = [
    ("Edges perfect on annotated cells\n(link + insert oracles together)", 0.1304, "combo"),
    ("Every division right\n(division Jaccard = 1)", 0.0800, "div"),
    ("Link every missed edge whose\ntwo GT nodes were detected", 0.0783, "link"),
    ("   … slow movers only (< 5 µm/frame)", 0.0461, "sub"),
    ("   … fast movers only (≥ 5 µm/frame)", 0.0318, "sub"),
    ("Insert every missed GT node\nwith its GT edges", 0.0515, "det"),
    ("Delete spurious edge endpoints\n(upper bound)", 0.0420, "dup"),
    ("Delete near-duplicate nodes\n(≤ 7 µm from the true one)", 0.0061, "dup"),
    ("Predicted node count =\nestimated true count", -0.0046, "count"),
]


def fig_oracle(png):
    labels = [o[0] for o in ORACLES]
    vals = np.array([o[1] for o in ORACLES])
    kinds = [o[2] for o in ORACLES]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    style(ax, xgrid=True, ygrid=False)
    colors = [GRAY if k == "combo" else ("#86b6ef" if k == "sub" else BLUE) for k in kinds]
    ax.barh(y, vals, height=0.6, color=colors, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + (0.0015 if v >= 0 else -0.0015), yy, fmt(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=9.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.6, color=INK)
    ax.set_xlim(-0.026, 0.147)
    ax.set_xlabel("Score gain if this error class were fixed (val40 replay, baseline 0.93941)")
    ax.set_title("The missing score is in links and divisions; node count has nothing to give", pad=26)
    ax.text(0, 1.035, "Oracle edits of our pipeline output, scored with the organizer metric on 40 held-out training videos. "
            "Gray: two oracles combined.",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    save(fig, "oracle_budget", png)


# Single-variable LB experiments: (change, base). Same list as docs/results.md "public vs private deltas".
PAIRS = [
    ("S57", "S56"), ("S58", "S56"), ("S59", "S57"), ("S60", "S56"), ("S61", "S56"), ("S62", "S56"), ("S63", "S56"),
    ("S65", "S57"), ("S67", "S57"), ("S66", "S57"), ("S68", "S57"), ("S69", "S57"), ("S71", "S57"), ("S70", "S57"),
    ("S72", "S57"), ("S73", "S57"), ("S75", "S57"), ("S76", "S57"), ("S77", "S57"), ("S78", "S57"), ("S79", "S57"),
    ("S80", "S57"), ("S81", "S57"), ("S82", "S57"), ("S83", "S57"), ("S84", "S57"), ("S85", "S83"), ("S86", "S83"),
    ("S87", "S83"), ("S89", "S83"), ("S90", "S83"), ("S93", "S83"), ("S88", "S83"), ("S91", "S86"), ("S92", "S86"),
    ("S97", "S86"), ("S99", "S86"), ("S100", "S86"), ("S99-ep16", "S86"), ("S99-ep18", "S86"), ("S99-ep20", "S86"),
    ("T01", "S86"), ("T05", "S86"), ("X01", "S86"), ("N01", "S86"), ("S86-repeat", "S86"), ("X2-EP1", "S86"),
    ("X2-EP5", "S86"), ("X2-EP10", "S86"), ("N05", "S86"), ("N06-pctl060", "N05"), ("N05-t1-ratio050", "N05"),
    ("N05-rw8", "N05"), ("N05-rw128", "N05"), ("N06-pctl040", "N05"), ("motion-gate", "N05"), ("x138gap", "N05"),
    ("x138all", "N05"), ("x138flow", "N05"), ("synthdet16", "N05"), ("synthdet199-ep28", "N05"),
    ("flowdivsafe", "x138flow"), ("flow-motion", "x138flow"), ("flowdivsafe-motion", "x138flow"),
    ("x138flow-csvfloat", "x138flow"), ("flow-edgeratio", "x138flow"), ("flow-sew080", "x138flow"),
    ("flow-iter2", "x138flow"), ("flow-tight8", "x138flow"), ("flow-combo30", "x138flow"),
    ("flow-v1284late", "x138flow"), ("flow-v1284", "x138flow"), ("iter3", "x138flow"),
    ("iter2-v1284late", "flow-iter2"), ("iter2-v1284", "flow-iter2"), ("iter2-ownhead", "flow-iter2"),
    ("iter2-swapfree-v1284", "iter2-v1284"), ("flow-ownhead", "flow-v1284"), ("flow-v1284-noguard", "flow-v1284"),
    ("flow-v1284-pmax8", "flow-v1284"), ("flow-v1284-divprune-chroma", "flow-v1284"),
    ("flow-v1284-headavg", "flow-v1284"), ("iter2-swapfree-v1284-noguard", "iter2-swapfree-v1284"),
    ("headavg-pmax8", "flow-v1284-headavg"), ("headavg-pmax7", "flow-v1284-headavg"),
    ("headavg-pmax75", "flow-v1284-headavg"), ("headavg-pmax8-chroma", "headavg-pmax8"),
    ("headens5-pmax8", "headavg-pmax8"),
]


def deltas(S):
    return [(c, b, S[c][0] - S[b][0], S[c][1] - S[b][1]) for c, b in PAIRS]


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fig_scatter(S, png):
    D = deltas(S)
    dp = np.array([d[2] for d in D])
    dq = np.array([d[3] for d in D])
    lim_x, lim_y = (-0.0125, 0.0075), (-0.0125, 0.0135)
    inside = (dp >= lim_x[0]) & (dp <= lim_x[1]) & (dq >= lim_y[0]) & (dq <= lim_y[1])
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    style(ax, xgrid=True)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.plot(lim_x, lim_x, color=GRAY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.0072, -0.0081, "public change = private change", color=INK2, fontsize=9, rotation=45,
            rotation_mode="anchor", transform_rotates_text=True, ha="left", va="bottom")
    ax.scatter(dp[inside], dq[inside], s=36, color=GRAY, edgecolor=SURFACE, linewidth=1, zorder=3)
    marks = {  # highlighted submissions: (color, label, label position)
        "flow-ownhead": (AQUA, "Our own head instead of the public head", (-0.0119, 0.0088)),
        "flow-v1284": (INK, "Public V1284 head", (0.0012, -0.0062)),
        "S73": (INK, "Third detector on dense pseudo-labels", (-0.0119, 0.0124)),
        "S83": (INK, "Image-evidence division gate", (0.0006, 0.0104)),
        "S57": (INK, "Our third detector", (0.0021, 0.0042)),
        "x138flow-csvfloat": (INK, "Float CSV coordinates", (-0.0090, -0.0108)),
    }
    for c, b, x, yv in D:
        if c in marks:
            col, text, (tx, ty) = marks[c]
            ax.scatter([x], [yv], s=72, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.annotate(text, xy=(x, yv), xytext=(tx, ty), fontsize=9.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(*lim_x)
    ax.set_ylim(*lim_y)
    ax.set_xlabel("Change on the public LB (vs the kernel it modified)")
    ax.set_ylabel("Change on the private LB (vs the kernel it modified)")
    ax.set_title("Public and private agreed on large effects, not on the small ones we tuned", pad=46)
    big = (np.abs(dp) >= 0.0005) & (np.abs(dq) >= 0.0005)
    agree = int(((np.sign(dp) == np.sign(dq)) & big).sum())
    sub = (f"{len(D)} submissions, each compared with the kernel it modified.\nSpearman rank correlation "
           f"{spearman(dp, dq):.2f}; the sign agrees in {agree} of {int(big.sum())} cases where both |change| \u2265 0.0005.\n"
           f"Outside the window (same sign on both): safe divisions off {fmt(S['S80'][0] - S['S57'][0], 3)} / "
           f"{fmt(S['S80'][1] - S['S57'][1], 3)}, weak-node removal {fmt(S['N01'][0] - S['S86'][0], 3)} / "
           f"{fmt(S['N01'][1] - S['S86'][1], 3)}.")
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    save(fig, "public_vs_private_deltas", png)
    return D


def main():
    png = "--png" in sys.argv
    S = scores()
    fig_milestones(S, png)
    fig_steps(S, png)
    fig_oracle(png)
    fig_scatter(S, png)
    print("wrote", sorted(p.name for p in FIG.glob("*")))


if __name__ == "__main__":
    main()
