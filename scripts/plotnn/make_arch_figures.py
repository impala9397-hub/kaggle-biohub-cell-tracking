"""Architecture figures in the PlotNeuralNet style (3D layer blocks).

Requires a local clone of PlotNeuralNet (MIT, https://github.com/HarisIqbal88/PlotNeuralNet;
only its `layers/` TikZ styles are used) and a LaTeX engine (tested with tectonic 0.17).
PlotNeuralNet is not vendored here.

    PLOTNN_DIR=/path/to/PlotNeuralNet python scripts/plotnn/make_arch_figures.py

Writes docs/figures/arch_detector_linker.{pdf,png} and arch_coordinate_head.{pdf,png}.
PNG rasterisation uses macOS `qlmanage`; on other systems convert the PDF with any tool.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
PLOTNN = Path(os.environ.get("PLOTNN_DIR", "")).expanduser()
SCALE = 0.2  # PlotNeuralNet box scale (layers/Box.sty)

COLORS = r"""
\definecolor{convc}{RGB}{255,214,140}
\definecolor{reluc}{RGB}{255,186,112}
\definecolor{poolc}{RGB}{204,72,52}
\definecolor{unpoolc}{RGB}{86,128,186}
\definecolor{attnc}{RGB}{130,205,170}
\definecolor{attnbandc}{RGB}{46,150,110}
\definecolor{linc}{RGB}{176,156,230}
\definecolor{linreluc}{RGB}{140,110,210}
\definecolor{headc}{RGB}{200,90,170}
\definecolor{outc}{RGB}{130,40,120}
\definecolor{sumc}{RGB}{60,180,110}
\definecolor{inputc}{RGB}{215,215,215}
"""


def head() -> str:
    layers = (PLOTNN / "layers").as_posix() + "/"
    return r"""\documentclass[border=14pt, multi, tikz]{standalone}
\usepackage{import}
\usepackage{amsmath,amssymb}
\subimport{""" + layers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d,calc}
""" + COLORS + r"""
\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
\tikzstyle{lab}=[font=\large, align=center, anchor=north]
\tikzstyle{note}=[font=\large, align=left]
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}
"""


def end() -> str:
    return "\n\\end{tikzpicture}\n\\end{document}\n"


def box(name, at, fill, h, d, w, xlabel="", zlabel="", offset="(0,0,0)", opacity=0.9):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{Box={{name={name}, caption={{ }}, xlabel={{{{{xlabel}, }}}}, zlabel={{{zlabel}}},
           fill={fill}, opacity={opacity}, height={h}, width={w}, depth={d}}}}};"""


def banded(name, at, fill, band, h, d, w1, w2, xl1="", xl2="", zlabel="", offset="(0,0,0)", opacity=0.9):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{RightBandedBox={{name={name}, caption={{ }}, xlabel={{{{{xl1}, {xl2}}}}}, zlabel={{{zlabel}}},
           fill={fill}, bandfill={band}, opacity={opacity}, height={h}, width={{{w1}, {w2}}}, depth={d}}}}};"""


def ball(name, at, logo, radius=2.0, offset="(0,0,0)"):
    return rf"""
\pic[shift={{{offset}}}] at {at}
    {{Ball={{name={name}, fill=sumc, opacity=0.7, radius={radius}, logo={{{logo}}}}}}};"""


def cap(name, text, depth, width_cm=4.0, dy=-1.25, dx=0.0):
    """Label under a box: anchored below the front-bottom edge (depth = unscaled box depth)."""
    return rf"""
\node[lab, text width={width_cm}cm] at ($({name}-south)+({dx},{dy},{depth * SCALE / 2:.2f})$) {{{text}}};"""


def conn(a, b):
    return rf"""
\draw [connection] ({a}-east) -- node {{\midarrow}} ({b}-west);"""


def skip(a, b, pos=1.25):
    return rf"""
\path ({a}-southeast) -- ({a}-northeast) coordinate[pos={pos}] ({a}-top);
\path ({b}-south) -- ({b}-north) coordinate[pos={pos}] ({b}-top);
\draw [copyconnection] ({a}-northeast) -- node {{\copymidarrow}} ({a}-top) -- node {{\copymidarrow}} ({b}-top) -- node {{\copymidarrow}} ({b}-north);"""


def legend(x, y, items, dx):
    out = []
    for i, (color, text) in enumerate(items):
        xi = x + i * dx
        out.append(rf"""
\fill[fill={color}, draw=black!45] ({xi},{y}) rectangle ++(0.75,0.75);
\node[anchor=west, font=\large] at ({xi + 0.95},{y + 0.37}) {{{text}}};""")
    return "".join(out)


def fig_detector_linker() -> str:
    a = [head()]
    a.append(r"""
\node[anchor=west, font=\LARGE] at (-1.5,9.6) {\textbf{UNetNodeTransformer: find cells in two frames, then score every $t \rightarrow t{+}1$ link}};
\node[anchor=west, font=\Large] at (-1.5,8.3) {2,076,706 parameters $=$ 3D U-Net 1,496,320 $+$ detection head 33 $+$ node transformer 580,353. All three detectors in our blend share this architecture.};""")
    # Row 1: temporal 3D U-Net + detection head
    a.append(box("inp", "(0,0,0)", "inputc", 40, 40, 1.0, opacity=0.7))
    a.append(box("inp2", "(inp-east)", "inputc", 40, 40, 1.0, offset="(0.45,0,0)", zlabel="$64^3$", opacity=0.7))
    a.append(cap("inp", r"\textbf{Input}\\frames $t$ and $t{+}1$", 40, 4.0, dx=0.3))
    a.append(banded("enc1", "(inp2-east)", "convc", "reluc", 40, 40, 2, 2, "32", "32", "$64^3$", offset="(2.4,0,0)"))
    a.append(conn("inp2", "enc1"))
    a.append(cap("enc1", r"\textbf{Encoder 1}", 40, 3.2))
    a.append(box("p1", "(enc1-east)", "poolc", 30, 30, 1, opacity=0.55))
    a.append(banded("enc2", "(p1-east)", "convc", "attnbandc", 30, 30, 3.5, 3.5, "64", "64", "$32^3$", offset="(2.6,0,0)"))
    a.append(conn("p1", "enc2"))
    a.append(cap("enc2", r"\textbf{Encoder 2}\\+ temporal attention", 30, 4.2))
    a.append(box("p2", "(enc2-east)", "poolc", 20, 20, 1, opacity=0.55))
    a.append(banded("bott", "(p2-east)", "convc", "attnbandc", 20, 20, 5, 5, "128", "128", "$16^3$", offset="(2.6,0,0)"))
    a.append(conn("p2", "bott"))
    a.append(cap("bott", r"\textbf{Bottleneck}\\+ temporal attention", 20, 4.2))
    a.append(box("u1", "(bott-east)", "unpoolc", 30, 30, 1, offset="(2.6,0,0)", opacity=0.6))
    a.append(conn("bott", "u1"))
    a.append(banded("dec1", "(u1-east)", "convc", "reluc", 30, 30, 3.5, 3.5, "64", "64", "$32^3$"))
    a.append(skip("enc2", "dec1", 1.25))
    a.append(cap("dec1", r"\textbf{Decoder 1}", 30, 3.2))
    a.append(box("u2", "(dec1-east)", "unpoolc", 40, 40, 1, offset="(2.6,0,0)", opacity=0.6))
    a.append(conn("dec1", "u2"))
    a.append(banded("dec2", "(u2-east)", "convc", "reluc", 40, 40, 2, 2, "32", "32", "$64^3$"))
    a.append(skip("enc1", "dec2", 1.25))
    a.append(cap("dec2", r"\textbf{Decoder 2}\\feature map", 40, 3.4))
    a.append(box("det", "(dec2-east)", "headc", 40, 40, 1.2, xlabel="1", zlabel="$64^3$", offset="(4.6,0,0)", opacity=0.85))
    a.append(conn("dec2", "det"))
    a.append(cap("det", r"\textbf{Detection head}\\1$\times$1$\times$1 conv + sigmoid", 40, 4.8, dx=0.9))
    a.append(r"""
\node[note, anchor=west] at ($(det-east)+(2.4,0.9,0)$) {cell-probability map};
\node[note, anchor=west] at ($(det-east)+(2.4,-0.3,0)$) {local peaks $\rightarrow$ $N_t$, $N_{t+1}$ cells};""")
    # Row 2: node transformer linker
    y = -15.5
    a.append(box("tok", f"(1.5,{y},0)", "linc", 26, 6, 2.4, xlabel="64", zlabel="$N$", opacity=0.9))
    a.append(cap("tok", r"\textbf{Node tokens}\\32 U-Net + 32 position", 6, 4.4))
    a.append(r"""
\draw[copyconnection, -Stealth] ($(dec2-south)+(0,-3.6,0)$) -- ++(0,-1.4,0) coordinate (kk) -| ($(tok-north)+(0,0.5,0)$);
\node[font=\large, fill=white, inner sep=3pt] at ($(kk)!0.5!(kk -| tok-north)$) {U-Net features of frame $t$ and $t{+}1$ sampled at each peak};""")
    a.append(box("emb", "(tok-east)", "linc", 26, 6, 3.6, xlabel="128", zlabel="$N$", offset="(4.6,0,0)", opacity=0.9))
    a.append(conn("tok", "emb"))
    a.append(cap("emb", r"\textbf{Linear}\\64$\rightarrow$128 + LayerNorm", 6, 4.2))
    prev = "emb"
    for i in range(1, 5):
        name = f"blk{i}"
        a.append(banded(name, f"({prev}-east)", "attnc", "attnbandc", 26, 6, 2.2, 2.2, "128", "",
                        "$N$" if i == 4 else "", offset="(1.1,0,0)" if i > 1 else "(2.8,0,0)", opacity=0.9))
        a.append(conn(prev, name))
        prev = name
    a.append(r"""
\node[lab, text width=9cm] at ($(blk2-south)!0.5!(blk3-south)+(0,-1.25,0.6)$) {\textbf{4 cross-attention blocks}\\each frame attends to the other, 4 heads};""")
    a.append(box("pair1", f"({prev}-east)", "linreluc", 22, 22, 4.0, xlabel="128", zlabel="", offset="(3.4,0,0)", opacity=0.85))
    a.append(conn(prev, "pair1"))
    a.append(box("pair2", "(pair1-east)", "linreluc", 22, 22, 2.4, xlabel="64", offset="(0.7,0,0)", opacity=0.85))
    a.append(box("pair3", "(pair2-east)", "linc", 22, 22, 0.8, xlabel="1", offset="(0.7,0,0)", opacity=0.9))
    a.append(r"""
\node[lab, text width=6cm] at ($(pair2-south)+(0,-1.25,2.2)$) {\textbf{Pair MLP}, for every pair $(i,j)$\\259$\rightarrow$128$\rightarrow$64$\rightarrow$1};""")
    a.append(box("smx", "(pair3-east)", "outc", 22, 22, 0.8, offset="(4.4,0,0)", opacity=0.85))
    a.append(conn("pair3", "smx"))
    a.append(cap("smx", r"\textbf{Softmax}\\over candidate parents", 22, 5.2, dx=0.6))
    a.append(r"""
\node[note, anchor=west] at ($(smx-east)+(1.2,0.9,0)$) {edge probability,};
\node[note, anchor=west] at ($(smx-east)+(1.2,-0.3,0)$) {$N_t \times N_{t+1}$ matrix};""")
    a.append(legend(-1.5, -25.0, [
        ("convc", r"2 $\times$ (Conv $3^3$, BatchNorm, ReLU)"),
        ("attnbandc", "temporal / cross attention"),
        ("poolc", r"MaxPool $\div 2$"),
        ("unpoolc", r"Upsample $\times 2$, concat skip"),
        ("linc", "Linear / tokens"),
        ("headc", "sigmoid head"),
        ("outc", "softmax output"),
    ], dx=8.9))
    a.append(end())
    return "".join(a)


def fig_coordinate_head() -> str:
    a = [head()]
    a.append(r"""
\node[anchor=west, font=\LARGE] at (-2.5,12.6) {\textbf{Coordinate head: move every detected centre by at most 2\,$\mu$m before linking}};
\node[anchor=west, font=\Large] at (-2.5,11.3) {Same 224 features, two small MLPs, bounded shifts averaged. Change vs no head (public / private LB): public head $+0.005$ / $-0.002$, our head $+0.003$ / $+0.005$, average $+0.007$ / $+0.001$.};""")
    a.append(banded("feat", "(0,0,0)", "convc", "reluc", 36, 36, 2, 2, "32", "", "$64^3$"))
    a.append(cap("feat", r"\textbf{U-Net feature map}\\(primary detector, frame $t$)", 36, 5.0))
    a.append(r"""
\fill[black] ($(feat-anchor)+(0.25,0.9,0.6)$) circle (4pt) coordinate (pk);
\draw[black, thick] (pk) -- ++(-1.6,3.6,0) node[above, font=\large] {detected peak $p$};""")
    a.append(box("vec", "(feat-east)", "linc", 30, 4, 1.6, xlabel="224", offset="(4.0,0,0)", opacity=0.9))
    a.append(conn("feat", "vec"))
    a.append(cap("vec", r"\textbf{7 samples}\\at $p$ and $p\pm1$ voxel in $z,y,x$:\\centre 32 + six differences 6$\times$32", 4, 6.2))
    heads = [
        ("pub", r"\textbf{Public V1284 head}\\CC0, trained on 20 videos", 5.4, "above"),
        ("own", r"\textbf{Our head}\\159 train videos, 103,806 detection--GT pairs", -5.4, "below"),
    ]
    for nm, title, yoff, where in heads:
        a.append(box(f"{nm}1", "(vec-east)", "linreluc", 14, 4, 2.6, xlabel="32", offset=f"(5.2,{yoff},0)", opacity=0.9))
        a.append(box(f"{nm}2", f"({nm}1-east)", "linc", 6, 4, 1.0, xlabel="3", offset="(1.3,0,0)", opacity=0.9))
        a.append(rf"""
\draw [connection] (vec-east) -- node {{\midarrow}} ({nm}1-west);""")
        desc = r"Linear 224$\rightarrow$32, SiLU, Linear 32$\rightarrow$3"
        if where == "above":
            a.append(rf"""
\node[font=\large, align=center, anchor=south] at ($({nm}1-north)!0.5!({nm}2-north)+(0,0.9,0)$) {{{title}\\[2pt]{desc}}};""")
        else:
            a.append(rf"""
\node[font=\large, align=center, anchor=north] at ($({nm}1-south)!0.5!({nm}2-south)+(0,-1.4,0.4)$) {{{title}\\[2pt]{desc}}};""")
    a.append(ball("avg", "(vec-east)", r"$\scriptstyle\frac{1}{2}\Sigma$", radius=2.6, offset="(14.6,0,0)"))
    a.append(r"""
\draw [connection] (pub2-east) -- node {\midarrow} (avg-west);
\draw [connection] (own2-east) -- node {\midarrow} (avg-west);
\node[font=\large, align=center, anchor=south] at ($(avg-north)+(0,0.7,0)$) {mean of the two\\bounded shifts};""")
    a.append(box("out", "(avg-east)", "outc", 8, 4, 1.0, xlabel="3", offset="(3.0,0,0)", opacity=0.85))
    a.append(conn("avg", "out"))
    a.append(r"""
\node[note, anchor=west] at ($(out-east)+(1.2,0,0)$) {\textbf{refined centre} $p+\Delta$\\float coordinates, $\lVert\Delta\rVert \le 2\,\mu$m\\[5pt]used everywhere downstream:\\trilinear features for the linker,\\relink distances, final CSV};""")
    a.append(r"""
\node[font=\large, align=center, anchor=north] at ($(avg-south)+(0,-3.4,0)$) {each head's raw output $\mathbf{d}$ is bounded\\as $2\,\mathbf{d}/(1+\lVert\mathbf{d}\rVert)$ $\mu$m};""")
    a.append(legend(-2.5, -13.2, [
        ("convc", "U-Net feature map"),
        ("linc", "feature vector / Linear"),
        ("linreluc", "Linear + SiLU"),
        ("sumc", "average of two heads"),
        ("outc", "3D shift"),
    ], dx=8.8))
    a.append(end())
    return "".join(a)


def build(name: str, tex: str) -> None:
    """Compile in a temporary directory so the .tex (which embeds the local PlotNeuralNet path) is not kept."""
    import tempfile

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / f"{name}.tex").write_text(tex)
        r = subprocess.run(["tectonic", "-X", "compile", f"{name}.tex"], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"tectonic failed for {name}")
        shutil.copy(tmp / f"{name}.pdf", OUT / f"{name}.pdf")
        subprocess.run(["qlmanage", "-t", "-s", "3000", "-o", str(tmp), str(tmp / f"{name}.pdf")], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(tmp / f"{name}.pdf.png", OUT / f"{name}.png")
    print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT), (OUT / f"{name}.png").relative_to(ROOT))


def main() -> int:
    if not (PLOTNN / "layers" / "init.tex").is_file():
        print("Set PLOTNN_DIR to a PlotNeuralNet clone (https://github.com/HarisIqbal88/PlotNeuralNet)", file=sys.stderr)
        return 1
    build("arch_detector_linker", fig_detector_linker())
    build("arch_coordinate_head", fig_coordinate_head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
