"""Draw docs/figures/pipeline.svg — the final pipeline, coloured by where each piece comes from.

    python scripts/make_pipeline_svg.py

Plain SVG (no dependencies). Text uses a common sans-serif stack; the layout leaves generous margins so small
font-metric differences between renderers do not cause overlaps.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "pipeline.svg"

W = 960
FONT = "Helvetica, Arial, 'DejaVu Sans', sans-serif"
INK, INK2 = "#0b0b0b", "#52514e"
BOX_FILL, BOX_STROKE = "#f4f3ef", "#d6d5cf"
SURFACE = "#fcfcfb"
SRC = {  # provenance -> (colour, legend text)
    "public": ("#a8a7a1", "public notebook lineage"),
    "ours": ("#2a78d6", "our addition"),
    "x138": ("#eb6834", "ported from the public x138 notebook (Apache-2.0)"),
}

STAGES = [
    ("Input", "", [
        ("public", "3D + time light-sheet video of one zebrafish embryo, nuclei labelled"),
        ("public", "100 frames × 64 × 256 × 256 voxels (1.625 × 0.406 × 0.406 µm)"),
    ]),
    ("1  Detect", "nuclei per frame", [
        ("public", "Primary + secondary TemporalUNet3D detectors (public weights), 8-view TTA"),
        ("ours", "Third detector: our own model (organizer recipe, epoch 100), blend weight 0.20"),
        ("public", "Peak picking on the blended logit map gives the candidate nuclei"),
    ]),
    ("2  Refine", "centres", [
        ("x138", "V1284 head: UNet features at each detection → MLP → shift of at most 2 µm"),
        ("ours", "Our own head (159 training videos, 104k pairs), averaged with the public head"),
    ]),
    ("3  Link", "frame to frame", [
        ("public", "Node-transformer edge probabilities with bidirectional fusion"),
        ("public", "Integer linear program (tracksdata) selects nodes and edges"),
    ]),
    ("4  Repair", "the graph", [
        ("x138", "Motion relink on a local flow field (median displacement of nearby cells)"),
        ("ours", "Learned-probability bonus on every relink candidate"),
        ("public", "Gap closing with a DeepCenter veto"),
        ("ours", "Division repair gated by image evidence z(dip) + z(sym) − z(div)"),
        ("ours", "Parent-to-daughter distance gate 9 → 8 µm"),
        ("public", "Short-track filter and line-fit smoothing"),
    ]),
    ("Output", "", [
        ("public", "submission.csv: nodes (t, z, y, x in integer voxels) and edges"),
    ]),
]

X0, X1 = 20, W - 20          # stage box extent
TITLE_X = X0 + 18            # stage title column
ITEM_X = X0 + 190            # item column (chip)
ROW = 27                     # item row height
PAD_T, PAD_B = 16, 12        # box padding
GAP = 30                     # vertical gap between boxes (holds the arrow)


def text(x, y, s, size=13.5, weight="normal", fill=INK, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def main() -> None:
    parts = []
    y = 20
    # legend
    lx = X0
    for key in ("public", "ours", "x138"):
        col, label = SRC[key]
        parts.append(f'<rect x="{lx}" y="{y}" width="14" height="14" rx="3" fill="{col}"/>')
        parts.append(text(lx + 22, y + 12, label, size=13, fill=INK2))
        lx += 22 + int(len(label) * 7.0) + 34
    y += 40
    centers = []
    for title, sub, items in STAGES:
        h = PAD_T + len(items) * ROW + PAD_B - 6
        parts.append(f'<rect x="{X0}" y="{y}" width="{X1 - X0}" height="{h}" rx="10" fill="{BOX_FILL}" '
                     f'stroke="{BOX_STROKE}" stroke-width="1"/>')
        ty = y + PAD_T + 13
        parts.append(text(TITLE_X, ty, title, size=15, weight="bold"))
        if sub:
            parts.append(text(TITLE_X, ty + 19, sub, size=12.5, fill=INK2))
        for k, (src, s) in enumerate(items):
            iy = y + PAD_T + k * ROW
            col = SRC[src][0]
            parts.append(f'<rect x="{ITEM_X}" y="{iy + 1}" width="6" height="17" rx="2" fill="{col}"/>')
            parts.append(text(ITEM_X + 16, iy + 14, s, size=13.5, fill=INK))
        centers.append((y, y + h))
        y += h + GAP
    # arrows between consecutive boxes, in the title column
    ax = TITLE_X + 40
    for (_, bottom), (top, _) in zip(centers, centers[1:]):
        y0, y1 = bottom + 4, top - 4
        parts.append(f'<line x1="{ax}" y1="{y0}" x2="{ax}" y2="{y1 - 7}" stroke="{INK2}" stroke-width="1.6"/>')
        parts.append(f'<path d="M {ax - 5} {y1 - 8} L {ax + 5} {y1 - 8} L {ax} {y1} Z" fill="{INK2}"/>')
    height = y - GAP + 20
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
           f'role="img" aria-label="Pipeline of the final submission: detect, refine centres, link, repair">\n'
           f'<rect width="{W}" height="{height}" fill="{SURFACE}"/>\n' + "\n".join(parts) + "\n</svg>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg)
    print(f"wrote {OUT} ({W} x {height})")


if __name__ == "__main__":
    main()
