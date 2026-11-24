"""Figures of the article, drawn from the saved measures of a run.

Usage::

    python scripts/figures_article.py results/d1-2b/metrics/metrics.json article/figures \
        --probe results/d1-2b-world/probe/probe.json

Needs matplotlib and numpy (not dependencies of the package).

For print, the four charts can be drawn on white::

    python scripts/figures_article.py results/d1-2b/metrics/metrics.json figures-print \
        --probe results/d1-2b-world/probe/probe.json --theme light --only steps confusion conflict reasoning-map

Colour code: by default every figure uses the "deep sea" style of the cover, a dark blue background with light
text; the light theme keeps the same roles on a white page. Colour carries judgement, with the meaning readers
expect: green for right answers, red for errors, never the only signal (labels and symbols always say the same).
Both errors weigh the same, so they share one red. The green-red pair on the dark background passes the
colour-vision-deficiency check only with a second signal, which is always present; the pair of the light theme
passes it on its own (worst simulated difference 10.4, deuteranopia).

Text must always fit: before saving, :func:`check_layout` stops with an error if any text leaves the
figure or its box, or touches another text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

# --- colour code -------------------------------------------------------------------------------------------
SURFACE = "#0b2a3f"  # figure background
INK, INK2, MUTED = "#f3f8fb", "#c4d9e6", "#8fa9b9"  # main, secondary and faint text
GRID, BASELINE = "#1f4761", "#4d6f84"  # grids and axes
GRID_LIGHT = "#1a4058"  # background grid of stacked panels
RIGHT, WRONG = "#48ab70", "#d9503f"  # right answer, error
CARD, PANEL = "#123850", "#0e3046"  # boxes and secondary boxes
PAPER = "#f3f8fb"  # light sheet: the rulebook in the diagrams, as on the cover
PAPER_MUTED = "#5a7486"  # faint text on the light sheet (contrast about 4.6:1)
HIGHLIGHT = "#f3e2b8"  # sand highlighter on the rule that decides
BAND = "#4f7890"  # confidence intervals
ACCENT, SAND, DEEP_INK = "#f2a541", "#d9b97c", "#0b2a3f"  # the question (orange), sand, dark text on light
LENGTHS = {"1": "Short", "2": "Medium", "3": "Long"}
DPI = 200

THEMES = {
    "dark": {
        "SURFACE": SURFACE,
        "INK": INK,
        "INK2": INK2,
        "MUTED": MUTED,
        "GRID": GRID,
        "BASELINE": BASELINE,
        "GRID_LIGHT": GRID_LIGHT,
        "RIGHT": RIGHT,
        "WRONG": WRONG,
        "CARD": CARD,
        "PANEL": PANEL,
        "BAND": BAND,
    },
    # White page for print. Text contrast on white: 16.4, 8.1 and 5.2 to 1; right 3.2 and error 5.9 to 1.
    "light": {
        "SURFACE": "#ffffff",
        "INK": "#14212b",
        "INK2": "#3e5261",
        "MUTED": "#5c6f7d",
        "GRID": "#d6dee4",
        "BASELINE": "#8a9ba7",
        "GRID_LIGHT": "#e1e8ed",
        "RIGHT": "#2fa35a",
        "WRONG": "#b93226",
        "CARD": "#f0f4f7",
        "PANEL": "#f5f8fa",
        "BAND": "#a9bfcd",
    },
}
"""The colour roles of the charts in each theme. The diagrams (cover, task, protocol) exist in the dark theme only."""

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "svg.fonttype": "none",  # text stays text: small SVG files, selectable words
        "svg.hashsalt": "bythebooktest",  # stable identifiers: the same file at every run
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def use_theme(name: str) -> None:
    """Set the colour roles of :data:`THEMES` and the matching matplotlib defaults."""
    globals().update(THEMES[name])
    plt.rcParams.update(
        {
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


# --- layout check and saving ------------------------------------------------------------------------------

FIT_PAIRS: list[tuple[Text, FancyBboxPatch]] = []
"""(text, box) pairs registered by the diagrams: the text must stay inside its box."""


def check_layout(fig: plt.Figure, name: str, tol: float = 1.0, gap: float = 3.0) -> None:
    """Stop if a text leaves the figure or its registered box, or comes closer than ``gap`` pixels to
    another text.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    fb = fig.bbox
    problems = []
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    for t in texts:
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - tol or b.x1 > fb.x1 + tol or b.y0 < fb.y0 - tol or b.y1 > fb.y1 + tol:
            problems.append(f"outside the figure: {t.get_text()[:50]!r}")
    for t, patch in FIT_PAIRS:
        if t.figure is not fig:
            continue
        b, p = t.get_window_extent(r), patch.get_window_extent(r)
        if b.x0 < p.x0 - tol or b.x1 > p.x1 + tol or b.y0 < p.y0 - tol or b.y1 > p.y1 + tol:
            problems.append(f"outside its box: {t.get_text()[:50]!r}")
    FIT_PAIRS[:] = [(t, p) for t, p in FIT_PAIRS if t.figure is not fig]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    for i, (t1, b1) in enumerate(boxes):
        for t2, b2 in boxes[i + 1 :]:
            if b1.x0 < b2.x1 + gap and b2.x0 < b1.x1 + gap and b1.y0 < b2.y1 + gap and b2.y0 < b1.y1 + gap:
                problems.append(f"texts too close: {t1.get_text()[:30]!r} and {t2.get_text()[:30]!r}")
    if problems:
        raise SystemExit(f"{name}: text does not fit\n  " + "\n  ".join(problems))


def save(fig: plt.Figure, out: Path, name: str, **kwargs: object) -> None:
    """Check the layout, then write ``name.png`` and ``name.svg``."""
    check_layout(fig, name)
    kwargs.setdefault("facecolor", SURFACE)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", **kwargs)  # type: ignore[arg-type]
    plt.close(fig)


# --- shared pieces ----------------------------------------------------------------------------------------


def axis_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    vertical: bool = False,
    label_pad: float = 0.012,
) -> None:
    """An arrow giving the direction of an axis, in axes fractions, with a word saying what grows."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, mutation_scale=11, shrinkA=0, shrinkB=0),
    )
    if vertical:
        ax.text(
            start[0] - label_pad,
            (start[1] + end[1]) / 2,
            label,
            transform=ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            color=MUTED,
            fontsize=8.5,
        )
    else:
        ax.text(
            end[0], start[1] - label_pad, label, transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8.5
        )


def _box(ax, x, y, w, h, *, edge=BASELINE, face=CARD, lw=1.2, r=0.012, aspect=1.0) -> FancyBboxPatch:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=aspect,
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(p)
    return p


def _inside(patch: FancyBboxPatch, *texts: Text) -> None:
    """Register texts that must stay inside a box (checked by :func:`check_layout`)."""
    FIT_PAIRS.extend((t, patch) for t in texts)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.4) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14 + 3 * (lw > 1.5), shrinkA=0, shrinkB=0),
    )


def _chip(ax, x, y, text, *, right=False, w=0.085, h=0.06, aspect=1.0, fs=10) -> None:
    """A possible answer, like the buttons of the cover: light border on dark; the right one filled with
    green, with dark text (contrast 5.2:1) and a tick.
    """
    if right:
        p = _box(ax, x, y, w, h, edge=RIGHT, face=RIGHT, lw=1.6, r=0.01, aspect=aspect)
        t = ax.text(
            x + w / 2, y + h / 2, "✓ " + text, ha="center", va="center", fontsize=fs, color=DEEP_INK, fontweight="bold"
        )
    else:
        p = _box(ax, x, y, w, h, edge=MUTED, face=SURFACE, lw=1.2, r=0.01, aspect=aspect)
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK2)
    _inside(p, t)


# --- illustrations ----------------------------------------------------------------------------------------


def figure_cover(out: Path) -> None:
    """Cover (2:1, 1300x650 for Hugging Face, also fine for LinkedIn previews): the paradox of the test in
    one picture, a cactus at the bottom of the sea.
    """
    import numpy as np
    from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

    W, H = 13.0, 6.5
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # sea: from lighter at the top to darker at the bottom
    top, bottom = np.array([0x0D, 0x4A, 0x6E]) / 255, np.array([0x05, 0x1C, 0x2C]) / 255
    grad = np.linspace(0, 1, 256)[:, None, None]
    ax.imshow(top * (1 - grad) + bottom * grad, extent=(0, 2, 0, 1), aspect="auto", zorder=0)
    # light rays from the surface
    for x0, w in ((1.10, 0.10), (1.38, 0.07), (1.62, 0.12), (1.86, 0.06)):
        ax.add_patch(
            Polygon(
                [(x0, 1), (x0 + w, 1), (x0 + w * 2.6 - 0.12, 0.18), (x0 - 0.14, 0.18)],
                closed=True,
                facecolor="#ffffff",
                alpha=0.045,
                linewidth=0,
                zorder=1,
            )
        )
    # wavy surface
    xs = np.linspace(0, 2, 400)
    ax.fill_between(xs, 0.965 + 0.012 * np.sin(xs * 18), 1.0, color="#1f6f96", alpha=0.9, zorder=2, linewidth=0)
    # sandy bottom
    ax.fill_between(
        xs, 0, 0.16 + 0.025 * np.sin(xs * 4.5 + 1.2) + 0.012 * np.sin(xs * 11), color=SAND, zorder=3, linewidth=0
    )
    ax.fill_between(xs, 0, 0.10 + 0.02 * np.sin(xs * 3.7), color="#c9a566", zorder=3, linewidth=0)
    # seaweed on the sides
    for bx, hgt, ph in ((1.02, 0.30, 0.0), (1.06, 0.22, 1.3), (1.93, 0.34, 0.6), (1.89, 0.24, 2.0)):
        ys = np.linspace(0.14, 0.14 + hgt, 60)
        ax.plot(bx + 0.018 * np.sin(ys * 22 + ph), ys, color="#2f7d5a", lw=5, solid_capstyle="round", zorder=3)
    # the cactus (green because cacti are green, not as a judgement)
    cg, cd = "#3c9a5f", "#2b7547"
    cx, base = 1.50, 0.15

    def stem(x, y, w, h):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={w / 2}",
                mutation_aspect=H / W * 2,
                facecolor=cg,
                edgecolor=cd,
                linewidth=2.5,
                zorder=5,
            )
        )
        for k in (0.3, 0.5, 0.7):  # ribs
            ax.plot([x + w * k, x + w * k], [y + 0.02, y + h - 0.03], color=cd, lw=1.6, zorder=6)

    stem(cx - 0.055, base, 0.11, 0.56)  # trunk
    stem(cx - 0.20, base + 0.24, 0.07, 0.22)  # left arm
    ax.add_patch(Rectangle((cx - 0.165, base + 0.24), 0.12, 0.06, facecolor=cg, edgecolor="none", zorder=4))
    stem(cx + 0.13, base + 0.30, 0.07, 0.20)  # right arm
    ax.add_patch(Rectangle((cx + 0.045, base + 0.30), 0.12, 0.06, facecolor=cg, edgecolor="none", zorder=4))
    rng = np.random.default_rng(7)  # spines
    for _ in range(40):
        px, py = cx - 0.05 + rng.random() * 0.10, base + 0.04 + rng.random() * 0.48
        ax.plot([px, px + 0.008], [py, py + 0.01], color="#e9f5e3", lw=1, zorder=7)
    # bubbles
    for bx, by, r in (
        (1.47, 0.80, 0.012),
        (1.52, 0.86, 0.008),
        (1.49, 0.91, 0.006),
        (1.30, 0.62, 0.010),
        (1.33, 0.70, 0.007),
        (1.72, 0.66, 0.011),
        (1.70, 0.75, 0.007),
        (1.75, 0.82, 0.005),
    ):
        ax.add_patch(
            Ellipse(
                (bx, by),
                r * 2 * H / W * 2,
                r * 2,
                facecolor="none",
                edgecolor="#bfe3f2",
                linewidth=1.6,
                alpha=0.9,
                zorder=6,
            )
        )
    # a puzzled fish looking at the cactus: orange is "the question" in every diagram
    fx, fy = 1.24, 0.47
    ax.add_patch(Ellipse((fx, fy), 0.11, 0.08, facecolor=ACCENT, edgecolor="none", zorder=6))
    ax.add_patch(
        Polygon(
            [(fx - 0.05, fy), (fx - 0.09, fy + 0.035), (fx - 0.09, fy - 0.035)],
            closed=True,
            facecolor=ACCENT,
            edgecolor="none",
            zorder=6,
        )
    )
    ax.add_patch(Circle((fx + 0.03, fy + 0.01), 0.008, facecolor="#0b0b0b", zorder=7))
    ax.text(fx - 0.02, fy + 0.07, "?", fontsize=30, fontweight="bold", color="#ffffff", zorder=7)
    # text on the left
    ax.text(0.08, 0.80, "The Hidden Conflict", fontsize=40, fontweight="bold", color="#ffffff", zorder=8)
    ax.text(0.08, 0.68, "in AI Decisions", fontsize=40, fontweight="bold", color="#ffffff", zorder=8)
    ax.text(
        0.08,
        0.575,
        "Does your AI follow the document,\nor what it already knows?",
        fontsize=17,
        color="#cfe6f2",
        va="top",
        linespacing=1.4,
        zorder=8,
    )
    # the rulebook as a light card, and the three answers, above the sand
    ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.335),
            0.74,
            0.105,
            boxstyle="round,pad=0,rounding_size=0.02",
            mutation_aspect=H / W * 2,
            facecolor="#ffffff",
            alpha=0.95,
            linewidth=0,
            zorder=8,
        )
    )
    ax.text(
        0.105, 0.3875, "Rulebook:  “A cactus grows under the sea.”", fontsize=15, color=DEEP_INK, va="center", zorder=9
    )
    for i, (t, cw) in enumerate((("TRUE", 0.20), ("FALSE", 0.21), ("UNKNOWN", 0.29))):
        x0 = 0.08 + [0, 0.22, 0.45][i]
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.215),
                cw,
                0.085,
                boxstyle="round,pad=0,rounding_size=0.02",
                mutation_aspect=H / W * 2,
                facecolor="none",
                edgecolor="#ffffff",
                linewidth=2,
                zorder=8,
            )
        )
        ax.text(
            x0 + cw / 2, 0.2575, t, ha="center", va="center", fontsize=14, color="#ffffff", fontweight="bold", zorder=9
        )
    # on the sand, in a dark brown that reads well
    ax.text(0.08, 0.045, "Introducing ByTheBookTest", fontsize=14, color="#4a3818", fontweight="bold", zorder=8)
    save(fig, out, "fig0-cover", dpi=100, facecolor="auto")


def figure_task(out: Path) -> None:
    """The task: rulebook -> claim -> one of three answers, with two examples: the rulebook settles the
    question (cactus) and the rulebook names Jupiter but says nothing about its size (claims L1-KB1-C0006 and
    L1-KB4-C0008 of the released data, both answered FALSE by d1-2B).
    """
    fig = plt.figure(figsize=(8.0, 4.6), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    asp = 4.6 / 8.0
    ax.text(0.03, 0.93, "The task", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        0.03,
        0.875,
        "The model reads the rulebook and judges one claim, using only the rulebook.",
        fontsize=8.5,
        color=INK2,
    )
    # The cover's code: the rulebook is a light sheet with dark text and the deciding rule highlighted; the
    # claim is the question, with the orange of the fish; the right answer is filled with green.
    for x, title in ((0.03, "RULEBOOK"), (0.405, "CLAIM"), (0.72, "ANSWER")):
        ax.text(x, 0.79, title, fontsize=8.5, color=SAND, fontweight="bold")
    rb = _box(ax, 0.03, 0.08, 0.32, 0.68, edge=PAPER, face=PAPER, aspect=asp)
    lines = [
        ("Napoleon was born in Australia.", False),
        ("A cactus grows under the sea.", True),
        ("Every gegibux can knit.", False),
        ("Fubetrir is a gegibux.", False),
        ("Jupiter is a trokulus.", False),
        ("…  about 120 to 2,400 statements", False),
    ]
    for i, (t, hl) in enumerate(lines):
        y = 0.69 - i * 0.1
        if hl:
            _box(ax, 0.04, y - 0.035, 0.30, 0.07, edge=HIGHLIGHT, face=HIGHLIGHT, lw=0, r=0.006, aspect=asp)
        dim = t.startswith("…")
        _inside(
            rb,
            ax.text(
                0.05,
                y,
                t,
                fontsize=8.5,
                color=PAPER_MUTED if dim else DEEP_INK,
                va="center",
                fontweight="bold" if hl else "normal",
            ),
        )
    _inside(
        rb,
        ax.text(
            0.05, 0.11, "Nothing about Jupiter's size.", fontsize=8.5, color=PAPER_MUTED, va="center", style="italic"
        ),
    )
    rows = [
        (0.58, "“A cactus does not grow\nunder the sea.”", "FALSE", "the rulebook settles it"),
        (0.24, "“Jupiter is the smallest\nplanet in the Solar System.”", "UNKNOWN", "the rulebook does not say"),
    ]
    for y, claim, ok, note in rows:
        cb = _box(ax, 0.405, y - 0.08, 0.265, 0.16, edge=ACCENT, face=CARD, lw=2.0, aspect=asp)
        _inside(cb, ax.text(0.42, y, claim, fontsize=8.5, color=INK, va="center", fontweight="bold"))
        _arrow(ax, 0.358, y, 0.4, y, color=INK2, lw=2.0)
        _arrow(ax, 0.675, y, 0.715, y, color=INK2, lw=2.0)
        cx = 0.72
        for t, cw in (("TRUE", 0.07), ("FALSE", 0.078), ("UNKNOWN", 0.108)):
            _chip(ax, cx, y - 0.035, t, right=(t == ok), w=cw, h=0.07, aspect=asp, fs=7.5)
            cx += cw + 0.008
        ax.text(0.72, y - 0.085, note, fontsize=8, color=INK2, va="top")
    ax.text(
        0.405, 0.04, "TRUE: the claim follows · FALSE: its opposite follows · UNKNOWN: neither", fontsize=8, color=MUTED
    )
    save(fig, out, "fig-task")


def figure_protocol(out: Path) -> None:
    """The protocol: fact tables -> generator -> independent check with Z3 -> model -> measures, with the
    knowledge probe as a branch from the same tables.
    """
    fig = plt.figure(figsize=(8.0, 4.4), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1), facecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    asp = 4.4 / 8.0
    ax.text(0.03, 0.93, "How ByTheBookTest is built and run", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        0.03,
        0.875,
        "Every expected answer is checked by a theorem prover before any model sees it.",
        fontsize=8.5,
        color=INK2,
    )
    steps = [
        ("Fact tables", "305 known facts,\nwith false versions;\n26 class abilities"),
        ("Generator", "rulebooks + claims\nfrom known proofs;\nfixed seed"),
        ("Z3 check", "keeps a claim only\nif the prover agrees\nwith its answer"),
        ("Model under test", "one answer per\nclaim: TRUE,\nFALSE or UNKNOWN"),
        ("Measures", "accuracy, error\ntypes, conflict and\nsilence effects"),
    ]
    # all boxes alike: green means "right answer", not "important step"
    w, gap, y, h, x0 = 0.168, 0.032, 0.45, 0.28, 0.02
    for i, (title, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        b = _box(ax, x, y, w, h, aspect=asp)
        _inside(
            b,
            ax.text(x + 0.011, y + h - 0.055, title, fontsize=8.8, fontweight="bold", color=INK),
            ax.text(x + 0.011, y + h - 0.105, body, fontsize=7.4, color=INK2, va="top", linespacing=1.35),
        )
        if i < len(steps) - 1:
            _arrow(ax, x + w + 0.004, y + h / 2, x + w + gap - 0.004, y + h / 2, color=INK2, lw=2.0)
    # branch: the knowledge probe, from the tables to the measures
    by, bx0, bx1 = 0.12, 0.25, 0.75
    kb = _box(ax, bx0, by, bx1 - bx0, 0.19, edge=GRID, face=PANEL, aspect=asp)
    _inside(
        kb,
        ax.text(bx0 + 0.015, by + 0.135, "Knowledge test (no rulebook)", fontsize=9, fontweight="bold", color=INK),
        ax.text(
            bx0 + 0.015,
            by + 0.085,
            "Ask the model each table statement and its negation,\nwithout any rulebook: which facts does it know?",
            fontsize=7.6,
            color=INK2,
            va="top",
            linespacing=1.35,
        ),
    )
    x_tab = x0 + w / 2
    x_mea = x0 + 4 * (w + gap) + w / 2
    my = by + 0.095
    ax.plot([x_tab, x_tab], [y, my], color=INK2, lw=2.0)
    _arrow(ax, x_tab, my, bx0 - 0.004, my, color=INK2, lw=2.0)
    ax.plot([bx1 + 0.004, x_mea], [my, my], color=INK2, lw=2.0)
    _arrow(ax, x_mea, my, x_mea, y - 0.008, color=INK2, lw=2.0)
    save(fig, out, "fig-protocol")


# --- charts -------------------------------------------------------------------------------------------------


def _depth(m: dict, lv: str) -> tuple[list[int], list[float], list[float], list[float]]:
    d = m["levels"][lv]["curves"]["depth_tf"]
    steps = sorted(int(k) for k in d)
    return (
        steps,
        [100 * d[str(s)]["value"] for s in steps],
        [100 * d[str(s)]["ci95"][0] for s in steps],
        [100 * d[str(s)]["ci95"][1] for s in steps],
    )


def figure_steps(m: dict, out: Path) -> None:
    """Right answers by number of reasoning steps: three stacked panels, one per rulebook length, on the
    same scale of steps (0-6), with the 95% interval. The value axis goes to 122% so that points at 100%
    and their labels are not cut.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.4), dpi=DPI, facecolor=SURFACE, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.10, hspace=0.42)
    # one background grid for the three panels: a vertical line per step, from the top panel to the bottom
    top, bottom = axes[0].get_position(), axes[-1].get_position()
    bg = fig.add_axes((bottom.x0, bottom.y0, bottom.width, top.y1 - bottom.y0), zorder=0)
    bg.set_xlim(-0.4, 6.4)
    bg.set_ylim(0, 1)
    bg.axis("off")
    for s in range(7):
        bg.axvline(s, color=GRID_LIGHT, linewidth=1)
    for ax, (lv, name) in zip(axes, LENGTHS.items()):
        ax.set_zorder(1)
        ax.set_facecolor("none")  # transparent: the background grid shows through
        steps, acc, lo, hi = _depth(m, lv)
        ax.set_xlim(-0.4, 6.4)
        ax.set_ylim(0, 122)
        ax.set_xticks(range(7))
        ax.set_yticks([])  # every point carries its value
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        axis_arrow(ax, (0.0, 0.0), (0.0, 0.86), "more right", vertical=True, label_pad=0.012)
        ax.axhline(100 / 3, color=GRID_LIGHT, linewidth=1, zorder=0)  # chance, like a grid line
        ax.annotate(
            "",
            xy=(1.0, 0.0),
            xytext=(0.0, 0.0),
            xycoords="axes fraction",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, mutation_scale=11, shrinkA=0, shrinkB=0),
        )
        if lv != "3":
            ax.tick_params(labelbottom=False)
        else:
            ax.tick_params(labelbottom=True)
            ax.text(
                1.0,
                -0.30,
                "more reasoning steps needed",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=MUTED,
                fontsize=8.5,
            )
        ax.fill_between(steps, lo, hi, color=RIGHT, alpha=0.22, linewidth=0)
        ax.plot(steps, acc, color=RIGHT, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.scatter(steps, acc, s=56, color=RIGHT, edgecolor=SURFACE, linewidth=2, zorder=4, clip_on=False)
        for s, a in zip(steps, acc):
            ax.annotate(
                f"{a:.0f}%", (s, a), xytext=(0, 9), textcoords="offset points", ha="center", color=INK2, fontsize=8.5
            )
        ax.set_title(f"{name} rulebooks", loc="left", pad=8, color=INK, fontsize=10, fontweight="bold")
    fig.text(
        0.03, 0.955, "Right answers by number of reasoning steps", fontsize=13, fontweight="bold", color=INK, ha="left"
    )
    fig.text(
        0.03,
        0.925,
        "Claims the rulebook settles (TRUE or FALSE). Shaded: 95% interval. Light horizontal line: chance (33%).",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    save(fig, out, "fig2b-steps-panels")


def figure_confusion(m: dict, out: Path) -> None:
    """Three confusion matrices (rows: right answer; columns: answer given), as shares of each row. Every
    cell has the same background; a bar gives the share, green on the diagonal (right) and red off it.
    """
    labels = ["TRUE", "FALSE", "UNKNOWN"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.9), dpi=DPI, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.15, wspace=0.14)
    for ax, (lv, name) in zip(axes, LENGTHS.items()):
        conf = m["levels"][lv]["lenient"]["confusion"]
        ax.set_facecolor(SURFACE)
        for i, e in enumerate(labels):
            row = conf[e]
            n = sum(row.values())
            for j, a in enumerate(labels):
                share = row[a] / n if n else 0
                x0, y0 = j + 0.04, 2 - i + 0.04
                ax.add_patch(
                    FancyBboxPatch(
                        (x0, y0), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0, facecolor=CARD
                    )
                )
                ax.text(
                    j + 0.5,
                    2 - i + 0.60,
                    f"{100 * share:.0f}%",
                    ha="center",
                    va="center",
                    color=INK if share > 0 else MUTED,
                    fontsize=10.5,
                    fontweight="bold" if i == j else "normal",
                )
                bx, bw, by, bh = j + 0.16, 0.68, 2 - i + 0.20, 0.09
                ax.add_patch(
                    FancyBboxPatch(
                        (bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.04", linewidth=0, facecolor=GRID
                    )
                )
                if share > 0:
                    ax.add_patch(
                        FancyBboxPatch(
                            (bx, by),
                            max(bw * share, 0.02),
                            bh,
                            boxstyle="round,pad=0,rounding_size=0.04",
                            linewidth=0,
                            facecolor=RIGHT if i == j else WRONG,
                        )
                    )
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        # the same letters on rows and columns: they are the same three answers
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(["T", "F", "U"])
        ax.set_yticks([2.5, 1.5, 0.5])
        ax.set_yticklabels(["T", "F", "U"])
        ax.tick_params(length=0, labelsize=10)
        ax.set_xlabel("Answer given", color=INK2, fontsize=9)
        if lv == "1":
            ax.set_ylabel("Right answer", color=INK2, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{name} rulebooks", fontsize=10, fontweight="bold", color=INK)
    fig.text(
        0.03,
        0.93,
        "Right answer (rows) against the answer given (columns)",
        fontsize=13,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    fig.text(
        0.03,
        0.865,
        "T = TRUE, F = FALSE, U = UNKNOWN. Share of each row. Green bar: right answers. Red bar: errors.",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    save(fig, out, "fig3-confusion")


def figure_conflict(m: dict, probe: dict | None, out: Path) -> None:
    """For every length: right answers when the rulebook agrees with the world (hollow dot, the reference)
    and when it contradicts it (filled dot); the red segment is the loss (the conflict effect). Last row,
    with the probe: long rulebooks, facts the model knows only.
    """
    rows = []
    for lv, name in LENGTHS.items():
        e = m["levels"][lv]["lenient"]
        rows.append((f"{name} rulebooks", 100 * e["agree_accuracy"]["value"], 100 * e["conflict_accuracy"]["value"]))
    if probe:
        k = next(iter(probe["models"].values()))["levels"]["3"]["known"]
        rows.append(("Long, facts it knows", 100 * k["agree_accuracy"]["value"], 100 * k["conflict_accuracy"]["value"]))
    fig = plt.figure(figsize=(8.0, 4.0), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0.25, 0.20, 0.68, 0.60), facecolor=SURFACE)
    ax.set_xlim(0, 103)  # room on the right: a dot at 100% must not be cut
    axis_arrow(ax, (0.0, -0.17), (100 / 103, -0.17), "more right answers")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=9.5)
    for i, (_, agree, conflict) in enumerate(rows):
        if abs(agree - conflict) > 0.5:
            ax.plot([conflict, agree], [i, i], color=WRONG, linewidth=4, solid_capstyle="round", zorder=2)
            ax.text(
                (agree + conflict) / 2,
                i - 0.28,
                f"{conflict - agree:+.0f}",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=9.5,
                fontweight="bold",
            )
        else:
            ax.text(agree, i - 0.28, "0", ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
        ax.scatter([agree], [i], s=80, color=SURFACE, edgecolor=MUTED, linewidth=2, zorder=3)
        ax.scatter([conflict], [i], s=80, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
    fig.text(
        0.03,
        0.93,
        "When the rulebook contradicts what the model knows",
        fontsize=13,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    fig.text(
        0.03,
        0.87,
        "Right answers when the rulebook agrees with the world (hollow) and when it contradicts "
        "it (filled). Red: the loss.",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    save(fig, out, "fig4-conflict")


GROUP_NAMES = [
    ("direct_fact", "Facts written in the rulebook"),
    ("modal", '"Can", "must", "generally"'),
    ("rule_chain", "Chains of rules"),
    ("negation", "Negations"),
    ("conjunction", 'Several conditions ("and")'),
    ("disjunction", '"Or", "either… or"'),
    ("iff", '"If and only if"'),
    ("exception", "Exceptions"),
    ("time", "Dates and intervals"),
    ("relation", "Relations between entities"),
    ("exists", '"Some"'),
    ("counting", 'Counts ("exactly", "at least")'),
    ("universal", '"Every" over a group'),
    ("shared_chain", "Chains sharing a link"),
]


def figure_reasoning_map(m: dict, out: Path) -> None:
    """Where the model gives way: in every cell a scale from 0 to 100%, the band of the 95% interval, a
    marker on the measured value and a short tick at 33% (chance). If the band crosses the tick, the result
    cannot be told apart from guessing.
    """
    fig = plt.figure(figsize=(8.0, 6.9), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0.37, 0.02, 0.60, 0.74), facecolor=SURFACE)
    n = len(GROUP_NAMES)
    ax.set_xlim(0, 3)
    ax.set_ylim(n, -1.1)
    pad, width = 0.10, 0.80  # in every column the scale runs from j+pad (0%) to j+pad+width (100%)

    def sx(j: int, v: float) -> float:
        return j + pad + width * v

    chance = 1 / 3
    for i, (gid, gname) in enumerate(GROUP_NAMES):
        y = i + 0.55
        ax.text(-0.06, y, gname, ha="right", va="center", color=INK2, fontsize=9.5)
        for j, lv in enumerate(LENGTHS):
            p = m["levels"][lv]["curves"]["group"].get(gid)
            if p is None:
                ax.text(j + 0.5, y, "–", ha="center", va="center", color=BASELINE, fontsize=10)
                continue
            v, (lo, hi) = p["value"], p["ci95"]
            ax.plot([sx(j, 0), sx(j, 1)], [y, y], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=1)
            ax.plot([sx(j, lo), sx(j, hi)], [y, y], color=BAND, linewidth=6, solid_capstyle="round", zorder=2)
            # the chance tick: short and grey, a reference that must not compete with the data
            ax.plot([sx(j, chance), sx(j, chance)], [y - 0.14, y + 0.14], color=MUTED, linewidth=1.0, zorder=3)
            ax.scatter([sx(j, v)], [y], s=46, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.text(
                sx(j, v),
                y - 0.22,
                f"{100 * v:.0f}%",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=8.5,
                fontweight="bold",
            )
    for j, name in enumerate(LENGTHS.values()):
        ax.text(j + 0.5, -0.74, name, ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
        ax.text(sx(j, 0), -0.38, "0%", ha="left", va="bottom", color=MUTED, fontsize=7.5)
        ax.text(sx(j, chance), -0.38, "33%", ha="center", va="bottom", color=MUTED, fontsize=7.5)
        ax.text(sx(j, 1), -0.38, "100%", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    ax.axis("off")
    fig.text(0.03, 0.955, "Where the model gives way", fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(
        0.03,
        0.922,
        "Right answers by kind of reasoning and rulebook length, 100 claims per cell.",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    # drawn legend: marker, band, tick
    leg = fig.add_axes((0.03, 0.865, 0.94, 0.03))
    leg.set_xlim(0, 1)
    leg.set_ylim(-1, 1)
    leg.axis("off")
    leg.scatter([0.01], [0], s=40, color=INK, edgecolor=SURFACE, linewidth=1.5)
    leg.text(0.025, 0, "measured share", va="center", color=INK2, fontsize=8.5)
    leg.plot([0.19, 0.23], [0, 0], color=BAND, linewidth=6, solid_capstyle="round")
    leg.text(0.245, 0, "95% interval", va="center", color=INK2, fontsize=8.5)
    leg.plot([0.39, 0.39], [-0.6, 0.6], color=MUTED, linewidth=1.0)
    leg.text(0.40, 0, "chance (guessing among three answers)", va="center", color=INK2, fontsize=8.5)
    fig.text(
        0.03,
        0.835,
        "If the interval crosses the chance tick, the result cannot be told apart from guessing. "
        "–: kind of reasoning not used at that length.",
        fontsize=8,
        color=MUTED,
        ha="left",
    )
    save(fig, out, "fig5-reasoning-map")


DIAGRAMS = ("cover", "task", "protocol")
CHARTS = ("steps", "confusion", "conflict", "reasoning-map")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the figures of the article.")
    ap.add_argument("metrics", type=Path, help="metrics.json of the benchmark run")
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument("--probe", type=Path, help="probe.json of the knowledge probe (adds a row to the conflict figure)")
    ap.add_argument("--theme", choices=sorted(THEMES), default="dark", help="dark (article, default) or light (print)")
    ap.add_argument("--only", nargs="+", choices=DIAGRAMS + CHARTS, help="draw only these figures (default: all)")
    args = ap.parse_args(argv)
    names = args.only or list(DIAGRAMS + CHARTS)
    if args.theme != "dark" and set(names) & set(DIAGRAMS):
        ap.error("the diagrams exist in the dark theme only: choose charts with --only")
    use_theme(args.theme)
    args.out.mkdir(parents=True, exist_ok=True)
    m = next(iter(json.loads(args.metrics.read_text(encoding="utf-8"))["models"].values()))
    probe = json.loads(args.probe.read_text(encoding="utf-8")) if args.probe else None
    draw = {
        "cover": lambda: figure_cover(args.out),
        "task": lambda: figure_task(args.out),
        "protocol": lambda: figure_protocol(args.out),
        "steps": lambda: figure_steps(m, args.out),
        "confusion": lambda: figure_confusion(m, args.out),
        "conflict": lambda: figure_conflict(m, probe, args.out),
        "reasoning-map": lambda: figure_reasoning_map(m, args.out),
    }
    for name in names:
        draw[name]()
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
