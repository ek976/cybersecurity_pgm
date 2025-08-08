# hybrid_plot.py (replace your plot_hybrid_pgm_structure with this)

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

PALETTE = {
    "accent": "#EFEBCE",  # T nodes
    "mid":    "#D8A48F",  # C nodes
    "dark":   "#BB8588",  # G nodes
    "edge":   "black",
}

def _trim_to_circle(x1, y1, x2, y2, r):
    """
    Return the segment (sx,sy)->(tx,ty) trimmed so it starts/ends on the
    circumference of circles centered at (x1,y1) and (x2,y2) with radius r.
    """
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return (x1, y1), (x2, y2)
    ux, uy = dx / L, dy / L
    sx, sy = x1 + ux * r, y1 + uy * r          # leave source circle
    tx, ty = x2 - ux * r, y2 - uy * r          # enter target circle
    return (sx, sy), (tx, ty)

def plot_hybrid_pgm_structure(use_ngrams=False, n_positions=10, figsize=None,
                              radius=0.28, xgap=1.35, arc_rad=0.32, arc_y_offset=0.32):
    """
    Hybrid PGM
    Arrows are trimmed to circle boundaries and arcs are cleanly offset above G row.
    """
    Gy, Cy, Ty = 0.9, 0.0, -0.9  # y positions for the three rows

    xmin, xmax = -0.5, (n_positions - 1) * xgap + 0.5
    ymin, ymax = Ty - 0.7, Gy + 0.7

    if figsize is None:
        data_aspect = (xmax - xmin) / (ymax - ymin)
        width = 14
        height = max(3.8, width / data_aspect)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0, adjustable="datalim")
    ax.axis("off")

    # --- draw nodes first (so edges can sit on top cleanly) ---
    G_centers = []
    for i in range(n_positions):
        x = i * xgap
        G_centers.append((x, Gy))

        g = Circle((x, Gy), radius, facecolor=PALETTE["dark"], edgecolor=PALETTE["edge"], lw=1.2, zorder=2)
        ax.add_patch(g); ax.text(x, Gy, f"G{i+1}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

        c = Circle((x, Cy), radius, facecolor=PALETTE["mid"], edgecolor=PALETTE["edge"], lw=1.2, zorder=2)
        ax.add_patch(c); ax.text(x, Cy, f"C{i+1}", ha="center", va="center", fontsize=10, fontweight="bold")

        t = Circle((x, Ty), radius, facecolor=PALETTE["accent"], edgecolor=PALETTE["edge"], lw=1.2, zorder=2)
        ax.add_patch(t); ax.text(x, Ty, f"T{i+1}", ha="center", va="center", fontsize=10, fontweight="bold")

        # vertical G->C->T (trimmed to circle edges)
        (sx, sy), (tx, ty) = _trim_to_circle(x, Gy, x, Cy, radius)
        ax.add_patch(FancyArrowPatch((sx, sy), (tx, ty), arrowstyle="->", lw=1.0, color=PALETTE["edge"], zorder=1))
        (sx, sy), (tx, ty) = _trim_to_circle(x, Cy, x, Ty, radius)
        ax.add_patch(FancyArrowPatch((sx, sy), (tx, ty), arrowstyle="->", lw=1.0, color=PALETTE["edge"], zorder=1))

    # bigram edges: G_{i-1} -> G_i (trim to circle boundaries)
    for i in range(1, n_positions):
        (x1, y1), (x2, y2) = G_centers[i-1], G_centers[i]
        (sx, sy), (tx, ty) = _trim_to_circle(x1, y1, x2, y2, radius)
        ax.add_patch(FancyArrowPatch((sx, sy), (tx, ty),
                                     arrowstyle="->", lw=1.2, color=PALETTE["edge"], zorder=1))

    # trigram edges: G_{i-2} -> G_i as curved arcs *above* the G row
    if use_ngrams:
        for i in range(2, n_positions):
            (x1, y1), (x2, y2) = G_centers[i-2], G_centers[i]
            # Start/end points trimmed, but nudged upward so the arc rides above nodes
            (sx, sy), (tx, ty) = _trim_to_circle(x1, y1, x2, y2, radius)
            sy += arc_y_offset; ty += arc_y_offset
            arc = FancyArrowPatch((sx, sy), (tx, ty),
                                  connectionstyle=f"arc3,rad={arc_rad}",
                                  arrowstyle="->", lw=1.0, color=PALETTE["mid"], alpha=0.95, zorder=1)
            ax.add_patch(arc)

    ax.set_title("Hybrid PGM Structure (G → C → T with n-grams)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# 2) Posteriors: simple bar chart of P(Correct)
# ------------------------------------------------------------
def plot_hybrid_posteriors(posteriors, observed_chars, timing_classes, target=None, figsize=(12, 3.5)):
    positions = list(range(1, min(11, (len(target) if target else 10) + 1)))
    probs = [posteriors.get(f"G{i}").values[1] if f"G{i}" in posteriors else 1.0 for i in positions]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(positions, probs, color=PALETTE["dark"], alpha=0.9, edgecolor=PALETTE["edge"])
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xlabel("Position"); ax.set_ylabel("P(Correct)")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax.set_title("Hybrid: Posterior P(Correct) by Position", fontsize=12, fontweight="bold")

    # annotate values (compact)
    for x, p in zip(positions, probs):
        ax.text(x, p + 0.03, f"{p:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    return fig

# ------------------------------------------------------------
# 3) Operational view: P(Correct) + timing class
# ------------------------------------------------------------
def plot_hybrid_operational(posteriors, observed_chars, timing_classes, target=None, figsize=(12, 3.5)):
    n = len(target) if target else 10
    positions = list(range(1, min(11, n + 1)))
    probs = [posteriors.get(f"G{i}").values[1] if f"G{i}" in posteriors else 1.0 for i in positions]

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(positions, probs, color=PALETTE["mid"], alpha=0.95, edgecolor=PALETTE["edge"], label="P(Correct)")
    ax1.set_ylim(0, 1); ax1.set_xlabel("Position"); ax1.set_ylabel("P(Correct)")
    ax1.set_title("Hybrid Operational View", fontsize=12, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(positions, (timing_classes or [])[:len(positions)], "o-", color=PALETTE["dark"], label="Timing class")
    ax2.set_ylabel("Timing Class")

    # single legend
    fig.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig

# ------------------------------------------------------------
# 4) Compact comparison: Timing vs Wordlike vs Hybrid
# ------------------------------------------------------------
def plot_attack_comparison(timing_results, wordlike_results, hybrid_results, target=None, figsize=(8, 4)):
    """
    timing_results: (posteriors_t, binary_guess, timing_probs)
    wordlike_results: (posteriors_w, guess_word, _)
    hybrid_results: (posteriors_h, guess_hybrid, _)
    """
    _, _, timing_probs = timing_results
    _, guess_word, _ = wordlike_results
    _, guess_hybrid, _ = hybrid_results

    if not target:
        raise ValueError("plot_attack_comparison requires `target` for accuracy comparison.")

    # rough timing "accuracy": positions flagged as correct (p>0.5) within length of target
    timing_acc = sum(p > 0.5 for p in timing_probs[:len(target)]) / len(target)
    wordlike_acc = sum(guess_word[i] == target[i] for i in range(len(target))) / len(target)
    hybrid_acc = sum(guess_hybrid[i] == target[i] for i in range(len(target))) / len(target)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Timing", "Wordlike", "Hybrid"], [timing_acc, wordlike_acc, hybrid_acc],
           color=[PALETTE["accent"], PALETTE["mid"], PALETTE["dark"]],
           edgecolor=PALETTE["edge"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Attack Method Comparison", fontsize=12, fontweight="bold")

    for x, v in zip([0, 1, 2], [timing_acc, wordlike_acc, hybrid_acc]):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    return fig
