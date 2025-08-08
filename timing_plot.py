# timing_plot.py
"""
Visualization functions for timing attack model.
Updated for a clean, simple style using custom palette.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Palette
PALETTE = {
    "primary": "#A3A380",   # muted green-beige
    "secondary": "#D8A48F", # pink
    "accent": "#EFEBCE",    # light cream
    "edge": "black"
}

# ----------------------------------
# STRUCTURE PLOT
# ----------------------------------
def plot_timing_pgm_structure(figsize=(12, 4)):
    """Show the causal structure of the timing attack PGM (G_i → T_i)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(10):
        # G nodes
        ax.add_patch(patches.Circle((i, 0.5), 0.25, facecolor=PALETTE["primary"], edgecolor=PALETTE["edge"], lw=1.5))
        ax.text(i, 0.5, f'G{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

        # T nodes
        ax.add_patch(patches.Circle((i, -0.5), 0.25, facecolor=PALETTE["secondary"], edgecolor=PALETTE["edge"], lw=1.5))
        ax.text(i, -0.5, f'T{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Edge G → T
        ax.arrow(i, 0.25, 0, -0.5, head_width=0.1, head_length=0.05, fc=PALETTE["edge"], ec=PALETTE["edge"])

    ax.text(4.5, -1.2, 'G = Correctness (0/1), T = Timing (short/med/long)',
            ha='center', fontsize=11, style='italic')
    plt.title('Timing Attack PGM Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ----------------------------------
# POSTERIOR PROBABILITIES
# ----------------------------------
def plot_timing_posteriors(posteriors, timing_classes=None, secret=None, figsize=(12, 6)):
    """Display posterior probabilities and timing class observations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    positions = list(range(1, 11))
    prob_correct = [posteriors.get(f"G{i}", None).values[1] if f"G{i}" in posteriors else 0.5 for i in positions]

    # Bar chart P(correct)
    bars = ax1.bar(positions, prob_correct, color=PALETTE["primary"], alpha=0.8, edgecolor=PALETTE["edge"])
    ax1.set(title="Posterior Probability of Correctness", xlabel="Position", ylabel="P(Correct)",
            ylim=(0, 1))
    ax1.axhline(0.5, color='gray', ls='--', alpha=0.5)

    # Labels
    for bar, p in zip(bars, prob_correct):
        ax1.text(bar.get_x() + bar.get_width()/2, p + 0.02, f"{p:.2f}", ha='center', fontsize=9)

    # Timing observations
    if timing_classes:
        class_colors = {0: PALETTE["secondary"], 1: PALETTE["accent"], 2: PALETTE["primary"]}
        class_labels = {0: 'Short', 1: 'Medium', 2: 'Long'}
        for i, t in enumerate(timing_classes[:10]):
            ax2.bar(i+1, 1, color=class_colors[t], edgecolor=PALETTE["edge"])
            ax2.text(i+1, 0.5, class_labels[t], ha='center', va='center', fontsize=9, rotation=90)

    ax2.set(title="Observed Timing Patterns", xlabel="Position", ylim=(0, 1.2), yticks=[])
    if secret:
        for i, ch in enumerate(secret[:10]):
            ax2.text(i+1, 0.1, ch, ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig

# ----------------------------------
# OPERATIONAL VIEW
# ----------------------------------
def plot_timing_operational(timings, secret=None, figsize=(12, 5)):
    """Show raw timing measurements and highlight potential correct guesses."""
    fig, ax = plt.subplots(figsize=figsize)
    pos = list(range(1, len(timings) + 1))

    ax.step(pos, timings, where='mid', lw=2, color=PALETTE["primary"])
    ax.plot(pos, timings, 'o', ms=8, color=PALETTE["secondary"])

    mean_t, std_t = np.mean(timings), np.std(timings)
    ax.axhline(mean_t, color='gray', ls='--', alpha=0.5, label='Mean')
    ax.axhline(mean_t + std_t, color=PALETTE["primary"], ls=':', alpha=0.5)
    ax.axhline(mean_t - std_t, color=PALETTE["secondary"], ls=':', alpha=0.5)

    if secret:
        for i, ch in enumerate(secret[:len(timings)]):
            ax.text(i+1, ax.get_ylim()[0] + 0.001, ch, ha='center', fontsize=10, fontweight='bold')

    ax.set(title="Timing Attack: Operational View", xlabel="Character Position", ylabel="Response Time (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

# ----------------------------------
# COMPARISON PLOT
# ----------------------------------
def plot_timing_comparison(correct_timings, incorrect_timings, figsize=(10, 5)):
    """Compare timing distributions for correct vs incorrect guesses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bins = np.linspace(0, max(max(correct_timings), max(incorrect_timings)), 20)

    # Histogram
    ax1.hist(incorrect_timings, bins=bins, alpha=0.7, label='Incorrect',
             color=PALETTE["secondary"], edgecolor=PALETTE["edge"])
    ax1.hist(correct_timings, bins=bins, alpha=0.7, label='Correct',
             color=PALETTE["primary"], edgecolor=PALETTE["edge"])
    ax1.set(title="Timing Distribution Comparison", xlabel="Response Time (s)", ylabel="Frequency")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Boxplot
    data = [incorrect_timings, correct_timings]
    bp = ax2.boxplot(data, labels=['Incorrect', 'Correct'], patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], [PALETTE["secondary"], PALETTE["primary"]]):
        patch.set_facecolor(color)
    ax2.set(title="Timing Statistics", ylabel="Response Time (s)")
    ax2.grid(alpha=0.3, axis='y')

    stats = (f"Incorrect: μ={np.mean(incorrect_timings):.3f}, σ={np.std(incorrect_timings):.3f}\n"
             f"Correct: μ={np.mean(correct_timings):.3f}, σ={np.std(correct_timings):.3f}")
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor=PALETTE["accent"], alpha=0.8))

    plt.suptitle('Timing Side-Channel Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
