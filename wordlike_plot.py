# wordlike_plot.py
"""
Visualization functions for character prediction model.
Clean, simple styling with palette: EFEBCE, D8A48F, BB8588.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from wordlike_model import CHARSET

# Palette
PALETTE = {
    "accent": "#EFEBCE",   # light cream
    "mid":    "#D8A48F",   # warm sand
    "dark":   "#BB8588",   # muted rose
    "edge":   "black",
    "text":   "black",
}


# -----------------------------------------------------------------------------
# STRUCTURE
# -----------------------------------------------------------------------------
def plot_wordlike_pgm_structure(figsize=(14, 4)):
    """
    Show the causal structure of the character prediction PGM.
    Only bigram edges (i-1 -> i) are shown.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Nodes
    for i in range(10):
        ax.add_patch(
            patches.Circle((i, 0), 0.3, facecolor=PALETTE["dark"], edgecolor=PALETTE["edge"], lw=1.5)
        )
        ax.text(i, 0, f'G{i+1}', ha='center', va='center', fontsize=11, fontweight='bold', color="white")

    # Bigram edges: (i-1) -> i
    for i in range(9):
        ax.arrow(i + 0.3, 0, 0.4, 0, head_width=0.08, head_length=0.06,
                 fc=PALETTE["edge"], ec=PALETTE["edge"], lw=1.3)

    plt.title('Character Prediction PGM Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig



# -----------------------------------------------------------------------------
# POSTERIORS
# -----------------------------------------------------------------------------
def plot_wordlike_posteriors(posteriors, observed=None, target=None, figsize=(14, 8)):
    """
    Show top-10 posterior character probabilities for each position (1..10).
    Simpler style: single-color bars, optional highlight for target char.
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for pos in range(1, 11):
        ax = axes[pos - 1]
        node = f"G{pos}"

        if observed and node in observed:
            char_idx = observed[node]
            char = CHARSET[char_idx]
            ax.set_facecolor(PALETTE["accent"])
            ax.text(0.5, 0.5, f"Observed:\n'{char}'",
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    transform=ax.transAxes, color=PALETTE["text"])
        elif node in posteriors:
            probs = posteriors[node].values
            top_k = min(10, len(probs))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            chars = [CHARSET[i] for i in top_indices]
            values = [probs[i] for i in top_indices]

            bars = ax.barh(range(top_k), values, color=PALETTE["dark"], alpha=0.85, edgecolor=PALETTE["edge"])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(chars, fontsize=9)
            ax.set_xlim([0, max(0.5, float(max(values)) * 1.1)])
            ax.invert_yaxis()

            # Values at end of bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8, color=PALETTE["text"])

            # Optional: highlight target char
            if target and pos <= len(target):
                target_char = target[pos - 1]
                if target_char in chars:
                    idx = chars.index(target_char)
                    bars[idx].set_edgecolor(PALETTE["mid"])
                    bars[idx].set_linewidth(2)

        ax.set_title(f'Position {pos}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Probability', fontsize=9)

    plt.suptitle('Character Probability Distributions by Position', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# OPERATIONAL VIEW
# -----------------------------------------------------------------------------
def plot_wordlike_operational(posteriors, observed=None, target=None, figsize=(14, 6)):
    """
    Show best-guess character per position with confidence and per-position entropy.
    Colors:
      - Observed: accent
      - Predicted: dark (alpha scaled by confidence)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

    positions = list(range(1, 11))
    best_chars, best_probs, entropies = [], [], []

    for pos in positions:
        node = f"G{pos}"
        if observed and node in observed:
            char_idx = observed[node]
            best_chars.append(CHARSET[char_idx])
            best_probs.append(1.0)
            entropies.append(0.0)
        elif node in posteriors:
            probs = posteriors[node].values
            best_idx = int(np.argmax(probs))
            best_chars.append(CHARSET[best_idx])
            best_probs.append(float(probs[best_idx]))
            entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
        else:
            best_chars.append('?')
            best_probs.append(0.0)
            entropies.append(np.log(len(CHARSET)))

    # Top panel: best char boxes
    for i, (ch, p) in enumerate(zip(best_chars, best_probs)):
        if observed and f"G{i+1}" in observed:
            face = PALETTE["accent"]
            edge = PALETTE["edge"]
            lw = 1.5
            alpha = 1.0
            txt_color = PALETTE["text"]
        else:
            face = PALETTE["dark"]
            edge = PALETTE["edge"]
            lw = 1.0
            alpha = max(0.25, min(1.0, p))  # scale by confidence
            txt_color = "white"

        rect = FancyBboxPatch((i - 0.4, -0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha)
        ax1.add_patch(rect)
        ax1.text(i, 0, ch, ha='center', va='center', fontsize=16, fontweight='bold', color=txt_color)
        ax1.text(i, -0.7, f'{p:.2f}', ha='center', va='center', fontsize=9, color='gray')

        if target and i < len(target):
            tgt = target[i]
            match = '✓' if ch == tgt else '✗'
            ax1.text(i, 0.7, f'{tgt} {match}', ha='center', va='center',
                     fontsize=10, color=(PALETTE["mid"] if ch == tgt else PALETTE["edge"]))

    ax1.set_xlim(-0.5, 9.5)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'P{i}' for i in positions])
    ax1.set_yticks([])
    ax1.set_title('Best Character Predictions', fontsize=12, fontweight='bold')

    # Bottom panel: entropy bars
    bars = ax2.bar(positions, entropies, color=PALETTE["mid"], alpha=0.9, edgecolor=PALETTE["edge"])
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Entropy', fontsize=11)
    ax2.set_title('Prediction Uncertainty by Position', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.25, axis='y')

    baseline_entropy = np.log(len(CHARSET))
    ax2.axhline(y=baseline_entropy, color=PALETTE["edge"], linestyle='--', alpha=0.5,
                label=f'Max entropy ({baseline_entropy:.2f})')
    ax2.legend(frameon=False)

    for i, e in enumerate(entropies):
        if e == 0:
            ax2.text(i + 1, 0.1, 'Obs', ha='center', fontsize=8, color=PALETTE["edge"])

    plt.suptitle('Character Prediction: Operational View', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# N-GRAM ANALYSIS
# -----------------------------------------------------------------------------
def plot_ngram_analysis(words, n=3, top_k=20, figsize=(14, 6)):
    """
    Analyze and visualize n-gram patterns in training data.
    """
    from collections import Counter
    from wordlike_model import CHARSET_INDEX

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Count n-grams
    ngram_counts = Counter()
    for word in words:
        word = [c for c in word if c in CHARSET_INDEX]
        for i in range(len(word) - n + 1):
            ngram = ''.join(word[i:i + n])
            ngram_counts[ngram] += 1

    top_ngrams = ngram_counts.most_common(top_k)

    # top n-grams
    if top_ngrams:
        ngrams, counts = zip(*top_ngrams)
        y = np.arange(len(ngrams))
        ax1.barh(y, counts, color=PALETTE["dark"], alpha=0.9, edgecolor=PALETTE["edge"])
        ax1.set_yticks(y)
        ax1.set_yticklabels(ngrams, fontsize=9)
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title(f'Top {top_k} {n}-grams', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        for i, v in enumerate(counts):
            ax1.text(v + max(counts) * 0.01, i, str(v), va='center', fontsize=8, color=PALETTE["text"])
    else:
        ax1.text(0.5, 0.5, 'No n-grams found', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, color=PALETTE["text"])

    # character frequencies
    char_counts = Counter()
    for word in words:
        for c in word:
            if c in CHARSET_INDEX:
                char_counts[c] += 1

    if char_counts:
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:26]
        if sorted_chars:
            chars, freqs = zip(*sorted_chars)
            ax2.bar(range(len(chars)), freqs, color=PALETTE["mid"], alpha=0.95, edgecolor=PALETTE["edge"])
            ax2.set_xticks(range(len(chars)))
            ax2.set_xticklabels(chars, fontsize=9)
            ax2.set_xlabel('Character', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Character Frequency Distribution', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.25, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No character data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color=PALETTE["text"])

    plt.suptitle(f'Training Data Analysis ({len(words)} words)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# PREDICTION TRAJECTORY
# -----------------------------------------------------------------------------
def plot_prediction_trajectory(model, partial, max_length=10, figsize=(12, 8)):
    """
    Show how predictions evolve as more characters are revealed.
    Observed: accent box; Predicted: dark box (alpha by confidence).
    """
    from wordlike_model import run_wordlike_inference, suggest_wordlike_guess, CHARSET_INDEX

    steps_to_show = max_length - len(partial)
    if steps_to_show <= 0:
        print("Partial password is already at max length")
        return None

    fig, axes = plt.subplots(steps_to_show, 1, figsize=figsize, sharex=True)
    if steps_to_show == 1:
        axes = [axes]

    current = partial
    positions = list(range(1, max_length + 1))

    for step, ax in enumerate(axes):
        observed = {f"G{i+1}": CHARSET_INDEX[c] for i, c in enumerate(current)}
        posteriors = run_wordlike_inference(model, observed)
        guess = suggest_wordlike_guess(posteriors, observed)

        for pos in positions:
            node = f"G{pos}"
            if pos <= len(current):
                ch, p = current[pos - 1], 1.0
                face, edge, alpha, txt = PALETTE["accent"], PALETTE["edge"], 1.0, PALETTE["text"]
            elif node in posteriors:
                probs = posteriors[node].values
                best_idx = int(np.argmax(probs))
                ch, p = CHARSET[best_idx], float(probs[best_idx])
                face, edge, alpha, txt = PALETTE["dark"], PALETTE["edge"], max(0.25, min(1.0, p)), "white"
            else:
                ch, p = "?", 0.0
                face, edge, alpha, txt = PALETTE["mid"], PALETTE["edge"], 0.4, "white"

            rect = plt.Rectangle((pos - 0.4, -0.4), 0.8, 0.8,
                                 facecolor=face, edgecolor=edge, alpha=alpha)
            ax.add_patch(rect)
            ax.text(pos, 0, ch, ha='center', va='center', fontsize=12, fontweight='bold', color=txt)

        ax.set_xlim(0.5, max_length + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(f'Step {step + 1}: Observed "{current}"', fontsize=10, loc='left')

        # Auto-reveal next char from guess if available
        if len(guess) > len(current) and step < steps_to_show - 1:
            next_pos = len(current) + 1
            if next_pos <= max_length and guess[next_pos - 1] != '?':
                current += guess[next_pos - 1]

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([f'P{i}' for i in positions])
    axes[-1].set_xlabel('Position', fontsize=11)

    plt.suptitle('Prediction Evolution as Characters are Revealed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
