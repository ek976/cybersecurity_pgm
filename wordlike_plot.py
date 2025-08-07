# wordlike_plot.py
"""
Visualization functions for character prediction model.
Provides structural, posterior, and operational views.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from wordlike_model import CHARSET


def plot_wordlike_pgm_structure(use_ngrams=True, figsize=(14, 4)):
    """
    Show the causal structure of the character prediction PGM.
    Shows character dependencies (bigram or trigram).
    
    Args:
        use_ngrams: Whether trigram model is used
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up plot area
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw character nodes
    for i in range(10):
        circle = patches.Circle((i, 0), 0.3, 
                               facecolor='lightgreen', 
                               edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(i, 0, f'G{i+1}', ha='center', va='center', 
                fontsize=11, fontweight='bold')
    
    # Draw edges for bigram dependencies
    for i in range(9):
        ax.arrow(i + 0.3, 0, 0.4, 0, head_width=0.1, head_length=0.05,
                fc='black', ec='black', linewidth=1.5)
    
    # Draw trigram edges if enabled
    if use_ngrams:
        for i in range(8):
            # Curved arrow for skip connection
            arc = FancyArrowPatch((i + 0.2, -0.3), (i + 1.8, -0.3),
                                 connectionstyle="arc3,rad=-.3",
                                 arrowstyle='->', color='blue', 
                                 linewidth=1, alpha=0.6)
            ax.add_patch(arc)
    
    # Add model type label
    model_type = "Trigram Model" if use_ngrams else "Bigram Model"
    ax.text(4.5, -0.8, f'{model_type}: Each character depends on previous {"1-2" if use_ngrams else "1"} character(s)', 
            ha='center', fontsize=11, style='italic')
    
    # Add legend
    if use_ngrams:
        ax.plot([], [], 'k-', linewidth=1.5, label='Direct dependency (i-1 → i)')
        ax.plot([], [], 'b-', linewidth=1, alpha=0.6, label='Skip dependency (i-2 → i)')
        ax.legend(loc='upper right')
    
    plt.title('Character Prediction PGM Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_wordlike_posteriors(posteriors, observed=None, target=None, figsize=(14, 8)):
    """
    Display posterior character probabilities for each position.
    
    Args:
        posteriors: Dictionary of posterior distributions
        observed: Dictionary of observed characters
        target: True password for comparison
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()
    
    for pos in range(1, 11):
        ax = axes[pos-1]
        node = f"G{pos}"
        
        if observed and node in observed:
            # Show observed character
            char_idx = observed[node]
            char = CHARSET[char_idx]
            ax.text(0.5, 0.5, f"Observed:\n'{char}'", 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
            ax.set_facecolor('#e6ffe6')
        elif node in posteriors:
            # Show probability distribution
            probs = posteriors[node].values
            top_k = 10
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            chars = [CHARSET[i] for i in top_indices]
            values = [probs[i] for i in top_indices]
            
            # Color bars based on probability
            colors = ['green' if v > 0.2 else 'orange' if v > 0.1 else 'gray' 
                     for v in values]
            
            bars = ax.barh(range(top_k), values, color=colors, alpha=0.7)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(chars)
            ax.set_xlim([0, max(0.5, max(values) * 1.1)])
            ax.invert_yaxis()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.3f}', 
                       va='center', fontsize=8)
            
            # Highlight if target character is known
            if target and pos <= len(target):
                target_char = target[pos-1]
                if target_char in chars:
                    idx = chars.index(target_char)
                    bars[idx].set_edgecolor('red')
                    bars[idx].set_linewidth(2)