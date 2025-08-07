# timing_plot.py
"""
Visualization functions for timing attack model.
Provides structural, posterior, and operational views.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch


def plot_timing_pgm_structure(figsize=(12, 4)):
    """
    Show the causal structure of the timing attack PGM.
    G_i → T_i for each position.
    
    Args:
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
    
    # Draw nodes
    for i in range(10):
        # G nodes (correctness)
        g_circle = patches.Circle((i, 0.5), 0.25, 
                                 facecolor='lightblue', 
                                 edgecolor='black', linewidth=2)
        ax.add_patch(g_circle)
        ax.text(i, 0.5, f'G{i+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # T nodes (timing)
        t_circle = patches.Circle((i, -0.5), 0.25, 
                                 facecolor='lightyellow', 
                                 edgecolor='black', linewidth=2)
        ax.add_patch(t_circle)
        ax.text(i, -0.5, f'T{i+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Edges G → T
        ax.arrow(i, 0.25, 0, -0.5, head_width=0.1, head_length=0.05, 
                fc='black', ec='black')
    
    # Add legend
    ax.text(4.5, -1.2, 'G = Correctness (0/1), T = Timing (short/med/long)', 
            ha='center', fontsize=11, style='italic')
    
    plt.title('Timing Attack PGM Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_timing_posteriors(posteriors, timing_classes=None, secret=None, figsize=(12, 6)):
    """
    Display posterior probabilities of correctness for each position.
    
    Args:
        posteriors: Dictionary of posterior distributions
        timing_classes: Observed timing bins
        secret: True password for comparison
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Extract probabilities
    positions = list(range(1, 11))
    prob_correct = []
    
    for i in positions:
        node = f"G{i}"
        if node in posteriors:
            prob_correct.append(posteriors[node].values[1])
        else:
            prob_correct.append(0.5)
    
    # Plot 1: Bar chart of P(correct)
    colors = ['green' if p > 0.7 else 'orange' if p > 0.3 else 'red' 
              for p in prob_correct]
    bars = ax1.bar(positions, prob_correct, color=colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('P(Correct)', fontsize=11)
    ax1.set_title('Posterior Probability of Correctness by Position', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, prob_correct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Timing observations
    if timing_classes:
        timing_colors = {0: 'red', 1: 'orange', 2: 'green'}
        timing_labels = {0: 'Short', 1: 'Medium', 2: 'Long'}
        
        for i, t_class in enumerate(timing_classes[:10]):
            ax2.bar(i+1, 1, color=timing_colors[t_class], alpha=0.6, edgecolor='black')
            ax2.text(i+1, 0.5, timing_labels[t_class], ha='center', va='center',
                    fontsize=9, rotation=90)
    
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Timing Class', fontsize=11)
    ax2.set_title('Observed Timing Patterns', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.2])
    ax2.set_yticks([])
    
    # Add actual password if provided
    if secret:
        for i, char in enumerate(secret[:10]):
            ax2.text(i+1, 0.1, char, ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_timing_operational(timings, secret=None, figsize=(12, 5)):
    """
    Show operational view of timing attack in action.
    Displays raw timing measurements and attack progression.
    
    Args:
        timings: List of timing measurements
        secret: True password
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = list(range(1, len(timings) + 1))
    
    # Create step plot showing timing progression
    ax.step(positions, timings, where='mid', linewidth=2, color='blue', label='Timing')
    ax.plot(positions, timings, 'o', markersize=8, color='darkblue')
    
    # Color regions based on timing
    for i, t in enumerate(timings):
        if t > np.mean(timings) + np.std(timings):
            ax.axvspan(i+0.5, i+1.5, alpha=0.2, color='green', label='Long (Correct)' if i == 0 else '')
        elif t < np.mean(timings) - np.std(timings):
            ax.axvspan(i+0.5, i+1.5, alpha=0.2, color='red', label='Short (Wrong)' if i == 0 else '')
        else:
            ax.axvspan(i+0.5, i+1.5, alpha=0.2, color='yellow')
    
    # Add threshold lines
    mean_time = np.mean(timings)
    ax.axhline(y=mean_time, color='gray', linestyle='--', alpha=0.5, label='Mean')
    ax.axhline(y=mean_time + np.std(timings), color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=mean_time - np.std(timings), color='red', linestyle=':', alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Character Position', fontsize=12)
    ax.set_ylabel('Response Time (seconds)', fontsize=12)
    ax.set_title('Timing Attack: Operational View', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add password characters if provided
    if secret:
        for i, char in enumerate(secret[:len(timings)]):
            ax.text(i+1, ax.get_ylim()[0] + 0.001, char, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add annotation explaining the attack
    ax.text(0.02, 0.98, 
            'Longer response time indicates correct character\n' +
            'Attack reveals password structure through timing',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_timing_comparison(correct_timings, incorrect_timings, figsize=(10, 5)):
    """
    Compare timing distributions for correct vs incorrect guesses.
    
    Args:
        correct_timings: List of timings for correct characters
        incorrect_timings: List of timings for incorrect characters
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram comparison
    bins = np.linspace(0, max(max(correct_timings), max(incorrect_timings)), 20)
    
    ax1.hist(incorrect_timings, bins=bins, alpha=0.5, label='Incorrect', 
             color='red', edgecolor='black')
    ax1.hist(correct_timings, bins=bins, alpha=0.5, label='Correct', 
             color='green', edgecolor='black')
    ax1.set_xlabel('Response Time (s)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Timing Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    data = [incorrect_timings, correct_timings]
    bp = ax2.boxplot(data, labels=['Incorrect', 'Correct'], 
                     patch_artist=True, notch=True)
    
    # Color the boxes
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Response Time (s)', fontsize=11)
    ax2.set_title('Timing Statistics', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Incorrect: μ={np.mean(incorrect_timings):.3f}, σ={np.std(incorrect_timings):.3f}\n"
    stats_text += f"Correct: μ={np.mean(correct_timings):.3f}, σ={np.std(correct_timings):.3f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Timing Side-Channel Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig