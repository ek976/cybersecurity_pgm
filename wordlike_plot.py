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
        
        ax.set_title(f'Position {pos}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Probability', fontsize=9)
    
    plt.suptitle('Character Probability Distributions by Position', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_wordlike_operational(posteriors, observed=None, target=None, figsize=(14, 6)):
    """
    Show operational view of character prediction in action.
    Displays best guess and alternatives for each position.
    
    Args:
        posteriors: Dictionary of posterior distributions
        observed: Dictionary of observed characters
        target: True password
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    positions = list(range(1, 11))
    best_chars = []
    best_probs = []
    entropies = []
    
    # Extract best predictions and entropy
    for pos in positions:
        node = f"G{pos}"
        
        if observed and node in observed:
            char_idx = observed[node]
            best_chars.append(CHARSET[char_idx])
            best_probs.append(1.0)
            entropies.append(0.0)
        elif node in posteriors:
            probs = posteriors[node].values
            best_idx = np.argmax(probs)
            best_chars.append(CHARSET[best_idx])
            best_probs.append(probs[best_idx])
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        else:
            best_chars.append('?')
            best_probs.append(0.0)
            entropies.append(np.log(len(CHARSET)))
    
    # Plot 1: Character display with confidence
    for i, (char, prob) in enumerate(zip(best_chars, best_probs)):
        # Draw character box
        if observed and f"G{i+1}" in observed:
            color = 'lightgreen'
            edgecolor = 'darkgreen'
            linewidth = 2
        else:
            # Color based on confidence
            if prob > 0.5:
                color = 'lightblue'
            elif prob > 0.2:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            edgecolor = 'black'
            linewidth = 1
        
        rect = FancyBboxPatch((i - 0.4, -0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=edgecolor,
                              linewidth=linewidth)
        ax1.add_patch(rect)
        
        # Add character
        ax1.text(i, 0, char, ha='center', va='center',
                fontsize=16, fontweight='bold')
        
        # Add probability below
        ax1.text(i, -0.7, f'{prob:.2f}', ha='center', va='center',
                fontsize=9, color='gray')
        
        # Add target character above if known
        if target and i < len(target):
            target_char = target[i]
            match = '✓' if char == target_char else '✗'
            color = 'green' if char == target_char else 'red'
            ax1.text(i, 0.7, f'{target_char} {match}', ha='center', va='center',
                    fontsize=10, color=color)
    
    ax1.set_xlim(-0.5, 9.5)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'P{i}' for i in positions])
    ax1.set_yticks([])
    ax1.set_title('Best Character Predictions', fontsize=12, fontweight='bold')
    
    # Add legend
    ax1.text(10.2, 0.5, 'Green: Observed', fontsize=9, color='darkgreen')
    ax1.text(10.2, 0.2, 'Blue: High conf', fontsize=9, color='blue')
    ax1.text(10.2, -0.1, 'Yellow: Med conf', fontsize=9, color='orange')
    ax1.text(10.2, -0.4, 'Red: Low conf', fontsize=9, color='red')
    
    # Plot 2: Entropy (uncertainty) per position
    bars = ax2.bar(positions, entropies, color='purple', alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Entropy', fontsize=11)
    ax2.set_title('Prediction Uncertainty by Position', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add baseline entropy
    baseline_entropy = np.log(len(CHARSET))
    ax2.axhline(y=baseline_entropy, color='red', linestyle='--', 
                alpha=0.5, label=f'Max entropy ({baseline_entropy:.2f})')
    ax2.legend()
    
    # Annotate observed positions
    for i, e in enumerate(entropies):
        if e == 0:
            ax2.text(i+1, 0.1, 'Obs', ha='center', fontsize=8, color='green')
    
    plt.suptitle('Character Prediction: Operational View', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_ngram_analysis(words, n=3, top_k=20, figsize=(14, 6)):
    """
    Analyze and visualize n-gram patterns in training data.
    
    Args:
        words: Training words
        n: N-gram size
        top_k: Number of top n-grams to show
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    from collections import Counter
    from wordlike_model import CHARSET_INDEX
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count n-grams
    ngram_counts = Counter()
    for word in words:
        word = [c for c in word if c in CHARSET_INDEX]
        for i in range(len(word) - n + 1):
            ngram = ''.join(word[i:i+n])
            ngram_counts[ngram] += 1
    
    # Get top n-grams
    top_ngrams = ngram_counts.most_common(top_k)
    
    if top_ngrams:
        # Plot 1: Top n-gram frequencies
        ngrams, counts = zip(*top_ngrams)
        y_pos = np.arange(len(ngrams))
        
        ax1.barh(y_pos, counts, color='steelblue', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ngrams, fontsize=9)
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title(f'Top {top_k} {n}-grams in Training Data', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(counts):
            ax1.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No n-grams found', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
    
    # Plot 2: Character frequency distribution
    char_counts = Counter()
    for word in words:
        for char in word:
            if char in CHARSET_INDEX:
                char_counts[char] += 1
    
    if char_counts:
        # Sort by frequency
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:26]
        if sorted_chars:
            chars, freqs = zip(*sorted_chars)
            
            ax2.bar(range(len(chars)), freqs, color='coral', alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(chars)))
            ax2.set_xticklabels(chars, fontsize=9)
            ax2.set_xlabel('Character', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Character Frequency Distribution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No character data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
    
    plt.suptitle(f'Training Data Analysis ({len(words)} words)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_prediction_trajectory(model, partial, max_length=10, figsize=(12, 8)):
    """
    Show how predictions evolve as more characters are revealed.
    
    Args:
        model: The character prediction model
        partial: Starting partial password
        max_length: Maximum length to predict
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    from wordlike_model import run_wordlike_inference, suggest_wordlike_guess, CHARSET_INDEX
    
    # Ensure we have steps to show
    steps_to_show = max_length - len(partial)
    if steps_to_show <= 0:
        print("Partial password is already at max length")
        return None
    
    fig, axes = plt.subplots(steps_to_show, 1, figsize=figsize, sharex=True)
    
    if steps_to_show == 1:
        axes = [axes]
    
    # Track predictions as we reveal more
    current = partial
    predictions_history = []
    
    for step, ax in enumerate(axes):
        # Current observed
        observed = {f"G{i+1}": CHARSET_INDEX[c] for i, c in enumerate(current)}
        
        # Run inference
        posteriors = run_wordlike_inference(model, observed)
        guess = suggest_wordlike_guess(posteriors, observed)
        predictions_history.append(guess)
        
        # Visualize current state
        positions = list(range(1, max_length + 1))
        
        for pos in positions:
            node = f"G{pos}"
            
            # Determine character and confidence
            if pos <= len(current):
                # Observed
                char = current[pos-1]
                confidence = 1.0
                color = 'green'
            elif node in posteriors:
                # Predicted
                probs = posteriors[node].values
                best_idx = np.argmax(probs)
                char = CHARSET[best_idx]
                confidence = probs[best_idx]
                
                if confidence > 0.5:
                    color = 'blue'
                elif confidence > 0.2:
                    color = 'orange'
                else:
                    color = 'red'
            else:
                char = '?'
                confidence = 0.0
                color = 'gray'
            
            # Draw character box
            rect = plt.Rectangle((pos - 0.4, -0.4), 0.8, 0.8,
                                facecolor=color, alpha=confidence,
                                edgecolor='black')
            ax.add_patch(rect)
            ax.text(pos, 0, char, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        
        ax.set_xlim(0.5, max_length + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(f'Step {step + 1}: Observed "{current}"', 
                    fontsize=10, loc='left')
        
        # Add next character for next iteration (if exists in guess)
        if len(guess) > len(current) and step < steps_to_show - 1:
            next_pos = len(current) + 1
            if next_pos <= max_length and guess[next_pos-1] != '?':
                current += guess[next_pos-1]
    
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([f'P{i}' for i in positions])
    axes[-1].set_xlabel('Position', fontsize=11)
    
    plt.suptitle('Prediction Evolution as Characters are Revealed', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig