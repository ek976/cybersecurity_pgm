def plot_attack_comparison(timing_results, wordlike_results, hybrid_results, 
                          target=None, figsize=(16, 10)):
    """
    Compare all three attack methods visually.
    
    Args:
        timing_results: (posteriors, binary_guess, correctness_probs) from timing attack
        wordlike_results: (posteriors, char_guess, top_predictions) from wordlike attack
        hybrid_results: (posteriors, guess, analysis) from hybrid attack
        target: True password
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # ========== TIMING ATTACK VISUALIZATION ==========
    # Left: Position correctness probabilities
    ax = axes[0, 0]
    _, binary_guess, correctness_probs = timing_results
    positions = list(range(1, 11))
    
    colors = ['green' if p > 0.7 else 'orange' if p > 0.3 else 'red' 
              for p in correctness_probs]
    bars = ax.bar(positions, correctness_probs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, prob in zip(bars, correctness_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('P(Correct)', fontsize=10)
    ax.set_title('Timing Attack: Position Correctness', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Right: Binary guess display
    ax = axes[0, 1]
    for i, bit in enumerate(binary_guess[:10]):
        if bit == '1':
            color = 'green'
            text = '✓'
        elif bit == '0':
            color = 'red'
            text = '✗'
        else:
            color = 'gray'
            text = '?'
        
        circle = plt.Circle((i, 0), 0.35, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(i, 0, text, ha='center', va='center', fontsize=16, 
               color='white', fontweight='bold')
        
        # Show actual character if target provided
        if target and i < len(target):
            ax.text(i, -0.7, target[i], ha='center', va='center',
                   fontsize=10, color='black')
    
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1, 0.6)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'P{i+1}' for i in range(10)])
    ax.set_yticks([])
    ax.set_title('Timing: Binary Correctness', fontsize=11, fontweight='bold')
    ax.text(4.5, -1.3, 'Green=Correct, Red=Incorrect', ha='center', fontsize=9, style='italic')
    
    # ========== CHARACTER PREDICTION VISUALIZATION ==========
    # Left: Character display with confidence
    ax = axes[1, 0]
    _, char_guess, top_preds = wordlike_results
    
    for i, char in enumerate(char_guess[:10]):
        # Get confidence from top predictions
        if i+1 in top_preds and top_preds[i+1]:
            conf = top_preds[i+1][0][1]  # Top prediction probability
            if conf > 0.5:
                color = 'darkgreen'
                facecolor = 'lightgreen'
            elif conf > 0.2:
                color = 'darkorange'
                facecolor = 'lightyellow'
            else:
                color = 'darkred'
                facecolor = 'lightcoral'
        else:
            conf = 0.0
            color = 'gray'
            facecolor = 'lightgray'
        
        rect = FancyBboxPatch((i - 0.35, -0.35), 0.7, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor=facecolor, alpha=0.7,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0, char, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(i, -0.6, f'{conf:.2f}', ha='center', va='center', fontsize=8, color='gray')
        
        # Show match/mismatch if target provided
        if target and i < len(target):
            if char == target[i]:
                ax.text(i, 0.5, '✓', ha='center', va='center', fontsize=10, color='green')
            else:
                ax.text(i, 0.5, '✗', ha='center', va='center', fontsize=10, color='red')
    
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.8, 0.7)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'P{i+1}' for i in range(10)])
    ax.set_yticks([])
    ax.set_title('Character Prediction: Best Guess', fontsize=11, fontweight='bold')
    
    # Right: Character confidence bars
    ax = axes[1, 1]
    confidences = []
    for i in range(1, 11):
        if i in top_preds and top_preds[i]:
            confidences.append(top_preds[i][0][1])
        else:
            confidences.append(0.0)
    
    colors = ['green' if c > 0.5 else 'orange' if c > 0.2 else 'red' 
              for c in confidences]
    bars = ax.bar(positions, confidences, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Confidence', fontsize=10)
    ax.set_title('Character: Prediction Confidence', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== HYBRID ATTACK VISUALIZATION ==========
    # Left: Combined character display
    ax = axes[2, 0]
    _, hybrid_guess, analysis = hybrid_results
    
    for i, char in enumerate(hybrid_guess[:10]):
        conf = analysis['confidence_scores'][i]
        p_correct = analysis['correctness_prob'][i]
        combined = conf * p_correct
        
        if combined > 0.5:
            color = 'darkgreen'
            facecolor = 'lightgreen'
            linewidth = 3
        elif combined > 0.2:
            color = 'darkorange'
            facecolor = 'lightyellow'
            linewidth = 2
        else:
            color = 'darkred'
            facecolor = 'lightcoral'
            linewidth = 1
        
        rect = FancyBboxPatch((i - 0.35, -0.35), 0.7, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor=facecolor, alpha=0.8,
                              edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)
        ax.text(i, 0, char, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(i, -0.6, f'{combined:.2f}', ha='center', va='center', 
                fontsize=8, color='gray')
        
        # Show match if target known
        if target and i < len(target):
            if char == target[i]:
                ax.plot(i, 0.5, 'g*', markersize=12)
            else:
                ax.plot(i, 0.5, 'rx', markersize=10)
    
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.8, 0.7)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'P{i+1}' for i in range(10)])
    ax.set_yticks([])
    ax.set_title('Hybrid Attack: Combined Result', fontsize=11, fontweight='bold')
    
    # Right: Multi-metric comparison
    ax = axes[2, 1]
    width = 0.25
    x = np.array(positions)
    
    # Extract metrics
    char_conf = analysis['confidence_scores'][:10]
    p_correct = analysis['correctness_prob'][:10]
    combined = [c * p for c, p in zip(char_conf, p_correct)]
    
    # Plot grouped bars
    bars1 = ax.bar(x - width, char_conf, width, 
                   label='Char Conf', color='green', alpha=0.7)
    bars2 = ax.bar(x, p_correct, width, 
                   label='P(Correct)', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, combined, width, 
                   label='Combined', color='purple', alpha=0.7)
    
    # Add subtle value labels for combined only
    for i, (bar, val) in enumerate(zip(bars3, combined)):
        if val > 0.1:  # Only label significant values
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Hybrid: Multi-Source Confidence', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    plt.suptitle('Attack Method Comparison', fontsize=14, fontweight='bold', y=1.02)
    
    # Add summary text at bottom
    if target:
        # Calculate accuracies
        timing_positions = sum(1 for i, b in enumerate(binary_guess[:len(target)]) 
                              if b == '1' and i < len(target))
        char_correct = sum(1 for i, c in enumerate(char_guess[:len(target)]) 
                          if c == target[i])
        hybrid_correct = sum(1 for i, c in enumerate(hybrid_guess[:len(target)]) 
                            if c == target[i])
        
        summary = f"Target: {target} | "
        summary += f"Timing: {timing_positions}/{len(target)} positions | "
        summary += f"Character: {char_correct}/{len(target)} correct | "
        summary += f"Hybrid: {hybrid_correct}/{len(target)} correct"
        
        fig.text(0.5, 0.02, summary, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_hybrid_pgm_structure():
    """
    Draws a simple hybrid PGM diagram showing how timing and wordlike models connect.
    """
    G = nx.DiGraph()

    # Example node setup
    max_len = 10
    for i in range(1, max_len + 1):
        G.add_node(f"G{i}", layer="chars")
        G.add_node(f"T{i}", layer="timing")
        G.add_edge(f"G{i}", f"T{i}")  # char influences timing
        if i > 1:
            G.add_edge(f"G{i-1}", f"G{i}")  # bigram
        if i > 2:
            G.add_edge(f"G{i-2}", f"G{i}")  # trigram (optional)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=1200, node_color="lightblue", arrowsize=15)
    plt.title("Hybrid Attack PGM Structure", fontsize=16)
    plt.show()


import matplotlib.pyplot as plt
import networkx as nx

def plot_hybrid_pgm_structure(use_ngrams=True, figsize=(16, 8)):
    """
    Draws the hybrid PGM structure showing character (G_i), correctness (C_i),
    and timing (T_i) nodes, plus dependencies.
    """
    G = nx.DiGraph()

    max_len = 10
    for i in range(1, max_len + 1):
        G.add_node(f"G{i}", layer="chars")        # Character node
        G.add_node(f"C{i}", layer="correctness")  # Correctness node
        G.add_node(f"T{i}", layer="timing")       # Timing node

        # Edges within each position
        G.add_edge(f"G{i}", f"C{i}")
        G.add_edge(f"C{i}", f"T{i}")

        # N-gram edges
        if i > 1:
            G.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            G.add_edge(f"G{i-2}", f"G{i}")

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", arrowsize=15)
    plt.title("Hybrid Attack PGM Structure", fontsize=16)
    return plt.gcf()


import matplotlib.pyplot as plt
import numpy as np

def plot_hybrid_posteriors(posteriors, observed_chars, timing_classes, target, figsize=(16, 10)):
    """
    Plots per-position posterior probabilities for hybrid attack.
    """
    positions = sorted(int(k[1:]) for k in posteriors.keys())
    probs_correct = [posteriors[f"G{i}"].values[1] if f"G{i}" in posteriors else 1.0 for i in positions]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(positions, probs_correct, color="skyblue", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xlabel("Position")
    ax.set_ylabel("P(Correct)")
    ax.set_title("Hybrid Attack: Posterior Probabilities")
    return fig

def plot_hybrid_operational(posteriors, observed_chars, timing_classes, target, figsize=(16, 8)):
    """
    Simple operational view showing observed chars, timing classes, and posteriors.
    """
    positions = range(1, len(target) + 1)
    probs_correct = [posteriors[f"G{i}"].values[1] if f"G{i}" in posteriors else 1.0 for i in positions]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(positions, probs_correct, color="lightgreen", alpha=0.6, label="P(Correct)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(Correct)")

    ax2 = ax1.twinx()
    ax2.plot(positions, timing_classes[:len(target)], 'o-', color="orange", label="Timing class")
    ax2.set_ylabel("Timing Class")

    fig.legend(loc="upper right")
    ax1.set_title("Hybrid Attack Operational View")
    return fig

def plot_attack_comparison(timing_results, wordlike_results, hybrid_results, target, figsize=(16, 10)):
    """
    Compares accuracy across timing, wordlike, and hybrid methods.
    """
    _, _, timing_probs = timing_results
    timing_acc = sum(p > 0.5 for p in timing_probs) / len(timing_probs)

    _, guess_word, _ = wordlike_results
    wordlike_acc = sum(guess_word[i] == target[i] for i in range(len(target))) / len(target)

    _, guess_hybrid, _ = hybrid_results
    hybrid_acc = sum(guess_hybrid[i] == target[i] for i in range(len(target))) / len(target)

    accuracies = [timing_acc, wordlike_acc, hybrid_acc]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Timing", "Wordlike", "Hybrid"], accuracies, color=["skyblue", "lightgreen", "orange"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Attack Method Comparison")
    return fig

