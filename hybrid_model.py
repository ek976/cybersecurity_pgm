# hybrid_model.py
"""
Hybrid model combining character prediction with timing feedback.
Demonstrates how multiple attack vectors can be combined for more powerful inference.
Combines: G_i ∈ CHARSET (characters), C_i ∈ {0,1} (correctness), T_i ∈ {0,1,2} (timing).
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import time

# Import components from other models
from wordlike_model import (
    CHARSET, CHARSET_INDEX, CHARSET_REVERSE,
    build_ngram_counts, normalize_counts, get_training_words
)
from timing_model import bin_timing_measurements


def build_hybrid_model(words=None, use_ngrams=True, target_password=None):
    """
    Build hybrid attack model combining character and timing information.
    
    Network structure:
    - G_i ∈ CHARSET: Character at position i
    - C_i ∈ {0,1}: Correctness at position i
    - T_i ∈ {0,1,2}: Timing observation at position i
    
    Edges:
    - G_i → G_{i+1}: Character dependencies (language model)
    - G_i → C_i: Character determines correctness
    - C_i → T_i: Correctness affects timing
    
    Args:
        words: Training words for language model
        use_ngrams: Use trigrams (True) or bigrams (False)
        target_password: If provided, creates deterministic correctness CPDs
    
    Returns:
        DiscreteBayesianNetwork: Configured hybrid model
    """
    if words is None:
        words = get_training_words(source="hybrid", limit=5000)
    
    model = DiscreteBayesianNetwork()
    
    # Build n-gram probabilities
    n = 3 if use_ngrams else 2
    ngram_counts = build_ngram_counts(words, n=n)
    ngram_probs = normalize_counts(ngram_counts)
    
    # Add all nodes
    for i in range(1, 11):
        model.add_node(f"G{i}")  # Character node
        model.add_node(f"C{i}")  # Correctness node  
        model.add_node(f"T{i}")  # Timing node
    
    # Add character sequence edges
    for i in range(2, 11):
        model.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            model.add_edge(f"G{i-2}", f"G{i}")
    
    # Add timing attack edges
    for i in range(1, 11):
        model.add_edge(f"G{i}", f"C{i}")
        model.add_edge(f"C{i}", f"T{i}")
    
    # ========== Character CPDs (from language model) ==========
    
    # First character CPD
    unigram = Counter(w[0] for w in words if w and w[0] in CHARSET_INDEX)
    total = sum(unigram.values())
    if total > 0:
        prob_start = [unigram.get(c, 1e-6) / total for c in CHARSET]
    else:
        prob_start = [1/len(CHARSET)] * len(CHARSET)
    
    cpd_g1 = TabularCPD("G1", len(CHARSET), [[p] for p in prob_start])
    model.add_cpds(cpd_g1)
    
    # Second character CPD (bigram)
    values = []
    for prev_char in CHARSET:
        dist = ngram_probs.get((prev_char,), {})
        prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
        prob_sum = sum(prob_vector)
        prob_vector = [p / prob_sum for p in prob_vector]
        values.append(prob_vector)
    values = list(map(list, zip(*values)))
    
    cpd_g2 = TabularCPD("G2", len(CHARSET), values, 
                        evidence=["G1"], evidence_card=[len(CHARSET)])
    model.add_cpds(cpd_g2)
    
    # Remaining character CPDs
    for i in range(3, 11):
        if use_ngrams:
            evidence = [f"G{i-2}", f"G{i-1}"]
            evidence_card = [len(CHARSET)] * 2
            values = []
            for c1 in CHARSET:
                for c2 in CHARSET:
                    dist = ngram_probs.get((c1, c2), {})
                    prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
                    prob_sum = sum(prob_vector)
                    prob_vector = [p / prob_sum for p in prob_vector]
                    values.append(prob_vector)
            values = list(map(list, zip(*values)))
        else:
            evidence = [f"G{i-1}"]
            evidence_card = [len(CHARSET)]
            values = []
            for prev_char in CHARSET:
                dist = ngram_probs.get((prev_char,), {})
                prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
                prob_sum = sum(prob_vector)
                prob_vector = [p / prob_sum for p in prob_vector]
                values.append(prob_vector)
            values = list(map(list, zip(*values)))
        
        cpd = TabularCPD(f"G{i}", len(CHARSET), values, 
                        evidence=evidence, evidence_card=evidence_card)
        model.add_cpds(cpd)
    
    # ========== Correctness CPDs ==========
    
    for i in range(1, 11):
        if target_password and i <= len(target_password):
            # Deterministic CPD based on actual password
            target_char = target_password[i-1]
            if target_char in CHARSET_INDEX:
                target_idx = CHARSET_INDEX[target_char]
                values = []
                for char_idx in range(len(CHARSET)):
                    if char_idx == target_idx:
                        values.append([0.0, 1.0])  # Correct
                    else:
                        values.append([1.0, 0.0])  # Incorrect
                values = list(map(list, zip(*values)))
            else:
                # Unknown character - uniform
                values = [[0.5] * len(CHARSET), [0.5] * len(CHARSET)]
        else:
            # No target or beyond password length - use heuristic
            # Higher probability of correctness for common characters
            common_chars = set('etaoinshrdlu')
            values = []
            for char in CHARSET:
                if char.lower() in common_chars:
                    values.append([0.7, 0.3])  # More likely correct
                else:
                    values.append([0.9, 0.1])  # Less likely correct
            values = list(map(list, zip(*values)))
        
        cpd_c = TabularCPD(
            f"C{i}", 2, values,
            evidence=[f"G{i}"], evidence_card=[len(CHARSET)]
        )
        model.add_cpds(cpd_c)
    
    # ========== Timing CPDs ==========
    
    for i in range(1, 11):
        cpd_t = TabularCPD(
            f"T{i}", 3,
            [[0.8, 0.05],  # P(short | incorrect, correct)
             [0.15, 0.25], # P(medium | incorrect, correct)
             [0.05, 0.7]], # P(long | incorrect, correct)
            evidence=[f"C{i}"], evidence_card=[2]
        )
        model.add_cpds(cpd_t)
    
    model.check_model()
    print(f"Hybrid model built with {len(words)} training words")
    return model


def run_hybrid_inference(model, observed_chars=None, timing_classes=None):
    """
    Run inference on hybrid model with both character and timing evidence.
    
    Args:
        model: The hybrid PGM
        observed_chars: Dict of observed characters {"G1": char_index, ...}
        timing_classes: List of timing observations [0, 1, 2, ...]
    
    Returns:
        dict: Posterior distributions for all unobserved nodes
    """
    infer = VariableElimination(model)
    
    # Combine all evidence
    evidence = {}
    
    # Add character evidence
    if observed_chars:
        evidence.update(observed_chars)
    
    # Add timing evidence
    if timing_classes:
        for i, t_class in enumerate(timing_classes[:10]):
            evidence[f"T{i+1}"] = t_class
    
    # Query for all relevant nodes
    posteriors = {}
    
    for i in range(1, 11):
        # Character posteriors (if not observed)
        g_var = f"G{i}"
        if g_var not in evidence:
            try:
                result = infer.query(variables=[g_var], evidence=evidence)
                posteriors[g_var] = result
            except:
                pass
        
        # Correctness posteriors
        c_var = f"C{i}"
        try:
            result = infer.query(variables=[c_var], evidence=evidence)
            posteriors[c_var] = result
        except:
            pass
    
    return posteriors


def suggest_hybrid_guess(posteriors, observed_chars=None):
    """
    Generate password guess from hybrid model posteriors.
    
    Args:
        posteriors: Dictionary of posterior distributions
        observed_chars: Dictionary of observed characters
    
    Returns:
        tuple: (character_guess, correctness_probabilities)
    """
    char_guess = []
    correctness_probs = []
    
    for i in range(1, 11):
        g_var = f"G{i}"
        c_var = f"C{i}"
        
        # Get character prediction
        if observed_chars and g_var in observed_chars:
            char_idx = observed_chars[g_var]
            char_guess.append(CHARSET[char_idx])
        elif g_var in posteriors:
            probs = posteriors[g_var].values
            best_idx = int(probs.argmax())
            char_guess.append(CHARSET[best_idx])
        else:
            char_guess.append("?")
        
        # Get correctness probability
        if c_var in posteriors:
            prob_correct = posteriors[c_var].values[1]
            correctness_probs.append(prob_correct)
        else:
            correctness_probs.append(0.5)
    
    return ''.join(char_guess), correctness_probs


def measure_hybrid_timing(user_input, secret):
    """
    Measure timing with character-aware delay for hybrid attack.
    
    Args:
        user_input: Password attempt
        secret: True password
    
    Returns:
        float: Total delay time
    """
    delay_base = 0.005
    delay_correct = 0.01
    
    total_delay = 0
    for i in range(len(user_input)):
        if i >= len(secret):
            break
        if user_input[i] == secret[i]:
            time.sleep(delay_correct)
            total_delay += delay_correct
        else:
            time.sleep(delay_base)
            total_delay += delay_base
            break
    
    return total_delay


def collect_hybrid_timing_data(partial_password, secret, test_chars=None):
    """
    Collect timing data for hybrid attack by testing different characters.
    
    Args:
        partial_password: Known prefix
        secret: True password
        test_chars: Characters to test (if None, uses common letters)
    
    Returns:
        list: Average timing for each position
    """
    if test_chars is None:
        test_chars = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']
    
    timings = []
    
    for pos in range(len(partial_password), min(len(secret) + 1, 10)):
        position_timings = []
        
        # Test different characters at this position
        for char in test_chars:
            if pos < len(partial_password):
                test = partial_password[:pos]
            else:
                test = partial_password + ''.join([
                    secret[i] if i < len(partial_password) else 'a' 
                    for i in range(pos - len(partial_password))
                ]) + char
            
            timing = measure_hybrid_timing(test[:pos+1], secret)
            position_timings.append(timing)
        
        # Use average timing for this position
        avg_timing = np.mean(position_timings)
        timings.append(avg_timing)
    
    # Pad with short timings if needed
    while len(timings) < 10:
        timings.append(0.005)
    
    return timings


def get_hybrid_analysis(posteriors, observed_chars=None):
    """
    Analyze hybrid model results with detailed statistics.
    
    Args:
        posteriors: Dictionary of posterior distributions
        observed_chars: Dictionary of observed characters
    
    Returns:
        dict: Analysis results including entropy, top predictions, etc.
    """
    analysis = {
        'character_entropy': [],
        'correctness_prob': [],
        'top_3_chars': [],
        'confidence_scores': []
    }
    
    for i in range(1, 11):
        g_var = f"G{i}"
        c_var = f"C{i}"
        
        # Character analysis
        if g_var in posteriors:
            probs = posteriors[g_var].values
            
            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            analysis['character_entropy'].append(entropy)
            
            # Top 3 predictions
            top_indices = np.argsort(probs)[-3:][::-1]
            top_3 = [(CHARSET[idx], probs[idx]) for idx in top_indices]
            analysis['top_3_chars'].append(top_3)
            
            # Confidence (max probability)
            analysis['confidence_scores'].append(np.max(probs))
        elif observed_chars and g_var in observed_chars:
            analysis['character_entropy'].append(0.0)
            char_idx = observed_chars[g_var]
            analysis['top_3_chars'].append([(CHARSET[char_idx], 1.0)])
            analysis['confidence_scores'].append(1.0)
        else:
            analysis['character_entropy'].append(np.log(len(CHARSET)))
            analysis['top_3_chars'].append([("?", 0.0)])
            analysis['confidence_scores'].append(0.0)
        
        # Correctness analysis
        if c_var in posteriors:
            prob_correct = posteriors[c_var].values[1]
            analysis['correctness_prob'].append(prob_correct)
        else:
            analysis['correctness_prob'].append(0.5)
    
    return analysis


# Demo function
def demo_hybrid_attack(secret="vegetable", known_prefix="veg", verbose=True):
    """
    Complete demo of hybrid attack combining timing and character prediction.
    
    Args:
        secret: Target password
        known_prefix: Known prefix of password
        verbose: Print detailed output
    
    Returns:
        tuple: (posteriors, guess, analysis)
    """
    if verbose:
        print(f"Hybrid Attack Demo")
        print(f"Target: {secret}")
        print(f"Known prefix: {known_prefix}")
        print("=" * 70)
    
    # Build model with target password for realistic correctness CPDs
    model = build_hybrid_model(use_ngrams=True, target_password=secret)
    
    # Set up observed characters
    observed_chars = {}
    for i, char in enumerate(known_prefix):
        observed_chars[f"G{i+1}"] = CHARSET_INDEX[char]
    
    # Collect realistic timing data
    timings = collect_hybrid_timing_data(known_prefix, secret)
    timing_classes = bin_timing_measurements(timings)
    
    if verbose:
        print(f"\nTiming observations: {timing_classes}")
        print("(0=short, 1=medium, 2=long)")
    
    # Run inference
    posteriors = run_hybrid_inference(model, observed_chars, timing_classes)
    
    # Get predictions and analysis
    guess, correctness = suggest_hybrid_guess(posteriors, observed_chars)
    analysis = get_hybrid_analysis(posteriors, observed_chars)
    
    if verbose:
        print("\nResults:")
        print("-" * 70)
        print("Pos | Char | P(Correct) | Top 3 Predictions      | Actual | Match")
        print("-" * 70)
        
        for i in range(10):
            actual = secret[i] if i < len(secret) else "-"
            predicted = guess[i]
            p_correct = correctness[i]
            
            # Format top predictions
            top_3 = ", ".join([
                f"{c}({p:.2f})" 
                for c, p in analysis['top_3_chars'][i][:3]
            ])
            
            match = "✓" if predicted == actual else "✗"
            observed = "*" if f"G{i+1}" in observed_chars else " "
            
            print(f" {i+1:2d}{observed}|  {predicted}   |   {p_correct:.3f}    | {top_3:22s} |   {actual}   |  {match}")
        
        print("-" * 70)
        print(f"Predicted: {guess}")
        print(f"Actual:    {secret:<10}")
        print(f"Match:     {''.join(['✓' if i < len(guess) and i < len(secret) and guess[i]==secret[i] else '✗' for i in range(min(10, max(len(guess), len(secret))))])}")
        print("\n* = observed character")
        
        # Show advantage over individual models
        print("\nHybrid Advantage:")
        print(f"- Uses language patterns to predict likely characters")
        print(f"- Uses timing to confirm/reject predictions")
        print(f"- Average confidence: {np.mean(analysis['confidence_scores']):.3f}")
        print(f"- Average P(correct): {np.mean(correctness):.3f}")
    
    return posteriors, guess, analysis


# Comparison function
def compare_attack_methods(secret="vegetable", known_prefix="veg"):
    """
    Compare all three attack methods on the same target.
    
    Args:
        secret: Target password
        known_prefix: Known prefix
    
    Returns:
        dict: Comparison results
    """
    from timing_model import demo_timing_attack
    from wordlike_model import demo_wordlike_attack
    
    print("=" * 70)
    print("ATTACK METHOD COMPARISON")
    print("=" * 70)
    
    # Timing-only attack
    print("\n1. TIMING-ONLY ATTACK")
    print("-" * 30)
    _, timing_guess, timing_probs = demo_timing_attack(secret, verbose=False)
    print(f"Binary positions: {timing_guess}")
    print(f"Average P(correct): {np.mean(timing_probs):.3f}")
    
    # Character-only attack
    print("\n2. CHARACTER-ONLY ATTACK")
    print("-" * 30)
    _, char_guess, _ = demo_wordlike_attack(known_prefix, secret, verbose=False)
    print(f"Character guess: {char_guess}")
    accuracy = sum(1 for i in range(min(len(char_guess), len(secret))) 
                   if char_guess[i] == secret[i]) / len(secret)
    print(f"Character accuracy: {accuracy:.3f}")
    
    # Hybrid attack
    print("\n3. HYBRID ATTACK")
    print("-" * 30)
    _, hybrid_guess, analysis = demo_hybrid_attack(secret, known_prefix, verbose=False)
    print(f"Hybrid guess: {hybrid_guess}")
    hybrid_accuracy = sum(1 for i in range(min(len(hybrid_guess), len(secret))) 
                         if hybrid_guess[i] == secret[i]) / len(secret)
    print(f"Character accuracy: {hybrid_accuracy:.3f}")
    print(f"Average confidence: {np.mean(analysis['confidence_scores']):.3f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Hybrid attack combines strengths of both methods")
    print("=" * 70)
    
    return {
        'timing_only': timing_guess,
        'character_only': char_guess,
        'hybrid': hybrid_guess,
        'hybrid_confidence': np.mean(analysis['confidence_scores'])
    }