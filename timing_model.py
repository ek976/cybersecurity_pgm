# timing_model.py
"""
Pure timing side-channel attack model.
Demonstrates how response time reveals password correctness at each position.
Binary model: G_i ∈ {0,1} for incorrect/correct at position i.
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import time


def measure_time(user_input, secret="hunter2xyz"):
    """
    Simulates timing attack - takes longer if more characters are correct.
    
    Args:
        user_input: Password attempt
        secret: True password
    
    Returns:
        bool: True only if entire password matches
    """
    delay_per_char = 0.01
    for i in range(len(user_input)):
        if i >= len(secret) or user_input[i] != secret[i]:
            return False
        time.sleep(delay_per_char)
    return len(user_input) == len(secret)


def collect_timing_data(secret="hunter2xyz"):
    """
    Collects timing data for progressively longer correct prefixes.
    
    Args:
        secret: True password to test against
    
    Returns:
        list: Timing measurements for each position
    """
    timings = []
    for i in range(1, min(len(secret) + 1, 11)):
        guess = secret[:i]
        start = time.perf_counter()
        measure_time(guess, secret)
        end = time.perf_counter()
        timings.append(end - start)
    
    # Pad with short timings if password < 10 chars
    while len(timings) < 10:
        timings.append(0.001)
    
    return timings


def bin_timing_measurements(timings):
    """
    Bins timing measurements into categories: short, medium, long.
    
    Args:
        timings: List of timing measurements
    
    Returns:
        list: Bin indices (0=short, 1=medium, 2=long)
    """
    if len(timings) == 0:
        return []
    
    # Use adaptive quantiles based on actual timing distribution
    quantiles = np.quantile(timings, [0.33, 0.66])
    bins = []
    
    for t in timings:
        if t < quantiles[0]:
            bins.append(0)  # short
        elif t < quantiles[1]:
            bins.append(1)  # medium
        else:
            bins.append(2)  # long
    
    return bins


def build_timing_model():
    """
    Builds a binary timing attack PGM.
    
    Network structure:
    - G_i ∈ {0,1}: Character i is incorrect (0) or correct (1)
    - T_i ∈ {0,1,2}: Timing observation (short/medium/long)
    - Edges: G_i → T_i (correctness affects timing)
    
    Returns:
        DiscreteBayesianNetwork: Configured timing attack model
    """
    model = DiscreteBayesianNetwork()
    
    # Add nodes
    for i in range(1, 11):
        model.add_node(f'G{i}')
        model.add_node(f'T{i}')
    
    # Add edges from guess nodes to timing nodes
    for i in range(1, 11):
        model.add_edge(f'G{i}', f'T{i}')
    
    # Create CPDs
    for i in range(1, 11):
        g = f'G{i}'
        t = f'T{i}'
        
        # Prior: uniform probability of correct/incorrect
        cpd_guess = TabularCPD(
            variable=g,
            variable_card=2,
            values=[[0.5], [0.5]]  # P(incorrect), P(correct)
        )
        
        # Timing depends on correctness
        # Column 0: P(T|G=0) incorrect
        # Column 1: P(T|G=1) correct
        cpd_time = TabularCPD(
            variable=t,
            variable_card=3,
            values=[
                [0.7, 0.1],  # P(short | incorrect, correct)
                [0.2, 0.3],  # P(medium | incorrect, correct)  
                [0.1, 0.6]   # P(long | incorrect, correct)
            ],
            evidence=[g],
            evidence_card=[2]
        )
        
        model.add_cpds(cpd_guess, cpd_time)
    
    model.check_model()
    return model


def run_timing_inference(model, timing_classes):
    """
    Run inference on the timing model given observed timing classes.
    
    Args:
        model: The timing attack PGM
        timing_classes: List of timing bins (0, 1, or 2) for each position
    
    Returns:
        dict: Posterior distributions for each G node
    """
    infer = VariableElimination(model)
    
    # Create evidence from timing observations
    evidence = {}
    for i in range(min(len(timing_classes), 10)):
        evidence[f"T{i+1}"] = timing_classes[i]
    
    # Query for each guess node
    posteriors = {}
    for i in range(1, 11):
        var = f"G{i}"
        result = infer.query(variables=[var], evidence=evidence)
        posteriors[var] = result
    
    return posteriors


def suggest_binary_guess(posteriors):
    """
    Suggests which character positions are likely correct.
    
    Args:
        posteriors: Dictionary of posterior distributions for G nodes
    
    Returns:
        str: Binary string where '1' = likely correct, '0' = likely incorrect
    """
    guess_bits = []
    
    for i in range(1, 11):
        node = f"G{i}"
        if node in posteriors:
            probs = posteriors[node].values
            # probs[0] = P(incorrect), probs[1] = P(correct)
            if probs[1] > probs[0]:
                guess_bits.append("1")  # likely correct
            else:
                guess_bits.append("0")  # likely incorrect
        else:
            guess_bits.append("?")  # unknown
    
    return ''.join(guess_bits)


def get_position_correctness_probabilities(posteriors):
    """
    Extract P(correct) for each position from posteriors.
    
    Args:
        posteriors: Dictionary of posterior distributions
    
    Returns:
        list: Probability of correctness for each position
    """
    probs = []
    for i in range(1, 11):
        node = f"G{i}"
        if node in posteriors:
            prob_correct = posteriors[node].values[1]
            probs.append(prob_correct)
        else:
            probs.append(0.5)  # Default uncertainty
    
    return probs


# Demo function
def demo_timing_attack(secret="hunter2xyz", verbose=True):
    """
    Complete demo of the timing attack model.
    
    Args:
        secret: Password to attack
        verbose: Print detailed output
    
    Returns:
        tuple: (posteriors, binary_guess, correctness_probs)
    """
    if verbose:
        print(f"Timing Attack Demo")
        print(f"Target: {secret}")
        print("=" * 50)
    
    # Collect timing data
    timings = collect_timing_data(secret)
    timing_classes = bin_timing_measurements(timings)
    
    if verbose:
        print(f"Timing measurements: {[f'{t:.3f}' for t in timings[:len(secret)]]}")
        print(f"Timing bins: {timing_classes[:len(secret)]}")
        print("(0=short, 1=medium, 2=long)\n")
    
    # Build and run inference
    model = build_timing_model()
    posteriors = run_timing_inference(model, timing_classes)
    
    # Get results
    binary_guess = suggest_binary_guess(posteriors)
    correctness_probs = get_position_correctness_probabilities(posteriors)
    
    if verbose:
        print("Results:")
        print("-" * 40)
        print("Pos | P(Correct) | Guess | Actual")
        print("-" * 40)
        for i in range(len(secret)):
            actual_char = secret[i] if i < len(secret) else "-"
            print(f" {i+1:2d} |   {correctness_probs[i]:.3f}    |   {binary_guess[i]}   | {actual_char}")
        print("-" * 40)
        print(f"Binary guess: {binary_guess}")
        print(f"Interpretation: Positions marked '1' are likely correct\n")
    
    return posteriors, binary_guess, correctness_probs