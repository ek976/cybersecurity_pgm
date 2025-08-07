# timing_model.py 

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from wordlike_model import CHARSET


def measure_time(user_input, secret="hunter2xyz"):
    import time
    delay_per_char = 0.01
    for i in range(len(user_input)):
        if i >= len(secret) or user_input[i] != secret[i]:
            return False
        time.sleep(delay_per_char)
    return len(user_input) == len(secret)



def collect_timing_data(secret="hunter2xyz"):
    timings = []
    for i in range(1, len(secret) + 1):
        guess = secret[:i]
        import time
        start = time.perf_counter()
        measure_time(guess)
        end = time.perf_counter()
        timings.append(end - start)
    return timings


def bin_timings(timings):
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


def build_model():
    edges = [(f'G{i}', f'T{i}') for i in range(1, 11)]
    model = DiscreteBayesianNetwork(edges)

    for i in range(1, 11):
        g = f'G{i}'
        t = f'T{i}'

        cpd_guess = TabularCPD(g, 2, [[0.5], [0.5]])
        cpd_time = TabularCPD(
            variable=t,
            variable_card=3,
            values=[
                [0.1, 0.7],  # short
                [0.3, 0.2],  # medium
                [0.6, 0.1]   # long
            ],
            evidence=[g],
            evidence_card=[2]
        )

        model.add_cpds(cpd_guess, cpd_time)

    model.check_model()
    return model


def run_inference(model, timing_classes):
    infer = VariableElimination(model)
    evidence = {f"T{i+1}": timing_classes[i] for i in range(10)}
    posteriors = {}
    for i in range(1, 11):
        result = infer.query(variables=[f"G{i}"], evidence=evidence)
        posteriors[f"G{i}"] = result
    return posteriors


def suggest_password_guess(posteriors):
    """
    Suggests the most probable password based on the posterior distributions.

    Args:
        posteriors (dict): Dictionary with keys like 'G1', 'G2', ..., 'G10',
                           and values as pgmpy inference results with .values

    Returns:
        str: Suggested binary guess, e.g. '1110001110'
    """
    guess_bits = []

    for i in range(1, 11):
        node = f"G{i}"
        if node in posteriors:
            probs = posteriors[node].values
            max_val = int(probs[1] > probs[0])  # MAP estimate
            guess_bits.append(str(max_val))
        else:
            guess_bits.append("?")  # unknown if missing

    return ''.join(guess_bits)


###### Below Functions are for a later example with timing 

import time

def collect_timing_characterwise(secret="hunter2xyz"):
    """
    Returns list of (duration, correct_flag) for each character guess
    """
    timings = []
    for i in range(len(secret)):
        guess = secret[:i+1]
        correct = secret[i]
        start = time.perf_counter()
        measure_time(guess, secret=secret)
        end = time.perf_counter()
        duration = end - start
        timings.append((duration, True))  # assume correct
    return timings

def simulate_wrong_timings(secret="hunter2xyz", noise=0.005):
    timings = []
    for i in range(len(secret)):
        # Guess with incorrect final character
        wrong_char = chr((ord(secret[i]) + 1) % 128)
        guess = secret[:i] + wrong_char
        start = time.perf_counter()
        measure_time(guess, secret=secret)
        end = time.perf_counter()
        duration = end - start
        timings.append((duration, False))  # incorrect
    return timings

import numpy as np

def build_empirical_timing_cpd(correct_timings, incorrect_timings, bins=[0.005, 0.015]):
    """
    Bins: thresholds for short/medium/long (in seconds)
    """
    all_timings = []

    # Bin as 0=short, 1=med, 2=long
    def bin_duration(d):
        if d < bins[0]: return 0
        elif d < bins[1]: return 1
        else: return 2

    counts = {
        0: [0, 0, 0],  # incorrect → short/med/long
        1: [0, 0, 0]   # correct   → short/med/long
    }

    for d, is_correct in correct_timings + incorrect_timings:
        label = 1 if is_correct else 0
        binned = bin_duration(d)
        counts[label][binned] += 1

    # Normalize into probabilities
    cpd = np.zeros((3, 2))  # 3 timing bins × 2 correctness classes
    for label in [0, 1]:
        total = sum(counts[label])
        if total == 0:
            cpd[:, label] = [1/3, 1/3, 1/3]  # default uniform
        else:
            cpd[:, label] = np.array(counts[label]) / total

    # Replicate across all CHARSET entries (approximation)
    return [cpd[:, 0] if i % 2 == 0 else cpd[:, 1] for i in range(len(CHARSET))]  # shape (len(CHARSET), 3)
