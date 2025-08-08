# wordlike_model.py
"""
Character prediction model using n-gram language models.
Predicts password characters based on patterns learned from training data.
Character model: G_i ∈ CHARSET for each position i.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np 
from collections import defaultdict, Counter
import string
from common_words import get_training_words

# ---------------------
# Character Set Definition (must be defined BEFORE any calls to get_training_words)
# ---------------------
CHARSET = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*")
CHARSET_INDEX = {c: i for i, c in enumerate(CHARSET)}
CHARSET_REVERSE = {i: c for c, i in CHARSET_INDEX.items()}

# ---------------------
# N-gram Analysis Functions
# ---------------------
def build_ngram_counts(words, n=2):
    """
    Build n-gram frequency counts from training words.
    """
    counts = defaultdict(lambda: Counter())
    for word in words:
        # Filter to valid charset
        word = [c for c in word if c in CHARSET_INDEX]
        for i in range(len(word) - n + 1):
            prefix = tuple(word[i:i + n - 1])
            next_char = word[i + n - 1]
            counts[prefix][next_char] += 1
    return counts

def normalize_counts(counts_dict):
    """
    Convert frequency counts to probability distributions.
    """
    probs = {}
    for prefix, counter in counts_dict.items():
        total = sum(counter.values())
        if total > 0:
            probs[prefix] = {c: count / total for c, count in counter.items()}
    return probs

# ---------------------
# Model Building
# ---------------------
def build_wordlike_model(words=None, use_ngrams=False, max_length=10):
    """
    Build character prediction PGM using language patterns.
    """
    if words is None:
        words = get_training_words(source="hybrid", limit=5000)
        # Filter words to contain only characters in CHARSET
        words = [''.join(c for c in w if c in CHARSET_INDEX) for w in words if w]

    model = DiscreteBayesianNetwork()
    n = 3 if use_ngrams else 2

    ngram_counts = build_ngram_counts(words, n=n)
    ngram_probs = normalize_counts(ngram_counts)

    # Add nodes
    for i in range(1, max_length + 1):
        model.add_node(f"G{i}")

    # Add edges
    for i in range(2, max_length + 1):
        model.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            model.add_edge(f"G{i-2}", f"G{i}")

    # CPD for first char (unigram)
    unigram = Counter(w[0] for w in words if w and w[0] in CHARSET_INDEX)
    total_unigrams = sum(unigram.values())
    if total_unigrams > 0:
        prob_start = [unigram.get(c, 1e-6) / total_unigrams for c in CHARSET]
    else:
        prob_start = [1 / len(CHARSET)] * len(CHARSET)
    cpd_g1 = TabularCPD("G1", len(CHARSET), [[p] for p in prob_start])
    model.add_cpds(cpd_g1)

    # CPD for second char (bigram)
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

    # CPDs for remaining characters
    for i in range(3, max_length + 1):
        curr = f"G{i}"
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

        cpd = TabularCPD(curr, len(CHARSET), values,
                         evidence=evidence, evidence_card=evidence_card)
        model.add_cpds(cpd)

    model.check_model()
    print(f"Wordlike model built with {len(words)} training words")
    return model

# ---------------------
# Inference Functions
# ---------------------
def run_wordlike_inference(model, observed):
    """
    Run inference to predict unobserved characters.
    """
    infer = VariableElimination(model)
    posteriors = {}
    for i in range(1, 11):
        var = f"G{i}"
        if var in observed:
            continue
        result = infer.query(variables=[var], evidence=observed)
        posteriors[var] = result
    return posteriors

def suggest_wordlike_guess(posteriors, observed=None):
    """
    Generate best character sequence from posteriors.
    """
    guess_chars = []
    for i in range(1, 11):
        var = f"G{i}"
        if observed and var in observed:
            char_idx = observed[var]
            guess_chars.append(CHARSET[char_idx])
        elif var in posteriors:
            probs = posteriors[var].values
            best_idx = int(probs.argmax())
            guess_chars.append(CHARSET[best_idx])
        else:
            guess_chars.append("?")
    return ''.join(guess_chars)

def get_top_k_predictions(posteriors, k=3):
    """
    Get top-k most likely characters for each position.
    """
    predictions = {}
    for i in range(1, 11):
        var = f"G{i}"
        if var in posteriors:
            probs = posteriors[var].values
            top_indices = np.argsort(probs)[-k:][::-1]
            predictions[i] = [(CHARSET[idx], probs[idx]) for idx in top_indices]
        else:
            predictions[i] = [("?", 0.0)]
    return predictions

def demo_wordlike_attack(partial="veg", target="vegetable", use_ngrams=False, k=3, verbose=True):
    """
    Quick demo for the character prediction attack.
    Builds the model, runs inference with a known prefix, and prints a summary.

    Returns: (posteriors, guess, top_predictions, observed, model)
    """
    # Build model
    model = build_wordlike_model(use_ngrams=use_ngrams)

    # Observed prefix -> indices
    observed = {f"G{i+1}": CHARSET_INDEX[c] for i, c in enumerate(partial) if c in CHARSET_INDEX}

    # Inference
    posteriors = run_wordlike_inference(model, observed)
    guess = suggest_wordlike_guess(posteriors, observed)
    top_predictions = get_top_k_predictions(posteriors, k=k)

    if verbose:
        print("Character Prediction Attack Demo")
        print(f"Target:  {target}")
        print(f"Prefix:  {partial}")
        print(f"Model:   {'trigram' if use_ngrams else 'bigram'}")
        print("-" * 50)
        print("Pos | Best | Top-k | Actual")
        print("-" * 50)
        for i in range(1, 11):
            node = f"G{i}"
            actual = target[i-1] if i <= len(target) else "-"
            if node in observed:
                best = CHARSET[observed[node]]
                topk_str = "observed"
            elif node in posteriors:
                probs = posteriors[node].values
                best_idx = int(np.argmax(probs))
                best = CHARSET[best_idx]
                preds = top_predictions.get(i, [])
                topk_str = ", ".join(f"{c}({p:.2f})" for c, p in preds)
            else:
                best, topk_str = "?", ""
            print(f"{i:>3} |  {best}   | {topk_str:20s} | {actual}")

        # Simple accuracy on known length
        n = min(len(guess), len(target))
        acc = sum(guess[i] == target[i] for i in range(n)) / n if n else 0.0
        print("-" * 50)
        print(f"Predicted: {guess}")
        print(f"Accuracy (first {n}): {acc:.2%}")

    return posteriors, guess, top_predictions, observed, model
