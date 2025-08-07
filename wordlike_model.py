# wordlike_model.py
"""
Character prediction model using n-gram language models.
Predicts password characters based on patterns learned from training data.
Character model: G_i ∈ CHARSET for each position i.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from collections import defaultdict, Counter
import string
from common_words import get_training_words

# ---------------------
# Character Set Definition
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
    
    Args:
        words: List of training words
        n: N-gram size (2 for bigrams, 3 for trigrams)
    
    Returns:
        dict: Mapping from prefix to character counts
    """
    counts = defaultdict(lambda: Counter())
    
    for word in words:
        # Filter to valid charset
        word = [c for c in word if c in CHARSET_INDEX]
        
        # Count n-grams
        for i in range(len(word) - n + 1):
            prefix = tuple(word[i:i + n - 1])
            next_char = word[i + n - 1]
            counts[prefix][next_char] += 1
    
    return counts


def normalize_counts(counts_dict):
    """
    Convert frequency counts to probability distributions.
    
    Args:
        counts_dict: Dictionary of character counts
    
    Returns:
        dict: Normalized probability distributions
    """
    probs = {}
    
    for prefix, counter in counts_dict.items():
        total = sum(counter.values())
        if total == 0:
            continue
        probs[prefix] = {c: count / total for c, count in counter.items()}
    
    return probs


# ---------------------
# Model Building
# ---------------------
def build_wordlike_model(words=None, use_ngrams=True, max_length=10):
    """
    Build character prediction PGM using language patterns.
    
    Network structure:
    - G_i ∈ CHARSET: Character at position i
    - Edges: G_{i-1} → G_i (bigram), optionally G_{i-2} → G_i (trigram)
    
    Args:
        words: Training words (if None, loads default dataset)
        use_ngrams: If True, use trigrams; if False, use bigrams
        max_length: Maximum password length to model
    
    Returns:
        DiscreteBayesianNetwork: Configured character prediction model
    """
    if words is None:
        words = get_training_words(source="hybrid", limit=5000)
    
    model = DiscreteBayesianNetwork()
    n = 3 if use_ngrams else 2
    
    # Build n-gram statistics
    ngram_counts = build_ngram_counts(words, n=n)
    ngram_probs = normalize_counts(ngram_counts)
    
    # Add character nodes
    for i in range(1, max_length + 1):
        model.add_node(f"G{i}")
    
    # Add edges for character dependencies
    for i in range(2, max_length + 1):
        model.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            model.add_edge(f"G{i-2}", f"G{i}")
    
    # CPD for first character (unigram distribution)
    unigram = Counter(w[0] for w in words if w and w[0] in CHARSET_INDEX)
    total_unigrams = sum(unigram.values())
    
    if total_unigrams > 0:
        prob_start = [unigram.get(c, 1e-6) / total_unigrams for c in CHARSET]
    else:
        prob_start = [1/len(CHARSET)] * len(CHARSET)
    
    cpd_g1 = TabularCPD("G1", len(CHARSET), [[p] for p in prob_start])
    model.add_cpds(cpd_g1)
    
    # CPD for second character (bigram from G1)
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
            # Trigram model
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
            # Bigram model
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
    
    Args:
        model: The character prediction PGM
        observed: Dict mapping node names to character indices
                 e.g., {"G1": CHARSET_INDEX['p'], "G2": CHARSET_INDEX['a']}
    
    Returns:
        dict: Posterior distributions for unobserved character nodes
    """
    infer = VariableElimination(model)
    posteriors = {}
    
    # Query for each unobserved character
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
    
    Args:
        posteriors: Dictionary of posterior distributions
        observed: Dictionary of observed characters
    
    Returns:
        str: Predicted password string
    """
    guess_chars = []
    
    for i in range(1, 11):
        var = f"G{i}"
        
        if observed and var in observed:
            # Use observed character
            char_idx = observed[var]
            guess_chars.append(CHARSET[char_idx])
        elif var in posteriors:
            # Use most probable character
            probs = posteriors[var].values
            best_idx = int(probs.argmax())
            guess_chars.append(CHARSET[best_idx])
        else:
            guess_chars.append("?")
    
    return ''.join(guess_chars)


def get_top_k_predictions(posteriors, k=3):
    """
    Get top-k most likely characters for each position.
    
    Args:
        posteriors: Dictionary of posterior distributions
        k: Number of top predictions to return
    
    Returns:
        dict: Position -> list of (character, probability) tuples
    """
    predictions = {}
    
    for i in range(1, 11):
        var = f"G{i}"
        if var in posteriors:
            probs = posteriors[var].values
            top_indices = np.argsort(probs)[-k:][::-1]
            
            predictions[i] = [
                (CHARSET[idx], probs[idx]) 
                for idx in top_indices
            ]
        else:
            predictions[i] = [("?", 0.0)]
    
    return predictions


def calculate_sequence_probability(model, password):
    """
    Calculate the probability of a complete password under the model.
    
    Args:
        model: The character prediction PGM
        password: String to evaluate
    
    Returns:
        float: Log probability of the sequence
    """
    if len(password) > 10:
        password = password[:10]
    
    # Convert to character indices
    observed = {}
    for i, char in enumerate(password):
        if char in CHARSET_INDEX:
            observed[f"G{i+1}"] = CHARSET_INDEX[char]
    
    # Calculate joint probability
    infer = VariableElimination(model)
    variables = [f"G{i+1}" for i in range(len(password))]
    
    try:
        result = infer.query(variables=variables)
        # Get probability of specific assignment
        prob = result.get_value(**observed)
        return np.log(prob) if prob > 0 else float('-inf')
    except:
        return float('-inf')


# Demo function
def demo_wordlike_attack(partial="veg", target="vegetable", verbose=True):
    """
    Demo character prediction attack with partial knowledge.
    
    Args:
        partial: Known prefix of password
        target: Complete target password
        verbose: Print detailed output
    
    Returns:
        tuple: (posteriors, guess, top_predictions)
    """
    if verbose:
        print(f"Character Prediction Attack Demo")
        print(f"Target: {target}")
        print(f"Known prefix: {partial}")
        print("=" * 50)
    
    # Build model
    model = build_wordlike_model(use_ngrams=True)
    
    # Set up observed characters
    observed = {}
    for i, char in enumerate(partial):
        observed[f"G{i+1}"] = CHARSET_INDEX[char]
    
    # Run inference
    posteriors = run_wordlike_inference(model, observed)
    
    # Get predictions
    guess = suggest_wordlike_guess(posteriors, observed)
    top_predictions = get_top_k_predictions(posteriors, k=3)
    
    if verbose:
        print("\nResults:")
        print("-" * 60)
        print("Pos | Best | Top 3 Predictions         | Actual")
        print("-" * 60)
        
        for i in range(10):
            actual = target[i] if i < len(target) else "-"
            predicted = guess[i]
            
            # Format top predictions
            if i + 1 in top_predictions:
                top_3 = ", ".join([
                    f"{c}({p:.2f})" 
                    for c, p in top_predictions[i + 1]
                ])
            else:
                top_3 = "observed"
            
            match = "✓" if predicted == actual else "✗"
            print(f" {i+1:2d} |  {predicted}   | {top_3:25s} | {actual}  {match}")
        
        print("-" * 60)
        print(f"Predicted: {guess}")
        print(f"Actual:    {target:<10}")
        print(f"Match:     {''.join(['✓' if i < len(guess) and i < len(target) and guess[i]==target[i] else '✗' for i in range(min(10, max(len(guess), len(target))))])}")
    
    return posteriors, guess, top_predictions