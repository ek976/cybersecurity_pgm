from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from collections import defaultdict, Counter
import string

# ---------------------
# Expanded Charset
# ---------------------
CHARSET = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*")
CHARSET_INDEX = {c: i for i, c in enumerate(CHARSET)}
CHARSET_REVERSE = {i: c for c, i in CHARSET_INDEX.items()}

COMMON_WORDS = ["password", "hunter", "Welcome123", "Admin@2021", "vegetable", "Monkey#", "Dragon42!"]

# ---------------------
# Bigram and Trigram Counts
# ---------------------
def build_ngram_counts(words, n=2):
    counts = defaultdict(lambda: Counter())
    for word in words:
        word = [c for c in word if c in CHARSET_INDEX]
        for i in range(len(word) - n + 1):
            prefix = tuple(word[i:i + n - 1])
            next_char = word[i + n - 1]
            counts[prefix][next_char] += 1
    return counts

def normalize_counts(counts_dict):
    probs = {}
    for prefix, counter in counts_dict.items():
        total = sum(counter.values())
        if total == 0:
            continue
        probs[prefix] = {b: count / total for b, count in counter.items()}
    return probs

# ---------------------
# Build Wordlike Model
# ---------------------
def build_wordlike_model(words=COMMON_WORDS, use_ngrams=True):
    model = DiscreteBayesianNetwork()
    n = 3 if use_ngrams else 2
    ngram_counts = build_ngram_counts(words, n=n)
    ngram_probs = normalize_counts(ngram_counts)

    # Add nodes
    for i in range(1, 11):
        model.add_node(f"G{i}")

    # Add edges
    for i in range(2, 11):
        model.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            model.add_edge(f"G{i-2}", f"G{i}")

    # Unigram start distribution
    unigram = Counter(w[0] for w in words if w[0] in CHARSET_INDEX)
    total_unigrams = sum(unigram.values())
    prob_start = [unigram.get(c, 0) / total_unigrams for c in CHARSET]
    cpd_g1 = TabularCPD("G1", len(CHARSET), [[p] for p in prob_start])
    model.add_cpds(cpd_g1)

    # G2 uses G1 only (bigram-style)
    values = []
    for prev_char in CHARSET:
        dist = ngram_probs.get((prev_char,), {})
        prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
        prob_sum = sum(prob_vector)
        prob_vector = [p / prob_sum for p in prob_vector]
        values.append(prob_vector)
    values = list(map(list, zip(*values)))
    cpd_g2 = TabularCPD("G2", len(CHARSET), values, evidence=["G1"], evidence_card=[len(CHARSET)])
    model.add_cpds(cpd_g2)

    # G3–G10 use bigrams or trigrams
    for i in range(3, 11):
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

        cpd = TabularCPD(curr, len(CHARSET), values, evidence=evidence, evidence_card=evidence_card)
        model.add_cpds(cpd)

    model.check_model()
    return model

# ---------------------
# Run Inference
# ---------------------
def run_word_inference(model, observed):
    infer = VariableElimination(model)
    posteriors = {}
    for i in range(1, 11):
        var = f"G{i}"
        if var in observed:
            continue
        result = infer.query(variables=[var], evidence=observed)
        posteriors[var] = result
    return posteriors

# ---------------------
# Suggest Best String
# ---------------------
def suggest_wordlike_guess(posteriors):
    guess_chars = []
    for i in range(1, 11):
        var = f"G{i}"
        if var in posteriors:
            probs = posteriors[var].values
            best_idx = int(probs.argmax())
            guess_chars.append(CHARSET[best_idx])
        else:
            guess_chars.append("?")
    return ''.join(guess_chars)
