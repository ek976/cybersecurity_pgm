# hybrid_model.py

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from wordlike_model import CHARSET, build_ngram_counts, normalize_counts
import numpy as np

def build_hybrid_model(words, timing_cpd, use_ngrams=True):
    model = DiscreteBayesianNetwork()

    # --- CHARACTER MODEL ---
    n = 3 if use_ngrams else 2
    ngram_counts = build_ngram_counts(words, n=n)
    ngram_probs = normalize_counts(ngram_counts)

    for i in range(1, 11):
        model.add_node(f"G{i}")
        model.add_node(f"T{i}")
        model.add_edge(f"G{i}", f"T{i}")

    for i in range(2, 11):
        model.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            model.add_edge(f"G{i-2}", f"G{i}")

    # --- G1 prior ---
    from collections import Counter
    unigram = Counter(w[0] for w in words if w[0] in CHARSET)
    total = sum(unigram.values())
    prob_start = [unigram.get(c, 0) / total for c in CHARSET]
    cpd_g1 = TabularCPD("G1", len(CHARSET), [[p] for p in prob_start])
    model.add_cpds(cpd_g1)

    # --- G2–G10 ---
    for i in range(2, 11):
        if i == 2:
            evidence = [f"G{i-1}"]
        elif use_ngrams:
            evidence = [f"G{i-2}", f"G{i-1}"]
        else:
            evidence = [f"G{i-1}"]

        evidence_card = [len(CHARSET)] * len(evidence)
        values = []

        if len(evidence) == 1:
            for c1 in CHARSET:
                dist = ngram_probs.get((c1,), {})
                prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
                prob_sum = sum(prob_vector)
                values.append([p / prob_sum for p in prob_vector])
        else:
            for c1 in CHARSET:
                for c2 in CHARSET:
                    dist = ngram_probs.get((c1, c2), {})
                    prob_vector = [dist.get(c, 1e-6) for c in CHARSET]
                    prob_sum = sum(prob_vector)
                    values.append([p / prob_sum for p in prob_vector])

        values = list(map(list, zip(*values)))
        cpd = TabularCPD(f"G{i}", len(CHARSET), values, evidence=evidence, evidence_card=evidence_card)
        model.add_cpds(cpd)

    # --- T_i | G_i ---
    for i in range(1, 11):
        cpd_timing = TabularCPD(
            variable=f"T{i}",
            variable_card=3,
            values=timing_cpd,  # shape: (3, len(CHARSET))
            evidence=[f"G{i}"],
            evidence_card=[len(CHARSET)]
        )
        model.add_cpds(cpd_timing)

    model.check_model()
    return model


def run_hybrid_inference(model, timing_classes):
    """
    timing_classes: list of int (0, 1, 2) per T1–T10
    """
    evidence = {f"T{i+1}": timing_classes[i] for i in range(10)}
    infer = VariableElimination(model)
    posteriors = {}
    for i in range(1, 11):
        posteriors[f"G{i}"] = infer.query(variables=[f"G{i}"], evidence=evidence)
    return posteriors
