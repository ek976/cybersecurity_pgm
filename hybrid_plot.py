# hybrid_plot.py

import daft
import matplotlib.pyplot as plt

def plot_hybrid_pgm(use_ngrams=True, save=False, title="Hybrid Password-Timing Model"):
    pgm = daft.PGM()

    # Layout: characters on top, timings below
    for i in range(1, 11):
        pgm.add_node(f"G{i}", f"$G_{{{i}}}$", i, 1)
        pgm.add_node(f"T{i}", f"$T_{{{i}}}$", i, 0)

    # Character dependencies
    for i in range(2, 11):
        pgm.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            pgm.add_edge(f"G{i-2}", f"G{i}")

    # Timing edges: G_i → T_i
    for i in range(1, 11):
        pgm.add_edge(f"G{i}", f"T{i}")

    # Render
    pgm.render()
    plt.title(title)
    if save:
        plt.savefig("hybrid_pgm.png", bbox_inches="tight")
    plt.show()
