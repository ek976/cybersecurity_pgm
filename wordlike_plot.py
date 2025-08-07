# wordlike_plot.py

import daft
import matplotlib.pyplot as plt

def plot_wordlike_pgm(use_ngrams=True, save=False, title="Wordlike Password PGM"):
    """
    Displays the structure of the PGM used for password modeling.
    """
    pgm = daft.PGM()

    # Layout G1–G10 horizontally
    for i in range(1, 11):
        pgm.add_node(f"G{i}", f"$G_{{{i}}}$", i, 1)

    # Add edges based on n-gram structure
    for i in range(2, 11):
        pgm.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            pgm.add_edge(f"G{i-2}", f"G{i}")

    pgm.render()
    plt.title(title)
    if save:
        plt.savefig("wordlike_pgm_structure.png", bbox_inches="tight")
    plt.show()
