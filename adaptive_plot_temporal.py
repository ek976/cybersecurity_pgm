# # adaptive_plot.py

import daft
import matplotlib.pyplot as plt

def plot_adaptive_attack_pgm(use_ngrams=True, save=False, title="Adaptive Hybrid Attack PGM"):
    pgm = daft.PGM()

    # Character guess nodes G1–G10
    for i in range(1, 11):
        pgm.add_node(f"G{i}", f"$G_{{{i}}}$", i, 2)

    # Timing nodes T1–T10
    for i in range(1, 11):
        pgm.add_node(f"T{i}", f"$T_{{{i}}}$", i, 1)
        pgm.add_edge(f"G{i}", f"T{i}")  # Timing dependency

    # Language model dependencies
    for i in range(2, 11):
        pgm.add_edge(f"G{i-1}", f"G{i}")
        if use_ngrams and i > 2:
            pgm.add_edge(f"G{i-2}", f"G{i}")

    # Render and show
    pgm.render()
    plt.title(title)
    if save:
        plt.savefig("adaptive_attack_pgm.png", bbox_inches="tight")
    plt.show()


# adaptive_plot_temporal.py

import daft
import matplotlib.pyplot as plt

def plot_temporal_adaptive_pgm(rounds=3, save=False, title="Temporal Adaptive Attack PGM"):
    pgm = daft.PGM()

    for r in range(1, rounds + 1):
        x = r * 2

        # G node (belief at round r)
        pgm.add_node(f"G1_r{r}", f"$G_1^{{({r})}}$", x, 2)

        # T node (timing feedback at round r)
        pgm.add_node(f"T1_r{r}", f"$T_1^{{({r})}}$", x, 1)

        # Edge: G → T (timing feedback)
        pgm.add_edge(f"G1_r{r}", f"T1_r{r}")

        # Edge: G (prev) → G (curr) (belief update)
        if r > 1:
            pgm.add_edge(f"G1_r{r-1}", f"G1_r{r}")

    pgm.render()
    plt.title(title)
    if save:
        plt.savefig("temporal_adaptive_attack_pgm.png", bbox_inches="tight")
    plt.show()
