# timing_plot.py
# Visualize using daft 

import daft
import matplotlib.pyplot as plt


def plot_pgm(save=False):
    pgm = daft.PGM()

    for i in range(10):
        pgm.add_node(f"G{i+1}", f"$G_{{{i+1}}}$", i, 1)
        pgm.add_node(f"T{i+1}", f"$T_{{{i+1}}}$", i, 0)
        pgm.add_edge(f"G{i+1}", f"T{i+1}")

    pgm.render()
    if save:
        plt.savefig("timing_attack_pgm.png", bbox_inches="tight")
    plt.show()

# timing_model.py 