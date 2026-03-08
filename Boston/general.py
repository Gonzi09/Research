from __future__ import annotations

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


def compute_lambda1_and_partition(G: nx.MultiGraph, weight_attr="weight"):
    L = nx.normalized_laplacian_matrix(G, weight=weight_attr).astype(float)

    eigenvalues, eigenvectors = eigsh(L, k=5, which="SM")

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda1 = None
    fiedler = None

    for i, val in enumerate(eigenvalues):
        if float(val) > 1e-8:
            lambda1 = float(val)
            fiedler = eigenvectors[:, i]
            break

    if lambda1 is None:
        raise ValueError("No non-zero eigenvalue found")

    nodes = list(G.nodes())
    partition = {}

    for i, n in enumerate(nodes):
        partition[n] = 1 if fiedler[i] >= 0 else 0

    return lambda1, partition


def main():
    ox.settings.use_cache = True
    ox.settings.log_console = False

    G = ox.graph_from_place(
        "Boston, Massachusetts, USA",
        network_type="drive",
        simplify=True,
    )

    G = ox.convert.to_undirected(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # TODOS LOS PESOS = 1
    for _, _, _, data in G.edges(keys=True, data=True):
        data["weight"] = 1.0

    lambda1, partition = compute_lambda1_and_partition(G)

    print("lambda2 (connectividad algebraica):", lambda1)

    edge_colors = []
    edge_widths = []

    for u, v, k in G.edges(keys=True):
        if partition[u] != partition[v]:
            edge_colors.append("red")
            edge_widths.append(1.5)
        else:
            edge_colors.append("black")
            edge_widths.append(0.4)

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        node_color="black",
        edge_linewidth=edge_widths,
        bgcolor="white",
        show=False,
        close=False,
    )

    fig.savefig("boston_unweighted_spectral_cut.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()