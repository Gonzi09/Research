from __future__ import annotations

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


def compute_lambda1_and_partition_unweighted(G: nx.Graph):
    L = nx.normalized_laplacian_matrix(G, weight=None).astype(float)

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
    partition = {nodes[i]: 1 if fiedler[i] >= 0 else 0 for i in range(len(nodes))}

    return lambda1, partition


def in_box(lat, lon, north, south, east, west):
    return (south <= lat <= north) and (west <= lon <= east)


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

    east_north = 42.395
    east_south = 42.365
    east_east = -70.975
    east_west = -71.040

    main_north = 42.372
    main_south = 42.345
    main_east = -71.000
    main_west = -71.075

    nodes_keep = []

    for n, data in G.nodes(data=True):
        lat = data["y"]
        lon = data["x"]

        if (
            in_box(lat, lon, east_north, east_south, east_east, east_west)
            or
            in_box(lat, lon, main_north, main_south, main_east, main_west)
        ):
            nodes_keep.append(n)

    G_sub = G.subgraph(nodes_keep).copy()

    largest_cc_sub = max(nx.connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_cc_sub).copy()

    # Crear grafo simple SOLO para espectro
    G_simple = nx.Graph(G_sub)

    lambda1, partition = compute_lambda1_and_partition_unweighted(G_simple)

    print("nodes:", G_simple.number_of_nodes())
    print("edges:", G_simple.number_of_edges())
    print("lambda1:", lambda1)

    # Ahora coloreamos usando G_sub (MultiGraph)
    # edge_colors = []
    edge_widths = []

    for u, v, k in G_sub.edges(keys=True):
        if partition[u] != partition[v]:
            edge_colors.append("red")
            edge_widths.append(1.5)
        else:
            edge_colors.append("black")
            edge_widths.append(0.4)

    fig, ax = ox.plot_graph(
        G_sub,
        node_size=5,
        edge_color="white",
        edge_linewidth=edge_widths,
        bgcolor="black",
        show=False,
        close=False,
    )

    fig.savefig("east_boston_unweighted_spectral_cutg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main() 