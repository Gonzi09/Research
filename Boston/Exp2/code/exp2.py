from __future__ import annotations

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import eigsh
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
GTFS_DIR = BASE_DIR / "data" / "MBTA_GTFS"


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


def compute_bus_counts(G: nx.MultiGraph):
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt", low_memory=False)
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")

    # Solo buses (route_type == 3)
    bus_routes = routes[routes["route_type"] == 3][["route_id"]]
    trips = trips.merge(bus_routes, on="route_id", how="inner")

    shape_ids = trips["shape_id"].dropna().unique()
    shapes = shapes[shapes["shape_id"].isin(shape_ids)].copy()

    # Muestreo para no colgar el sistema
    shapes = shapes.sample(n=min(8000, len(shapes)), random_state=0)

    lons = shapes["shape_pt_lon"].to_numpy(dtype=float)
    lats = shapes["shape_pt_lat"].to_numpy(dtype=float)

    edges = ox.distance.nearest_edges(G, X=lons, Y=lats)

    counts = {}

    for e in edges:
        if len(e) == 3:
            u, v, k = e
        else:
            u, v = e
            k = 0

        counts[(u, v, k)] = counts.get((u, v, k), 0) + 1

    return counts


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

    print("Calculando conteo de buses...")
    counts = compute_bus_counts(G)

    print("Asignando pesos...")
    for u, v, k, data in G.edges(keys=True, data=True):
        c = counts.get((u, v, k), 0)

        # Peso = intensidad de buses
        # usamos log para evitar que valores enormes dominen
        length = float(data.get("length", 1.0))
        data["weight"] = length * (1 + 0.05 * c)

    lambda1, partition = compute_lambda1_and_partition(G)

    print("lambda1 (MBTA):", lambda1)

    edge_colors = []
    edge_widths = []

    for u, v, k in G.edges(keys=True):
        if partition[u] != partition[v]:
            edge_colors.append("red")
            edge_widths.append(2.0)
        else:
            edge_colors.append("gray")
            edge_widths.append(0.6)

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        node_color="white",
        node_edgecolor="black",
        node_zorder=3,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor="white",
        show=False,
        close=False,
    )

    fig.savefig("exp2_mbta_cut.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
