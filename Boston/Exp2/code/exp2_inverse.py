from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse.linalg import eigsh


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


def load_boston() -> nx.MultiGraph:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    G = ox.graph_from_place(
        "Boston, Massachusetts, USA",
        network_type="drive",
        simplify=True,
    )
    G = ox.convert.to_undirected(G)

    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def compute_bus_counts(G: nx.MultiGraph, sample_n: int = 8000):
    routes = pd.read_csv(GTFS_DIR / "routes.txt")
    trips = pd.read_csv(GTFS_DIR / "trips.txt", low_memory=False)
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")

    # Solo buses (GTFS route_type == 3)
    bus_routes = routes[routes["route_type"] == 3][["route_id"]]
    trips = trips.merge(bus_routes, on="route_id", how="inner")

    shape_ids = trips["shape_id"].dropna().unique()
    shapes = shapes[shapes["shape_id"].isin(shape_ids)].copy()

    # Muestreo para que no se cuelgue
    shapes = shapes.sample(n=min(sample_n, len(shapes)), random_state=0)

    lons = shapes["shape_pt_lon"].to_numpy(dtype=float)
    lats = shapes["shape_pt_lat"].to_numpy(dtype=float)

    edges = ox.distance.nearest_edges(G, X=lons, Y=lats)

    counts = {}
    for e in edges:
        if isinstance(e, tuple) and len(e) == 3:
            u, v, k = e
        else:
            u, v = e
            k = 0
        counts[(int(u), int(v), int(k))] = counts.get((int(u), int(v), int(k)), 0) + 1

    return counts


def plot_cut_streets(G: nx.MultiGraph, partition, out_png: str):
    edge_colors = []
    edge_widths = []

    for u, v, k in G.edges(keys=True):
        if partition[u] != partition[v]:
            edge_colors.append("red")
            edge_widths.append(1.6)
        else:
            edge_colors.append("black")
            edge_widths.append(0.5)

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor="white",
        show=False,
        close=False,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    G = load_boston()

    print("Contando buses (GTFS)...")
    counts = compute_bus_counts(G, sample_n=8000)

    # PESO INVERSO (estable):
    # w = length / (1 + gamma * bus_count)
    # Más buses => denominador mayor => peso menor => más barato cortar.
    gamma = 0.05

    for u, v, k, data in G.edges(keys=True, data=True):
        c = counts.get((u, v, k), 0)
        length = float(data.get("length", 1.0))
        data["bus_count"] = c
        data["weight"] = length / (1.0 + gamma * c)

    lambda1, partition = compute_lambda1_and_partition(G)
    print("lambda1 (inverse bus weights):", lambda1)

    plot_cut_streets(G, partition, "exp2_inverse_cut.png")


if __name__ == "__main__":
    main()
