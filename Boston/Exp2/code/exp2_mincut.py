from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GTFS_DIR = BASE_DIR / "data" / "MBTA_GTFS"


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

    bus_routes = routes[routes["route_type"] == 3][["route_id"]]
    trips = trips.merge(bus_routes, on="route_id", how="inner")

    shape_ids = trips["shape_id"].dropna().unique()
    shapes = shapes[shapes["shape_id"].isin(shape_ids)].copy()

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


def pick_north_south_nodes(G):
    nodes = list(G.nodes())

    north = max(nodes, key=lambda n: G.nodes[n]["y"])
    south = min(nodes, key=lambda n: G.nodes[n]["y"])

    return north, south


def build_flow_digraph_from_multigraph(G: nx.MultiGraph, counts, gamma: float = 0.2):
    # Capacidad = 1 + gamma * bus_count
    # (si c=0 => cap=1 para no dejar aristas con 0 capacidad)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())

    for u, v, k, data in G.edges(keys=True, data=True):
        c = counts.get((u, v, k), 0)
        cap = 1.0 + gamma * float(c)

        # Para "undirected", agregamos dos direcciones con misma capacidad
        # Si existen múltiples edges entre u-v, sumamos capacidades
        if H.has_edge(u, v):
            H[u][v]["capacity"] += cap
        else:
            H.add_edge(u, v, capacity=cap)

        if H.has_edge(v, u):
            H[v][u]["capacity"] += cap
        else:
            H.add_edge(v, u, capacity=cap)

    return H


def cut_edges_from_partition(H: nx.DiGraph, S: set, T: set):
    cut = []
    for u in S:
        for v in H.successors(u):
            if v in T:
                cut.append((u, v))
    return cut

def plot_cut(G, cut_uv_pairs, s, t, out_png):
    import matplotlib.pyplot as plt
    import osmnx as ox

    cut_set = set()
    for u, v in cut_uv_pairs:
        cut_set.add((u, v))
        cut_set.add((v, u))

    edge_colors = []
    edge_widths = []

    for u, v, k in G.edges(keys=True):
        if (u, v) in cut_set:
            edge_colors.append("red")
            edge_widths.append(1.8)
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

    # marcar s y t
    ax.scatter(G.nodes[s]["x"], G.nodes[s]["y"], c="blue", s=60)
    ax.scatter(G.nodes[t]["x"], G.nodes[t]["y"], c="green", s=60)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main():
    G = load_boston()

    print("Contando buses (GTFS)...")
    counts = compute_bus_counts(G, sample_n=8000)

    # Para escoger s,t usamos versión simple del grafo (sin multiedges)
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G.nodes())
    for u, v in G.edges():
        G_simple.add_edge(u, v)

    s, t = pick_north_south_nodes(G)
    print("s:", s, "t:", t)

    H = build_flow_digraph_from_multigraph(G, counts, gamma=0.2)

    # Min-cut exacto (equivale a max-flow)
    cut_value, (S, T) = nx.minimum_cut(H, s, t, capacity="capacity")
    print("min-cut value:", float(cut_value))
    print("|S|:", len(S), "|T|:", len(T))

    cut_uv = cut_edges_from_partition(H, S, T)
    print("cut edges (directed count):", len(cut_uv))

    plot_cut(G, cut_uv, s, t, "exp2_mincut_cut.png")



if __name__ == "__main__":
    main()
