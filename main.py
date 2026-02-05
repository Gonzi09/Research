from graph import Graph
from analyzer import SpectralAnalyzer
from visualizer import GraphVisualizer
from typing import List, Tuple


# Define graph
edges_disconnected: List[Tuple[int, int, int]] = [
    (0, 1, 1),
    (1, 2, 1),
    (1, 3, 1),
    (4, 5, 1),
    (4, 6, 1),
    (4, 7, 1),
    (5, 6, 1),
    (5, 7, 1),
    (6, 7, 1)
]


# Configuration
EIGENVECTORS_TO_COMPARE = [4,5]
SCALE_FACTOR = 2000
LAYOUT = 'spring'


# Create graph
graph: Graph = Graph(edges_disconnected, 8, "Disconnected")

# Create analyzer
analyzer: SpectralAnalyzer = SpectralAnalyzer(graph)

# Create visualizer
viz: GraphVisualizer = GraphVisualizer(graph, analyzer)


# Print eigenvalues
analyzer.print_eigenvalues()

# Print all eigenvectors
for i in range(graph.n_nodes):
    analyzer.print_eigenvector(i)


# Compare eigenvectors visually
viz.compare_eigenvectors(EIGENVECTORS_TO_COMPARE, layout=LAYOUT, scale_factor=SCALE_FACTOR)
