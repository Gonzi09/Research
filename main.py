from graph import Graph
from analyzer import SpectralAnalyzer
from visualizer import GraphVisualizer
from typing import List, Tuple


edges_disconnected: List[Tuple[int, int, float]] = [
    (0, 1, 1),
    (1, 2, 1),
]



# Configuration
EIGENVECTORS_TO_COMPARE = [0,1]
SCALE_FACTOR = 2000
LAYOUT = 'spring'


# Graph
graph: Graph = Graph(edges_disconnected, 3, "Disconnected")


# Analyzer
analyzer: SpectralAnalyzer = SpectralAnalyzer(graph)

# Visualizer
viz: GraphVisualizer = GraphVisualizer(graph, analyzer)


# Print eigenvalues
analyzer.print_eigenvalues()

# Print all eigenvectors
for i in range(graph.n_nodes):
    analyzer.print_eigenvector(i)


# Compare eigenvectors
viz.compare_eigenvectors(EIGENVECTORS_TO_COMPARE, layout=LAYOUT, scale_factor=SCALE_FACTOR)
