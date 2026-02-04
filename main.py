from spectral_graph_analysis import Graph, SpectralAnalyzer, GraphVisualizer, GraphComparator

# Define graphs
edges_disconnected = [
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

edges_connected = edges_disconnected + [(2, 4, 1), (3, 5, 1)]

# Create graphs
graph1 = Graph(edges_disconnected, 8, "Disconnected")
graph2 = Graph(edges_connected, 8, "Connected")

# Create analyzers
analyzer1 = SpectralAnalyzer(graph1)
analyzer2 = SpectralAnalyzer(graph2)

# Create visualizers
viz1 = GraphVisualizer(graph1, analyzer1)
viz2 = GraphVisualizer(graph2, analyzer2)

# Create comparator
comparator = GraphComparator()
comparator.add_graph(graph1, analyzer1, viz1)
comparator.add_graph(graph2, analyzer2, viz2)

# Print eigenvalues comparison
comparator.print_eigenvalues_comparison()

# Compare Fiedler vector (eigenvector 1)
comparator.compare_eigenvector(1, scale_factor=2000)

# View multiple eigenvectors from one graph
viz1.compare_eigenvectors([0, 1, 2], scale_factor=2000)