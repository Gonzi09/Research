import matplotlib.pyplot as plt

class GraphComparator:
    """
    Compares multiple graphs and their spectral properties.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.graphs = []
        self.analyzers = []
        self.visualizers = []
    
    def add_graph(self, graph, analyzer, visualizer):
        """
        Add a graph with its analyzer and visualizer.
        
        Parameters:
        -----------
        graph : Graph
            Instance of Graph
        analyzer : SpectralAnalyzer
            Instance of SpectralAnalyzer
        visualizer : GraphVisualizer
            Instance of GraphVisualizer
        """
        self.graphs.append(graph)
        self.analyzers.append(analyzer)
        self.visualizers.append(visualizer)
    
    def compare_eigenvector(self, eigenvector_index, layout='spring', scale_factor=2000, figsize=None):
        """
        Compare the same eigenvector across multiple graphs.
        
        Parameters:
        -----------
        eigenvector_index : int
            Index of the eigenvector
        layout : str
            Layout type
        scale_factor : float
            Scale factor for node sizes
        figsize : tuple, optional
            Figure size
        """
        n_graphs = len(self.graphs)
        if n_graphs == 0:
            print("No graphs to compare")
            return
        
        if figsize is None:
            figsize = (5 * n_graphs, 5)
        
        fig, axes = plt.subplots(1, n_graphs, figsize=figsize)
        
        if n_graphs == 1:
            axes = [axes]
        
        for i, visualizer in enumerate(self.visualizers):
            visualizer.plot_eigenvector(eigenvector_index, layout=layout, scale_factor=scale_factor, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
    