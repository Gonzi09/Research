import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    """
    Visualizes graphs and their spectral properties.
    """
    
    def __init__(self, graph, analyzer):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        graph : Graph
            Instance of Graph
        analyzer : SpectralAnalyzer
            Instance of SpectralAnalyzer
        """
        self.graph = graph
        self.analyzer = analyzer
    
    def plot_eigenvector(self, eigenvector_index, layout='spring', 
                        scale_factor=2000, min_size=100, 
                        figsize=(10, 8), ax=None, show_title=True):
        """
        Visualize the graph with node sizes and colors based on an eigenvector.
        
        Parameters:
        -----------
        eigenvector_index : int
            Index of the eigenvector to visualize
        layout : str
            Layout type ('spring', 'circular', 'kamada_kawai', 'spectral')
        scale_factor : float
            Factor to scale node sizes
        min_size : float
            Minimum node size
        figsize : tuple
            Figure size
        ax : matplotlib axis, optional
            Axis to draw on
        show_title : bool
            Whether to show title
        """
        eigenvector = self.analyzer.get_eigenvector(eigenvector_index)
        eigenvalue = self.analyzer.get_eigenvalue(eigenvector_index)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(self.graph.get_adjacency_matrix())
        
        # Calculate node sizes (proportional to absolute value)
        node_sizes = np.abs(eigenvector) * scale_factor
        node_sizes = np.maximum(node_sizes, min_size)
        
        # Calculate node colors (blue=positive, red=negative)
        node_colors = ['blue' if val >= 0 else 'red' for val in eigenvector]
        
        # Choose layout
        layouts = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout
        }
        pos = layouts.get(layout, nx.spring_layout)(G, seed=42)
        
        # Create figure if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Draw graph
        nx.draw(G, pos,
                with_labels=True,
                node_size=node_sizes,
                node_color=node_colors,
                font_size=12,
                font_weight='bold',
                edge_color='gray',
                width=2,
                edgecolors='black',
                linewidths=1.5,
                ax=ax)
        
        # Add edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        # Add title
        if show_title:
            title = f'{self.graph.name} - Eigenvector {eigenvector_index} (λ={eigenvalue:.3f})\nBlue=Positive, Red=Negative'
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.axis('off')
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def compare_eigenvectors(self, eigenvector_indices, layout='spring', 
                            scale_factor=2000, figsize=(15, 5)):
        """
        Compare multiple eigenvectors side by side.
        
        Parameters:
        -----------
        eigenvector_indices : list of int
            List of eigenvector indices
        layout : str
            Layout type
        scale_factor : float
            Scale factor for node sizes
        figsize : tuple
            Figure size
        """
        n_plots = len(eigenvector_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, eig_idx in enumerate(eigenvector_indices):
            eigenvalue = self.analyzer.get_eigenvalue(eig_idx)
            axes[i].set_title(f'Eigenvector {eig_idx}\n(λ={eigenvalue:.3f})', 
                            fontsize=12, fontweight='bold')
            self.plot_eigenvector(eig_idx, layout=layout, 
                                scale_factor=scale_factor, 
                                ax=axes[i], show_title=False)
        
        plt.tight_layout()
        plt.show()