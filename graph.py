import numpy as np

class Graph:
    """
    Represents a graph and its fundamental matrices.
    """
    
    def __init__(self, edges, n_nodes, name="Graph"):
        """
        Initialize the graph.
        
        Parameters:
        -----------
        edges : list of tuples
            List of edges (i, j, weight)
        n_nodes : int
            Number of nodes in the graph
        name : str
            Descriptive name for the graph
        """
        self.edges = edges
        self.n_nodes = n_nodes
        self.name = name
        
        self.W = None  # Adjacency matrix
        self.D = None  # Degree matrix
        self.L = None  # Laplacian matrix
        
        self._compute_matrices()
    
    def _compute_matrices(self):
        """Compute W (adjacency), D (degree), and L (Laplacian) matrices."""
        self.W = np.zeros((self.n_nodes, self.n_nodes))
        
        for i, j, w in self.edges:
            self.W[i, j] = w
            self.W[j, i] = w
        
        self.D = np.diag(np.sum(self.W, axis=1))
        self.L = self.D - self.W
    
    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        return self.W
    
    def get_degree_matrix(self):
        """Return the degree matrix."""
        return self.D
    
    def get_laplacian_matrix(self):
        """Return the Laplacian matrix."""
        return self.L