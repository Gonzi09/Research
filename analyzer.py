import numpy as np
from .graph import Graph

class SpectralAnalyzer:
    """
    Performs spectral analysis on a graph.
    """
    
    def __init__(self, graph):
        """
        Initialize the analyzer with a graph.
        
        Parameters:
        -----------
        graph : Graph
            Instance of the Graph class
        """
        self.graph = graph
        self.eigenvalues = None
        self.eigenvectors = None
        
        self._compute_spectrum()
    
    def _compute_spectrum(self):
        """Compute eigenvalues and eigenvectors of the Laplacian."""
        L = self.graph.get_laplacian_matrix()
        eigvals, eigvecs = np.linalg.eig(L)
        
        # Sort by eigenvalue (smallest to largest)
        idx = np.argsort(eigvals)
        self.eigenvalues = eigvals[idx]
        self.eigenvectors = eigvecs[:, idx]
    
    def get_eigenvalue(self, index):
        """
        Get a specific eigenvalue.
        
        Parameters:
        -----------
        index : int
            Index of the eigenvalue
        
        Returns:
        --------
        float : The eigenvalue
        """
        return self.eigenvalues[index].real
    
    def get_eigenvector(self, index):
        """
        Get a specific eigenvector.
        
        Parameters:
        -----------
        index : int
            Index of the eigenvector
        
        Returns:
        --------
        numpy.ndarray : The eigenvector
        """
        return self.eigenvectors[:, index].real
    
    def get_all_eigenvalues(self):
        """Return all eigenvalues."""
        return self.eigenvalues
    
    def get_all_eigenvectors(self):
        """Return all eigenvectors."""
        return self.eigenvectors
    
    def print_eigenvalues(self):
        """Print all eigenvalues."""
        print(f"\n{'='*70}")
        print(f"EIGENVALUES - {self.graph.name}")
        print(f"{'='*70}")
        for i, ev in enumerate(self.eigenvalues):
            print(f"  λ_{i} = {ev:.6f}")
    
    def print_eigenvector(self, index):
        """
        Print detailed information about a specific eigenvector.
        
        Parameters:
        -----------
        index : int
            Index of the eigenvector to print
        """
        print(f"\n{'='*70}")
        print(f"EIGENVECTOR {index} - {self.graph.name}")
        print(f"{'='*70}")
        print(f"Eigenvalue: λ_{index} = {self.get_eigenvalue(index):.6f}")
        print(f"Eigenvector:")
        
        eigenvector = self.get_eigenvector(index)
        for j, val in enumerate(eigenvector):
            print(f"  Node {j}: {val:+.6f}")