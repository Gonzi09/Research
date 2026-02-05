import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from graph import Graph

class SpectralAnalyzer:
    """
    Performs spectral analysis on a graph.
    """
    
    def __init__(self, graph: Graph) -> None:
        """
        Initialize the analyzer with a graph.
        
        Parameters:
        -----------
        graph : Graph
            Instance of the Graph class
        """
        self.graph: Graph = graph
        self.eigenvalues: NDArray[np.complex128] = None
        self.eigenvectors: NDArray[np.complex128] = None
        
        self._compute_spectrum()
    
    def _compute_spectrum(self) -> None:
        """Compute eigenvalues and eigenvectors of the Laplacian."""
        L = self.graph.get_laplacian_matrix()
        eigvals, eigvecs = np.linalg.eig(L)
        
        # Sort by eigenvalue (smallest to largest)
        idx = np.argsort(eigvals)
        self.eigenvalues = eigvals[idx]
        self.eigenvectors = eigvecs[:, idx]
    
    def get_eigenvalue(self, index: int) -> float:
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
    
    def get_eigenvector(self, index: int) -> NDArray[np.float64]:
        """
        Get a specific eigenvector.
        
        Parameters:
        -----------
        index : int
            Index of the eigenvector
        
        Returns:
        --------
        NDArray[np.float64] : The eigenvector
        """
        return self.eigenvectors[:, index].real
    
    def get_all_eigenvalues(self) -> NDArray[np.complex128]:
        """Return all eigenvalues."""
        return self.eigenvalues
    
    def get_all_eigenvectors(self) -> NDArray[np.complex128]:
        """Return all eigenvectors."""
        return self.eigenvectors
    
    def print_eigenvalues(self) -> None:
        """Print all eigenvalues as a list."""
        formatted = ', '.join([f'{val:.2f}' for val in self.eigenvalues.real])
        print(f"\nEigenvalues ({self.graph.name}):")
        print(f"[{formatted}]")
    
    def print_eigenvector(self, index: int) -> None:
        """
        Print a specific eigenvector as a list.
        
        Parameters:
        -----------
        index : int
            Index of the eigenvector to print
        """
        eigenvector = self.get_eigenvector(index)
        formatted = ', '.join([f'{val:.2f}' for val in eigenvector])
        print(f"\nEigenvector {index} (λ={self.get_eigenvalue(index):.2f}):")
        print(f"[{formatted}]")
    
    def print_all_eigenvectors(self) -> None:
        """Print all eigenvectors as a matrix."""
        print(f"\nAll Eigenvectors ({self.graph.name}):")
        print(self.eigenvectors.real)