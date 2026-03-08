import osmnx as ox
from pathlib import Path

class BostonVisualizer:
    """
    Visualizes the Boston street network.
    """
    
    def __init__(self, output_dir: str = "plots"):
        ox.settings.log_console = True
        ox.settings.use_cache = True
        self.G = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_graph(self, place: str = "Boston, Massachusetts, USA", network_type: str = "drive"):
        """
        Load street network from OpenStreetMap.
        """
        self.G = ox.graph_from_place(place, network_type=network_type)
        return self.G
    
    def plot_simple(self, save: bool = False, filename: str = "boston_simple.png"):
        """
        Simple plot with default settings (shows nodes and edges).
        """
        if self.G is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")
        
        if save:
            filepath = self.output_dir / filename
            ox.plot_graph(self.G, save=True, filepath=str(filepath), show=False)
            print(f"Simple plot saved to: {filepath}")
        else:
            ox.plot_graph(self.G)
    
    def plot_streets(self, save: bool = False, filename: str = "boston_streets.png"):
        """
        Plot just the streets (no nodes, black and white).
        """
        if self.G is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")
        
        if save:
            filepath = self.output_dir / filename
            ox.plot_graph(
                self.G,
                node_size=0,
                edge_color="black",
                edge_linewidth=0.7,
                bgcolor="white",
                save=True,
                filepath=str(filepath),
                show=False
            )
            print(f"Streets plot saved to: {filepath}")
        else:
            ox.plot_graph(
                self.G,
                node_size=0,
                edge_color="black",
                edge_linewidth=0.7,
                bgcolor="white"
            )