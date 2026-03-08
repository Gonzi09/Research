from boston_visualizer import BostonVisualizer

viz = BostonVisualizer(output_dir="plots")
viz.load_graph()

# Save both plots
viz.plot_simple(save=True, filename="boston_with_nodes.png")
viz.plot_streets(save=True, filename="boston_streets_only.png")