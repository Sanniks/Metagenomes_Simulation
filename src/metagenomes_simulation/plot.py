import matplotlib.pyplot as plt
import igraph as ig


def plot_network(G):
    layout = G.layout("auto")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    visual_style = {
        "vertex_size": 6,
        "vertex_color": "lightblue",
        "edge_width": 1,
        "layout": layout,
    }
    ig.plot(G, target=ax, **visual_style)
