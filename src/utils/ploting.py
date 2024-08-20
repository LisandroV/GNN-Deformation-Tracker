# pip install torch-geometric matplotlib networkx

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_graph(graph: Data):
    # Convert to NetworkX graph (directed)
    G = to_networkx(graph, to_undirected=False)


    import matplotlib.pyplot as plt

    # Plotting the directed graph
    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G)  # Positions the nodes using a force-directed algorithm
    pos = {i: (graph.x[i, 0].item(), graph.x[i, 1].item()) for i in range(48)}

    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=500, arrows=True)

    # Optionally, you can add labels to edges (e.g., edge weights)
    edge_labels = {(i, j): f'' for i, j in G.edges()} # f'{i}->{j}'
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()