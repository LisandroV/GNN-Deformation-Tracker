# pip install torch-geometric matplotlib networkx


import torch
import numpy as np
from torch_geometric.data import Data
from data.velocity.train2 import level_set
from data.train2_data import finger_data

TIME_STEP=30
polygon = np.array(level_set)[TIME_STEP][:,:2]
finger = np.array(finger_data)


# FIND CLOSEST POINT TO FINGER
first_polygon = np.array(level_set)[0][:,:2]
finger_position = finger[0,:2]
min_dist = 1000
min_node_index = -1
for i in range(47):
    dist = np.linalg.norm(finger_position - first_polygon[i])
    if dist < min_dist:
        min_dist=dist
        min_node_index=i

# Example of a directed graph
forward = list(range(47))
backward = list(range(1,47)) + [0]
closest_index = 7 #train1
closest_index = 34 #train2

edge_index = torch.tensor([forward+backward + [47], backward+forward+[closest_index]], dtype=torch.long) # double connected graph

x = np.append(polygon, [finger[TIME_STEP,:2]], axis=0)

data = Data(x=x, edge_index=edge_index)



import networkx as nx
from torch_geometric.utils import to_networkx

# Convert to NetworkX graph (directed)
G = to_networkx(data, to_undirected=False)


import matplotlib.pyplot as plt

# Plotting the directed graph
plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G)  # Positions the nodes using a force-directed algorithm
pos = {i: (x[i, 0].item(), x[i, 1].item()) for i in range(48)}

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=500, arrows=True)

# Optionally, you can add labels to edges (e.g., edge weights)
edge_labels = {(i, j): f'' for i, j in G.edges()} # f'{i}->{j}'
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
