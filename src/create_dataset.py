import torch
import numpy as np
from torch_geometric.data import Data
from data.velocity.train1 import level_set
from data.train1_data import finger_data

polygon = np.array(level_set)[30][:,:2]
finger = np.array(finger_data)


# FIND CLOSEST POINT TO FINGER
first_polygon = polygon = np.array(level_set)[0][:,:2]
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
edge_index = torch.tensor([forward+backward, backward+forward], dtype=torch.long) # double connected graph

x = np.append(polygon,[finger_position])  # Random node features (4 nodes, 2 features per node)

data = Data(x=x, edge_index=edge_index)