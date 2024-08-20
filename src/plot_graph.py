import torch
import numpy as np
from torch_geometric.data import Data
from data.velocity.train2 import level_set
from data.train2_data import finger_data
from utils.ploting import plot_graph

TIME_STEP=30
polygon = np.array(level_set)[TIME_STEP][:,:2]
finger = np.array(finger_data)


# Example of a directed graph
forward = list(range(47))
backward = list(range(1,47)) + [0]
closest_index = 7 #train1
closest_index = 34 #train2

edge_index = torch.tensor([forward+backward + [47], backward+forward+[closest_index]], dtype=torch.long) # double connected graph

x = np.append(polygon, [finger[TIME_STEP,:2]], axis=0)

edge_attr = [[0,0]] * 94 # force on all nodes is zero
edge_attr = finger[TIME_STEP,2:4] # only consider the force from the finger

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


plot_graph(graph)




# CREATE GRAPH ------------------------------------------------------------------------------------------------------------------------------

# polygon = np.array(level_set)[TIME_STEP]
# x = np.append(polygon, [finger[TIME_STEP,:2]], axis=0)
# edge_attr = torch.tensor([[0.0, 0.0]]*47*2 + , dtype=torch.float)
