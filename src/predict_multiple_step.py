import torch
import torch.nn as nn
import numpy as np
from utils.ploting import plot_graph, plot_predicted_poligon
from data.DataLoader import create_data_loader, get_graphs
from data.velocity.train1 import finger_data
from torch_geometric.data import Data

TIME_STEPS = 100

model = torch.load("src/trained_gnn_model.pth")
# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode

graphs = get_graphs()

# Predict the next state

with torch.no_grad():
    predicted_polygon = model(graphs[0])


velocities = predicted_polygon - graphs[0].x[:,:2]
new_polygon = np.append(predicted_polygon, velocities, axis=1)[:47] # current polygon with velocity, and not predicted finger data


for time_step in range (1, TIME_STEPS-1):
    #add finger data
    finger_node = np.append(finger_data[time_step,:2], finger_data[time_step,4:])
    x = torch.tensor(np.append(new_polygon, [finger_node], axis=0), dtype=torch.float) # add finger position to the graph

    new_graph = Data(
        x=x, # use predicted points
        edge_index=graphs[time_step].edge_index,
        edge_attr=graphs[time_step].edge_attr
    )

    with torch.no_grad():
        predicted_polygon = model(new_graph)

    velocities = predicted_polygon - x[:,:2]
    new_polygon = np.append(predicted_polygon, velocities, axis=1)[:47] # current polygon with velocity, and not predicted finger data

    print(new_polygon)
