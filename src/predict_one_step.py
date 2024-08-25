import torch
import torch.nn as nn
import numpy as np
from utils.ploting import plot_graph, plot_predicted_poligon
from data.DataLoader import create_data_loader, get_graphs

model = torch.load("src/trained_gnn_model.pth")
# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode

graphs = get_graphs()

# Predict the next state
with torch.no_grad():
    predicted_next_state = model(graphs[40])

print("Predicted Future State:")
print(predicted_next_state)
print(predicted_next_state.size())

plot_predicted_poligon(predicted_next_state[:-1])