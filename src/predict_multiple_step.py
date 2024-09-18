import torch
import torch.nn as nn
import numpy as np
from utils.ploting import plot_graph, plot_predicted_poligon
from data.DataLoader import get_graphs, get_expected_level_set
import data.velocity.train1 as train1_data
import data.velocity.validation as validation_data
import data.velocity.test as test_data
from torch_geometric.data import Data
from plot_data import plot_data

TIME_STEPS = 100

model = torch.load("src/trained_gnn_model.pth")
# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode


def predict(dataset):
    graphs = get_graphs(dataset)

    predicted_level_set = []

    with torch.no_grad():
        predicted_polygon = model(graphs[0])
        predicted_level_set.append(predicted_polygon[:-1, :2])


    velocities = predicted_polygon - graphs[0].x[:,:2]
    new_polygon = np.append(predicted_polygon, velocities, axis=1)[:47] # current polygon with velocity, and not predicted finger data


    for time_step in range (1, TIME_STEPS-1):
        #add finger data
        finger_node = np.append(dataset.finger_data[time_step,:2], dataset.finger_data[time_step,4:])
        x = torch.tensor(np.append(new_polygon, [finger_node], axis=0), dtype=torch.float) # add finger position to the graph

        new_graph = Data(
            x=x, # use predicted points
            edge_index=graphs[time_step].edge_index,
            edge_attr=graphs[time_step].edge_attr
        )

        with torch.no_grad():
            predicted_polygon = model(new_graph)
            predicted_level_set.append(predicted_polygon[:-1, :2])

        velocities = predicted_polygon - x[:,:2]
        new_polygon = np.append(predicted_polygon, velocities, axis=1)[:47] # current polygon with velocity, and not predicted finger data

    return np.array(predicted_level_set)

def get_overall_loss(dataset, prediction):
    loss_fn = nn.MSELoss()  # Example for regression

    expected_level_set = get_expected_level_set(dataset)

    loss = loss_fn(
        torch.tensor(prediction, dtype=torch.float),
        torch.tensor(expected_level_set, dtype=torch.float)
    )

    return loss


# PLOT PREDICTION ------------------------------------------
predicted_level_set = predict(train1_data)
plot_data(predicted_level_set, train1_data.finger_data)
loss = get_overall_loss(train1_data, predicted_level_set)
print("Train1 Loss: "+ str(loss))


predicted_level_set = predict(validation_data)
plot_data(predicted_level_set, validation_data.finger_data)
loss = get_overall_loss(validation_data, predicted_level_set)
print("Validation Loss: "+ str(loss))


predicted_level_set = predict(test_data)
plot_data(predicted_level_set, test_data.finger_data)
loss = get_overall_loss(test_data, predicted_level_set)
print("Test Loss: "+ str(loss))