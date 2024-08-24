import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from data.velocity.validation import level_set, finger_data
from utils.ploting import plot_graph
from utils.PropagationGNN import PropagationGNN

def get_closest_point(finger, velocity_polygon):
    """Returns the index of the closest node to the finger position."""
    polygon = velocity_polygon[:,:2]
    finger_position = finger[:2]
    min_dist = 1000
    min_node_index = -1
    for i in range(47):
        dist = np.linalg.norm(finger_position - polygon[i])
        if dist < min_dist:
            min_dist=dist
            min_node_index=i

    return min_node_index


def get_expected_polygon(level_set, finger_data, time_step):
    """Returns only the node positions at a given time_step"""
    expected_polygon = np.append(level_set[time_step][:,:2], [finger_data[time_step][:2]], axis=0)

    return torch.tensor(expected_polygon, dtype=torch.float)


def create_graph(level_set, finger_data, time_step):
    polygon = level_set[time_step]
    finger = finger_data

    # Example of a directed graph
    forward = list(range(47))
    backward = list(range(1,47)) + [0]

    closest_index = 7 # train1
    closest_index = 34 # train2
    closest_index = 11 # validation
    closest_index = get_closest_point(finger_data[time_step], level_set[time_step])

    edge_index = torch.tensor([forward+backward + [closest_index, 47] , backward+forward+[47 ,closest_index]], dtype=torch.long) # double connected graph

    finger_node = np.append(finger[time_step,:2], finger[time_step,4:]) # not consider finger force
    x = torch.tensor(np.append(polygon, [finger_node], axis=0), dtype=torch.float) # add finger position to the graph

    zero_force = np.array([[0,0]] * 95) # force on all nodes is zero
    edge_attr = np.append(zero_force, [finger[time_step,2:4]], axis=0) # only consider the force from the finger
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    expected = get_expected_polygon(level_set, finger_data, time_step+1)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=expected)

    return graph


graph = create_graph(level_set, finger_data, 25)


# TRAIN ------------------------------------------------------------------------------------------------------------------------------

model = PropagationGNN(in_dim_node=4, in_dim_edge=2, out_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred = model(graph)

    # Compute loss
    loss = loss_fn(pred, graph.y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode

# Predict the next state
with torch.no_grad():
    predicted_next_state = model(graph)

print("Predicted Future State:")
print(predicted_next_state)
print(predicted_next_state.size())