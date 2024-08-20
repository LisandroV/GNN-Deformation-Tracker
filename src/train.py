
import torch
import torch.nn as nn
from torch_geometric.data import Data
from PropagationGNN import PropagationGNN

# PREPARE THE DATA

# Example node features: [position_x, position_y, velocity_x, velocity_y]
node_features = torch.tensor([
    [1.0, 2.0, 0.5, 0.2],
    [2.0, 3.0, 0.1, 0.1]
], dtype=torch.float)

# Example edge index (connectivity between nodes)
edge_index = torch.tensor([
    [0, 1],
    [1, 0]
], dtype=torch.long)

# Example edge attributes (e.g., force magnitude between nodes)
edge_attr = torch.tensor([[0.7, 0.5], [0.3, 0.3]], dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# True future states (this would be known in a training context)
true_next_states = torch.tensor([
    [1.2, 2.1, 0.6, 0.3],
    [2.1, 3.1, 0.2, 0.15]
], dtype=torch.float)




#TRAIN THE MODEL
model = PropagationGNN(in_dim_node=4, in_dim_edge=2, out_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred = model(data)

    # Compute loss
    loss = loss_fn(pred, true_next_states)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')



# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode

# Suppose you have new initial conditions
new_node_features = torch.tensor([
    [1.5, 2.5, 0.4, 0.2],
    [2.5, 3.5, 0.2, 0.2]
], dtype=torch.float)

# Using the same edge_index and edge_attr as before
new_data = Data(x=new_node_features, edge_index=edge_index, edge_attr=edge_attr)

# Predict the next state
with torch.no_grad():
    predicted_next_state = model(new_data)

print("Predicted Future State:")
print(predicted_next_state)
