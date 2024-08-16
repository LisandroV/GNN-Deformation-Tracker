import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class PropagationGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(PropagationGNN, self).__init__()
        # Adjust GCNConv to account for node and edge features
        self.conv1 = pyg_nn.GCNConv(node_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)

        # Linear layer to combine edge features after message passing
        self.edge_fc = nn.Linear(edge_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        # First GCN layer (only on node features)
        x = self.conv1(x, edge_index).relu()

        # Combine node features with transformed edge features
        edge_features = self.edge_fc(edge_attr).relu() # FIXME: meter x, como en la definición

        # Message passing considering updated edge features
        x = self.conv2(x, edge_index).relu()

        # Final output
        out = self.fc(x)
        return out


# class PropagationGNN(nn.Module):
#     def __init__(self, node_dim, hidden_dim, output_dim):
#         super(PropagationGNN, self).__init__()
#         # Define the GCN layers with the correct input dimensions
#         self.conv1 = pyg_nn.GCNConv(node_dim, hidden_dim)
#         self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)

#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, edge_index, edge_attr=None):
#         # If edge_attr is used to influence the model, it must be correctly expanded or processed
#         # For now, we'll skip edge_attr to focus on node processing
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index).relu()

#         # Final linear transformation
#         out = self.fc(x)
#         return out



# PREPARE THE DATA
from torch_geometric.data import Data

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
edge_attr = torch.tensor([0.7, 0.3], dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# True future states (this would be known in a training context)
true_next_states = torch.tensor([
    [1.2, 2.1, 0.6, 0.3],
    [2.1, 3.1, 0.2, 0.15]
], dtype=torch.float)




#TRAIN THE MODEL
model = PropagationGNN(node_dim=4, edge_dim=1, hidden_dim=64, output_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred = model(data.x, data.edge_index, data.edge_attr)

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
    predicted_next_state = model(new_data.x, new_data.edge_index, new_data.edge_attr)

print("Predicted Future State:")
print(predicted_next_state)
