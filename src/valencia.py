# LOAD AND PREPROCESS DATA #############################################################################
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# Example data
num_points = 100
num_features = 3  # x, y, z positions
num_forces = 3  # force in x, y, z directions

# Initial positions of points
x = torch.randn((num_points, num_features), dtype=torch.float)

# Forces applied to each point
forces = torch.randn((num_points, num_forces), dtype=torch.float)

# New positions of points after deformation (dummy data for example)
y = x + 0.1 * forces  # For simplicity, we just add some scaled force to the initial positions

# Combine initial positions and forces as node features
node_features = torch.cat([x, forces], dim=1)

# Define edges based on k-nearest neighbors
k = 5  # Number of nearest neighbors
edge_index = knn_graph(x, k)





# DEFINE GNN #############################################################################
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DeformationGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DeformationGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Define the model
model = DeformationGNN(in_channels=node_features.size(1), hidden_channels=64, out_channels=num_features)







# TRAIN MODEL #############################################################################
from torch_geometric.loader import DataLoader

# Create Data object
data = Data(x=node_features, edge_index=edge_index, y=y)

# Create DataLoader
loader = DataLoader([data], batch_size=1, shuffle=True)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(100):  # Number of epochs
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')






# EVALUATE MODEL #############################################################################
model.eval()
with torch.no_grad():
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y)
        print(f'Test Loss: {loss.item()}')
