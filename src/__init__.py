import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Example graph data
edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 0, 3, 2]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1], [2]], dtype=torch.float)

# Create a PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)

# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.tensor([0, 1, 0, 1], dtype=torch.long))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Inference
model.eval()
_, pred = model(data).max(dim=1)
print(f'Predicted classes: {pred}')