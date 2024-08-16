import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class DeformationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeformationGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.fc(x)
        return x

model = DeformationGNN(input_dim=4, hidden_dim=32, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Train
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(x, edge_index, edge_weight)
    loss = loss_fn(out, true_displacements)
    loss.backward()
    optimizer.step()

predicted_displacements = model(x, edge_index, edge_weight)
new_positions = x[:, :3] + predicted_displacements
