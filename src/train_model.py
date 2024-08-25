import torch
import torch.nn as nn
import numpy as np
from utils.ploting import plot_graph, plot_predicted_poligon
from utils.PropagationGNN import PropagationGNN
from data.DataLoader import create_data_loader, get_graphs

model = PropagationGNN(in_dim_node=4, in_dim_edge=2, out_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

data_loader = create_data_loader(batch_size=20)
graphs = get_graphs()


# Training loop
epochs = 100
for epoch in range(epochs):
    print(f"training epoch [{epoch}]")
    for batch_graph in data_loader:
        model.train()
        optimizer.zero_grad()

        pred = model(batch_graph)

        # Compute loss
        loss = loss_fn(pred, batch_graph.y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


torch.save(model, "src/trained_gnn_model.pth")
