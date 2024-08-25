import torch
import torch.nn as nn
import numpy as np
from utils.ploting import plot_graph
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
    print("Aaaaaaa")
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


# PREDICT FUTURE STATES
model.eval()  # Set the model to evaluation mode

# Predict the next state
with torch.no_grad():
    predicted_next_state = model(graphs[40])

print("Predicted Future State:")
print(predicted_next_state)
print(predicted_next_state.size())