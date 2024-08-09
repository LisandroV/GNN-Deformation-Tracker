# Graph Neural Network to predict the deformation of elastic objects

 Using Graph Neural Network (GNN) as a data-driven approach to predict the deformation of an object. Instead of relying on physical material properties, the model can learn to predict the deformation directly from data, using the observed relationships between applied forces and resulting displacements.

Alternative Approach Without Explicit Material Properties
In this approach, the GNN model will learn the relationship between applied forces and point displacements purely from data, without explicitly incorporating material-specific parameters like Young's Modulus or Poisson's ratio.

1. Problem Setup:

* Input:
    * Point cloud representing the object's initial positions.
    * Applied forces at specific points or across the entire object.
* Output:
    * New positions of the points in the cloud after the force is applied.

2. Graph Neural Network (GNN) Model:
    * The model should be designed to learn the mapping from the current positions and applied forces to the predicted displacements.

    ```
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
    ````

 * Input: The node features x could include the current positions and forces applied to the nodes.
Output: The output would be the predicted displacements for each node in the graph.

3. Training the Model:
* Prepare the dataset consisting of initial positions, applied forces, and corresponding displacements.

* Train the GNN to minimize the error between the predicted displacements and the ground truth displacements.

    ```
    model = DeformationGNN(input_dim=6, hidden_dim=64, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)
        loss = loss_fn(out, true_displacements)
        loss.backward()
        optimizer.step()
    ```

 * Input Dim: In this case, the input could be 6-dimensional, with 3 dimensions for the initial position and 3 for the applied forces.
* Output Dim: The output is 3-dimensional, representing the displacement in the x, y, and z directions.

4. Prediction:

* Once trained, the model can predict the new positions of the points after deformation.

```
predicted_displacements = model(x, edge_index, edge_weight)
new_positions = x[:, :3] + predicted_displacements
```

## Advantages of This Approach:
* No Need for Material Properties: Since the model learns directly from the data, there is no need to know or input material-specific properties like Young's Modulus or Poisson's ratio.

* Data-Driven: This method can adapt to complex and non-linear deformations that might be difficult to model explicitly with traditional physics-based approaches.

## Considerations:
* Data Quality: The success of this approach heavily depends on the quality and quantity of the training data. The model needs to see a wide variety of deformations to generalize well.

* Physics Consistency: While this approach is purely data-driven, ensuring that the model outputs physically plausible results can be challenging without the guidance of explicit physical laws.

This alternative approach aligns more closely with data-driven machine learning techniques, which may be beneficial in scenarios where the material properties are unknown or difficult to measure.

# Setup GNN-Deformation-Tracker
Predict deformation on objects using  Graph Neural Networks (GNN)

## Requirements
```
pyenv
pyenv-virtualenv
```

## Setup
1. Clone repo
2. Go to the repo and setup python virtual env:
    ```
    pyenv install 3.12
    pyenv virtualenv 3.12 gnn_env
    pyenv local gnn_env # link repo directory to its own virtualenv
    ```

3. Install libraries
    ```
    pip install -r requirements.txt
    ```

## Train model

## Make deformation predictions
