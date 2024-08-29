import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np

TIME_STEPS = 100

def _get_closest_point(finger, velocity_polygon):
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


def _get_expected_polygon(level_set, finger_data, time_step):
    """Returns only the node positions at a given time_step"""
    expected_polygon = np.append(level_set[time_step][:,:2], [finger_data[time_step][:2]], axis=0)

    return torch.tensor(expected_polygon, dtype=torch.float)


def _create_graph(level_set, finger_data, time_step):
    polygon = level_set[time_step]
    finger = finger_data

    # Example of a directed graph
    forward = list(range(47))
    backward = list(range(1,47)) + [0]

    closest_index = 7 # train1
    closest_index = 34 # train2
    closest_index = 11 # validation
    closest_index = _get_closest_point(finger_data[time_step], level_set[time_step])

    edge_index = torch.tensor([forward+backward + [closest_index, 47] , backward+forward+[47 ,closest_index]], dtype=torch.long) # double connected graph

    finger_node = np.append(finger[time_step,:2], finger[time_step,4:]) # not consider finger force
    x = torch.tensor(np.append(polygon, [finger_node], axis=0), dtype=torch.float) # add finger position to the graph

    zero_force = np.array([[0,0]] * 95) # force on all nodes is zero
    edge_attr = np.append(zero_force, [finger[time_step,2:4]], axis=0) # only consider the force from the finger
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    expected = _get_expected_polygon(level_set, finger_data, time_step+1)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=expected)

    return graph


def get_graphs(dataset):
    graphs = []
    for time_step in range(TIME_STEPS-1): # the last graph doesn't have an expected value
        graph = _create_graph(dataset.level_set, dataset.finger_data, time_step)
        graphs.append(graph)

    return graphs

def get_expected_level_set(dataset):
    return dataset.level_set[1:, :, :2]

def create_data_loader(datasets, batch_size=4):
    """
    Factory for Torch Data Loader.

    Input:
        dataset: list of datasets
            [{
                level_set: np.array() of shape(100,47,2),
                finger_data: np.array() of shape(100,2),
            }]

        batch_size: Data loader batch size.
    """
    all_graphs = []
    for dataset in datasets:
        graphs = get_graphs(dataset)
        all_graphs += graphs

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    return loader

