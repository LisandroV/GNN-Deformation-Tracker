import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
from data.velocity.validation import level_set, finger_data
from utils.ploting import plot_graph
from PropagationGNN import PropagationGNN

