import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import tqdm

# NetworkX is a Python package used to create, manipulate, and mine graphs
import networkx as nx

# further libraries for working with graphs
import torch_geometric
from torch_geometric.nn import GCNConv, pool
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# For visualization
import phate


def plot_metrics(train_metrics, test_metrics, xlabel, ylabel, title, fname, subdir):
    """ Function for visualizing loss functions over training epochs. """
    x = np.array(range(len(train_metrics))) + 1
    plt.plot(x, train_metrics, label="train")
    plt.plot(x, test_metrics, label="test")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fname = subdir + fname
    plt.savefig(fname) 
    plt.show()