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

# Function to add padding to target matrices to match the output dimensions
def pad_graphs(input_x, input_edge_index, input_edge_attr, D = 82):
    extra_x = int((D/2) - input_x.shape[0])
    padded_x = F.pad(input_x, (0, 0, 0, extra_x), value=0)

    extra_edge_index = int(D - input_edge_index.shape[1])
    padded_edge_index = F.pad(input_edge_index, (0, extra_edge_index, 0, 0), value=0)

    extra_edge_attr = int(D - input_edge_attr.shape[0])
    padded_edge_attr = F.pad(input_edge_attr, (0, 0, 0, extra_edge_attr), value=0)

    return padded_x, padded_edge_index, padded_edge_attr


# Function to reshape the edge index representation into an adjacency matrix
# and unfold it into a vector
def reshape_index(input_index, D=82):
    D = D + 1
    adj = np.zeros((D,D))
    for pair in input_index.T:
        i = pair[0].numpy()
        j = pair[1].numpy()
        adj[i, j] = 1
    adj = adj[1:D, 1:D]
    
    linearized = adj[np.triu_indices(D-1, k=1)]
    return torch.tensor(linearized)

# Once we have predicted edges,
# need some way to unfold into a valid symmetric adjacency matrix
def unfold_index(linearized, D=82):
    adj = np.zeros((D, D))
    
    counter = 0
    for j in range(D-1):
        segment = D - j - 1
        adj[j, j+1:D] = linearized[counter:counter + segment]
        counter += segment
    symmetric = adj + adj.T - np.diag(np.diag(adj + adj.T))
    return symmetric



class DecodeNet(nn.Module):
    """
    Initialize an MLP for re-creating graph representations from latent space.
    Input (x) will be latent space embeddings (tensor 32).
    We need to be able to predict x, edge_index, edge_attr.
    The number of nodes and edges will also be dynamically predicted.
    """
    def __init__(
        self,
        num_features: int = 32, # This is the size of the latent space representations.
        D: int = 82, # TODO figure this out
        p: float = 0.0,
    ):
        super().__init__()
        self.D = D

        # Use an MLP to expand dimensions of latent space
        self.mlp = nn.Sequential(
            Linear(num_features, 128),
            nn.ReLU(),
            Linear(128, 128),
            nn.ReLU()
        )
        
        # one head for predicting features x
        x_dim = int(D * 3 / 2)
        self.x_fc = nn.Sequential(
            Linear(128, 128),
            nn.ReLU(),
            Linear(128, 64),
            nn.ReLU(),
            Linear(64, x_dim))

        # one head for predicting edge_index
        self.edge_index_fc = nn.Sequential(
            Linear(128, 64),
            nn.Sigmoid(),
            Linear(64, 128),
            nn.Sigmoid(),
            Linear(128, int(D*(D-1)/2)),
            nn.Sigmoid())

        # one head for predicting edge_attr
        self.edge_attr_fc = nn.Sequential(
            Linear(128, 128),
            nn.ReLU(),
            Linear(128, 64),
            nn.ReLU(),
            Linear(64, 1*D))


    def forward(self, data):
        
        # Apply the MLP
        h = self.mlp(data)
        
        # Predict x and reshape
        x = self.x_fc(h)
        x = x.view(-1, int(self.D / 2), 3)
        x = torch.squeeze(x, 0)

        # Predict edge_index and reshape
        edge_index = self.edge_index_fc(h)
        edge_index = torch.squeeze(edge_index, 0)

        # Predict edge_attr and reshape
        edge_attr = self.edge_attr_fc(h)
        edge_attr = edge_attr.view(-1, self.D, 1)
        edge_attr = torch.squeeze(edge_attr, 0)

        return x, edge_index, edge_attr 


def train_decoder_epoch(model, optimizer, train_loader):
    """Train the model for one epoch.
    Args:
        model: the model
        optimizer: the optimizer
        train_loader: contains all information needed for training, including graphs and their latent space representations.
    Returns:
        train_loss: the loss of the epoch
    """
    
    model.train()
    optimizer.zero_grad()
    loss_epoch = 0

    # what loss functions are used for each of our three targets?
    x_criterion = nn.MSELoss()
    edge_index_criterion = nn.BCELoss() 
    edge_attr_criterion = nn.MSELoss()
    
    # evaluate on the train nodes
    for data in train_loader:
        target_x = data.x
        target_edge_index = data.edge_index
        target_edge_attr = data.edge_attr
        
        batch_size = data.num_graphs

        # get the outputs
        x, edge_index, edge_attr  = model(data.hidden_values)

        # calculate loss for each of the three outputs
        x_loss = x_criterion(x, target_x)
        edge_index_loss = edge_index_criterion(edge_index.to(torch.float), target_edge_index.to(torch.float))  ## something strange w data types here
        edge_attr_loss = edge_attr_criterion(edge_attr, target_edge_attr)

        # calculate L1 regularization loss
        l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        
        # incentivize predicting edge loss
        loss = edge_index_loss + loss_lambda * x_loss + loss_lambda * edge_attr_loss
        
        loss.backward()
        loss_epoch += loss.detach().numpy() * batch_size

        optimizer.step()

    # calculate training loss for the epoch
    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch 
     
    
def test_decoder_epoch(model, train_loader):
    """Test the model for one epoch.
    Args:
        model: the model
        train_loader: contains all information needed for training, including graphs and their latent space representations.
    Returns:
        train_loss: the loss of the epoch
    """
    model.eval()  # set model to evaluation mode

    loss_epoch = 0
    
    # what loss functions are used for each of our three targets?
    x_criterion = nn.MSELoss()
    edge_index_criterion = nn.BCELoss() 
    edge_attr_criterion = nn.MSELoss()
    
    with torch.no_grad():  # disable gradient calculation

        for data in train_loader:
            target_x = data.x
            target_edge_index = data.edge_index
            target_edge_attr = data.edge_attr
            
            batch_size = data.num_graphs
    
            # get the outputs
            x, edge_index, edge_attr  = model(data.hidden_values)
    
            # calculate loss for each of the three outputs
            x_loss = x_criterion(x, target_x)
            edge_index_loss = edge_index_criterion(edge_index.to(torch.float), target_edge_index.to(torch.float))  ## something strange w data types here
            edge_attr_loss = edge_attr_criterion(edge_attr, target_edge_attr)
            
            # calculate L1 regularization loss
            l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            
            # aggregate the total loss
      #      loss = x_loss + edge_index_loss + edge_attr_loss + regularization_lambda * l1_norm

            # incentivize predicting edge loss
            loss = edge_index_loss + loss_lambda * x_loss + loss_lambda * edge_attr_loss
         
            loss_epoch += loss.detach().numpy() * batch_size

        # calculate test loss for the epoch
        loss_epoch = loss_epoch / len(test_loader.dataset)
                
    return loss_epoch

def train_decoder(model, train_loader, test_loader, optimizer, epochs=5):
    """Train the model.
    Args:
        model: the model
        loss_fn: the loss function
        train_loader: the training data loader
        test_loader: the testing data loader
        optimizer: the optimizer
        epochs: the number of epochs to train
    Returns:
        train_losses: the training losses
        test_losses: the testing losses
    """
    train_losses = []
    
    test_losses = []
    
    loop = tqdm.tqdm(range(1, epochs + 1))

    for epoch in loop:

        # train the model for one epoch
        train_loss_epoch = train_decoder_epoch(model, optimizer, train_loader)
        
        # test the model for one epoch        
        test_loss_epoch = test_decoder_epoch(model, train_loader)

        # put into our storage vectors
        train_losses.append(train_loss_epoch)
        test_losses.append(test_loss_epoch)
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(train_loss=train_loss_epoch, test_loss=test_loss_epoch)
    
    return train_losses, test_losses