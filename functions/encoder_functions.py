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

# Graph scattering functionality
from LEGS_module import *

# Home-grown functions
from utils import *

global smoothness_lambda 
smoothness_lambda = 1e-2 # for calculating composite loss


class GCN(nn.Module):
    """
    Initialize a graph convolutional network
    """
    def __init__(
        self,
        num_features: int = 3, # TODO: replace this with the proper dimension
        num_classes: int = 1,
        p: float = 0.0,
    ):
        super().__init__()

        # GCNConv takes in two arguments: in_channels and out_channels
        # for now, we choose an arbitrary number of channels
        self.conv1 = GCNConv(num_features, 32)

        # GRASSY would replace the GCNConv with scattering function

        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 16) # increase dimensionality here; can go all the way to 128

        self.linear1 = Linear(16, 16)
        self.dropout = nn.Dropout(p)
        self.embedding_layer = Linear(16, 16)
        
        self.classifier = Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x, edge_index)
        h = h.relu() # TODO: maybe pick a different non-linearity?
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()

        # sum pool
        h = pool.global_add_pool(h, batch)

        embedding = self.embedding_layer(h)
        
        h = embedding.relu()
        
        # last layer is a classifier
        output = self.classifier(h)

        return embedding, output 



class ScatterNet(nn.Module):
    """
    We improve on the graph convolutional network approach by using the graph scattering transform.
    """
    def __init__(
        self,
        num_features: int = 3, 
        num_classes: int = 1,
        p: float = 0.0,
    ):
        super().__init__()
        
        self.scatter = Scatter(num_features, trainable_laziness=False)  
    
        self.linear1 = Linear(99, 128)  # note that the output size of the scattering layer changes with the number
                                            # of moments that are calculated

        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 64)
        self.embedding_layer = Linear(64, 32)

        self.act = torch.nn.LeakyReLU()
        
        self.dropout = nn.Dropout(p)
        
        self.classifier = Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h, __ = self.scatter(data)
        h = self.act(h)  
        h = self.linear1(h)
   
        h = self.dropout(h)
        h = self.act(h)  
        h = self.linear2(h)

        h = self.dropout(h)
        h = self.act(h)  
        h = self.linear3(h)
        
        embedding = self.embedding_layer(h)
        
        h = self.act(embedding)  
        
        # last layer is a classifier
        output = self.classifier(embedding)

        return embedding, output 


def train_encoder_epoch(model, loss_fn, optimizer, train_loader, embeddings_list):
    """Train the model for one epoch.
    Args:
        model: the model
        loss_fn: the loss function
        optimizer: the optimizer
    Returns:
        train_loss: the loss of the epoch
    """
    
    model.train()
    optimizer.zero_grad()
    base_loss_epoch = 0
    smoothness_loss_epoch = 0
    composite_loss_epoch = 0

    # calculate smoothness loss
    smoothness_loss = calculate_smoothness_loss(embeddings_list, k=3)
    smoothness_loss_epoch += smoothness_loss.detach().numpy()

    # evaluate on the train nodes
    for data in train_loader:
        target = data.y
        batch_size = data.num_graphs

        # get the outputs
        __, out = model(data)

        # calculate base loss (MSE)

        base_loss = loss_fn(out, target.unsqueeze(1))
        base_loss_epoch += base_loss.detach().numpy() * batch_size

        # calculate composite loss
        loss = base_loss + smoothness_lambda * smoothness_loss
        composite_loss_epoch += loss.detach().numpy() * batch_size

        loss.backward()

        # perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    # calculate training loss for the epoch
    base_loss_epoch = base_loss_epoch / len(train_loader.dataset)
    composite_loss_epoch = composite_loss_epoch / len(train_loader.dataset)

    return base_loss_epoch, smoothness_loss_epoch, composite_loss_epoch

def test_encoder_epoch(model, loss_fn, test_loader, embeddings_list):
    """Test the model for one epoch.
    Args:
        model: the model
        loss_fn: the loss function
    Returns:
        test_loss: the loss of the epoch
    """
    model.eval()  # set model to evaluation mode

    base_loss_epoch = 0
    smoothness_loss_epoch = 0
    composite_loss_epoch = 0

    # calculate smoothness loss
    smoothness_loss = calculate_smoothness_loss(embeddings_list, k=3)
    smoothness_loss_epoch += smoothness_loss.detach().numpy()
    
    with torch.no_grad():  # disable gradient calculation

        for data in test_loader:
            target = data.y
            batch_size = data.num_graphs
        
            __, out = model(data)
            
            # calculate base loss (MSE)
            base_loss = loss_fn(out, target.unsqueeze(1))
            base_loss_epoch += base_loss.detach().numpy() * batch_size
            
            # calculate composite loss
            loss = base_loss + smoothness_lambda * smoothness_loss
            composite_loss_epoch += loss.detach().numpy() * batch_size

        # calculate test loss for the epoch
        base_loss_epoch = base_loss_epoch / len(test_loader.dataset)
        composite_loss_epoch = composite_loss_epoch / len(test_loader.dataset)
                
    return base_loss_epoch, smoothness_loss_epoch, composite_loss_epoch

def train_encoder(model, loss_fn, train_loader, test_loader, optimizer, epochs=5):
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
    train_base_losses = []
    train_smoothness_losses = []
    train_composite_losses = []
    
    test_base_losses = []
    test_smoothness_losses = []
    test_composite_losses = []
    
    loop = tqdm.tqdm(range(1, epochs + 1))

    for epoch in loop:
        # get the embeddings of the latent space, in order to calculate smoothness loss
        embeddings_list = []
        
        for data in train_loader:
            embedding, __ = model(data)
            embeddings_list.append(embedding.detach().numpy())
       
        embeddings_list = torch.squeeze(torch.Tensor(np.array(embeddings_list)))
        
        embeddings_list = torch.reshape(embeddings_list, (-1, 16))

        # train the model for one epoch
        train_base_loss_epoch, train_smoothness_loss_epoch, train_composite_loss_epoch = train_encoder_epoch(model, loss_fn, optimizer, train_loader, embeddings_list)
        
        # test the model for one epoch        
        test_base_loss_epoch, test_smoothness_loss_epoch, test_composite_loss_epoch = test_encoder_epoch(model, loss_fn, test_loader, embeddings_list)

        # put into our storage vectors
        train_base_losses.append(train_base_loss_epoch)
        train_smoothness_losses.append(train_smoothness_loss_epoch)
        train_composite_losses.append(train_composite_loss_epoch)
        test_base_losses.append(test_base_loss_epoch)
        test_smoothness_losses.append(test_smoothness_loss_epoch)
        test_composite_losses.append(test_composite_loss_epoch)
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(train_loss=train_base_loss_epoch, test_loss=test_base_loss_epoch)
    
    return train_base_losses, train_smoothness_losses, train_composite_losses, test_base_losses, test_smoothness_losses, test_composite_losses

def calculate_smoothness_loss(embedding, k=3):
    """
    Calculate smoothness loss, when given a set of intermediate hidden values/embeddings.
    Input:
    embedding = set of points from the latent space. Must be detached tensor/numpy array 
    k = parameter for KNN graph clustering

    Output:
    smoothness_loss = scalar quantifying the amount of loss related to smoothness
    """   
    # construct the KNN graph
    knn = NearestNeighbors(n_neighbors = k, algorithm='auto').fit(embedding)
    distances, indices = knn.kneighbors(embedding)

    # get the interpolated latent points
    interpolated_points = []
    for i in range(len(embedding)):
        for j in range(k):
            alpha = torch.rand(1).item()
            point = alpha * embedding[i] + (1 - alpha) * embedding[indices[i, j]]
            interpolated_points.append(point)
    
    decoded_embedding_points = embedding # TODO: Decode embedding and interpolated points
    decoded_interpolated_points = interpolated_points
    
    # Calculate smoothness loss
    smoothness_loss = 0
    for i in range(len(embedding)):
        for j in range(k):
            x1 = decoded_embedding_points[indices[i, 0]]
            x2 = decoded_embedding_points[indices[i, j]]
            xi = decoded_interpolated_points[i]
            
            term1 = torch.norm(x1 - xi, p=2)
            term2 = torch.norm(x2 - xi, p=2)
            term3 = torch.norm(x1 - x2, p=2)
            
            smoothness_loss  += F.relu((term1 + term2) / 2 - term3)
    
    smoothness_loss = smoothness_loss / k
    
    return smoothness_loss
