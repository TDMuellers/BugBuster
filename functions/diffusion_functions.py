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


from diffusers import DDPMScheduler, UNet2DModel


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

class DiffuseNet(nn.Module):
    def __init__(
        self, 
        input_size=32,  # dimension of the latent space
        class_emb_size=1):
        
        super().__init__()
        
        # embedding for the class
        self.property_embedding = nn.Linear(class_emb_size, 32)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=(4, 8),  # target size = size of latent space embedding
            in_channels=1 + class_emb_size,  # Additional input channels for class cond.
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the property value as an additional argument
    def forward(self, x, t, property_label):
        bs = x.size(0)
        
        # class conditioning in right shape to add as additional input channels
        property_label = self.property_embedding(property_label.view(bs, -1))
        property_label = property_label.unsqueeze(-1).unsqueeze(-1)
        property_label = property_label.view(bs, 1, 4, 8)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, property_label), 1)  
        
        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  


def train_diffusion_epoch(model, optimizer, train_loader):
    """Train the model for one epoch.
    Args:
        model: the model
        optimizer: the optimizer
        train_loader: contains all information needed for training, including graphs and their latent space representations.
    Returns:
        train_loss: the loss of the epoch
    """

    optimizer.zero_grad()
    loss_epoch = 0

    # Our loss function
    loss_fn = nn.MSELoss()
    
    for data in train_loader:
        batch_size = data.num_graphs 
        
        # Get some data and prepare the corrupted version
        x = data.hidden_values
        
        # reshape the 32-long vector to be more "image-like"
        x = x.view(1, 1, 4, 8)
        y = data.y
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long()
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = model(noisy_x, timesteps, y)  # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise)  # How close is the output to the noise

        # Backprop and update the params:
        loss.backward()
        loss_epoch += loss.detach().numpy() * batch_size
        
        optimizer.step()

    # calculate training loss for the epoch
    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch



def train_diffusion(model, train_loader, optimizer, epochs=5):
    """Train the model.
    Args:
        model: the model
        loss_fn: the loss function
        train_loader: the training data loader
        optimizer: the optimizer
        epochs: the number of epochs to train
    Returns:
        train_losses: the training losses
    """
    train_losses = []
        
    loop = tqdm.tqdm(range(1, epochs + 1))

    for epoch in loop:

        # train the model for one epoch
        train_loss_epoch = train_diffusion_epoch(model, optimizer, train_loader)
        
        # put into our storage vectors
        train_losses.append(train_loss_epoch)
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(train_loss=train_loss_epoch)
    
    return train_losses

def generate_new_points(model, noise_scheduler, desired_logP):
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(1, 1, 4, 8)
    y = torch.ones(1, 1).float() * desired_logP  # Create a batch of labels
    
    # Sampling loop
    for i, t in tqdm.tqdm(enumerate(noise_scheduler.timesteps)):
    
        # Get model pred
        with torch.no_grad():
            residual = model(x, t, y)  # Again, note that we pass in our labels y
    
        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    return x

    
def plot_generated_point(generated_new_points, desired_logP_list, index):
    # Show the results
    x = generated_new_points[index]
    img = x[0] 
    img = img.detach().numpy()
    
    # squeeze to remove the channel dimension.
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    else:
        img = np.transpose(img, (1, 2, 0))
    
    plt.figure(figsize=(8, 4))
    plt.imshow(img, cmap='magma' if img.ndim == 2 else None, interpolation='nearest')
    plt.title(f"Generated values conditioned on logP = {desired_logP_list[index]}")
    plt.savefig("./training-figs/diffusion/generated-values-" + str(index) + "-4x8.png") 
    plt.show()
