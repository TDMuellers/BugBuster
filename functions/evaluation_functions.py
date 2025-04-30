import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch_geometric
from torch_geometric.utils import to_networkx

import networkx as nx

def unpad_graphs(generated_graphs):
    '''
    This function does two things:
    First, it determines if generated graphs are valid. It reports the percent valid and creates a new dataset of valid graphs.
    Then, it removes the zero padding on generated graphs
    '''
    temp = []
    unconnected = np.zeros(len(generated_graphs))
    valid_graphs = []
    
    for i, G in enumerate(generated_graphs):
        adj_matrix = np.round(G.edge_index)
        dim = adj_matrix.shape[0] # track the dimensions to know how much padding to remove
        row_sum = adj_matrix.sum(axis=0) # get the sums within each row, which is the same as the column sum
        zero_indices = np.where(row_sum == 0)[0] # get all zero indices
        
        # see if zero indices has increments of one
        # this indicates if the graph is unconnected
        for p in range(1, len(zero_indices)):
            if zero_indices[p] != zero_indices[p - 1] + 1:
                unconnected[i] = 1 # if the graph is unconnected, flag it
                
        idx = zero_indices[0] if zero_indices.size > 0 else 0 # get first zero index
        adjusted_matrix = adj_matrix[0:idx, 0:idx] # only take the nonzero matrix portion of valid graphs
        
        G_update = copy.deepcopy(G) # copy to avoid overwriting
        G_update.edge_index = adjusted_matrix
        
        if unconnected[i] == 0:
            valid_graphs.append(G_update) # this only keeps the valid graphs
        
    print(f'Percent of valid graphs: {100*(len(unconnected)-sum(unconnected))/len(unconnected)}')
    return valid_graphs

def graph_isomorphism_within_set(generated_graphs):
    '''
    This function takes the generated graphs and compares them to determine if generated graphs are isomorphic.
    The function returns the number of unique graphs and their percent
    '''
    isomorphic_tracker = np.zeros(len(generated_graphs))
    
    for i, new_graph0 in enumerate(generated_graphs):
        
        G0 = nx.from_numpy_array(np.round(new_graph0.edge_index), create_using = nx.MultiGraph())
        
        for p, new_graph1 in enumerate(generated_graphs):
            G1 = nx.from_numpy_array(np.round(new_graph1.edge_index), create_using = nx.MultiGraph())
            if i != p:
                if nx.is_isomorphic(G0, G1, node_match=None, edge_match=None) == True:
                    isomorphic_tracker[i] = p+1 # move these values off of zero
            if i == p:
                continue
    
    n_unique_graphs = 0
    duplicate_tracker = []
    
    for val in isomorphic_tracker:
        if val == 0:
            n_unique_graphs += 1
        if val != 0:
            duplicate_tracker.append(val)

    n_unique_graphs += len(set(duplicate_tracker))/2

    percent_unique = 100*n_unique_graphs/len(generated_graphs)
    
    return print(f'There are {n_unique_graphs} unique graphs, with an overall {percent_unique}% unique graphs generated')


def graph_isomorphism_between_sets(initial_graphs, generated_graphs):
    '''
    This function compares initial graphs and generated graphs and returns the % unique of generated graphs
    '''
    isomorphic_tracker = np.zeros(len(generated_graphs))
    
    for i, new_graph in enumerate(generated_graphs):
        
        G_new = nx.from_numpy_array(np.round(new_graph.edge_index), create_using = nx.MultiGraph())
        
        for original_graph in initial_graphs:
            G_old = to_networkx(original_graph, to_undirected=True)
            if nx.is_isomorphic(G_new, G_old, node_match=None, edge_match=None) == True:
                isomorphic_tracker[i] = 1
                
    percent_unique = 100*(len(generated_graphs) - sum(isomorphic_tracker))/len(generated_graphs)
    return print(f'Compared to the initial graphs used for training, generated graphs are {percent_unique}% unique')