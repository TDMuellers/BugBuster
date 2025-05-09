import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch_geometric
from torch_geometric.utils import to_networkx

import rdkit
from rdkit import Chem

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

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7))
}

# only specify bond type
e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'HYDROGEN',
        'THREECENTER',
    ]
}

def mol_reconstruct(valid_graphs):
    '''
    This function takes valid graphs with adjacency matrices at their edge index position
    and tries to reconstruct a valid molecule
    '''
    
    mol_out = []
    valid_mol = []
    validity_tracker = 0
    
    # first reconstruct the node features, edge indices, and edge features in appropriate format
    for graph in valid_graphs:
        graph_temp = copy.deepcopy(graph) 

        D = graph_temp.edge_index.shape[0] # determine dimension of adjacency matrix
        
        # this block takes the adjacency matrix and reconstructs edge indices needed for RDKIT
        G_temp= nx.from_numpy_array(graph_temp.edge_index, create_using = nx.MultiGraph())
        nx.write_adjlist(G_temp, "gtemp.adjlist")
        G_edge_pairs = nx.read_adjlist("gtemp.adjlist")
        
        edge_pair_inverse = G_edge_pairs.edges
        graph_temp.edge_index = torch.swapaxes(torch.tensor(np.array(edge_pair_inverse, dtype=int)), 0, 1)
        D_edge = graph_temp.edge_index.shape[1]

        # this block takes the node features and rounds them needed for RDKIT
        graph_temp.x = np.array(np.round(graph_temp.x), dtype=int)[0:D]

        # this block takes the node features and rounds them needed for RDKIT
        graph_temp.edge_attr = np.array(np.round(graph_temp.edge_attr), dtype=int)[0:D_edge]

        # now attempt to reconstruct valid molecules
        mol = Chem.RWMol()

        for i in range(graph_temp.num_nodes):
            atom = Chem.Atom(int(graph_temp.x[i, 0].item()))
            atom.SetFormalCharge(x_map['formal_charge'][int(graph_temp.x[i, 2].item())])
            mol.AddAtom(atom)
        
        edges = [tuple(i) for i in graph_temp.edge_index.t().tolist()]
        
        for i in range(len(edges)):
            src, dst = edges[i]
        
            bond_type = Chem.BondType.values[int(graph_temp.edge_attr[i, 0].item())]
            mol.AddBond(src, dst, bond_type)
        
        reconstructed_mol = mol.GetMol()
        mol_out.append(reconstructed_mol)

        try:
            Chem.SanitizeMol(mol)
            is_sanitized = True
            valid_mol.append(reconstructed_mol)
            validity_tracker += 1
            
        except Chem.AtomValenceException as e:
            is_sanitized = False            
            
    print(f'There are {validity_tracker} valid molecules, with a percent validity of {100*(validity_tracker/len(valid_graphs))}%')
    
    return mol_out, valid_mol