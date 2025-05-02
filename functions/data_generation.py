import pandas as pd
import numpy as np

# for web scraping
import requests
from bs4 import BeautifulSoup as bs
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib.request
from urllib.error import HTTPError
from urllib.request import urlopen

# cheminformatics
import rdkit
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.Chem.QED import properties
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import AllChem
import cirpy

# other packages
from tqdm import tqdm
from typing import Any
import torch
import torch_geometric
from rdkit import Chem, RDLogger
from torch_geometric.data import Data


# function to get content from https://www.pesticideinfo.org/
# based on https://realpython.com/beautiful-soup-web-scraper-python/
def pesticideinfo_get(PRI_start, PRI_end):
    """
    this function takes starting and ending ids for this database and constructs a range.
    Based on the range of PRIs, it extracts data from each PesticideInfo page. 
    PRI_start = integer
    PRI_end = integer
    """
    # webdriver workaround from https://stackoverflow.com/questions/76928765/attributeerror-str-object-has-no-attribute-capabilities-in-selenium
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    cservice = webdriver.ChromeService(
        executable_path="C:/Users/tobia/OneDrive/Documents/GitHub/BugBuster/chromedriver-win64/chromedriver-win64/chromedriver.exe",
        chrome_options=options)
    driver = webdriver.Chrome(service = cservice)
    # driver code based on https://stackoverflow.com/questions/52687372/beautifulsoup-not-returning-complete-html-of-the-page
    
    # make range
    PRIs = np.arange(PRI_start, PRI_end, 1)

    # set up storage for a dataframe
    pris = []
    names = []
    casrns = []
    classes = []
    mws = []
    uses =[]

    for pri in PRIs:
        URL = "https://www.pesticideinfo.org/chemical/PRI"+str(pri)
        driver.get(URL)
        time.sleep(0.25) # wait for page to load
        page = driver.page_source
        soup = bs(page, "html.parser") # parse page
        table = soup.find_all("div", {"class": "data-table-key-value"}) #get values from table of interest
        
        if len(table) < 5:
            print(f'{pri} does not exist')
        else:
            # now extract desired information 
            pris.append(pri)
            name = str(table[0]).split('</div>')[1][5:]
            names.append(name)
            casrn = str(table[2]).split('</div>')[1][5:]
            casrns.append(casrn)
            chem_class = str(table[4]).split('</div>')[1][5:]
            classes.append(chem_class)
            mw = str(table[5]).split('</div>')[1][5:]
            mws.append(mw)
            use = str(table[6]).split('</div>')[1][5:]
            uses.append(use)

    data = {'name': names, 'pri': pris, 'casrn': casrns, 
            'class': classes, 'mw': mws, 'use': uses}
    df = pd.DataFrame(data)
    
    driver.quit()
    
    return df

# add smiles based on casrn
# use cirpy 
def add_smiles(df, cas_col, name_col):
    temp_df = df.copy() # avoid overwrite
    casrns = temp_df[cas_col]
    names = temp_df[name_col]
    smiles_cirpy_cas_storage = []
    smiles_cirpy_name_storage = []

    for cas in tqdm(casrns):
        if cas == "Not Listed":
            smiles_cirpy_casrn = float('NaN')
        else:
            smiles_cirpy_casrn = cirpy.resolve(cas, 'smiles')
        smiles_cirpy_cas_storage.append(smiles_cirpy_casrn)

    for name in tqdm(names):
        smiles_cirpy_name = cirpy.resolve(name, 'smiles')
        smiles_cirpy_name_storage.append(smiles_cirpy_name)

    temp_df['smiles_cirpy_casrn'] = smiles_cirpy_cas_storage
    temp_df['smiles_cirpy_name'] = smiles_cirpy_name_storage

    return temp_df

# remove no smiles
def merge_and_remove_smiles(df, cirpy_casrn_smiles, cirpy_name_smiles):
    temp = df.copy()
    idx = range(0, temp.shape[0])
    cirpy_casrn_smiles = temp[cirpy_casrn_smiles]
    cirpy_name_smiles = temp[cirpy_name_smiles]

    smiles_storage = []
    for i in idx:
        if pd.isnull(cirpy_casrn_smiles[i]) == False:
            smiles = cirpy_casrn_smiles[i]
        else:
            if pd.isnull(cirpy_name_smiles[i]) == False:
                smiles = cirpy_name_smiles[i]
            else:
                smiles = float('NaN')
        smiles_storage.append(smiles)

    temp['smiles'] = smiles_storage
    
    return temp

# remove salts, based on https://www.rdkit.org/docs/source/rdkit.Chem.SaltRemover.html
def no_salts(df, smiles_col):
    temp = df.copy()
    smiles = temp[smiles_col]
    cleaned = []
    for smi in smiles:

        # for chlorine
        ions = ['Cl-', 'Br-', 'NH4+', 'Na+', 'Ca+', 'F-', 'Li+', 'K+', 'Mg++', 'Ca++']
        if any(x in smi for x in ions): # https://stackoverflow.com/questions/3389574/check-if-multiple-strings-exist-in-another-string
            remover = SaltRemover.SaltRemover(defnData="[Cl-]", defnFormat='smiles')
            res = remover.StripMol(Chem.MolFromSmiles(smi), dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Br-]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[NH4+]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Na+]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Ca+]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[F-]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Li+]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[K+]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Mg++]", defnFormat='smiles')
            res = remover.StripMol(res, dontRemoveEverything=True)
            remover = SaltRemover.SaltRemover(defnData="[Ca++]", defnFormat='smiles')
            res = Chem.MolToSmiles(remover.StripMol(res, dontRemoveEverything=True))
            cleaned.append(res)
        else:
            cleaned.append(smi)
                
    temp['no_salt'] = cleaned
    return temp

def neutralize_mol(df, col):
    '''
    adapted from https://www.rdkit.org/docs/Cookbook.html
    Author: Noel O’Boyle (Vincent Scalfani adapted code for RDKit)
    Source: https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html
    Index ID#: RDKitCB_33
    Summary: Neutralize charged molecules by atom.
    '''
    neutral_storage = []

    smiles = df[col]

    for smi in smiles:
        mol = rdkit.Chem.MolFromSmiles(smi)
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        neutral = Chem.MolToSmiles(mol)
        neutral_storage.append(neutral)

    df['neutral_smiles'] = neutral_storage
    
    return df

def calc_prop(df, col):
    '''
    calculates MW using https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
    calculates logP using QED ALogP  https://www.rdkit.org/docs/source/rdkit.Chem.QED.html
    calculates number of heavy atoms using 
    error reporting supported by Clarity
    
    ''' 
    mw_storage = []
    logp_storage = []
    nheavy_storage = []
    valid_smiles_idx = []

    smiles = df[col]
    
    for n, smi in enumerate(smiles):
        try:
            mol = rdkit.Chem.MolFromSmiles(smi)
            if mol is not None:
                mw = rdkit.Chem.Descriptors.MolWt(mol)
                qed = rdkit.Chem.QED.properties(mol)
                nheavy = rdkit.Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol)
                logp = qed[1]
                mw_storage.append(mw)
                logp_storage.append(logp)
                nheavy_storage.append(nheavy)
                valid_smiles_idx.append(n)
            else:
                print(f"Invalid SMILES: {smi}")
                
        except Exception as e:
            print(f"Error processing SMILES: {smi} - Error: {e}")

    df_valid = df.iloc[valid_smiles_idx].copy()
    df_valid['mw'] = mw_storage
    df_valid['nheavy'] = nheavy_storage
    df_valid['alogp'] = logp_storage

    return df_valid

def aliphatic_string(df, col):
    '''
    For each carbon string of 3 or more, create a new molecule with one less carbon and one with one more.
    Clarity used for code improvements
    '''
    df_temp = df.copy().reset_index(drop=True)

    augmented_addC = []
    augmented_minusC = []
    
    for index, row in df_temp.iterrows():
        smi = row[col]
        
        if 'CCC' in smi:
            original = row.copy()
            
            # Create a molecule with one more carbon
            plus_C = smi.replace('CCC', 'CCCC', 1) # Only do one replacement
            plus_C_row = original.copy()
            plus_C_row['PREFERRED NAME'] = 'aug_plus'
            plus_C_row['CASRN'] = 'aug_plus'
            plus_C_row[col] = plus_C
            augmented_addC.append(plus_C_row)
            
            # Create a molecule with one less carbon
            minus_C = smi.replace('CCC', 'CC', 1)
            minus_C_row = original.copy()
            minus_C_row['PREFERRED NAME'] = 'aug_minus'
            minus_C_row['CASRN'] = 'aug_minus'
            minus_C_row[col] = minus_C
            augmented_minusC.append(minus_C_row)
            
    addC = pd.DataFrame(augmented_addC)
    minusC = pd.DataFrame(augmented_minusC)
    
    # combine original DataFrame with the augmented DataFrame
    out_df = pd.concat([df_temp, addC, minusC], ignore_index=True)
    
    return out_df
    
# from https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/utils/smiles.html
# revised per comments
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

def from_smiles(smiles: str) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
    """
    RDLogger.DisableLog('rdApp.*')
    
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.float).view(-1, 3)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))

        edge_indices += [[i, j], [j, i]] # this creates the adjacency matrix
        edge_attrs += [e, e] # this creates the edge attributes matrix

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)



def to_smiles(data: 'torch_geometric.data.Data') -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0].item()))
        atom.SetFormalCharge(x_map['formal_charge'][int(data.x[i, 2].item())])
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0].item())]
        mol.AddBond(src, dst, bond_type)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

# define function to create data
# based on adapted code from https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
def create_py_geom_dataset(x_smiles, y):
    '''
    x_smiles = column with input smiles
    y = column with relevant property value
    '''
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):

        initial_data = from_smiles(smiles)

        X = initial_data.x.detach()
        E = initial_data.edge_index.detach()
        EF = initial_data.edge_attr.detach()
        y_tensor = torch.tensor(y_val, dtype = torch.float)
        
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list

def create_smiles_data(py_geom_df):
    '''
    Creates a dataset with smiles and y values
    Inputs:
    py_geom_df = input dataset that is a pytorch geometric with a y tensor
    '''
    data_list = []
    idx = range(len(py_geom_df))
    for i in idx:

        smiles = to_smiles(py_geom_df[i])
        y = py_geom_df[i].y.detach().tonumpy()
        
        data_list.append(smiles)
        data_list.append(y)
    df_out = pd.DataFrame(data_list)
    return data_list