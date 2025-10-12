# %%
import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import rdkit.Chem.Draw as rdcd
import torch_geometric.datasets as tcgd
import random
import sys
import os
import urllib.request
sys.path.append('..')
from utils.clustering import clustering

random.seed(0)
np.random.seed(0)

# URLs for datasets
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
HCE_URL = "https://raw.githubusercontent.com/aspuru-guzik-group/Tartarus/refs/heads/main/datasets/hce.csv"

def download_dataset(url, output_path):
    """Download a dataset from URL."""
    print(f"Downloading {os.path.basename(output_path)} from {url}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
# %%
# Load QM9
database = tcgd.QM9(root='qm9_data')

# Extract SMILES and Gibbs free energy of atomization from QM9
smiles_list = [di.smiles for di in database]
data_qm9 = pd.DataFrame(smiles_list, columns=['smiles'])
data_qm9['dga'] = [di.y[:,15].item() for di in database] # Place 15 is Gibbs free energy
print(f"Dataset size: {len(data_qm9.index)}")

# Filter QM9 with roundtrip
data_qm9['mol'] = [rdc.MolFromSmiles(si) for si in list(data_qm9['smiles'].values)]
data_qm9 = data_qm9.dropna(ignore_index=True)
data_qm9 = data_qm9.drop(columns=['mol'])
print(f"Dataset size after filtering: {len(data_qm9.index)}")

# Apply clustering to QM9
data_qm9 = clustering(data_qm9, target_column='dga', output_dir='../clustered_data/qm9', dataset_name='qm9')

# %%
# ESOL
download_dataset(ESOL_URL, "../data/esol.csv")

database = pd.read_csv("../data/esol.csv")
data_esol = pd.DataFrame()
data_esol['smiles'] = database['smiles'].values
data_esol['solubility'] = database['measured log solubility in mols per litre'].values
print(f"Dataset size: {len(data_esol.index)}")

# Apply clustering to ESOL
data_esol = clustering(data_esol, target_column='solubility', output_dir='../clustered_data/esol', dataset_name='esol')
# %%
# HCE - Simple quantile-based clustering
download_dataset(HCE_URL, "../data/hce.csv")

database = pd.read_csv("../data/hce.csv")
data_hce = pd.DataFrame()
data_hce['smiles'] = database['smiles'].values
data_hce['pce_1'] = database['pce_1'].values
print(f"Dataset size: {len(data_hce.index)}")

# For HCE we just do simple clustering based on property quantiles
data_hce['cluster'] = pd.qcut(data_hce['pce_1'], q=4, labels=[0, 1, 2, 3])

os.makedirs('../clustered_data/hce', exist_ok=True)
for ci in range(4):
    smiles_list = list(data_hce.loc[data_hce['cluster'] == ci, 'smiles'])
    print(f"Cluster {ci}: {len(smiles_list)} Molecules")
    molecules_list = [rdc.MolFromSmiles(si) for si in smiles_list]
    grid = rdcd.MolsToGridImage(molecules_list[:9], returnPNG=False)
    grid.save(f'clustered_data/hce/{ci}.png')

data_hce.to_csv("clustered_data/hce/hce.csv", index=False)
print("\nAll datasets processed successfully!")
# %%
