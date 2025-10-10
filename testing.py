# %%
import rdkit
import torch_geometric
import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import rdkit.DataStructs as rdd
import rdkit.Chem.AllChem as rdca
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Draw as rdcd
import sklearn.decomposition as skd
import sklearn.cluster as skc
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import scipy as sp
import torch_geometric.datasets as tcgd
import random

random.seed(0)
np.random.seed(0)
# %%
# Load QM9
database = tcgd.QM9(root='qm9_data')

# Extract SMILES and Gibbs free energy of atomization from QM9
smiles_list = [di.smiles for di in database]
data = pd.DataFrame(smiles_list, columns=['smiles'])
data['dga'] = [di.y[:,15].item() for di in database]
# data = data.sample(n=int(len(data.index) * 0.10), random_state=0)
print(f"Dataset size: {len(data.index)}")

# Filter QM9 with roundtrip
data['mol'] = [rdc.MolFromSmiles(si) for si in list(data['smiles'].values)]
data = data.dropna(ignore_index=True)
print(f"Dataset size: {len(data.index)}")

# %%
# ESOL
database = pd.read_csv("data/ESOL.csv")
data = pd.DataFrame()
data['smiles'] = database['smiles'].values
data['solubility'] = database['measured log solubility in mols per litre'].values
print(f"Dataset size: {len(data.index)}")
# %%
fingerprint_size = 4096
number_of_molecules = len(data['smiles'].values)
number_of_bins = 100
number_of_samples = 4
number_of_references = number_of_bins * number_of_samples
fingerprint_generator = rdca.GetRDKitFPGenerator(fpSize=fingerprint_size)

# Compute fingerprints
fingerprints = [fingerprint_generator.GetFingerprint(rdc.MolFromSmiles(si)) for si in list(data['smiles'].values)]

# %%
data['quantile'] = pd.qcut(data.loc[:,'solubility'], number_of_bins, labels=list(range(number_of_bins)), duplicates='drop').values

#data['quantile'] = pd.qcut(data.loc[:,'dga'], number_of_bins, labels=list(range(number_of_bins)), duplicates='drop').values
fingerprints_references = [fingerprints[ii] for ii in data.groupby('quantile', group_keys=False, observed=False).apply(lambda x: x.sample(n=number_of_samples, random_state=0), include_groups=False).index.tolist()]
# %%
# Initialize similarity matrix
similarities = np.zeros((number_of_molecules, number_of_references), dtype=np.float32) # could be changed back to np.float32 if memory is no concern

# Compute pairwise similarities
for ri in range(number_of_molecules):
  similarities[ri, :] = rdd.BulkTanimotoSimilarity(fingerprints[ri], fingerprints_references)
# %%
# Perform PCA with all components and a random subset of the molecules
subset_size = int(0.05 * number_of_molecules) if number_of_molecules > 20000 else min(number_of_molecules, 1000)
pca = skd.IncrementalPCA(n_components=number_of_references)
pca.fit(similarities[np.random.choice(similarities.shape[0], size=subset_size, replace=False), :])

# Only take principal components that cumulatively explain x% of the total variance
total_variance_explained = 0.80 # hyperparameter
number_of_components = len([np.sum(pca.explained_variance_ratio_[:ni]) for ni in range(number_of_molecules) if np.sum(pca.explained_variance_ratio_[:ni]) < total_variance_explained])
print(f'Number of dominant principal components: {number_of_components}')

# Redo PCA with significant components
pca = skd.IncrementalPCA(n_components=number_of_components)
transformed_similarities = pca.fit_transform(similarities)
# %%
# Initialize list of silhouette scores
silhouette_averages = list()
lower_cluster_limit = 2
upper_cluster_limit = 25
clusterization_range = range(lower_cluster_limit, upper_cluster_limit)

# Perform clustering
for ni in clusterization_range:
    clustering = skc.MiniBatchKMeans(n_clusters=ni, random_state=0)
    cluster_labels = clustering.fit_predict(transformed_similarities)
    silhouette_averages.append(skm.silhouette_score(transformed_similarities, cluster_labels))
    print("done")

# Find best number of clusters
number_of_clusters = silhouette_averages.index(max(silhouette_averages)) + lower_cluster_limit
print(f'Best number of clusters: {number_of_clusters}')

# Repeat clustering with best number of clusters
clustering = skc.MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0)
cluster_labels = clustering.fit_predict(transformed_similarities)
silhouette_average = skm.silhouette_score(transformed_similarities, cluster_labels)

# Plot silhouette score as a function of the number of clusters
plt.plot(clusterization_range, silhouette_averages)
plt.plot(number_of_clusters, silhouette_average, marker='x', color='black')
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette score')
plt.show()

# %%
# Apply clustering to original dataframe
data['cluster'] = cluster_labels

# Plot example molecules for each cluster
for ci in range(number_of_clusters):
  smiles_list = list(data.loc[data['cluster'] == ci, 'smiles'])
  print(f"Cluster {ci}: {len(smiles_list)} Molecules")
  molecules_list = [rdc.MolFromSmiles(si) for si in smiles_list]
  grid = rdcd.MolsToGridImage(molecules_list[:9], returnPNG=False)
  grid.save(f'clustered_data/esol/{ci}.png')

# %%
data.to_csv("clustered_data/esol/esol.csv", index=False)
# %%
# For HCE we just do clustering based on the property quantiles
database = pd.read_csv("data/hce.csv")
data = pd.DataFrame()
data['smiles'] = database['smiles'].values
data['pce_1'] = database['pce_1'].values
data['cluster'] = pd.qcut(data['pce_1'], q=4, labels=[0, 1, 2, 3])

# %%
for ci in range(4):
    smiles_list = list(data.loc[data['cluster'] == ci, 'smiles'])
    print(f"Cluster {ci}: {len(smiles_list)} Molecules")
    molecules_list = [rdc.MolFromSmiles(si) for si in smiles_list]
    grid = rdcd.MolsToGridImage(molecules_list[:9], returnPNG=False)
    grid.save(f'clustered_data/hce/{ci}.png')

data.to_csv("clustered_data/hce/hce.csv", index=False)
# %%
