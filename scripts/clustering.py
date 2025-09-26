
import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import rdkit.DataStructs as rdd
import rdkit.Chem.AllChem as rdca
import sklearn.decomposition as skd
import sklearn.cluster as skc
import sklearn.metrics as skm
from typing import Dict, List, Union


def cluster(dataset: Union[pd.DataFrame, str], 
           smiles_column: str = "smiles",
           variance_threshold: float = 0.80,
           fp_size: int = 2048,
           random_state: int = 91234) -> Dict[str, List[str]]:
    """
    Cluster molecules based on Tanimoto similarity and return grouped SMILES.
    
    Args:
        dataset: DataFrame with SMILES or path to CSV file
        smiles_column: Name of column containing SMILES strings
        variance_threshold: PCA variance threshold for dimensionality reduction
        fp_size: RDKit fingerprint size
        random_state: Random state for reproducible clustering
        
    Returns:
        Dictionary with cluster names as keys and lists of SMILES as values
    """
    # Load data if path is provided
    if isinstance(dataset, str):
        data = pd.read_csv(dataset)
    else:
        data = dataset.copy()
    
    # Generate fingerprints
    fingerprint_generator = rdca.GetRDKitFPGenerator(fpSize=fp_size)
    smiles_list = list(data[smiles_column].values)
    fingerprints = [fingerprint_generator.GetFingerprint(rdc.MolFromSmiles(smile)) 
                   for smile in smiles_list]
    
    number_of_molecules = len(fingerprints)
    
    # Calculate similarity matrix
    similarities = np.zeros((number_of_molecules, number_of_molecules))
    for ri in range(number_of_molecules):
        similarities[ri, :] = rdd.BulkTanimotoSimilarity(fingerprints[ri], fingerprints)
    
    # PCA dimensionality reduction
    pca = skd.PCA(number_of_molecules)
    pca.fit(similarities)
    
    # Find number of components for desired variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    number_of_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    pca = skd.PCA(number_of_components)
    transformed_similarities = pca.fit_transform(similarities)
    
    # Find optimal number of clusters using silhouette score
    silhouette_averages = []
    clusterization_range = range(2, max(3, int(number_of_molecules / 10)))
    
    for n_clusters in clusterization_range:
        kmeans = skc.KMeans(n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(transformed_similarities)
        silhouette_averages.append(skm.silhouette_score(transformed_similarities, cluster_labels))
    
    number_of_clusters = silhouette_averages.index(max(silhouette_averages)) + 2
    
    # Final clustering
    kmeans = skc.KMeans(n_clusters=number_of_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(transformed_similarities)
    
    # Create molecule groups dictionary
    molecule_groups = {}
    for i in range(number_of_clusters):
        cluster_name = f"cluster_{i+1}"
        cluster_smiles = [smiles_list[j] for j, label in enumerate(cluster_labels) if label == i]
        molecule_groups[cluster_name] = cluster_smiles
    
    return molecule_groups


# Example usage
if __name__ == "__main__":
    # Test the function
    molecule_groups = cluster("data/train_ESOL.csv")
    print(f"Found {len(molecule_groups)} clusters:")
    for cluster_name, smiles_list in molecule_groups.items():
        print(f"{cluster_name}: {len(smiles_list)} molecules")
        print(f"  Example: {smiles_list[0] if smiles_list else 'None'}")
        print()