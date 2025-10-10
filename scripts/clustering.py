
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
           fp_size: int = 2048,
           n_clusters: int = None,
           threshold: float = 0.5,
           branching_factor: int = 50) -> Dict[str, List[str]]:
    """
    Cluster molecules using BIRCH algorithm (memory-efficient).
    
    Args:
        dataset: DataFrame with SMILES or path to CSV file
        smiles_column: Name of column containing SMILES strings
        fp_size: RDKit fingerprint size
        n_clusters: Number of clusters (if None, BIRCH auto-determines clusters)
        threshold: BIRCH threshold for subcluster radius
        branching_factor: BIRCH branching factor
        
    Returns:
        Dictionary with cluster names as keys and lists of SMILES as values
    """
    if isinstance(dataset, str):
        data = pd.read_csv(dataset)
    else:
        data = dataset.copy()
    
    print(f"Generating fingerprints for {len(data)} molecules...")
    fingerprint_generator = rdca.GetRDKitFPGenerator(fpSize=fp_size)
    smiles_list = list(data[smiles_column].values)
    fingerprints = [fingerprint_generator.GetFingerprint(rdc.MolFromSmiles(smile)) 
                   for smile in smiles_list]
    
    fp_arrays = []
    for fp in fingerprints:
        arr = np.zeros((fp_size,), dtype=np.int8)
        rdd.ConvertToNumpyArray(fp, arr)
        fp_arrays.append(arr)
    fp_matrix = np.array(fp_arrays)
    
    number_of_molecules = len(fingerprints)
    
    print(f"Clustering {number_of_molecules} molecules using BIRCH...")
    birch = skc.Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters
    )
    
    cluster_labels = birch.fit_predict(fp_matrix)
    number_of_clusters = len(np.unique(cluster_labels))
    print(f"BIRCH identified {number_of_clusters} clusters")
    
    molecule_groups = {}
    for i in range(number_of_clusters):
        cluster_name = f"cluster_{i+1}"
        cluster_smiles = [smiles_list[j] for j, label in enumerate(cluster_labels) if label == i]
        molecule_groups[cluster_name] = cluster_smiles
        print(f"  {cluster_name}: {len(cluster_smiles)} molecules")
    
    return molecule_groups


if __name__ == "__main__":
    molecule_groups = cluster("data/train_ESOL.csv")
    print(f"Found {len(molecule_groups)} clusters:")
    for cluster_name, smiles_list in molecule_groups.items():
        print(f"{cluster_name}: {len(smiles_list)} molecules")
        print(f"  Example: {smiles_list[0] if smiles_list else 'None'}")
        print()