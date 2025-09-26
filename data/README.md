This folder contains the data on which the models in trained_models are trained. data_handling.py shows how the files have been processed.

This includes the three original datasets: hce.csv, qm9.csv and ESOL.csv (originally delaney-processed.csv)

qm9 and ESOL are from https://moleculenet.org/datasets-1. hce is from https://github.com/aspuru-guzik-group/Tartarus/tree/main/datasets.

For ESOL, we used all data points (1128). We dropped all columns except for the SMILES and target column (measured log solubility in mols per litre), and split it into 80% training data and 20% test data with sklearn's https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split function with random_seed 19237. 

For qm9, we randomly sampled 1128 data points from the entire dataset (with random_seed 19237), then dropped all columns except for the SMILES and target column (g298_atom), and split it into 80% training data and 20% test data similarly with sklearn function with random_seed 19237.

For hce, we randomly sampled 1128 data points from the entire dataset (with random_seed 19237), then dropped all columns except for the SMILES and target column (pce_1), and split it into 80% training data and 20% test data similarly with sklearn function with random_seed 19237.
