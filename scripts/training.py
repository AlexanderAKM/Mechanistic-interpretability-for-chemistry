# %%
import pandas as pd
import argparse
from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_model
# %%
RANDOM_SEED = 19237

# Make an args parser
esol_defaults = {
    'train_csv': 'data/train_ESOL.csv',
    'test_csv': 'data/test_ESOL.csv',
    'target_column': 'measured log solubility in mols per litre',
    'smiles_column': 'smiles',
    'output_dir': 'trained_models',
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.0008045,
    'l1_lambda': 2.596e-05,
    'l2_lambda': 5e-05,
    'dropout': 0.3408,
    'hidden_channels': 87,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

esol_parser = argparse.Namespace(**esol_defaults)

# %%
train_esol = pd.read_csv("data/train_ESOL.csv")
test_esol = pd.read_csv("data/test_ESOL.csv")
norm_train_esol, esol_scaler = normalize_csv(train_esol, target_col=esol_parser.target_column)
norm_test_esol, _ = normalize_csv(test_esol, target_col=esol_parser.target_column, scaler=esol_scaler)

# %%
esol_results = train_chemberta_model(esol_parser, norm_train_esol, norm_test_esol, esol_scaler)
esol_results

# %%

# %%
hce_defaults = {
    'train_csv': 'data/train_hce.csv',
    'test_csv': 'data/test_hce.csv',
    'target_column': 'pce_1',
    'smiles_column': 'smiles',
    'output_dir': 'trained_models',
    'epochs': 10,
    'batch_size': 16,
    'lr': 0.0008045,
    'l1_lambda': 2.596e-05,
    'l2_lambda': 5e-05,
    'dropout': 0.3408,
    'hidden_channels': 87,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

hce_parser = argparse.Namespace(**hce_defaults)


train_hce = pd.read_csv("data/train_hce.csv")
test_hce = pd.read_csv("data/test_hce.csv")
norm_train_hce, hce_scaler = normalize_csv(train_hce, target_col="pce_1")
norm_test_hce, _ = normalize_csv(test_hce, target_col="pce_1", scaler=hce_scaler)
hce_results = train_chemberta_model(hce_parser, norm_train_hce, norm_test_hce, hce_scaler, device=DEVICE)
hce_results

# %% [markdown]
# Now we'll be doing the same for qm9
qm9_defaults = {
    'train_csv': 'data/train_qm9.csv',
    'test_csv': 'data/test_qm9.csv',
    'target_column': 'g298_atom',
    'smiles_column': 'smiles',
    'output_dir': 'trained_models',
    'epochs': 15,
    'batch_size': 16,
    'lr': 0.0008045,
    'l1_lambda': 2.596e-05,
    'l2_lambda': 5e-05,
    'dropout': 0.3408,
    'hidden_channels': 87,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

qm9_parser = argparse.Namespace(**qm9_defaults)

train_qm9 = pd.read_csv("data/train_qm9.csv")
test_qm9 = pd.read_csv('data/test_qm9.csv')
norm_train_qm9, qm9_scaler = normalize_csv(train_qm9, target_col="g298_atom")
norm_test_qm9, _ = normalize_csv(test_qm9, target_col="g298_atom", scaler=qm9_scaler)
qm9_results = train_chemberta_model(qm9_parser, norm_train_qm9, norm_test_qm9, qm9_scaler)
qm9_results
