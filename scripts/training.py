# %%
import sys
import os
import pandas as pd
import argparse
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_model
# %%
RANDOM_SEED = 19237
# %%
# For esol

# Make an args parser
esol_defaults = {
    'train_csv': '../clustered_data/esol/train_esol.csv',
    'test_csv': '../clustered_data/esol/test_esol.csv',
    'target_column': 'solubility',
    'smiles_column': 'smiles',
    'output_dir': '../trained_models',
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

esol_parser = argparse.Namespace(**esol_defaults)

# %%
train_esol = pd.read_csv(esol_parser.train_csv)
test_esol = pd.read_csv(esol_parser.test_csv)
norm_train_esol, esol_scaler = normalize_csv(train_esol, target_col=esol_parser.target_column)
norm_test_esol, _ = normalize_csv(test_esol, target_col=esol_parser.target_column, scaler=esol_scaler)

# %%
esol_results = train_chemberta_model(esol_parser, norm_train_esol, norm_test_esol, esol_scaler)
esol_results


# %%
# For hce
hce_defaults = {
    'train_csv': '../clustered_data/hce/train_hce.csv',
    'test_csv': '../clustered_data/hce/test_hce.csv',
    'target_column': 'pce_1',
    'smiles_column': 'smiles',
    'output_dir': '../trained_models',
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

hce_parser = argparse.Namespace(**hce_defaults)


train_hce = pd.read_csv(hce_parser.train_csv)
test_hce = pd.read_csv(hce_parser.test_csv)
norm_train_hce, hce_scaler = normalize_csv(train_hce, hce_parser.target_column)
norm_test_hce, _ = normalize_csv(test_hce, hce_parser.target_column, scaler=hce_scaler)

# %%
hce_results = train_chemberta_model(hce_parser, norm_train_hce, norm_test_hce, hce_scaler)
hce_results

# %% 
# For qm9
qm9_defaults = {
    'train_csv': '../clustered_data/qm9/train_qm9.csv',
    'test_csv': '../clustered_data/qm9/test_qm9.csv',
    'target_column': 'dga',
    'smiles_column': 'smiles',
    'output_dir': '../trained_models',
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

qm9_parser = argparse.Namespace(**qm9_defaults)

train_qm9 = pd.read_csv("../clustered_data/qm9/train_qm9.csv")
val_qm9 = pd.read_csv("../clustered_data/qm9/validation_qm9.csv")
test_qm9 = pd.read_csv('../clustered_data/qm9/test_qm9.csv')
norm_train_qm9, qm9_scaler = normalize_csv(train_qm9, target_col=qm9_parser.target_column)
norm_val_qm9, _ = normalize_csv(val_qm9, target_col=qm9_parser.target_column, scaler=qm9_scaler)
norm_test_qm9, _ = normalize_csv(test_qm9, target_col=qm9_parser.target_column, scaler=qm9_scaler)

norm_train_qm9_05 = norm_train_qm9.sample(n=int(0.005 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 0.5%
norm_train_qm9_1 = norm_train_qm9.sample(n=int(0.01 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 1%
norm_train_qm9_2 = norm_train_qm9.sample(n=int(0.02 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 2%
norm_train_qm9_5 = norm_train_qm9.sample(n=int(0.05 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 5%

# %%
qm9_results_05 = train_chemberta_model(qm9_parser, norm_train_qm9_05, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_05")
qm9_results_1 = train_chemberta_model(qm9_parser, norm_train_qm9_1, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_1")
qm9_results_2 = train_chemberta_model(qm9_parser, norm_train_qm9_2, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_2")
qm9_results_5 = train_chemberta_model(qm9_parser, norm_train_qm9_5, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_5")

# %%
# The point where improvement kind of stalls is X, so we're retraining on that percentage and testing on test set and keeping that as 
# the final model
qm9_results = train_chemberta_model(qm9_parser, norm_train_qm9_05, norm_test_qm9, qm9_scaler)