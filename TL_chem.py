# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
# # ChemBERTa × TransformerLens: mechanistic interpretability notebook
#
# **Goal** – Load a fine‑tuned ChemBERTa checkpoint, port its encoder into
# [TransformerLens](https://github.com/neelnanda‑io/TransformerLens) (TL),
# validate it is functionally the same as the original model, and run a
# round of mechanistic interpretability analyses and visualizations
# (neuron ablations and regression-lens probes).  
# This notebook is used for development. After a technique works, it is moved to
# an independent Python file in utils/ and imported to ensure modularity.

#%%
from pathlib import Path
import torch
import seaborn as sns
import pandas as pd
import argparse

from utils.tl_conversion import load_chemberta_models
from utils.tl_validation import validate_conversion, test_prediction_equivalence
from utils.tl_ablation import run_ablation_analysis_with_metrics
from utils.tl_regression import run_regression_lens, plot_individual_molecules_regression_lens
from utils.tl_regression import compare_molecule_groups_regression_lens, plot_group_molecules_regression_lens
from utils.chemberta_workflows import train_chemberta_model
from utils.normalizing import normalize_csv
from clustering import cluster
# %%
sns.set_theme(context="notebook", style="whitegrid")

MODEL_PATH = "trained_models/train_ESOL/chemberta/chemberta_model_final.bin"
TEST_PATH = "data/test_ESOL.csv"
TRAIN_PATH = "data/train_ESOL.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "trained_models/train_ESOL/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "measured log solubility in mols per litre"
print(DEVICE)
# %%
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results["final_output"] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results["is_equivalent"]}")
# %% [markdown]
# Let's run ablation studies to see the effect of misssing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['smiles'].to_list()
targets = test_data[TARGET_COLUMN].to_list()

print(f"Testing ablation on {len(test_molecules)} molecules")
print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, output_dir=Path(f"results/ESOL"), n_seeds=2, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest and smallest target value to showcase the technique
# on the training data
min_max_molecules = [train_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0], train_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0]]

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("results/ESOL/regression_lens"))

# %% [markdown]
# Now we do regression lens on groups of molecules
# First example group
# example_molecule_groups = {
#     "Simple Alcohols": ["CCO", "CC(C)O", "CCCO"],
#     "Aromatic": ["c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1"],  
#     "Carboxylic Acids": ["CC(=O)O", "CCC(=O)O", "c1ccc(C(=O)O)cc1"],
#     "Alkanes": ["CC", "CCC", "CCCCCCCCCC"]
# }

# With clustering
molecule_groups = cluster(train_data)

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
plot_group_molecules_regression_lens(group_results, results_dir=Path("results/ESOL/regression_lens"))



# %% [markdown] 
# Now we try to explore the above techniques on two other datasets. We'll first try this on the HCE dataset:
# https://github.com/aspuru-guzik-group/Tartarus/blob/main/datasets/hce.csv we'll be predicting pce_1
# We'll also do it on the qm9 dataset which is nicely normally distributed. 
# Same for hce although there's a spike at 0, should be fine I think
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results["final_output"] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results["is_equivalent"]}")
# %% [markdown]
# Let's run ablation studies to see the effect of misssing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['smiles'].to_list()
targets = test_data[TARGET_COLUMN].to_list()

print(f"Testing ablation on {len(test_molecules)} molecules")
print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, output_dir=Path(f"Results/ESOL"), n_seeds=2, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest and smallest target value to showcase the technique
# on the training data
min_max_molecules = [train_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0], train_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0]]

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_molecules, tokenizer)
results
plot_individual_molecules_regression_lens(results)

# %% [markdown]
# Now we do regression lens on groups of molecules
molecule_groups = {
    "Simple Alcohols": ["CCO", "CC(C)O", "CCCO"],
    "Aromatic": ["c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1"],  
    "Carboxylic Acids": ["CC(=O)O", "CCC(=O)O", "c1ccc(C(=O)O)cc1"],
    "Alkanes": ["CC", "CCC", "CCCCCCCCCC"]
}

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
plot_group_molecules_regression_lens(group_results)
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


# %%
train_esol = pd.read_csv("data/train_ESOL.csv")
test_esol = pd.read_csv("data/test_ESOL.csv")
norm_train_esol, esol_scaler = normalize_csv(train_esol, target_col=esol_parser.target_column)
norm_test_esol, _ = normalize_csv(test_esol, target_col=esol_parser.target_column, scaler=esol_scaler)

# %%
#esol_results = train_chemberta_model(esol_parser, norm_train_esol, norm_test_esol, esol_scaler)
#esol_results

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
# %%

hce = pd.read_csv("data/test_hce.csv")
for i in range(len(hce["pce_1"])):
    print(hce["pce_1"][i])
# %%
esol = pd.read_csv("data/delaney-post-processed.csv")
len(esol)

# %%
# TODO: activation patching, see thesis repo
