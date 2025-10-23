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
#
# Essentially, this is just a file which runs the techniques shown in the paper
# For **all three datasets**. There is lots of repetitive code, but 
# this way it's easy to go through the methods and develop them.
# %%
from pathlib import Path
import torch
import pandas as pd
import os
import sys

from utils.tl_conversion import load_chemberta_models
from utils.tl_validation import validate_conversion, test_prediction_equivalence
from utils.tl_ablation import run_ablation_analysis_with_metrics, plot_ablation_metrics
from utils.tl_regression import run_regression_lens, plot_individual_molecules_regression_lens
from utils.tl_regression import compare_molecule_groups_regression_lens, plot_group_molecules_regression_lens

# %%
# For ESOL
MODEL_PATH = "trained_models/train_esol/chemberta/chemberta_model_final.bin"
FULL_PATH = "clustered_data/esol/esol.csv"
TEST_PATH = "clustered_data/esol/test_esol.csv"
TRAIN_PATH = "clustered_data/esol/train_esol.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "trained_models/train_esol/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "solubility"
print(DEVICE)

# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO" # arbitrary
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

esol_results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=Path("results/esol"), n_seeds=10, scaler=scaler)
plot_ablation_metrics(esol_results, Path("results/esol"))

# %%
import pickle

# Load saved ablation results
with open("results/esol/ablation/all_results.pkl", "rb") as f:
    esol_results = pickle.load(f)
    
plot_ablation_metrics(esol_results, Path("results/esol"))

# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]
min_max_median_molecules = [
    full_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # max
    median_molecule,  # median
    full_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # min
]
# Get actual target values for these molecules
actual_targets = [
    full_data.nlargest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # max value
    full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN],  # median value
    full_data.nsmallest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # min value
]
min_max_median_molecules

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
print(results)
plot_individual_molecules_regression_lens(results, results_dir=Path("results/esol/example_regression_lens"), molecule_labels = ["Molecule 1", "Molecule 2", "Molecule 3"], actual_targets=actual_targets, target_labels=["maximum", "median", "minimum"])
# %% [markdown]
# Now we do regression lens on groups of molecules
# First example group
# example_molecule_groups = {
#     "Simple Alcohols": ["CCO", "CC(C)O", "CCCO"],
#     "Aromatic": ["c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1"],  
#     "Carboxylic Acids": ["CC(=O)O", "CCC(=O)O", "c1ccc(C(=O)O)cc1"],
#     "Alkanes": ["CC", "CCC", "CCCCCCCCCC"]
# }
# example_group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, example_molecule_groups, tokenizer, DEVICE)
# plot_group_molecules_regression_lens(example_group_results, results_dir=Path("results/ESOL/example_regression_lens"))

# With clustering
molecule_groups = {f"Cluster {cluster + 1}": group['smiles'].tolist() 
                   for cluster, group in full_data.groupby('cluster')}

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
for name, results in group_results.items():
    print(name)
plot_group_molecules_regression_lens(group_results, results_dir=Path("results/esol/regression_lens"))


# %% 
# Now for **qm9 dataset**
MODEL_PATH = "trained_models/train_qm9_1/chemberta/chemberta_model_final.bin"
FULL_PATH = "clustered_data/qm9/qm9.csv"
TEST_PATH = "clustered_data/qm9/test_qm9.csv"
TRAIN_PATH = "clustered_data/qm9/train_qm9.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "trained_models/train_qm9_1/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "dga"
print(DEVICE)
# %%
full_data = pd.read_csv(FULL_PATH)
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

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=Path("results/qm9"), n_seeds=10, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
import pickle
with open("results/qm9_1/ablation/all_results.pkl", "rb") as f:
    qm9_results = pickle.load(f)
    
plot_ablation_metrics(qm9_results, Path("results/qm9_1"), title = "QM9")

# %%
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]
min_max_median_molecules = [
    full_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # max
    median_molecule, # median
    full_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # min
]
# Get actual target values for these molecules
actual_targets = [
    full_data.nlargest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # max value
    full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN],  # median value
    full_data.nsmallest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # min value
]
min_max_median_molecules

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("results/qm9_1/example_regression_lens"), molecule_labels = ["Molecule 4", "Molecule 5", "Molecule 6"], y_label = "Gibbs Free Energies of Atomization At 298K", title = "QM9", actual_targets=actual_targets, target_labels=["maximum", "median", "minimum"])

# %% 
molecule_groups = {f"Cluster {cluster + 1}": group['smiles'].tolist() 
                   for cluster, group in full_data.groupby('cluster')}

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
plot_group_molecules_regression_lens(group_results, results_dir=Path("results/qm9_1/regression_lens"), mean_y_label = "Mean Gibbs Free Energies of Atomization At 298K", var_y_label = "Variance Gibbs Free Energies of Atomization At 298K", title = "QM9")

# %%
# Now for **hce dataset**
MODEL_PATH = "trained_models/train_hce/chemberta/chemberta_model_final.bin"
FULL_PATH = "clustered_data/hce/hce.csv"
TEST_PATH = "clustered_data/hce/test_hce.csv"
TRAIN_PATH = "clustered_data/hce/train_hce.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "trained_models/train_hce/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "pce_1"
print(DEVICE)
# %%
full_data = pd.read_csv(FULL_PATH)
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

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=Path("results/hce"), n_seeds=10, scaler=scaler)

plot_ablation_metrics(results, Path("results/hce"))
# %% 
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
import pickle
with open("results/hce/ablation/all_results.pkl", "rb") as f:
    qm9_results = pickle.load(f)
    
plot_ablation_metrics(qm9_results, Path("results/hce"), title = "HCE")

# %%
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]
min_max_median_molecules = [
    full_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # max
    median_molecule, # median
    full_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # min
]
# Get actual target values for these molecules
actual_targets = [
    full_data.nlargest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # max value
    full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN],  # median value
    full_data.nsmallest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # min value
]
min_max_median_molecules

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("results/hce/example_regression_lens"), molecule_labels = ["Molecule 7", "Molecule 8", "Molecule 9"], y_label = "Power Conversion Efficiency", title = "HCE", actual_targets=actual_targets, target_labels=["maximum", "median", "minimum"])

# %% 
molecule_groups = {f"Cluster {cluster + 1}": group['smiles'].tolist() 
                   for cluster, group in full_data.groupby('cluster')}

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
plot_group_molecules_regression_lens(group_results, results_dir=Path("results/hce/regression_lens"), mean_y_label = "Mean Power Conversion Efficiency", var_y_label = "Variance Power Conversion Efficiency", title = "HCE")


# %%

# TODO: activation patching, see thesis repo
