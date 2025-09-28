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
# For **all three datasets**. There is lots of repetitive code, but easiest
# and this way it's easy to go through the methods and develop them.
# %%
from pathlib import Path
import torch
import pandas as pd
import os
import sys
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tl_conversion import load_chemberta_models
from utils.tl_validation import validate_conversion, test_prediction_equivalence
import utils.tl_ablation
from utils.tl_ablation import run_ablation_analysis_with_metrics, plot_ablation_metrics
from utils.tl_regression import run_regression_lens, plot_individual_molecules_regression_lens
from utils.tl_regression import compare_molecule_groups_regression_lens, plot_group_molecules_regression_lens
from scripts.clustering import cluster

# %%
# **First for ESOL**
MODEL_PATH = "../trained_models/train_ESOL/chemberta/chemberta_model_final.bin"
TEST_PATH = "../data/test_ESOL.csv"
TRAIN_PATH = "../data/train_ESOL.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "../trained_models/train_ESOL/chemberta/normalization_scaler.pkl"
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

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, output_dir=Path(f"../results/ESOL"), n_seeds=10, scaler=scaler)

# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest and smallest target value to showcase the technique
# on the training data
min_max_molecules = [train_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0], train_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0]]

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("../results/ESOL/regression_lens"))

# %%
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
# plot_group_molecules_regression_lens(example_group_results, results_dir=Path("../results/ESOL/regression_lens")

# With clustering
molecule_groups = cluster(train_data)

group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, DEVICE)
plot_group_molecules_regression_lens(group_results, results_dir=Path("../results/ESOL/regression_lens"))



# %% 
# Now for **qm9 dataset**
MODEL_PATH = "../trained_models/train_qm9/chemberta/chemberta_model_final.bin"
TEST_PATH = "../data/test_qm9.csv"
TRAIN_PATH = "../data/train_qm9.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "../trained_models/train_qm9/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "g298_atom"
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

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, output_dir=Path(f"../results/qm9"), n_seeds=5, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest and smallest target value to showcase the technique
# on the training data
min_max_molecules = [train_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0], train_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0]]

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("../results/qm9/regression_lens"))

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
plot_group_molecules_regression_lens(group_results, results_dir=Path("../results/qm9/regression_lens"))

# %%
# Now for **hce dataset**
MODEL_PATH = "../trained_models/train_hce/chemberta/chemberta_model_final.bin"
TEST_PATH = "../data/test_hce.csv"
TRAIN_PATH = "../data/train_hce.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = "../trained_models/train_hce/chemberta/normalization_scaler.pkl"
TARGET_COLUMN = "pce_1"
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

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, output_dir=Path(f"../results/hce"), n_seeds=5, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest and smallest target value to showcase the technique
# on the training data
min_max_molecules = [train_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0], train_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0]]

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=Path("../results/hce/regression_lens"))

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
plot_group_molecules_regression_lens(group_results, results_dir=Path("../results/hce/regression_lens"))

# TODO: activation patching, see thesis repo
