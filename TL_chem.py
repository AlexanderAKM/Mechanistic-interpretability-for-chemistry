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
# (activation patching, neuron ablations, logit-lens probes,
# attention maps, residual‑stream norms).  
# This notebook is used for development. After a technique works, it is moved to
# an independent Python file in utils/ and imported to ensure modularity.

#%%
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import transformer_lens as tl

from utils.tl_conversion import load_chemberta_models
from utils.tl_validation import validate_conversion, test_prediction_equivalence


# %%
sns.set_theme(context="notebook", style="whitegrid")

MODEL_PATH = "trained_models/ESOL/chemberta/chemberta_model_final.bin"
TEST_PATH = "data/test_ESOL.csv"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZATION_PIPELINE_PATH = "trained_models/ESOL/chemberta/normalization_pipeline.pkl"
TARGET_COLUMN = "measured log solubility in mols per litre"
print(DEVICE)
# %%
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, normalization_pipeline = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, NORMALIZATION_PIPELINE_PATH
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, normalization_pipeline)
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
# %%
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl
from transformers import RobertaTokenizerFast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.tl_conversion import FaithfulTLRegressor
from utils.tl_ablation import ablate_attention_heads_by_percentage, ablate_neurons_by_percentage
from utils.tl_validation import run_evaluation_metrics
# %%
# hf_metrics = run_evaluation_metrics(hf_regressor, test_data, tokenizer, use_tl_model=False, normalization_pipeline=normalization_pipeline)
# tl_metrics = run_evaluation_metrics(tl_regressor, test_data, tokenizer, use_tl_model=True, normalization_pipeline=normalization_pipeline)
# hf_metrics
# %% [markdown]
# Let's run regression lens to see how the prediction of models changes over time

print(tl_metrics)
print(hf_metrics)

# %%
ablated = ablate_attention_heads_by_percentage(tl_encoder, 1.0)
ablated = ablate_neurons_by_percentage(ablated, 1.0)
ablated_regressor = FaithfulTLRegressor(ablated, tl_regressor.mlp_head)
ablated_metrics = run_evaluation_metrics(ablated_regressor, test_data, tokenizer, use_tl_model=True)
# %%
normalization_pipeline
ablated_metrics
# %% [markdown]
# TODO: activation patching, see thesis repo

# %%
