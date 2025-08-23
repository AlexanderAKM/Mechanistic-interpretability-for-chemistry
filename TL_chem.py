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
import json
import transformer_lens as tl

from utils.tl_conversion import load_chemberta_models
from utils.tl_validation import validate_conversion


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
hf_encoder, tl_encoder, tokenizer, hf_regressor, normalization_pipeline = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, NORMALIZATION_PIPELINE_PATH, TARGET_COLUMN
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, normalization_pipeline)
# %% [markdown]
# Validating conversation (check whether the two models have the same output, extremely important!)
# %%
test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results["final_output"] < 0.001}")