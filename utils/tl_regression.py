"""Regression lens analysis module for ChemBERTa interpretability.

This module implements "regression lens" analysis by applying the final readout layer
after each transformer block to see what predictions the model would make at
each intermediate layer. This reveals how the model's "thinking" evolves through
the layers.

Key functions:
- run_regression_lens(): Apply readout after each layer for one or more molecules
- plot_regression_lens_results(): Visualize prediction changes through layers
- compare_molecules_regression_lens(): Compare regression lens patterns across molecules
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import transformer_lens as tl
from transformers import RobertaTokenizerFast

from .tl_conversion import FaithfulTLRegressor


def run_regression_lens(
        tl_model: tl.HookedEncoder,
        regressor: FaithfulTLRegressor,
        scaler,
        smiles: List[str],
        tokenizer: RobertaTokenizerFast,
        device: Optional[str] = None,
        batch_size: int = 32,
) -> Dict:
    """Run regression lens analysis for one or more molecules (batched for efficiency).
    
    Applies the readout layer after each transformer block to see what
    the model would predict based on representations at each layer.
    
    Args:
        tl_model: TransformerLens encoder model
        regressor: Full regressor with MLP head for readout
        scaler: Scaler for denormalization
        smiles: List of SMILES strings to analyze
        tokenizer: Tokenizer for the model
        device: Device to use for computation
        batch_size: Number of molecules to process at once (default: 32)
        
    Returns:
        Dictionary with layer-wise predictions for each molecule
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    
    # Process in batches for efficiency
    for batch_start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[batch_start:batch_start + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # Run model with cache for entire batch
            _, cache = tl_model.run_with_cache(
                inputs["input_ids"],
                one_zero_attention_mask=inputs["attention_mask"]
            )

            # Process each molecule in batch
            for batch_idx, smile in enumerate(batch_smiles):
                result = {}
                
                for layer in range(tl_model.cfg.n_layers + 1):
                    if layer == 0:
                        # After embedding but before any transformer blocks
                        representation = cache["hook_full_embed"][batch_idx, 0, :]
                        layer_name = "Embedding"
                    else:
                        representation = cache[f"blocks.{layer-1}.hook_normalized_resid_post"][batch_idx, 0, :]
                        layer_name = f"{layer}"
                
                    norm_prediction = regressor.mlp_head(representation).squeeze().item()
                    prediction = norm_prediction * scaler.scale_[0] + scaler.mean_[0]

                    result[layer_name] = prediction

                results[smile] = result
    
    return results

def compare_molecule_groups_regression_lens(
        tl_model: tl.HookedEncoder,
        regressor: FaithfulTLRegressor,
        scaler,
        group_smiles: Dict,
        tokenizer: RobertaTokenizerFast,
        device: Optional[str] = None,
        batch_size: int = 64,
) -> Dict:
    """Run regression lens analysis for one or more molecule groups (batched).
    
    Applies the readout layer after each transformer block to see what
    the model would predict based on representations at each layer.
    
    Args:
        tl_model: TransformerLens encoder model
        regressor: Full regressor with MLP head for readout
        scaler: Scaler for denormalization
        group_smiles: Dictionary mapping group names to lists of SMILES
        tokenizer: Tokenizer for the model
        device: Device to use for computation
        batch_size: Number of molecules to process at once (default: 64)
        
    Returns:
        Dictionary with layer-wise predictions for each molecule group
    """
    results = {}

    for group, smiles in group_smiles.items():
        print(f"Processing {group}: {len(smiles)} molecules...")
        group_results = run_regression_lens(tl_model, regressor, scaler, smiles, tokenizer, device, batch_size)
        results[group] = group_results

        # Compute per-layer mean and std across all molecules in the group
        layer_names = list(next(iter(group_results.values())).keys())
        means = {}
        variances = {}

        for layer in layer_names:
            values = np.array([group_results[smile][layer] for smile in smiles], dtype=np.float64)
            means[layer] = np.mean(values)
            variances[layer] = np.var(values)

        results[group]["mean"] = means
        results[group]["variance"] = variances
    
    return results


def plot_individual_molecules_regression_lens(
        results: dict,
        results_dir: str = "results/regression_lens",
        x_axis_labels: list = ["After Embedding \n Layer", "After Transformer \n Layer 1", "After Transformer \n Layer 2", "After Transformer \n Layer 3"],
        molecule_labels: list = ["Molecule 0", "Molecule 1", "Molecule 2"],
        y_label: str = "Log Solubility",
        title: str = "ESOL",
        actual_targets: Optional[List[float]] = None,
        target_labels: Optional[List[str]] = None
):
    """Plot regression lens results for individual molecules.
    
    Args:
        results: Dictionary mapping SMILES to layer-wise predictions
        results_dir: Directory to save the plot
        x_axis_labels: Labels for x-axis (layers)
        molecule_labels: Labels for each molecule
        y_label: Label for y-axis
        title: Plot title
        actual_targets: Optional list of actual target values for each molecule (in same order as results)
        target_labels: Optional list of labels for target types (e.g., ["max", "median", "min"])
    """
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    
    # Plot predictions for each molecule
    for i, (smile, smile_results) in enumerate(results.items()):
        # Add target label if provided
        label = molecule_labels[i]
        if target_labels and i < len(target_labels):
            label = f"{molecule_labels[i]} ({target_labels[i]})"
        
        plt.plot(range(len(smile_results)), smile_results.values(), 'o-', alpha=0.7, label=label)
        
        # Add dashed horizontal line for actual target value if provided
        if actual_targets and i < len(actual_targets):
            plt.axhline(y=actual_targets[i], color=f'C{i}', linestyle='--', alpha=0.5, linewidth=2)

    plt.title(title, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(range(len(smile_results)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "individual_molecules.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_molecules_regression_lens(
        results: dict,
        results_dir: str = "results/regression_lens",
        x_axis_labels: list = ["After Embedding \n Layer", "After Transformer \n Layer 1", "After Transformer \n Layer 2", "After Transformer \n Layer 3"],
        mean_y_label: str = "Mean Log Solubility",
        var_y_label: str = "Variance Log Solubility",
        title: str = "ESOL",
):
    os.makedirs(results_dir, exist_ok=True)

    # Determine layer order from the first group's mean dict
    first_group = next(iter(results.values()))
    layer_names = list(first_group["mean"].keys())

    # Plot group means
    plt.figure(figsize=(12, 8))
    for group_name, group_data in results.items():
        mean_values = [group_data["mean"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), mean_values, 'o-', alpha=0.8, label=group_name)

    plt.title(title, fontsize=18)
    plt.ylabel(mean_y_label, fontsize=16)
    plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "group_means.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot group variances
    plt.figure(figsize=(12, 8))
    for group_name, group_data in results.items():
        std_values = [group_data["variance"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), std_values, 'o-', alpha=0.8, label=group_name)

    plt.title(title, fontsize=18)
    plt.ylabel(var_y_label, fontsize=16)
    plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "group_vars.pdf", dpi=300, bbox_inches="tight")
    plt.close()