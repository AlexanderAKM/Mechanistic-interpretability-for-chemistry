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
from sklearn.metrics import r2_score
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
        targets: List[float],
        results_dir: Optional[str] = None,
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
        targets: Optional list of actual target values (same order as concatenated smiles)
        results_dir: Optional directory to save predictions CSV
        device: Device to use for computation
        batch_size: Number of molecules to process at once (default: 64)
        
    Returns:
        Dictionary with layer-wise predictions for each molecule group, 
        plus 'variance_ratio' key with variance ratios if targets provided
    """
    results = {}
    all_smiles_ordered = []
    
    for group, smiles in group_smiles.items():
        print(f"Processing {group}: {len(smiles)} molecules...")
        group_results = run_regression_lens(tl_model, regressor, scaler, smiles, tokenizer, device, batch_size)
        results[group] = group_results
        all_smiles_ordered.extend(smiles)

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
        
    # Collect all predictions across all groups for each layer
    all_predictions_by_layer = {}
    for layer in layer_names:
        all_preds = []
        for group, smiles in group_smiles.items():
            for smile in smiles:
                all_preds.append(results[group][smile][layer])
        all_predictions_by_layer[layer] = np.array(all_preds)
    
    # Compute variance ratio
    target_variance = np.var(targets)
    variance_ratios = {}
    for layer in layer_names:
        pred_variance = np.var(all_predictions_by_layer[layer])
        variance_ratios[layer] = pred_variance / target_variance

    print("Targets (first 20):", targets[:20])
    print(f"max target and min: {min(targets), max(targets)}")
    for layer in layer_names:
        print(f"max and min are {max(all_predictions_by_layer[layer]), min(all_predictions_by_layer[layer])}")
        print(f"Layer {layer} predictions (first 20):", all_predictions_by_layer[layer][:20])
    
    # Compute R^2 per layer
    r2_scores = {}
    for layer in layer_names:
            r2_scores[layer] = r2_score(targets, all_predictions_by_layer[layer])
    
    results["target_variance"] = target_variance
    results["variance_ratio"] = variance_ratios
    results["r2_scores"] = r2_scores
    print(f"Variance ratios by layer: {variance_ratios}")
    print(f"R² scores by layer: {r2_scores}")
    
    # Save predictions to CSV if directory provided
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
            
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

    # Define special keys that are not group data
    special_keys = {"variance_ratio", "target_variance", "r2_scores"}
    
    # Determine layer order from the first group's mean dict (skip special keys)
    first_group = next(iter({k: v for k, v in results.items() if k not in special_keys}.values()))
    layer_names = list(first_group["mean"].keys())
    
    # Set up continuous color palette for all groups (excluding special keys)
    group_items = [(k, v) for k, v in results.items() if k not in special_keys]
    n_groups = len(group_items)
    colors = plt.cm.turbo(np.linspace(0, 1, n_groups))

    # Plot group means
    plt.figure(figsize=(12, 8))
    for i, (group_name, group_data) in enumerate(group_items):
        mean_values = [group_data["mean"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), mean_values, 'o-', alpha=0.8, label=group_name, color=colors[i])

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
    for i, (group_name, group_data) in enumerate(group_items):
        std_values = [group_data["variance"][layer] for layer in layer_names]
        plt.plot(range(len(layer_names)), std_values, 'o-', alpha=0.8, label=group_name, color=colors[i])

    plt.title(title, fontsize=18)
    plt.ylabel(var_y_label, fontsize=16)
    plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "group_vars.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot variance ratio if it exists in results
    if "variance_ratio" in results:
        variance_ratios = results["variance_ratio"]
        ratios = [variance_ratios[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(layer_names)), ratios, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        
        # Add horizontal line at y=1 to show where variance equals target variance
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Equal Variance')
        
        plt.title(f"{title} - Variance Ratio Across Layers", fontsize=18)
        plt.ylabel("Variance Ratio (Predictions / Targets)", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance ratio times R^2 if it exists in results
    if "variance_ratio" and "r2_scores" in results:
        variance_ratios = results["variance_ratio"]
        r2_scores = results["r2_scores"]
        ratios = [variance_ratios[layer] * r2_scores[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(layer_names)), ratios, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        
        # Add horizontal line at y=1 to show where variance equals target variance
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Equal Variance')
        
        plt.title(f"{title} - Variance Ratio Across Layers", fontsize=18)
        plt.ylabel("Variance Ratio (Predictions / Targets) x R^2", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance and R^2 if it exists in results
    if "variance_ratio" and "r2_scores" in results:
        variance_ratios = results["variance_ratio"]
        r2_scores = results["r2_scores"]
        
        # Extract values in the same order as layer_names
        variance_ratio_values = [variance_ratios[layer] for layer in layer_names]
        r2_values = [r2_scores[layer] for layer in layer_names]
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.turbo(np.linspace(0, 1, 2))

        plt.plot(range(len(layer_names)), variance_ratio_values, 'o-', linewidth=2, markersize=10, color=colors[0], label='Variance Ratio')
        plt.plot(range(len(layer_names)), r2_values, 'o-', linewidth=2, markersize=10, color=colors[1], label='R²')

        # Add horizontal line at y=1 to show where variance equals target variance
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Equal Variance')
        
        plt.title(f"{title} - Variance and R2 Across Layers", fontsize=18)
        plt.ylabel("Variance Ratio and R^2", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "variance_ratio_and_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot variance ratio × R² for each cluster/group
    if "variance_ratio" and "r2_scores" and "target_variance" in results:
        r2_scores = results["r2_scores"]
        r2_values = [r2_scores[layer] for layer in layer_names]
        target_variance = results["target_variance"]
        
        plt.figure(figsize=(12, 8))
        
        # Recreate colors array for groups (was overwritten earlier)
        group_colors = plt.cm.turbo(np.linspace(0, 1, n_groups))
        
        # Plot variance ratio × R² for each group
        for i, (group_name, group_data) in enumerate(group_items):
            # Calculate variance ratio for this group: group variance / target variance
            variance_ratios = [group_data["variance"][layer] / target_variance for layer in layer_names]
            variance_ratio_times_r2 = [var_ratio * r2 for var_ratio, r2 in zip(variance_ratios, r2_values)]
            plt.plot(range(len(layer_names)), variance_ratio_times_r2, 'o-', alpha=0.8, 
                    label=group_name, color=group_colors[i], linewidth=2, markersize=8)
        
        plt.title(f"{title} - Group Variance Ratio × R² Across Layers", fontsize=18)
        plt.ylabel("Variance Ratio × R²", fontsize=16)
        plt.xticks(range(len(layer_names)), x_axis_labels, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(results_dir) / "group_variance_ratio_times_R2.pdf", dpi=300, bbox_inches="tight")
        plt.close()
