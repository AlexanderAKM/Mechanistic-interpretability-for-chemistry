"""Ablation analysis module for ChemBERTa interpretability.

This module provides systematic ablation studies to understand the importance
of different model components (MLP neurons, attention heads) by measuring 
performance degradation when they are removed or zeroed out.

from utils.tl_ablation import run_ablation_analysis_with_metrics

# Assuming you have true target values for your test molecules
results = run_ablation_analysis_with_metrics(
    tl_model=tl_encoder,
    regressor=tl_regressor,
    test_molecules=test_smiles,
    true_targets=test_targets,  
    tokenizer=tokenizer,
    ablation_percentages=[0.0, 0.2, 0.5, 0.8],
    n_seeds=3,
    output_dir=Path("results"),
)


# Results now include metric degradation and combined ablation:
print(f"50% MLP ablation MAE: {results['mlp_ablation'][0.5]['mean_mae_denorm']:.4f}")
print(f"50% attention ablation R²: {results['attention_ablation'][0.5]['mean_r2_denorm']:.4f}")
print(f"50% combined ablation MAE: {results['combined_ablation'][0.5]['mean_mae_denorm']:.4f}")
```

Key functions:
- ablate_neurons_by_percentage(): Remove random percentages of MLP neurons 
- ablate_attention_heads_by_percentage(): Remove random percentages of attention heads
- ablate_both_components_by_percentage(): Remove percentages of both MLP neurons AND attention heads
- run_ablation_analysis_with_metrics(): Comprehensive ablation study with metrics
- plot_ablation_metrics(): Visualize metric degradation vs ablation percentage
"""

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

from .tl_conversion import FaithfulTLRegressor

def ablate_neurons_by_percentage(
    tl_model: tl.HookedEncoder,
    ablation_percentage: float,
    layers_to_ablate: Optional[List[int]] = None,
    seed: int = 42
) -> tl.HookedEncoder:
    """Create a copy of the model with a percentage of MLP neurons ablated.
    
    Args:
        tl_model: The TransformerLens model to ablate
        ablation_percentage: Percentage of neurons to ablate (0.0 to 1.0)
        layers_to_ablate: Specific layers to ablate, if None ablates all layers
        seed: Random seed for reproducible ablation
        
    Returns:
        New model with specified neurons ablated
    """
    # Create a deep copy to avoid modifying original
    ablated_model = tl.HookedEncoder(tl_model.cfg).to(tl_model.cfg.device)
    ablated_model.load_state_dict(tl_model.state_dict())
    
    if layers_to_ablate is None:
        layers_to_ablate = list(range(tl_model.cfg.n_layers))
    
    random.seed(seed)
    
    for layer_idx in layers_to_ablate:
        # Get MLP block
        mlp_block = ablated_model.blocks[layer_idx].mlp
        
        # Calculate number of neurons to ablate
        d_mlp = mlp_block.W_in.shape[1]  # Number of neurons in this layer
        n_ablate = int(d_mlp * ablation_percentage)
        
        if n_ablate > 0:
            # Randomly select neurons to ablate
            neurons_to_ablate = random.sample(range(d_mlp), n_ablate)
            
            # Zero out weights and biases for selected neurons
            # This guarantees the neurons_to_ablate units have activations of 0.
            # We *should not* ablate b_out. These are not tied to any specific hidden neuron.
            with torch.no_grad():
                mlp_block.W_in[:, neurons_to_ablate] = 0.0
                mlp_block.b_in[neurons_to_ablate] = 0.0  # Per-neuron bias
                mlp_block.W_out[neurons_to_ablate, :] = 0.0
 
    return ablated_model


def ablate_attention_heads_by_percentage(
    tl_model: tl.HookedEncoder,
    ablation_percentage: float,
    layers_to_ablate: Optional[List[int]] = None,
    seed: int = 42
) -> tl.HookedEncoder:
    """Create a copy of the model with a percentage of attention heads ablated.
    
    Args:
        tl_model: The TransformerLens model to ablate
        ablation_percentage: Percentage of attention heads to ablate (0.0 to 1.0)
        layers_to_ablate: Specific layers to ablate, if None ablates all layers
        seed: Random seed for reproducible ablation
        
    Returns:
        New model with specified attention heads ablated
    """
    # Create a deep copy to avoid modifying original
    ablated_model = tl.HookedEncoder(tl_model.cfg).to(tl_model.cfg.device)
    ablated_model.load_state_dict(tl_model.state_dict())
    
    if layers_to_ablate is None:
        layers_to_ablate = list(range(tl_model.cfg.n_layers))
    
    random.seed(seed)
    
    for layer_idx in layers_to_ablate:
        # Get attention block
        attn_block = ablated_model.blocks[layer_idx].attn
        n_heads = tl_model.cfg.n_heads
        
        # Calculate number of heads to ablate
        n_ablate = int(n_heads * ablation_percentage)
        
        if n_ablate > 0:
            # Randomly select heads to ablate
            heads_to_ablate = random.sample(range(n_heads), n_ablate)
            
            # Zero out weights for selected heads
            # Similarly, we shouldn't zero the output bias b_O
            with torch.no_grad():
                attn_block.W_Q[heads_to_ablate, :, :] = 0.0
                attn_block.W_K[heads_to_ablate, :, :] = 0.0  
                attn_block.W_V[heads_to_ablate, :, :] = 0.0
                attn_block.W_O[heads_to_ablate, :, :] = 0.0
                attn_block.b_Q[heads_to_ablate, :] = 0.0
                attn_block.b_K[heads_to_ablate, :] = 0.0
                attn_block.b_V[heads_to_ablate, :] = 0.0
    
    return ablated_model