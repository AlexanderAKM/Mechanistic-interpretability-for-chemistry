# tl_validation.py - Validation and testing utilities for faithful TL conversion
"""
This module provides comprehensive validation and testing functions for 
verifying that TransformerLens models are faithful to their HuggingFace counterparts.

Key functions:
- validate_conversion(): Layer-by-layer equivalence checking
- run_evaluation_metrics(): Performance evaluation (RMSE, R²)
- test_prediction_equivalence(): End-to-end prediction comparison
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
import transformer_lens as tl

from .tl_conversion import load_chemberta_models, FaithfulTLRegressor
from .chemberta_dataset import ChembertaDataset
from models.chemberta_regressor import ChembertaRegressorWithFeatures


def validate_conversion(hf_model: RobertaModel, tl_model: tl.HookedEncoder, 
                       test_input_ids: torch.Tensor, test_attention_mask: torch.Tensor) -> Dict[str, float]:
    """Verify that the TL model produces identical outputs to the HF model.
    
    Args:
        hf_model: Original HuggingFace RoBERTa model
        tl_model: Converted TransformerLens model
        test_input_ids: Test input token IDs
        test_attention_mask: Test attention mask
        
    Returns:
        Dictionary with maximum absolute differences at each layer
    """
    with torch.no_grad():
        # Run both models
        hf_out = hf_model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        _, tl_cache = tl_model.run_with_cache(
            test_input_ids,
            one_zero_attention_mask=test_attention_mask
        )
        
        # Also get the encoder output for final comparison
        tl_encoder_out = tl_model.encoder_output(
            test_input_ids,
            test_attention_mask
        )
    # Compare outputs layer by layer
    diffs = {}
    
    # Embedding layer
    embed_diff = (tl_cache["hook_full_embed"] - hf_out.hidden_states[0]).abs().max().item()
    diffs["embedding"] = embed_diff
    
    # Each transformer layer (important to specify after final LN for TL)
    for l in range(len(hf_model.encoder.layer)):
        layer_diff = (tl_cache[f"blocks.{l}.hook_normalized_resid_post"] - hf_out.hidden_states[l + 1]).abs().max().item()
        diffs[f"layer_{l}"] = layer_diff
    
    # Final encoder output (most important for interpretability)
    final_diff = (tl_encoder_out - hf_out.last_hidden_state).abs().max().item()
    diffs["final_output"] = final_diff
    
    return diffs


def run_evaluation_metrics(model_path: str, test_csv: str, tokenizer_name: str,
                          smiles_col: str = "smiles", target_col: str = "target", 
                          batch_size: int = 64, device: Optional[str] = None,
                          use_tl_model: bool = False,
                          normalization_pipeline: Optional[Dict] = None,
                          target_column: Optional[str] = None) -> Dict[str, float]:
    """Run evaluation metrics (RMSE, R²) on test data.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to test CSV file
        tokenizer_name: Name of the tokenizer
        smiles_col: Name of SMILES column
        target_col: Name of target column
        batch_size: Batch size for evaluation
        device: Device to use
        use_tl_model: Whether to use TL model instead of HF model
        normalization_pipeline: Normalization pipeline containing training set statistics
        target_column: Target column name for normalization pipeline lookup
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    
    # Load test data
    df = pd.read_csv(test_csv)
    texts = df[smiles_col].tolist()
    labels = df[target_col].astype("float32").values
    
    # Use training set normalization parameters if available
    if (normalization_pipeline and target_column and 
        'scaler' in normalization_pipeline and 
        'numeric_cols' in normalization_pipeline):
        
        numeric_cols = normalization_pipeline['numeric_cols']
        if target_column in numeric_cols:
            target_idx = numeric_cols.index(target_column)
            scaler = normalization_pipeline['scaler']
            
            # Use training set normalization parameters
            mean = scaler.mean_[target_idx]
            std = scaler.scale_[target_idx]  # scale_ is the standard deviation
            
            print(f"Using training set normalization: mean={mean:.4f}, std={std:.4f}")
        else:
            print(f"Warning: Target column '{target_column}' not found in numeric_cols, using test set stats")
            mean, std = labels.mean(), labels.std()
    else:
        print("Warning: No normalization pipeline provided, using test set statistics")
        # Normalize targets (should match training normalization)
        mean, std = labels.mean(), labels.std()
    
    labels_norm = (labels - mean) / std
    
    ds = ChembertaDataset(texts, labels_norm, tokenizer, features=None)
    loader = DataLoader(ds, batch_size=batch_size)
    
    if use_tl_model:
        # Load TL model
        hf_encoder, tl_encoder, _, hf_regressor, _ = load_chemberta_models(model_path, tokenizer_name, device)
        
        # Create TL regressor
        tl_head = torch.nn.Linear(384, 1, bias=True).to(device).eval()
        tl_head.load_state_dict(hf_regressor.mlp.model[0].state_dict())
        model = FaithfulTLRegressor(tl_encoder, tl_head, dropout_p=hf_regressor.dropout.p).to(device).eval()
        
        def predict_fn(input_ids, attention_mask):
            return model(input_ids, attention_mask)
    else:
        # Load HF model
        model = ChembertaRegressorWithFeatures(
            pretrained=tokenizer_name,
            num_features=0,
            dropout=0.34,
            hidden_channels=384,
            num_mlp_layers=1,
        ).to(device).eval()
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        
        def predict_fn(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
    
    # Run evaluation
    preds, lbls = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            
            y_hat = predict_fn(ids, attention_mask)
            preds.append(y_hat.cpu())
            lbls.append(y.cpu())
    
    preds = torch.cat(preds).numpy()
    lbls = torch.cat(lbls).numpy()
    
    # Calculate metrics
    rmse_norm = math.sqrt(mean_squared_error(lbls, preds))
    r2_norm = r2_score(lbls, preds)
    rmse = math.sqrt(mean_squared_error(lbls * std + mean, preds * std + mean))
    r2 = r2_score(lbls * std + mean, preds * std + mean)
    
    return {
        "rmse": rmse,
        "r2": r2,
        "rmse_norm": rmse_norm,
        "r2_norm": r2_norm,
        "model_type": "TL" if use_tl_model else "HF"
    }


def test_prediction_equivalence(model_path: str, test_molecules: list[str], 
                               tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
                               device: Optional[str] = None) -> Dict[str, float]:
    """Test prediction equivalence between HF and TL models.
    
    Args:
        model_path: Path to the trained model
        test_molecules: List of SMILES strings to test
        tokenizer_name: Name of the tokenizer
        device: Device to use
        
    Returns:
        Dictionary with prediction differences
    """
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    hf_encoder, tl_encoder, tokenizer, hf_regressor, _ = load_chemberta_models(
        model_path, tokenizer_name, device
    )
    
    # Create TL regressor
    tl_head = torch.nn.Linear(384, 1, bias=True).to(device).eval()
    tl_head.load_state_dict(hf_regressor.mlp.model[0].state_dict())
    tl_regressor = FaithfulTLRegressor(tl_encoder, tl_head, dropout_p=hf_regressor.dropout.p).to(device).eval()
    
    results = {}
    max_diff = 0.0
    
    with torch.no_grad():
        for smiles in test_molecules:
            # Tokenize
            inputs = tokenizer(smiles, return_tensors="pt").to(device)
            
            # Get predictions
            # HF class was made to also output loss, hence the .logits
            hf_pred = hf_regressor(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).logits.item()
            
            tl_pred = tl_regressor(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).item()
            
            diff = abs(hf_pred - tl_pred)
            max_diff = max(max_diff, diff)
            
            results[smiles] = {
                "hf_prediction": hf_pred,
                "tl_prediction": tl_pred,
                "difference": diff
            }
    
    results["max_difference"] = max_diff
    results["is_equivalent"] = max_diff < 1e-5
    
    return results


def comprehensive_validation(model_path: str, test_csv: str, 
                           tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
                           test_molecules: Optional[list[str]] = None,
                           device: Optional[str] = None,
                           target: str = "measured log solubility in mols per litre",
                           smiles: str = "smiles") -> Dict:
    """Run comprehensive validation of TL conversion.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to test CSV file
        tokenizer_name: Name of the tokenizer
        test_molecules: Optional list of test molecules
        device: Device to use
        
    Returns:
        Dictionary with all validation results
    """
    from .tl_conversion import load_chemberta_models
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔍 Running comprehensive validation...")
    
    # Load models
    print("Loading models...")
    hf_encoder, tl_encoder, tokenizer, hf_regressor, _ = load_chemberta_models(
        model_path, tokenizer_name, device
    )
    
    # Test conversion with a simple molecule
    print("Validating conversion...")
    test_smiles = "CCO"
    inputs = tokenizer(test_smiles, return_tensors="pt").to(device)
    conversion_results = validate_conversion(
        hf_encoder, tl_encoder,
        inputs["input_ids"], inputs["attention_mask"]
    )
    
    # Test evaluation metrics
    print("Running evaluation metrics...")
    hf_metrics = run_evaluation_metrics(
        model_path, test_csv, tokenizer_name, device=device, use_tl_model=False, smiles_col=smiles, target_col=target
    )
    tl_metrics = run_evaluation_metrics(
        model_path, test_csv, tokenizer_name, device=device, use_tl_model=True, smiles_col=smiles, target_col=target
    )
    
    # Test prediction equivalence
    if test_molecules is None:
        test_molecules = ["CCO", "c1ccccc1", "CC(C)O"]
    
    print("Testing prediction equivalence...")
    prediction_results = test_prediction_equivalence(
        model_path, test_molecules, tokenizer_name, device
    )
    
    # Compile results
    results = {
        "conversion_validation": conversion_results,
        "hf_metrics": hf_metrics,
        "tl_metrics": tl_metrics,
        "prediction_equivalence": prediction_results,
        "summary": {
            "conversion_faithful": conversion_results["final_output"] < 1e-5,
            "predictions_equivalent": prediction_results["is_equivalent"],
            "metrics_difference": {
                "rmse_diff": abs(hf_metrics["rmse"] - tl_metrics["rmse"]),
                "r2_diff": abs(hf_metrics["r2"] - tl_metrics["r2"])
            }
        }
    }
    
    # Print summary
    print("\n✅ Validation Summary:")
    print(f"  Conversion faithful: {results['summary']['conversion_faithful']}")
    print(f"  Predictions equivalent: {results['summary']['predictions_equivalent']}")
    print(f"  Final output diff: {conversion_results['final_output']:.2e}")
    print(f"  Max prediction diff: {prediction_results['max_difference']:.2e}")
    print(f"  RMSE difference: {results['summary']['metrics_difference']['rmse_diff']:.6f}")
    
    return results 