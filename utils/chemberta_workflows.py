"""
Utility module to handle ChemBERTa training workflows.

This module contains functions for training ChemBERTa models with preprocessing,
training, evaluation, and explainability analysis.
"""

import os
import json
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    RobertaModel,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ..data.data_processing import get_numerical_features
from ...data_exploration.combine_and_reduce import perform_dimensionality_reduction
from ..data.inverse_transform import inverse_transform_target
from ..data.normalizing import preprocess_csv
from ..explainability.gradients import run_integrated_gradients
from ..explainability.shap import extended_shap_analysis
from ..visualization.plotting import (
    plot_errors,
    plot_losses,
    plot_predictions_vs_targets,
)
from .build_model import build_model

# Default pretrained model
DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


class L1Trainer(Trainer):
    """
    Custom trainer that adds L1 regularization to the loss function.
    """

    def __init__(self, l1_lambda=0.0, verbose=0, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda
        self.verbose_level = verbose
        self.regularized_losses = []  # Track regularized losses for plotting

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        raw_mse_loss = outputs.loss  # This is the raw MSE loss from the model

        # For training, add L1 regularization
        if self.training:
            if self.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                regularized_loss = raw_mse_loss + self.l1_lambda * l1_norm

                # Log difference between raw loss and regularized loss
                if self.verbose_level > 1:
                    print(
                        f"DEBUG - Training: Raw MSE: {raw_mse_loss.item():.6f} | L1 term: {(self.l1_lambda * l1_norm).item():.6f} | Regularized loss: {regularized_loss.item():.6f}"
                    )

                # Store the regularized loss value for history tracking
                self.regularized_losses.append(regularized_loss.item())

                loss = regularized_loss
            else:
                loss = raw_mse_loss
                self.regularized_losses.append(raw_mse_loss.item())
        else:
            # For validation, use pure MSE without regularization
            loss = raw_mse_loss
            if self.verbose_level > 1:
                print(f"DEBUG - Validation: Pure MSE loss: {loss.item():.6f}")

        return (loss, outputs) if return_outputs else loss

    def training(self):
        """Check if the model is in training mode"""
        return self.model.training if hasattr(self.model, "training") else True

    def log(self, logs, *args, **kwargs):
        """Override log method to include regularized loss in history"""
        if self.model.training and self.regularized_losses:
            logs["regularized_loss"] = self.regularized_losses[-1]
        super().log(logs, *args, **kwargs)


class ChembertaRegressorWithFeatures(nn.Module):
    """
    ChemBERTa regression model with optional numerical features.
    """

    def __init__(
        self,
        pretrained,
        num_features=0,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
        verbose=0,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        # Freeze the base language model to avoid overfitting. 
        # Currently commented for better performance.
        # TODO: Choose which parts of the model actually require finetuning and freeze all other parts.
        # for param in self.roberta.parameters():
        #     param.requires_grad = False
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.verbose_level = verbose

        # Handle the case with no numerical features
        self.has_features = num_features > 0
        if self.has_features:
            # Create the numerical branch next to ChemBERTa branch
            # TODO: EncoderMLP should be updated to reflect 
            # Linear interpolation instead.
            from scripts.modeling.models.basic.encoder_mlp import EncoderMLP

            self.num_encoder = EncoderMLP(
                input_dim=num_features,
                hidden_channels=hidden_channels,
                num_layers=num_mlp_layers,
                output_dim=hidden_size,
                dropout=dropout,
            )
            num_input_features = hidden_size * 3  # cls, feat, and product
        else:
            num_input_features = hidden_size  # just cls

        self.mlp = build_model(
            model_name="mlp",
            hidden_channels=hidden_channels,
            num_numerical_features=num_input_features,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
        )
        print("mlp architecture:")
        print(
            {
                "hidden_channels": hidden_channels,
                "num_numerical_features": num_input_features,
                "num_mlp_layers": num_mlp_layers,
                "dropout": dropout,
            }
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None):
        try:
            # Debug input shapes
            if input_ids is None:
                print("WARNING: input_ids is None in forward pass")

            # Call the base model
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            #print(f"cls_emb.shape: {cls_emb.shape}, cls_emb: {cls_emb}")

            # Handle different feature availability
            if self.has_features and features is not None:
                # Use the encoder MLP
                feat_emb = self.num_encoder(features)
                x = torch.cat([cls_emb, feat_emb, cls_emb * feat_emb], dim=1)
            else:
                x = cls_emb

            x = self.dropout(x)
            logits = self.mlp(x).squeeze(-1)
            loss = None
            if labels is not None:
                criterion = nn.MSELoss()
                raw_loss = criterion(logits, labels)
                # Print the raw MSE loss value for debugging
                if self.verbose_level > 1:
                    print(f"DEBUG - Raw MSE Loss: {raw_loss.item():.6f} | \
                           Mean logits: {logits.mean().item():.6f} | \
                           Mean labels: {labels.mean().item():.6f}")

                # This is the loss that will be used for training
                loss = raw_loss
            # We return this class of output for the Trainer to work
            # To get the last hidden state, call 
            # ChembertaRegressorWithFeatures.roberta(input_ids, attention_mask).last_hidden_state
            return SequenceClassifierOutput(loss=loss, logits=logits.unsqueeze(-1))
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            print(f"Input types: input_ids={type(input_ids)}, attention_mask={type(attention_mask)}")
            if input_ids is not None:
                print(f"Input shapes: input_ids={input_ids.shape}")
            if attention_mask is not None:
                print(f"attention_mask={attention_mask.shape}")
            if features is not None:
                print(f"Features shape: {features.shape}")
            raise


class ChembertaDataset(Dataset):
    """
    Dataset for ChemBERTa model that handles tokenized SMILES strings and numerical features.
    """

    def __init__(self, texts, targets, tokenizer, features=None):
        # Pre-tokenize all texts at initialization time
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.features = None
        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.targets[idx],
        }

        if self.features is not None and hasattr(self, "features"):
            item["features"] = self.features[idx]

        return item


def get_compute_metrics_fn(verbose_level, pipeline, target_column):
    """
    Returns a function that computes evaluation metrics based on model predictions.

    Args:
        verbose_level: Level of verbosity for logging
        pipeline: Preprocessing pipeline for inverse transforms
        target_column: Name of the target column

    Returns:
        function: A function that computes evaluation metrics
    """

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.squeeze(-1)
        labels = eval_pred.label_ids

        # Store normalized metrics (these are on the transformed/preprocessed data)
        normalized_mae = mean_absolute_error(labels, preds)
        normalized_mse = mean_squared_error(
            labels, preds
        )  # This is pure MSE without any L1 regularization
        normalized_rmse = np.sqrt(normalized_mse)
        normalized_r2 = r2_score(labels, preds)

        # Manually re-compute the MSE loss used by PyTorch
        def pytorch_mse(predictions, targets):
            return np.mean((predictions - targets) ** 2)

        pytorch_style_mse = pytorch_mse(preds, labels)

        # Calculate raw values for debugging
        mean_pred = np.mean(preds)
        mean_label = np.mean(labels)
        min_pred = np.min(preds)
        max_pred = np.max(preds)
        min_label = np.min(labels)
        max_label = np.max(labels)

        # Print diagnostic information if verbose
        if verbose_level > 1:
            print(
                f"METRICS DEBUG - MSE: {normalized_mse:.6f}, PyTorch MSE: {pytorch_style_mse:.6f}"
            )
            print(
                f"METRICS DEBUG - Predictions: mean={mean_pred:.6f}, min={min_pred:.6f}, max={max_pred:.6f}"
            )
            print(
                f"METRICS DEBUG - Labels: mean={mean_label:.6f}, min={min_label:.6f}, max={max_label:.6f}"
            )

        # Get original-scale metrics too if pipeline is available
        # Invert transform for original-scale metrics
        if pipeline is not None:
            try:
                orig_preds = inverse_transform_target(preds, pipeline, target_column)
                orig_labels = inverse_transform_target(labels, pipeline, target_column)

                # Compute regression metrics on original scale
                original_mae = mean_absolute_error(orig_labels, orig_preds)
                original_mse = mean_squared_error(orig_labels, orig_preds)
                original_rmse = np.sqrt(original_mse)
                original_r2 = r2_score(orig_labels, orig_preds)

                # Return both normalized and original metrics
                # Set the eval_loss to the pure MSE value without L1 regularization for early stopping
                return {
                    "loss": normalized_mse,  # This is pure MSE without L1 regularization
                    "mae": normalized_mae,
                    "mse": normalized_mse,
                    "pytorch_mse": pytorch_style_mse,
                    "rmse": normalized_rmse,
                    "r2": normalized_r2,
                    "original_mae": original_mae,
                    "original_mse": original_mse,
                    "original_rmse": original_rmse,
                    "original_r2": original_r2,
                    # Debugging info
                    "mean_pred": mean_pred,
                    "mean_label": mean_label,
                    "min_pred": min_pred,
                    "max_pred": max_pred,
                    "min_label": min_label,
                    "max_label": max_label,
                }
            except Exception as e:
                print(f"Warning: Error in inverse transform: {e}")

        # Return only normalized metrics if inverse transform isn't available or fails
        # Set the eval_loss to the MSE value for early stopping
        return {
            "loss": normalized_mse,  # This is pure MSE without L1 regularization
            "mae": normalized_mae,
            "mse": normalized_mse,
            "pytorch_mse": pytorch_style_mse,
            "rmse": normalized_rmse,
            "r2": normalized_r2,
            # Debugging info
            "mean_pred": mean_pred,
            "mean_label": mean_label,
            "min_pred": min_pred,
            "max_pred": max_pred,
            "min_label": min_label,
            "max_label": max_label,
        }

    return compute_metrics


def preprocess_data_for_chemberta(
    train_csv,
    test_csv,
    target_column,
    smiles_column,
    transform_type=None,
    dim_reduce=False,
    n_components=100,
    dim_method="pca",
    exclude_columns=None,
):
    """
    Load and preprocess training and test data for ChemBERTa models.

    Args:
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        target_column: Name of the target column
        smiles_column: Name of the SMILES column
        transform_type: Type of normalization to apply
        dim_reduce: Whether to perform dimensionality reduction
        n_components: Number of components for dimensionality reduction
        dim_method: Method for dimensionality reduction
        exclude_columns: Columns to exclude from numerical features

    Returns:
        tuple: (df_train_processed, df_test_processed, numeric_features, pipeline)
    """
    print(f"Loading training data from {train_csv}")
    df_train = pd.read_csv(train_csv)

    print(f"Loading test data from {test_csv}")
    df_test = pd.read_csv(test_csv)

    # Print column information
    print(f"\nTraining data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    # Ensure both datasets have the same columns
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols

    if missing_in_test:
        print(
            f"\nWARNING: Found {len(missing_in_test)} columns in training data that are missing in test data"
        )
        if len(missing_in_test) <= 10:
            print(list(missing_in_test))
        else:
            print(list(missing_in_test)[:10], "... and more")

    if missing_in_train:
        print(
            f"\nWARNING: Found {len(missing_in_train)} columns in test data that are missing in training data"
        )
        if len(missing_in_train) <= 10:
            print(list(missing_in_train))
        else:
            print(list(missing_in_train)[:10], "... and more")

    # Get common columns and use only those
    common_cols = list(train_cols.intersection(test_cols))
    print(f"Using {len(common_cols)} common columns between datasets")

    # Make sure target and SMILES columns are present in common columns
    if smiles_column not in common_cols:
        raise ValueError(f"SMILES column '{smiles_column}' not found in both datasets")

    if target_column not in common_cols:
        raise ValueError(f"Target column '{target_column}' not found in both datasets")

    # Filter datasets to common columns
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]

    # Apply normalization if requested
    if transform_type is not None:
        print("Applying normalization...")
        df_train_normalized, scaler, skewed_features, min_values, numeric_cols = (
            preprocess_csv(
                df=df_train.copy(deep=True),
                target_col=target_column,  # Pass target column to exclude it from automatic log transformation
                skewed_features=None,
                fit_scaler=True,
                transform_type=transform_type,
            )
        )

        df_test_normalized, _, _, _, _ = preprocess_csv(
            df=df_test.copy(deep=True),
            target_col=target_column,  # Pass target column to exclude it from automatic log transformation
            skewed_features=skewed_features,
            scaler=scaler,
            fit_scaler=False,
            transform_type=transform_type,
        )
    else:
        print("Skipping normalization as per user settings")
        df_train_normalized = df_train.copy()
        df_test_normalized = df_test.copy()
        scaler = None
        skewed_features = []
        min_values = {}
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # Apply dimensionality reduction if requested
    pca_scalar = None
    if dim_reduce:
        print("Applying dimensionality reduction...")
        df_train_processed, pca_scalar = perform_dimensionality_reduction(
            df_train_normalized.copy(deep=True),
            n_components=n_components,
            target_col=target_column,
            method=dim_method,
            pca_scalar=None,
            fit=True,
        )

        df_test_processed, _ = perform_dimensionality_reduction(
            df_test_normalized.copy(deep=True),
            n_components=n_components,
            target_col=target_column,
            method=dim_method,
            pca_scalar=pca_scalar,
            fit=False,
        )
    else:
        print("Skipping dimensionality reduction as per user settings")
        df_train_processed = df_train_normalized
        df_test_processed = df_test_normalized

    # Create preprocessing pipeline for inverse transforms
    pipeline = {
        "pca": pca_scalar,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "log_transform": {
            "columns": skewed_features if skewed_features is not None else [],
            "min_values": min_values if min_values is not None else {},
        },
    }

    # Get numerical features for the model

    numeric_features = get_numerical_features(
        df_train_processed, [smiles_column, target_column] + (exclude_columns or [])
    )

    print(f"Using {len(numeric_features)} numerical features with ChemBERTa")

    return df_train_processed, df_test_processed, numeric_features, pipeline


def train_chemberta_model(
    args, df_train, df_test, numeric_features, pipeline, device=None
):
    """
    Train a ChemBERTa model for regression on SMILES data with numerical features.

    Args:
        args: Command-line arguments
        df_train: Training dataframe containing SMILES and target
        df_test: Test dataframe containing SMILES and target
        numeric_features: List of columns to use as numerical features
        pipeline: Dict containing preprocessing pipeline info for inverse transform
        device: PyTorch device (optional)

    Returns:
        dict: Results including model, metrics, predictions, etc.
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    # Setup SMILES column and target column
    smiles_col = args.smiles_column
    target_col = args.target_column

    # Count available numerical features
    # Handle cases where numerical features are not provided
    numeric_features = numeric_features or []
    num_available_features = len(numeric_features)
    print(f"Using {num_available_features} numerical features with ChemBERTa")

    # Create datasets
    texts_train = df_train[smiles_col].tolist()
    targets_train = df_train[target_col].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    targets_test = df_test[target_col].values.astype(np.float32)

    features_train = (
        df_train[numeric_features].values.astype(np.float32)
        if num_available_features > 0
        else None
    )
    features_test = (
        df_test[numeric_features].values.astype(np.float32)
        if num_available_features > 0
        else None
    )

    train_dataset = ChembertaDataset(
        texts_train, targets_train, tokenizer, features_train
    )
    test_dataset = ChembertaDataset(texts_test, targets_test, tokenizer, features_test)

    # Create model
    model = ChembertaRegressorWithFeatures(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_features=num_available_features,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        verbose=args.verbose if hasattr(args, "verbose") else 0,
    )

    # Setup training arguments
    dataset_name = os.path.splitext(os.path.basename(args.train_csv))[0]
    output_dir = os.path.join(args.output_dir, dataset_name, "chemberta")
    os.makedirs(output_dir, exist_ok=True)

    # Set up callback list for early stopping if needed
    callbacks = []
    use_early_stopping = (hasattr(args, "early_stopping") and args.early_stopping and 
                         hasattr(args, "patience") and args.patience is not None and args.patience > 0)
    
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
        print(f"Early stopping enabled with patience={args.patience}")
        # Only evaluate during training when early stopping is enabled
        evaluation_strategy = "epoch"
        save_strategy = "epoch"
        load_best_model_at_end = True
    else:
        print("Early stopping disabled - training for full epoch count")
        # No evaluation during training to prevent data leakage
        evaluation_strategy = "no"
        save_strategy = "epoch"
        load_best_model_at_end = False

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=1,
        learning_rate=args.lr,
        weight_decay=args.l2_lambda,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_strategy="epoch",
        logging_first_step=True,
        seed=args.random_seed,
        report_to="none",  # Disable wandb reports
        disable_tqdm=args.verbose < 2 if hasattr(args, "verbose") else False,
    )

    # Create compute_metrics function
    compute_metrics = get_compute_metrics_fn(
        verbose_level=args.verbose if hasattr(args, "verbose") else 0,
        pipeline=pipeline,
        target_column=target_col,
    )

    # Initialize trainer
    trainer = L1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if use_early_stopping else None,
        compute_metrics=compute_metrics,
        l1_lambda=args.l1_lambda,
        verbose=args.verbose if hasattr(args, "verbose") else 0,
        callbacks=callbacks,
    )

    # Train model
    print("\nTraining ChemBERTa model...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = trainer.evaluate(eval_dataset=test_dataset)

    # Generate predictions for plotting
    predictions_output = trainer.predict(test_dataset)
    preds = predictions_output.predictions.squeeze(-1)
    labels = predictions_output.label_ids

    # Measure inference time
    model.eval()
    inference_start_time = time.time()
    
    with torch.no_grad():
        # Run inference on test dataset to measure timing
        _ = trainer.predict(test_dataset)
    
    inference_time = time.time() - inference_start_time
    inference_time_per_sample = inference_time / len(test_dataset)

    # Invert transform to original scale if possible
    if pipeline is not None:
        try:
            orig_preds = inverse_transform_target(preds, pipeline, target_col)
            orig_labels = inverse_transform_target(labels, pipeline, target_col)
        except Exception as e:
            print(f"Warning: Could not inverse transform targets: {e}")
            orig_preds = preds
            orig_labels = labels
    else:
        orig_preds = preds
        orig_labels = labels

    # Print metrics
    if "eval_original_mae" in metrics:
        print(f"\nTest metrics (original scale):")
        print(f"  MAE:  {metrics['eval_original_mae']:.4f}")
        print(f"  MSE:  {metrics['eval_original_mse']:.4f}")
        print(f"  RMSE: {metrics['eval_original_rmse']:.4f}")
        print(f"  R²:   {metrics['eval_original_r2']:.4f}")
    else:
        print(f"\nTest metrics (normalized scale):")
        print(f"  MAE:  {metrics['eval_mae']:.4f}")
        print(f"  MSE:  {metrics['eval_mse']:.4f}")
        print(f"  RMSE: {metrics['eval_rmse']:.4f}")
        print(f"  R²:   {metrics['eval_r2']:.4f}")

    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print(f"\nInference performance:")
    print(f"  Inference time: {inference_time:.4f} seconds ({inference_time_per_sample * 1000:.2f} ms/sample)")

    # Generate and save plots
    # 1. Predictions vs Targets plot
    plot_predictions_vs_targets(
        predictions=orig_preds,
        targets=orig_labels,
        output_path=output_dir,
        output_filename="preds_vs_targets.pdf",
    )

    # 2. Error plot
    plot_errors(
        predictions=orig_preds,
        targets=orig_labels,
        output_path=output_dir,
        output_filename="prediction_errors.png",
        title="ChemBERTa - Prediction Errors",
    )

    # 3. Loss plot
    # Convert HF training history to format expected by plot_losses
    history_for_plot = {"train_loss": [], "val_loss": []}

    for entry in trainer.state.log_history:
        # Use regularized_loss for training loss if available
        if "regularized_loss" in entry:
            history_for_plot["train_loss"].append(entry["regularized_loss"])
        elif "loss" in entry:
            history_for_plot["train_loss"].append(entry["loss"])

        if "eval_loss" in entry:
            history_for_plot["val_loss"].append(entry["eval_loss"])

    # Only plot if we have data
    if history_for_plot["train_loss"] or history_for_plot["val_loss"]:
        plot_losses(
            history_for_plot,
            output_dir,
            title="ChemBERTa - Training and Validation Loss",
        )

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(DEFAULT_PRETRAINED_NAME).save_pretrained(output_dir)

    model_path = os.path.join(output_dir, f"chemberta_model_final.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save hyperparameters
    hyperparams = {
        "hidden_channels": args.hidden_channels,
        "lr": args.lr,
        "l2_lambda": args.l2_lambda,  # Changed from weight_decay to l2_lambda for consistency
        "l1_lambda": args.l1_lambda,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "num_mlp_layers": args.num_mlp_layers,
        "epochs": args.epochs,
        "dim_reduce": args.dim_reduce if hasattr(args, "dim_reduce") else False,
        "n_components": args.n_components if hasattr(args, "n_components") else None,
    }
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Save normalization pipeline for later use in interpretability analysis
    if pipeline is not None:
        pipeline_path = os.path.join(output_dir, "normalization_pipeline.pkl")
        try:
            # Add target column to pipeline for easier reference
            pipeline_to_save = pipeline.copy()
            pipeline_to_save["target_column"] = target_col
            
            with open(pipeline_path, "wb") as f:
                pickle.dump(pipeline_to_save, f)
            print(f"Normalization pipeline saved to {pipeline_path}")
        except Exception as e:
            print(f"Warning: Could not save normalization pipeline: {e}")

    # Save log file
    log_path = os.path.join(output_dir, "evaluation_log.txt")
    with open(log_path, "w") as f:
        f.write("=== Final Evaluation Metrics ===\n\n")

        f.write("Normalized Scale Metrics:\n")
        f.write(f"MAE: {metrics['eval_mae']:.6f}\n")
        f.write(f"MSE: {metrics['eval_mse']:.6f}\n")
        f.write(f"RMSE: {metrics['eval_rmse']:.6f}\n")
        f.write(f"R²: {metrics['eval_r2']:.6f}\n\n")

        if "eval_original_mae" in metrics:
            f.write("Original Scale Metrics:\n")
            f.write(f"MAE: {metrics['eval_original_mae']:.6f}\n")
            f.write(f"MSE: {metrics['eval_original_mse']:.6f}\n")
            f.write(f"RMSE: {metrics['eval_original_rmse']:.6f}\n")
            f.write(f"R²: {metrics['eval_original_r2']:.6f}\n\n")

        f.write("=== Model Architecture ===\n")
        f.write(str(model) + "\n\n")
        
        # Calculate and write parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write("=== Model Parameters ===\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Frozen Parameters: {total_params - trainable_params:,}\n\n")

        f.write("=== Model Hyperparameters ===\n")
        for param, value in hyperparams.items():
            f.write(f"{param}: {value}\n")

        f.write("\n=== Training Information ===\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Epochs trained: {len(history_for_plot['train_loss'])}\n")

        if len(history_for_plot["train_loss"]) > 0:
            f.write(f"Final training loss: {history_for_plot['train_loss'][-1]:.6f}\n")
        if len(history_for_plot["val_loss"]) > 0:
            f.write(f"Final validation loss: {history_for_plot['val_loss'][-1]:.6f}\n")

        f.write("\n=== Inference Performance ===\n")
        f.write(f"Inference time: {inference_time:.4f} seconds\n")
        f.write(f"Inference time per sample: {inference_time_per_sample:.6f} seconds ({inference_time_per_sample * 1000:.2f} ms/sample)\n")

    print(f"Log saved to {log_path}")

    # Get data loaders for explainability analyses if needed
    need_explainability = (
        (hasattr(args, "shap_analysis") and args.shap_analysis)
        or (hasattr(args, "integrated_gradients") and args.integrated_gradients)
        or (hasattr(args, "mc_dropout") and args.mc_dropout)
    )

    if need_explainability:
        # Create datasets for explainability
        train_dataset = ChembertaDataset(
            texts=df_train[args.smiles_column].tolist(),
            targets=df_train[args.target_column].values,
            tokenizer=tokenizer,
            features=(
                numeric_features["train"]
                if numeric_features and "train" in numeric_features
                else None
            ),
        )

        test_dataset = ChembertaDataset(
            texts=df_test[args.smiles_column].tolist(),
            targets=df_test[args.target_column].values,
            tokenizer=tokenizer,
            features=(
                numeric_features["test"]
                if numeric_features and "test" in numeric_features
                else None
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Run explainability analyses
        explainability_results = run_explainability_analyses(
            args=args,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            pipeline=pipeline,
            target_column=args.target_column,
            df_test_normalized=df_test,
            output_dir=output_dir,
        )
    else:
        explainability_results = {}

    # Return results
    return {
        "model": model,
        "trainer": trainer,
        "metrics": metrics,
        "predictions": orig_preds,
        "targets": orig_labels,
        "history": trainer.state.log_history,
        "training_time": training_time,
        "output_dir": output_dir,
        "hyperparams": hyperparams,
        "explainability": explainability_results,
    }


def chemberta_training_workflow(args):
    """
    Run the standard training workflow for ChemBERTa models.

    Args:
        args: Command-line arguments

    Returns:
        dict: Results from training and evaluation
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocess data
    df_train, df_test, numeric_features, pipeline = preprocess_data_for_chemberta(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        target_column=args.target_column,
        smiles_column=args.smiles_column,
        transform_type=args.transform_type if hasattr(args, "transform_type") else None,
        dim_reduce=args.dim_reduce if hasattr(args, "dim_reduce") else False,
        n_components=args.n_components if hasattr(args, "n_components") else 100,
        dim_method=args.dim_method if hasattr(args, "dim_method") else "pca",
        exclude_columns=(
            args.exclude_columns if hasattr(args, "exclude_columns") else []
        ),
    )

    # Train model
    results = train_chemberta_model(
        args=args,
        df_train=df_train,
        df_test=df_test,
        numeric_features=numeric_features,
        pipeline=pipeline,
        device=device,
    )

    print("\nChemBERTa training and evaluation completed successfully.")
    print(f"All results saved to: {results['output_dir']}")

    return results