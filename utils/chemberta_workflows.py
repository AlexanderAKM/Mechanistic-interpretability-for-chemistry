"""
Utility module to handle ChemBERTa training workflows.

This module contains functions for training ChemBERTa models with preprocessing,
training, and evaluation.
"""

import os
import json
import time
import pickle
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

from models.simple_mlp import SimpleMLP

from .normalizing import inverse_transform
from .plotting import plot_predictions_vs_targets

DEFAULT_PRETRAINED_NAME = "DeepChem/ChemBERTa-77M-MLM"


class L1Trainer(Trainer):
    """
    Custom trainer that adds L1 regularization to the loss function.
    """

    def __init__(self, l1_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda
        self.regularized_losses = []  # Track regularized losses for plotting

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        raw_mse_loss = outputs.loss  # This is the raw MSE loss from the model

        # For training, add L1 regularization
        if self.training:
            if self.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                regularized_loss = raw_mse_loss + self.l1_lambda * l1_norm

                self.regularized_losses.append(regularized_loss.item())
                loss = regularized_loss
            else:
                loss = raw_mse_loss
                self.regularized_losses.append(raw_mse_loss.item())
        else:
            loss = raw_mse_loss

        return (loss, outputs) if return_outputs else loss

    def training(self):
        """Check if the model is in training mode"""
        return self.model.training if hasattr(self.model, "training") else True

    def log(self, logs, *args, **kwargs):
        """Override log method to include regularized loss in history"""
        if self.model.training and self.regularized_losses:
            logs["regularized_loss"] = self.regularized_losses[-1]
        super().log(logs, *args, **kwargs)


class ChembertaRegressor(nn.Module):
    """
    ChemBERTa regression model
    """

    def __init__(
        self,
        pretrained,
        num_features=0,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        # Freeze the base language model to avoid overfitting. 
        # Currently commented for better performance.
        # TODO: Choose which parts of the model actually require finetuning and freeze all other parts.
        # for param in self.roberta.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        num_input_features = self.roberta.config.hidden_size
        self.mlp = SimpleMLP(num_input_features, hidden_channels, num_mlp_layers, 1, dropout)


    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :] # Take CLS token TODO: Investigate if pooling is better

        x = self.dropout(cls_emb)
        logits = self.mlp(x).squeeze(-1)
        loss = None
        criterion = nn.MSELoss()
        loss = criterion(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits.unsqueeze(-1))


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


def get_compute_metrics_fn(scaler, target_column):
    """
    Returns a function that computes evaluation metrics based on model predictions.

    Args:
        scaler: Preprocessing scaler for inverse transforms
        target_column: Name of the target column

    Returns:
        function: A function that computes evaluation metrics
    """

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.squeeze(-1)
        labels = eval_pred.label_ids

        # Store normalized metrics (these are on the transformed/preprocessed data)
        normalized_mae = mean_absolute_error(labels, preds)
        normalized_mse = mean_squared_error(labels, preds)  # This is pure MSE without any L1 regularization
        normalized_rmse = np.sqrt(normalized_mse)
        normalized_r2 = r2_score(labels, preds)

        original_preds = inverse_transform(preds, scaler)
        original_labels = inverse_transform(labels, scaler)

        mae = mean_absolute_error(original_labels, original_preds)
        mse = mean_squared_error(original_labels, original_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(original_labels, original_preds)

        return {
            "normalized_mae": normalized_mae,
            "normalized_mse": normalized_mse,
            "normalized_rmse": normalized_rmse,
            "normalized_r2": normalized_r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
    return compute_metrics

def evaluate_chemberta_model(
    model, 
    dataset, 
    scaler, 
    target_column="pce_1",
    output_dir=None, 
    trainer=None, 
    batch_size=32,
    plot_filename="preds_vs_targets.pdf",
    smiles_column="smiles",
    tokenizer=None,
):
    """
    Evaluate a ChemBERTa model on a dataset and optionally plot results.

    Args:
        model: The ChemBERTa model to evaluate
        dataset: The dataset to evaluate on (ChembertaDataset or pandas DataFrame)
        scaler: Scaler used for normalization (for inverse transform)
        target_column: Name of the target column (default: "pce_1")
        output_dir: Directory to save plots and results (optional)
        trainer: Existing trainer instance (optional, will create one if not provided)
        batch_size: Batch size for evaluation (default: 32)
        plot_filename: Name of the plot file (default: "preds_vs_targets.pdf")
        smiles_column: Name of SMILES column if dataset is a DataFrame (default: "smiles")
        tokenizer: Tokenizer to use if dataset is a DataFrame (required if dataset is DataFrame)

    Returns:
        dict: Results including metrics, predictions, targets
    """
    
    if isinstance(dataset, pd.DataFrame):
        if tokenizer is None:
            tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)
        
        texts = dataset[smiles_column].tolist()
        targets = dataset[target_column].values.astype(np.float32)
        dataset = ChembertaDataset(texts, targets, tokenizer)
    
    if trainer is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )
        
        compute_metrics = get_compute_metrics_fn(scaler=scaler, target_column=target_column)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
        )
    
    print("\nEvaluating model...")
    predictions_output = trainer.predict(dataset)
    preds = predictions_output.predictions.squeeze(-1)
    labels = predictions_output.label_ids
    metrics = predictions_output.metrics
    orig_preds = inverse_transform(preds, scaler)
    orig_labels = inverse_transform(labels, scaler)
    
    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print(f"\nEvaluation Metrics:")
    for key, value in metrics.items():
        if not key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key[5:]}: {value:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_predictions_vs_targets(
        predictions=orig_preds,
        targets=orig_labels,
        output_path=output_dir,
        output_filename=plot_filename,
    )
    print(f"\nPlot saved to {os.path.join(output_dir, plot_filename)}")
    
    results = {
        "metrics": metrics,
        "predictions": orig_preds.tolist() if hasattr(orig_preds, 'tolist') else list(orig_preds),
        "targets": orig_labels.tolist() if hasattr(orig_labels, 'tolist') else list(orig_labels),
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")
    
    return results


def train_chemberta_model(
    args, df_train, df_test, scaler, device=None, evaluate_after_training=True, dataset_name=None
):
    """
    Train a ChemBERTa model for regression on SMILES data with one target value.

    Args:
        args: Command-line arguments
        df_train: Training dataframe containing SMILES and target
        df_test: Test dataframe containing SMILES and target
        scaler: scaler used for normalization
        device: PyTorch device (optional)
        evaluate_after_training: Whether to evaluate on test set after training (default: True)
        dataset_name: name of the dataset

    Returns:
        dict: Results including model, metrics, predictions, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    smiles_col = args.smiles_column
    target_col = args.target_column

    texts_train = df_train[smiles_col].tolist()
    targets_train = df_train[target_col].values.astype(np.float32)

    texts_test = df_test[smiles_col].tolist()
    targets_test = df_test[target_col].values.astype(np.float32)

    train_dataset = ChembertaDataset(texts_train, targets_train, tokenizer)
    test_dataset = ChembertaDataset(texts_test, targets_test, tokenizer)

    model = ChembertaRegressor(
        pretrained=DEFAULT_PRETRAINED_NAME,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
    )

    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(args.train_csv))[0]

    output_dir = os.path.join(args.output_dir, dataset_name, "chemberta")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        weight_decay=args.l2_lambda,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_strategy="epoch",
        logging_first_step=True,
        seed=args.random_seed,
        report_to="none", 
    )

    compute_metrics = get_compute_metrics_fn(
        scaler=scaler,
        target_column=target_col,
    )

    trainer = L1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        l1_lambda=args.l1_lambda,
    )

    print("\nTraining ChemBERTa model...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(DEFAULT_PRETRAINED_NAME).save_pretrained(output_dir)

    model_path = os.path.join(output_dir, f"chemberta_model_final.bin")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    hyperparams = {
        "hidden_channels": args.hidden_channels,
        "lr": args.lr,
        "l2_lambda": args.l2_lambda,
        "l1_lambda": args.l1_lambda,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "num_mlp_layers": args.num_mlp_layers,
        "epochs": args.epochs,
    }
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    if scaler is not None:
        scaler_path = os.path.join(output_dir, "normalization_scaler.pkl")
        
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Normalization scaler saved to {scaler_path}")

    eval_results = None
    if evaluate_after_training:
        print("Evaluating model on test set...")
        eval_results = evaluate_chemberta_model(
            model=model,
            dataset=test_dataset,
            scaler=scaler,
            target_column=target_col,
            output_dir=output_dir,
            trainer=trainer,
            batch_size=args.batch_size,
            plot_filename="preds_vs_targets.pdf",
        )

    results = {
        "history": trainer.state.log_history,
        "training_time": training_time,
        "output_dir": output_dir,
        "hyperparams": hyperparams,
    }
    
    if eval_results:
        results.update({
            "metrics": eval_results["metrics"],
            "predictions": eval_results["predictions"].tolist() if hasattr(eval_results["predictions"], 'tolist') else list(eval_results["predictions"]),
            "targets": eval_results["targets"].tolist() if hasattr(eval_results["targets"], 'tolist') else list(eval_results["targets"]),
        })

    complete_results_path = os.path.join(output_dir, "all_results.json")
    with open(complete_results_path, "w") as f:
        json.dump(results, f, indent=2)
        
    return results


