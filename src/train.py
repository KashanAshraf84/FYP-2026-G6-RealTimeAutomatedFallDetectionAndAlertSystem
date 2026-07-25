"""
Fall Detection System - Training Pipeline
==========================================
Complete training pipeline with:
  - Training/validation loop with early stopping
  - Learning rate scheduling
  - Class-weighted loss
  - Comprehensive evaluation metrics
  - Model checkpointing
  - Training visualization
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from config import SystemConfig, CONFIG
from fall_detector_model import create_model, count_parameters, export_to_onnx
from dataset import (
    load_dataset,
    create_data_loaders,
    generate_synthetic_data,
    preprocess_dataset_folder,
    LABEL_NAMES,
)
from pose_estimator import PoseEstimator


class FallDetectionTrainer:
    """
    Complete training pipeline for the fall detection model.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or CONFIG
        self.device = torch.device(self.config.get_device())
        print(f"Using device: {self.device}")

        # Create model
        self.model = create_model(self.config.model).to(self.device)
        print(f"Model: {self.config.model.architecture}")
        print(f"Parameters: {count_parameters(self.model):,}")

        # Loss function with class weights
        weights = torch.tensor(
            self.config.model.class_weights, dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.model.lr_scheduler_factor,
            patience=self.config.model.lr_scheduler_patience,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop with validation and early stopping.

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.model.epochs
        save_dir = save_dir or os.path.dirname(self.config.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        patience_counter = 0
        best_epoch = 0

        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1 = self._validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print progress
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model (by F1 score for imbalanced data)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                model_path = os.path.join(save_dir, "fall_detector_best.pth")
                self._save_checkpoint(model_path, epoch, val_loss, val_f1)
                print(f"  ✓ New best model saved (F1: {val_f1:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.model.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        # Save final model
        final_path = os.path.join(save_dir, "fall_detector_final.pth")
        self._save_checkpoint(final_path, epoch, val_loss, val_f1)

        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best epoch: {best_epoch} | Val F1: {best_val_f1:.4f} | Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}")

        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average="weighted")
        return avg_loss, accuracy, f1

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on test set.

        Returns:
            Dictionary with metrics and confusion matrix
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x)
            probs = torch.softmax(logits, dim=1)

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Classification report
        target_names = [LABEL_NAMES[i] for i in range(3)]
        report = classification_report(
            all_labels, all_preds, target_names=target_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        results = {
            "accuracy": float(report["accuracy"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {
                target_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        # Print results
        print(f"\n{'='*60}")
        print("TEST SET EVALUATION")
        print(f"{'='*60}")
        print(f"Accuracy:    {results['accuracy']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        print(f"Macro F1:    {results['macro_f1']:.4f}")
        print(f"\nPer-class metrics:")
        for name in target_names:
            m = results["per_class"][name]
            print(
                f"  {name:10s}: P={m['precision']:.4f} R={m['recall']:.4f} "
                f"F1={m['f1']:.4f} Support={m['support']}"
            )
        print(f"\nConfusion Matrix:")
        print(f"  {'':>10s}  {'Normal':>8s} {'Warning':>8s} {'Fall':>8s}")
        for i, name in enumerate(target_names):
            row = " ".join(f"{cm[i][j]:>8d}" for j in range(3))
            print(f"  {name:>10s}  {row}")
        print(f"{'='*60}")

        return results

    def _save_checkpoint(
        self, path: str, epoch: int, val_loss: float, val_f1: float
    ):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "config": {
                    "architecture": self.config.model.architecture,
                    "input_size": self.config.model.input_size,
                    "hidden_size": self.config.model.hidden_size,
                    "num_layers": self.config.model.num_layers,
                    "num_classes": self.config.model.num_classes,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(Val F1: {checkpoint['val_f1']:.4f})"
        )

    def export_onnx(self, path: Optional[str] = None):
        """Export model to ONNX for edge deployment."""
        path = path or self.config.onnx_model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        export_to_onnx(
            self.model,
            path,
            sequence_length=self.config.features.sequence_length,
            input_size=self.config.model.input_size,
        )


def plot_training_history(history: Dict, save_path: str = "training_plots.png"):
    """Generate training history plots."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Fall Detection Training History", fontsize=16, fontweight="bold")

        # Loss
        axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Val", linewidth=2)
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history["train_acc"], label="Train", linewidth=2)
        axes[0, 1].plot(history["val_acc"], label="Val", linewidth=2)
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score
        axes[1, 0].plot(history["val_f1"], label="Val F1", linewidth=2, color="green")
        axes[1, 0].set_title("Validation F1 Score")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(history["lr"], label="LR", linewidth=2, color="red")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training plots saved to {save_path}")
    except ImportError:
        print("matplotlib not available - skipping plot generation")


# =============================================================================
# Main training entry point
# =============================================================================

def train_pipeline(
    data_path: Optional[str] = None,
    use_synthetic: bool = False,
    config: Optional[SystemConfig] = None,
):
    """
    Complete training pipeline.

    Args:
        data_path: Path to preprocessed .npz dataset, or raw video folder
        use_synthetic: If True, generate synthetic data for testing
        config: System configuration
    """
    config = config or CONFIG

    # --- Step 1: Prepare Data ---
    print("\n" + "=" * 60)
    print("STEP 1: Preparing Data")
    print("=" * 60)

    if use_synthetic:
        print("Generating synthetic training data...")
        X, y = generate_synthetic_data(
            num_samples_per_class=500,
            sequence_length=config.features.sequence_length,
            num_features=config.model.input_size,
        )
        print(f"Generated {len(X)} samples")
    elif data_path and data_path.endswith(".npz"):
        print(f"Loading preprocessed data from {data_path}")
        X, y, norm_mean, norm_std = load_dataset(data_path)
    elif data_path and os.path.isdir(data_path):
        print(f"Preprocessing video dataset from {data_path}")
        output_dir = config.processed_data_dir
        pose_est = PoseEstimator(config.pose)
        X, y, _, _ = preprocess_dataset_folder(
            data_path, output_dir, pose_est, config.features
        )
        pose_est.release()
    else:
        print("No data path provided. Using synthetic data for demonstration.")
        X, y = generate_synthetic_data(
            num_samples_per_class=500,
            sequence_length=config.features.sequence_length,
            num_features=config.model.input_size,
        )

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Step 2: Create Data Loaders ---
    print("\n" + "=" * 60)
    print("STEP 2: Creating Data Loaders")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        batch_size=config.model.batch_size,
        val_split=0.15,
        test_split=0.15,
        augment_train=True,
        class_weights=config.model.class_weights,
    )

    # --- Step 3: Train Model ---
    print("\n" + "=" * 60)
    print("STEP 3: Training Model")
    print("=" * 60)

    trainer = FallDetectionTrainer(config)
    history = trainer.train(train_loader, val_loader)

    # --- Step 4: Evaluate ---
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating on Test Set")
    print("=" * 60)

    # Load best model for evaluation
    best_model_path = os.path.join(
        os.path.dirname(config.model_save_path), "fall_detector_best.pth"
    )
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    results = trainer.evaluate(test_loader)

    # --- Step 5: Save & Export ---
    print("\n" + "=" * 60)
    print("STEP 5: Saving Model & Artifacts")
    print("=" * 60)

    # Save evaluation results
    results_path = os.path.join(
        os.path.dirname(config.model_save_path), "evaluation_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Export to ONNX
    try:
        trainer.export_onnx()
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Plot training history
    plot_path = os.path.join(
        os.path.dirname(config.model_save_path), "training_plots.png"
    )
    plot_training_history(history, plot_path)

    return trainer, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fall Detection Model")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--arch",
        type=str,
        default="cnn_lstm",
        choices=["lstm", "cnn_lstm", "transformer"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    config = SystemConfig()
    config.model.architecture = args.arch
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.lr

    train_pipeline(
        data_path=args.data,
        use_synthetic=args.synthetic or (args.data is None),
        config=config,
    )
