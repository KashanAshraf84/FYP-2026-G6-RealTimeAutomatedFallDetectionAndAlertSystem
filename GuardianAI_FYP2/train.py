"""
Fall Detection System - Enhanced Training Pipeline (Pillar 1)
=============================================================
Complete training pipeline with:
  - Benchmark Dataset Integration (URFD, Le2i, MCFD, Synthetic)
  - Biomechanical & Temporal Data Augmentation (Jitter, Mirroring, Noise, Occlusion)
  - Focal Loss for severe class imbalance & hard-example mining
  - Multi-architecture support: CNN-LSTM, Transformer, Bi-LSTM
  - Learning rate scheduling with warm-up & ReduceLROnPlateau
  - Early stopping and model checkpointing
  - Comprehensive clinical evaluation: Precision, Recall, F1, Confusion Matrix, Sensitivity & Specificity
  - ONNX export for high-throughput edge deployment
"""

import os
import sys
import json
import time
import argparse
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

try:
    from fyp1_config import SystemConfig, CONFIG
except ImportError:
    from config import SystemConfig, CONFIG

from fall_detector_model import create_model, count_parameters, export_to_onnx
from dataset import (
    load_dataset,
    create_data_loaders,
    generate_synthetic_data,
    preprocess_dataset_folder,
    LABEL_NAMES,
)
from focal_loss import FocalLoss
from augmentations import PoseSequenceAugmenter
from benchmark_datasets import BenchmarkStandardizer, generate_standard_benchmark_mock


class FallDetectionTrainer:
    """
    Advanced training pipeline for skeleton-based temporal fall detection.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.05,
    ):
        self.config = config or CONFIG
        self.device = torch.device(self.config.get_device())
        print(f"\n[Trainer] Using device: {self.device}")

        # Instantiate neural network architecture
        self.model = create_model(self.config.model).to(self.device)
        print(f"[Trainer] Model Architecture: {self.config.model.architecture}")
        print(f"[Trainer] Trainable Parameters: {count_parameters(self.model):,}")

        # Class weights setup
        weights_list = class_weights or self.config.model.class_weights
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(self.device)

        # Loss Function Setup
        self.loss_type = loss_type.lower()
        if self.loss_type == "focal":
            print(f"[Trainer] Loss: Multi-Class Focal Loss (gamma={focal_gamma}, weights={weights_list})")
            self.criterion = FocalLoss(
                alpha=weights_tensor,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                reduction="mean",
            )
        else:
            print(f"[Trainer] Loss: CrossEntropyLoss (label_smoothing={label_smoothing}, weights={weights_list})")
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_tensor,
                label_smoothing=label_smoothing,
            )

        # Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )

        # LR Scheduler
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
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Execute the training loop with early stopping and best checkpoint saving.
        """
        epochs = epochs or self.config.model.epochs
        save_dir = save_dir or os.path.dirname(self.config.model_save_path) or "weights"
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        patience_counter = 0
        best_epoch = 0

        print(f"\n{'='*65}")
        print(f"  Starting Training: {epochs} Epochs | Loss: {self.loss_type.upper()}")
        print(f"{'='*65}\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)

            # Evaluate on validation split
            val_loss, val_acc, val_f1, _ = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% F1: {val_f1:.3f} | "
                f"LR: {current_lr:.2e}"
            )

            # Save best checkpoint based on validation loss and macro F1
            if val_loss < best_val_loss or (val_loss <= best_val_loss * 1.02 and val_f1 > best_val_f1):
                best_val_loss = min(val_loss, best_val_loss)
                best_val_f1 = max(val_f1, best_val_f1)
                best_epoch = epoch
                patience_counter = 0

                best_path = os.path.join(save_dir, "fall_detector.pth")
                self.save_checkpoint(best_path, epoch, val_loss, val_f1, norm_mean, norm_std)
                best_path_best = os.path.join(save_dir, "fall_detector_best.pth")
                self.save_checkpoint(best_path_best, epoch, val_loss, val_f1, norm_mean, norm_std)
                print(f"  [+] Saved new best model checkpoints -> {best_path} and {best_path_best}")
            else:
                patience_counter += 1
                if patience_counter >= getattr(self.config.model, 'patience', 15):
                    print(f"\n[Early Stopping] Triggered at epoch {epoch} (best epoch: {best_epoch})")
                    break

        print(f"\n[Training Finished] Best Val Loss: {best_val_loss:.4f} | Best Val F1: {best_val_f1:.3f} (Epoch {best_epoch})")
        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(X)
            loss = self.criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        return total_loss / max(1, total), (correct / max(1, total)) * 100.0

    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, Dict]:
        """
        Evaluate model with complete clinical metrics:
        Precision, Recall, F1, Per-class breakdown, Confusion Matrix.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        loss = total_loss / max(1, len(all_labels))
        acc = (all_preds == all_labels).mean() * 100.0 if len(all_labels) > 0 else 0.0
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Per-class metrics
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1, 2], zero_division=0
        )

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

        # Clinical Fall Specificity & Sensitivity
        # Class 2 is Fall:
        # TP = cm[2, 2], FN = cm[2, 0] + cm[2, 1]
        # TN = sum of non-fall correctly predicted as non-fall: cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]
        # FP = cm[0, 2] + cm[1, 2]
        tp_fall = cm[2, 2]
        fn_fall = cm[2, 0] + cm[2, 1]
        fp_fall = cm[0, 2] + cm[1, 2]
        tn_fall = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]

        fall_sensitivity = (tp_fall / (tp_fall + fn_fall)) * 100.0 if (tp_fall + fn_fall) > 0 else 0.0
        fall_specificity = (tn_fall / (tn_fall + fp_fall)) * 100.0 if (tn_fall + fp_fall) > 0 else 0.0

        metrics = {
            "loss": loss,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.tolist(),
            "fall_sensitivity": fall_sensitivity,
            "fall_specificity": fall_specificity,
            "per_class": {
                LABEL_NAMES[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1_per_class[i]),
                }
                for i in range(3)
            },
        }

        return loss, acc, f1_macro, metrics

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        val_f1: float,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "model_config": self.config.model.__dict__,
            "history": self.history,
        }
        torch.save(checkpoint, path)


def run_pipeline(args):
    """Entrypoint for the training pipeline."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(project_dir, args.save_dir)
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_dir, args.data_dir)
    if args.data_path and not os.path.isabs(args.data_path):
        args.data_path = os.path.join(project_dir, args.data_path)

    cfg = SystemConfig()
    cfg.model.architecture = args.arch
    cfg.model.epochs = args.epochs
    cfg.model.batch_size = args.batch_size
    cfg.model.learning_rate = args.lr

    print(f"\n{'='*65}")
    print(f"  GuardianAI Fall Detection - Training Pipeline (Pillar 1)")
    print(f"{'='*65}")

    # 1. Dataset Loading / Benchmark Standardization
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading preprocessed dataset from: {args.data_path}")
        X, y, norm_mean, norm_std = load_dataset(args.data_path)
    elif args.benchmark and args.benchmark.lower() != "none":
        bench = args.benchmark.lower()
        standardizer = BenchmarkStandardizer(sequence_length=cfg.features.sequence_length)

        if bench == "synthetic":
            data_file = "data/synthetic_benchmark.npz"
            X, y = generate_standard_benchmark_mock(
                output_path=data_file,
                num_normal=args.num_samples,
                num_warning=int(args.num_samples * 0.4),
                num_fall=int(args.num_samples * 0.5),
            )
            data_loaded = np.load(data_file)
            norm_mean, norm_std = data_loaded["norm_mean"], data_loaded["norm_std"]

        elif bench == "urfd":
            data_file = "data/urfd_benchmark.npz"
            X, y = standardizer.parse_urfd(args.data_dir or "data/urfd", data_file)
            data_loaded = np.load(data_file)
            norm_mean, norm_std = data_loaded["norm_mean"], data_loaded["norm_std"]

        elif bench == "le2i":
            data_file = "data/le2i_benchmark.npz"
            X, y = standardizer.parse_le2i(args.data_dir or "data/le2i", data_file)
            data_loaded = np.load(data_file)
            norm_mean, norm_std = data_loaded["norm_mean"], data_loaded["norm_std"]

        elif bench in ["generic", "custom"]:
            data_file = "data/custom_benchmark.npz"
            if not args.data_dir:
                raise ValueError("Please provide --data-dir <path_to_github_dataset_folder>")
            X, y = standardizer.parse_generic(args.data_dir, data_file)
            data_loaded = np.load(data_file)
            norm_mean, norm_std = data_loaded["norm_mean"], data_loaded["norm_std"]
        else:
            raise ValueError(f"Unknown benchmark: {bench}")



    else:
        print("No existing dataset specified. Generating standard training dataset...")
        X, y, norm_mean, norm_std = generate_synthetic_data(
            num_normal=args.num_samples,
            num_warning=int(args.num_samples * 0.4),
            num_fall=int(args.num_samples * 0.5),
        )

    # 2. Create DataLoaders with optional PoseSequenceAugmenter
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y,
        batch_size=args.batch_size,
        augment_train=args.augment,
        class_weights=cfg.model.class_weights if args.weighted_sampler else None,
    )

    # 3. Instantiate Trainer with Focal Loss or CrossEntropy
    trainer = FallDetectionTrainer(
        config=cfg,
        loss_type=args.loss,
        focal_gamma=args.gamma,
        class_weights=cfg.model.class_weights,
        label_smoothing=args.smoothing,
    )

    # 4. Train Model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # 5. Final Evaluation on Held-Out Test Set
    print(f"\n{'='*65}")
    print(f"  Final Evaluation on Held-Out Test Set")
    print(f"{'='*65}")
    test_loss, test_acc, test_f1, test_metrics = trainer.evaluate(test_loader)

    print(f"Test Accuracy:    {test_acc:.2f}%")
    print(f"Test Macro F1:    {test_f1:.4f}")
    print(f"Fall Sensitivity: {test_metrics['fall_sensitivity']:.2f}% (True Positive Rate for Falls)")
    print(f"Fall Specificity: {test_metrics['fall_specificity']:.2f}% (Rejection of False Alarms)")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in test_metrics["per_class"].items():
        print(f"  {cls_name.upper():<8} -> Precision: {vals['precision']:.3f} | Recall: {vals['recall']:.3f} | F1: {vals['f1']:.3f}")

    print("\nConfusion Matrix [Normal, Warning, Fall]:")
    print(np.array(test_metrics["confusion_matrix"]))

    # Save metrics JSON
    metrics_path = os.path.join(args.save_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[+] Saved test evaluation metrics to {metrics_path}")

    # 6. Optional ONNX Export
    if args.export_onnx:
        onnx_path = os.path.join(args.save_dir, "fall_detector.onnx")
        try:
            export_to_onnx(trainer.model, onnx_path)
            print(f"[+] Exported model to ONNX: {onnx_path}")
        except Exception as e:
            print(f"[!] Note: ONNX export skipped ({e}). PyTorch model checkpoints were saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuardianAI Fall Detection Training Pipeline")
    parser.add_argument("--arch", type=str, default="cnn_lstm", choices=["cnn_lstm", "transformer", "lstm"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "cross_entropy"], help="Loss function")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable pose data augmentations")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentations")
    parser.add_argument("--weighted-sampler", action="store_true", default=False, help="Use weighted random sampler in DataLoader")
    parser.add_argument("--benchmark", type=str, default="synthetic", choices=["synthetic", "urfd", "le2i", "generic", "custom", "none"], help="Benchmark dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw benchmark video directory")
    parser.add_argument("--data-path", type=str, default=None, help="Path to existing preprocessed .npz dataset")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save model weights and metrics")
    parser.add_argument("--num-samples", type=int, default=300, help="Number of samples for synthetic benchmark")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export best model to ONNX")

    args = parser.parse_args()
    run_pipeline(args)
