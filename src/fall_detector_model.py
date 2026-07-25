"""
Fall Detection System - Model Architectures
=============================================
Multiple neural network architectures for temporal fall detection:
  1. LSTM
  2. CNN-LSTM (default, recommended)
  3. Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import ModelConfig


# =============================================================================
# 1. LSTM Model
# =============================================================================

class FallDetectorLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal action classification.
    Good baseline with solid temporal modeling.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )

        lstm_output_size = (
            self.config.hidden_size * 2
            if self.config.bidirectional
            else self.config.hidden_size
        )

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        logits = self.classifier(context)
        return logits


# =============================================================================
# 2. CNN-LSTM Model (Default / Recommended)
# =============================================================================

class FallDetectorCNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid architecture.
    - 1D CNN extracts local temporal patterns from keypoint features.
    - LSTM models longer-range temporal dependencies.
    - Attention mechanism focuses on critical time steps.

    This is the recommended architecture for balanced accuracy and speed.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # 1D Convolutional feature extractor
        cnn_layers = []
        in_channels = self.config.input_size
        for out_channels in self.config.cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, self.config.cnn_kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.config.dropout * 0.5),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.config.cnn_filters[-1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )

        lstm_output_size = (
            self.config.hidden_size * 2
            if self.config.bidirectional
            else self.config.hidden_size
        )

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.permute(0, 2, 1)  # Back to (batch, seq, features)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)

        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        logits = self.classifier(context)
        return logits


# =============================================================================
# 3. Transformer Model
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


import numpy as np


class FallDetectorTransformer(nn.Module):
    """
    Transformer-based architecture for fall detection.
    Best accuracy but higher compute cost.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        d_model = self.config.hidden_size

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(self.config.dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_transformer_layers,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS token output for classification
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits


# =============================================================================
# Model Factory
# =============================================================================

def create_model(config: Optional[ModelConfig] = None) -> nn.Module:
    """
    Factory function to create the appropriate model architecture.

    Args:
        config: ModelConfig instance

    Returns:
        nn.Module: The fall detection model
    """
    config = config or ModelConfig()

    models = {
        "lstm": FallDetectorLSTM,
        "cnn_lstm": FallDetectorCNNLSTM,
        "transformer": FallDetectorTransformer,
    }

    if config.architecture not in models:
        raise ValueError(
            f"Unknown architecture: {config.architecture}. "
            f"Choose from: {list(models.keys())}"
        )

    model = models[config.architecture](config)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    sequence_length: int = 30,
    input_size: int = 51,
):
    """Export model to ONNX format for edge deployment."""
    model.eval()
    dummy_input = torch.randn(1, sequence_length, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX: {save_path}")
