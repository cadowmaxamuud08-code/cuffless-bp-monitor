"""
Deep Learning Models for Cuff-Less Blood Pressure Estimation.

Implements:
1. MultiScaleConvBlock: Parallel 1D-CNN kernels (fine, medium, coarse receptive fields)
   capturing localized pulse morphology (dicrotic notch, systolic peak, baseline).
2. TemporalAttention: Additive multi-head temporal attention over cardiac cycles.
3. PPG_CNN_BiLSTM_Attention: End-to-end multi-task neural network mapping
   raw/filtered continuous PPG windows (e.g., 5s @ 125 Hz -> 625 samples)
   to continuous Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP).
4. MultiTaskBPLoss: Robust Huber loss with physiological Mean Arterial Pressure (MAP)
   consistency constraint: MAP = 1/3 * SBP + 2/3 * DBP.
"""

from typing import Dict, Tuple, Optional
import numpy as np

# PyTorch imports (lazy guarded for environments during setup)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if torch is not None:

    class MultiScaleConvBlock(nn.Module):
        """
        Multi-Scale 1D Convolutional Block.
        Captures short-term morphological inflections (dicrotic notch) alongside
        broader pulse envelopes using parallel kernel sizes.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            assert out_channels % 3 == 0, "out_channels must be divisible by 3 for 3 branches"
            branch_channels = out_channels // 3

            # Fine scale (small receptive field for sharp peak & notch features)
            self.branch1 = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU()
            )
            # Medium scale (pulse upstroke and ejection slope)
            self.branch2 = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7),
                nn.BatchNorm1d(branch_channels),
                nn.GELU()
            )
            # Coarse scale (waveform envelope and low-frequency hemodynamic trend)
            self.branch3 = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=31, padding=15),
                nn.BatchNorm1d(branch_channels),
                nn.GELU()
            )
            # Residual 1x1 projection
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out3 = self.branch3(x)
            concat = torch.cat([out1, out2, out3], dim=1)
            res = self.shortcut(x)
            return self.act(concat + res)


    class TemporalAttention(nn.Module):
        """
        Self-Attention mechanism over temporal features to highlight critical
        cardiac phases (systolic peak, diastolic reflection) across the window.
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters:
                x: [Batch, Sequence_Length, Hidden_Dim]
            Returns:
                context_vector: [Batch, Hidden_Dim]
                attn_weights: [Batch, Sequence_Length, 1]
            """
            scores = self.attention_weights(x)  # [B, T, 1]
            weights = F.softmax(scores, dim=1)  # [B, T, 1]
            context = torch.sum(x * weights, dim=1)  # [B, Hidden_Dim]
            return context, weights


    class PPG_CNN_BiLSTM_Attention(nn.Module):
        """
        Hybrid Multi-Scale 1D-CNN + BiLSTM + Attention Network
        for continuous cuff-less blood pressure estimation.
        
        Input Shape: [Batch, Channels (1 or 2), Window_Length (e.g. 625 samples)]
        Output Shape: [Batch, 2] -> (Predicted SBP, Predicted DBP)
        """
        def __init__(
            self,
            in_channels: int = 1,
            conv_channels: int = 96,
            lstm_hidden: int = 128,
            lstm_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()

            # Stage 1: Multi-Scale Spatial/Temporal Feature Maps
            self.conv1 = MultiScaleConvBlock(in_channels, conv_channels)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 625 -> 312
            self.dropout1 = nn.Dropout(dropout * 0.5)

            self.conv2 = MultiScaleConvBlock(conv_channels, conv_channels)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 312 -> 156
            self.dropout2 = nn.Dropout(dropout * 0.5)

            self.conv3 = MultiScaleConvBlock(conv_channels, conv_channels)
            self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 156 -> 78
            self.dropout3 = nn.Dropout(dropout)

            # Stage 2: Bidirectional LSTM for sequential cardiac dynamics
            self.lstm = nn.LSTM(
                input_size=conv_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0
            )

            # Stage 3: Temporal Attention Pooling
            self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)

            # Stage 4: Multi-Task Dense Regression Head
            self.regressor = nn.Sequential(
                nn.Linear(lstm_hidden * 2, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2)  # Outputs [SBP, DBP]
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters:
                x: [Batch, 1, Seq_Len]
            Returns:
                bp_preds: [Batch, 2] -> columns [SBP, DBP]
                attn_weights: [Batch, Seq_Len_Downsampled, 1]
            """
            # Conv block feature extraction
            feat = self.conv1(x)
            feat = self.dropout1(self.pool1(feat))
            feat = self.conv2(feat)
            feat = self.dropout2(self.pool2(feat))
            feat = self.conv3(feat)
            feat = self.dropout3(self.pool3(feat))

            # Transpose to [Batch, Time_Steps, Channels] for recurrent layer
            feat_seq = feat.permute(0, 2, 1)

            # BiLSTM temporal processing
            lstm_out, _ = self.lstm(feat_seq)  # [B, T_downsampled, 2 * hidden]

            # Attention pooling
            context, attn_weights = self.attention(lstm_out)  # [B, 2 * hidden]

            # Final prediction
            bp_preds = self.regressor(context)  # [B, 2]
            return bp_preds, attn_weights


    class MultiTaskBPLoss(nn.Module):
        """
        Multi-task Loss with Huber penalty and physiological MAP consistency constraint.
        
        Loss = alpha * Huber(y_sbp, y_pred_sbp)
             + beta  * Huber(y_dbp, y_pred_dbp)
             + gamma * Huber(y_map, 1/3 * y_pred_sbp + 2/3 * y_pred_dbp)
        """
        def __init__(
            self,
            alpha: float = 0.45,
            beta: float = 0.45,
            gamma: float = 0.10,
            delta: float = 2.0
        ):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.huber = nn.HuberLoss(delta=delta)

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """
            Parameters:
                y_pred: [Batch, 2] -> [pred_sbp, pred_dbp]
                y_true: [Batch, 2] -> [true_sbp, true_dbp]
            """
            pred_sbp, pred_dbp = y_pred[:, 0], y_pred[:, 1]
            true_sbp, true_dbp = y_true[:, 0], y_true[:, 1]

            # Loss components for SBP and DBP
            loss_sbp = self.huber(pred_sbp, true_sbp)
            loss_dbp = self.huber(pred_dbp, true_dbp)

            # Hemodynamic MAP consistency constraint
            pred_map = (1.0 / 3.0) * pred_sbp + (2.0 / 3.0) * pred_dbp
            true_map = (1.0 / 3.0) * true_sbp + (2.0 / 3.0) * true_dbp
            loss_map = self.huber(pred_map, true_map)

            total_loss = self.alpha * loss_sbp + self.beta * loss_dbp + self.gamma * loss_map
            return total_loss

else:
    # Fallback placeholders when torch is being initialized
    MultiScaleConvBlock = None
    TemporalAttention = None
    PPG_CNN_BiLSTM_Attention = None
    MultiTaskBPLoss = None
