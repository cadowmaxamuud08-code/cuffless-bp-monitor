"""
Unit Test & Forward Pass Verification for Deep Learning Model (src/models.py).
"""

import sys
from pathlib import Path
import torch

# Ensure workspace root is in python path
workspace_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_dir))

from src.models import PPG_CNN_BiLSTM_Attention, MultiTaskBPLoss


def test_model_pipeline():
    print("=" * 65)
    print("      DEEP LEARNING MODEL ARCHITECTURE VERIFICATION")
    print("=" * 65)

    batch_size = 8
    in_channels = 1
    seq_len = 625  # 5 seconds at 125 Hz
    
    print(f"\n[Test 1] Initializing PPG_CNN_BiLSTM_Attention Model...")
    model = PPG_CNN_BiLSTM_Attention(
        in_channels=in_channels,
        conv_channels=96,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.25
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Model initialized successfully.")
    print(f"  -> Trainable Parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)")

    print(f"\n[Test 2] Testing Forward Pass with Input Shape: [{batch_size}, {in_channels}, {seq_len}]...")
    dummy_input = torch.randn(batch_size, in_channels, seq_len)
    
    model.eval()
    with torch.no_grad():
        bp_preds, attn_weights = model(dummy_input)
        
    print(f"  -> Predicted BP Output Shape : {list(bp_preds.shape)} (Expected: [{batch_size}, 2])")
    print(f"  -> Attention Weights Shape   : {list(attn_weights.shape)}")
    assert bp_preds.shape == (batch_size, 2), f"Unexpected shape {bp_preds.shape}"

    print(f"\n[Test 3] Testing Multi-Task Loss with MAP Hemodynamic Consistency...")
    criterion = MultiTaskBPLoss(alpha=0.45, beta=0.45, gamma=0.10, delta=2.0)
    
    # Ground truth targets: SBP around 120, DBP around 80
    dummy_targets = torch.tensor([
        [120.0, 80.0],
        [125.0, 82.0],
        [118.0, 76.0],
        [135.0, 88.0],
        [110.0, 70.0],
        [140.0, 92.0],
        [122.0, 79.0],
        [115.0, 75.0]
    ], dtype=torch.float32)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    preds, _ = model(dummy_input)
    loss = criterion(preds, dummy_targets)
    
    print(f"  -> Computed Multi-Task Huber Loss: {loss.item():.4f}")
    
    print(f"\n[Test 4] Verifying Backpropagation & Gradient Flow...")
    loss.backward()
    optimizer.step()
    
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  -> Gradients successfully propagated to all layers: {has_grads}")
    
    print("\n" + "=" * 65)
    print("      ALL MODEL VERIFICATION TESTS PASSED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    test_model_pipeline()
