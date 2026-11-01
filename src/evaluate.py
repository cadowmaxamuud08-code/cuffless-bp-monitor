"""
Clinical Evaluation and Statistical Validation Module for Cuff-less Blood Pressure.

Implements:
1. Clinical Standard Metrics:
   - AAMI SP10 standard (Mean Error <= 5 mmHg, SD <= 8 mmHg)
   - British Hypertension Society (BHS) Grading (Grade A/B/C/D based on cumulative error <= 5, 10, 15 mmHg)
   - Standard Regression Metrics: MAE, RMSE, Pearson r
2. Visual Diagnostics:
   - Bland-Altman plots with 95% limits of agreement (Mean +/- 1.96 * SD)
   - Correlation scatter plots (Predicted vs. Reference Ground Truth)
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes AAMI SP10 and BHS clinical validation metrics.
    """
    errors = y_pred - y_true
    me = float(np.mean(errors))
    std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # Pearson correlation
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    
    # AAMI Standard Compliance: |ME| <= 5 mmHg and SD <= 8 mmHg
    aami_pass = bool(abs(me) <= 5.0 and std <= 8.0)
    
    # BHS Cumulative Error Percentages
    abs_err = np.abs(errors)
    pct_5 = float(np.mean(abs_err <= 5.0) * 100.0)
    pct_10 = float(np.mean(abs_err <= 10.0) * 100.0)
    pct_15 = float(np.mean(abs_err <= 15.0) * 100.0)
    
    # BHS Grading determination
    if pct_5 >= 60.0 and pct_10 >= 85.0 and pct_15 >= 95.0:
        bhs_grade = "Grade A"
    elif pct_5 >= 50.0 and pct_10 >= 75.0 and pct_15 >= 90.0:
        bhs_grade = "Grade B"
    elif pct_5 >= 40.0 and pct_10 >= 65.0 and pct_15 >= 85.0:
        bhs_grade = "Grade C"
    else:
        bhs_grade = "Grade D"
        
    return {
        "mean_error_me": me,
        "std_error_sd": std,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": corr,
        "aami_compliant": aami_pass,
        "bhs_grade": bhs_grade,
        "bhs_percentages": {
            "<=5mmHg": pct_5,
            "<=10mmHg": pct_10,
            "<=15mmHg": pct_15
        }
    }


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Bland-Altman Analysis",
    unit: str = "mmHg",
    save_path: str = None
) -> plt.Figure:
    """
    Generates a clinical Bland-Altman agreement plot with 95% Limits of Agreement.
    """
    means = (y_true + y_pred) / 2.0
    diffs = y_pred - y_true
    md = np.mean(diffs)
    sd = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
    
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.6, color='tab:blue', edgecolors='none', s=40)
    
    # Reference lines
    ax.axhline(md, color='crimson', linestyle='-', linewidth=1.5, label=f'Mean Bias: {md:+.2f} {unit}')
    ax.axhline(loa_upper, color='darkorange', linestyle='--', linewidth=1.2, label=f'+1.96 SD: {loa_upper:+.2f} {unit}')
    ax.axhline(loa_lower, color='darkorange', linestyle='--', linewidth=1.2, label=f'-1.96 SD: {loa_lower:+.2f} {unit}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Mean of Reference & Predicted ({unit})", fontsize=10)
    ax.set_ylabel(f"Difference (Predicted - Reference) ({unit})", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig
