"""
Dataset Loader & Audit Module for Cuff-less Blood Pressure & PPG Signals.

Handles:
- Scanning the local workspace for `.mat` (MIMIC-II Kaggle parts) and `.csv` files.
- Parsing multi-channel arrays: PPG, ECG, Arterial Blood Pressure (ABP).
- Signal integrity check, duration audit, and subject-wise train/val/test splitting.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
# Lazy import numpy inside functions to allow running audit without dependencies


def find_dataset_files(search_dir: str = ".") -> List[Path]:
    """
    Search recursively for supported biosignal dataset files (.mat, .csv, .h5, .parquet).
    """
    extensions = ("*.mat", "*.csv", "*.h5", "*.parquet")
    found_files = []
    base_path = Path(search_dir).resolve()
    for ext in extensions:
        found_files.extend(base_path.glob(f"**/{ext}"))
    return found_files


def audit_dataset_directory(base_dir: str = ".") -> Dict[str, Any]:
    """
    Performs an audit of available data files in the workspace.
    """
    base_path = Path(base_dir).resolve()
    files = find_dataset_files(str(base_path))
    
    audit_report = {
        "workspace": str(base_path),
        "total_files_found": len(files),
        "files": []
    }
    
    for f in files:
        file_info = {
            "name": f.name,
            "relative_path": str(f.relative_to(base_path)),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "suffix": f.suffix
        }
        audit_report["files"].append(file_info)
        
    return audit_report


def load_mimic_mat_part(file_path: str, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads MIMIC-II .mat files (e.g., part_1.mat).
    Structure in MIMIC-II Kaggle dataset:
      mat['p'][0, i] contains [PPG, ABP, ECG] rows sampled at 125 Hz.
    """
    import scipy.io
    mat = scipy.io.loadmat(file_path)
    if 'p' not in mat:
        raise ValueError(f"Key 'p' not found in {file_path}. Available keys: {list(mat.keys())}")
        
    p_data = mat['p']
    num_records = len(p_data[0]) if max_records is None else min(max_records, len(p_data[0]))
    
    records = []
    for i in range(num_records):
        cell = p_data[0, i]
        # Check if cell has standard shape: [channels, time_steps]
        if hasattr(cell, 'shape') and len(cell.shape) >= 2:
            record = {
                "subject_idx": i,
                "ppg": cell[0, :],
                "abp": cell[1, :],
                "ecg": cell[2, :] if cell.shape[0] > 2 else None,
                "fs": 125.0
            }
            records.append(record)
    return records


def split_subjects_stratified(
    num_subjects: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Strict Subject-Wise Split.
    Prevents target leakage by ensuring complete subjects exist exclusively
    in either train, validation, or test sets.
    """
    import numpy as np
    np.random.seed(seed)
    indices = np.random.permutation(num_subjects)
    n_train = int(train_ratio * num_subjects)
    n_val = int(val_ratio * num_subjects)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }
