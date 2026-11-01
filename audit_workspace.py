"""
Workspace & Dataset Audit Script.
Run this script to inspect available data files and system packages.
"""

import os
import sys
from pathlib import Path

# Add src to python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from src.dataset import audit_dataset_directory

def main():
    print("=" * 60)
    print("   PULSE OXIMETER & BLOOD PRESSURE PIPELINE AUDIT")
    print("=" * 60)
    print(f"Working Directory : {os.getcwd()}")
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    
    # Check packages
    print("\n--- Package Verification ---")
    packages = [
        "numpy", "scipy", "pandas", "matplotlib", "seaborn",
        "sklearn", "torch", "lightgbm"
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [x] {pkg:<12} (v{version})")
        except ImportError:
            print(f"  [ ] {pkg:<12} (NOT INSTALLED)")
            
    # Check dataset files
    print("\n--- Dataset Ingestion Audit ---")
    audit = audit_dataset_directory(str(current_dir))
    print(f"Total dataset files detected in workspace: {audit['total_files_found']}")
    if audit['total_files_found'] == 0:
        print("  Notice: No .mat, .csv, or .h5 files found in 'data/raw/' or root yet.")
        print("  Place your dataset (e.g. part_1.mat or sensor logs) into: data/raw/")
    else:
        for f in audit["files"]:
            print(f"  - {f['name']} ({f['size_mb']} MB) at {f['relative_path']}")
            
    print("=" * 60)

if __name__ == "__main__":
    main()
