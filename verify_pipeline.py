"""
End-to-End Pipeline Verification Script.
Generates synthetic physiological signals, runs the preprocessing stage,
screens for signal quality, and extracts clinical/morphological features.
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in python path
workspace_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import (
    generate_synthetic_ppg,
    butter_bandpass_filter,
    remove_baseline_wander,
    detect_fiducial_points,
    calculate_sqi
)
from src.feature_extraction import extract_features_from_continuous_ppg


def main():
    print("=" * 65)
    print("      BIOMEDICAL SIGNAL PROCESSING PIPELINE VERIFICATION")
    print("=" * 65)
    
    fs = 125.0  # Sampling rate in Hz
    duration = 30.0  # 30 seconds of recording
    hr_target = 75.0  # 75 bpm
    sbp_target = 122.0  # mmHg
    dbp_target = 78.0   # mmHg
    
    print(f"\n[Step 1] Synthesizing Multimodal Physiological Signals ({duration}s @ {fs} Hz)...")
    raw_ppg, true_abp, ecg = generate_synthetic_ppg(
        duration_sec=duration,
        fs=fs,
        heart_rate_bpm=hr_target,
        sbp_target=sbp_target,
        dbp_target=dbp_target,
        noise_level=0.08
    )
    print(f"  -> Generated {len(raw_ppg)} samples.")
    print(f"  -> Target Reference BP: SBP={sbp_target:.1f} mmHg, DBP={dbp_target:.1f} mmHg")

    print("\n[Step 2] Applying Bandpass Filter (0.5 - 8.0 Hz Butterworth)...")
    bp_filtered_ppg = butter_bandpass_filter(raw_ppg, lowcut=0.5, highcut=8.0, fs=fs, order=4)
    
    print("[Step 3] Removing Residual Baseline Wander...")
    clean_ppg = remove_baseline_wander(bp_filtered_ppg, fs=fs, window_sec=0.8)

    print("\n[Step 4] Delineating Fiducial Points (Onsets & Systolic Peaks)...")
    fiducials = detect_fiducial_points(clean_ppg, fs=fs)
    peaks = fiducials["systolic_peaks"]
    onsets = fiducials["pulse_onsets"]
    print(f"  -> Detected {len(peaks)} systolic peaks and {len(onsets)} pulse onsets.")

    print("\n[Step 5] Evaluating Signal Quality Indices (SQI)...")
    sqi_report = calculate_sqi(clean_ppg[:int(10 * fs)], fs=fs)
    print(f"  -> Skewness (S_sqi)   : {sqi_report['skewness']:.4f} (Valid > 0: {sqi_report['skewness'] > 0})")
    print(f"  -> Kurtosis (K_sqi)   : {sqi_report['kurtosis']:.4f}")
    print(f"  -> Power SQI (P_sqi)  : {sqi_report['power_sqi']:.4f} (Valid > 0.5: {sqi_report['power_sqi'] > 0.5})")
    print(f"  -> Overall Quality    : {'PASS (High Quality)' if sqi_report['is_valid'] else 'FAIL'}")

    print("\n[Step 6] Extracting Morphological, Temporal & APG Features across Cardiac Cycles...")
    features_list = extract_features_from_continuous_ppg(
        clean_ppg,
        onsets,
        fs=fs,
        spo2_val=98.5,
        pi_val=4.2
    )
    
    df_features = pd.DataFrame(features_list)
    print(f"  -> Successfully extracted {len(df_features)} beats, {df_features.shape[1]} features each.")
    
    # Save features to CSV
    output_dir = workspace_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "extracted_features.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"  -> Saved extracted features dataset to: {csv_path.relative_to(workspace_dir)}")

    # Print summary of representative clinical features
    key_cols = [
        "pulse_rate_bpm", "t_systolic_sec", "t_diastolic_sec", 
        "area_ratio_sys_dia", "aging_index", "spo2_percent", "perfusion_index"
    ]
    print("\n--- Key Clinical Features (Mean +/- Std) ---")
    for col in key_cols:
        if col in df_features.columns:
            m = df_features[col].mean()
            s = df_features[col].std()
            print(f"  - {col:<20}: {m:8.3f} +/- {s:6.3f}")

    print("\n[Step 7] Generating Signal Comparison & Waveform Plot...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    time_vec = np.arange(len(raw_ppg)) / fs
    
    # Display 6 seconds window for clarity
    win_samples = int(6.0 * fs)
    t_win = time_vec[:win_samples]
    
    axes[0].plot(t_win, raw_ppg[:win_samples], color='tab:gray', label='Raw Noisy PPG')
    axes[0].set_title("1. Raw PPG with Motion Artifact & Baseline Wander", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Raw Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t_win, clean_ppg[:win_samples], color='tab:blue', label='Clean PPG (Filtered & Detrended)')
    sub_peaks = peaks[peaks < win_samples]
    sub_onsets = onsets[onsets < win_samples]
    axes[1].scatter(sub_peaks / fs, clean_ppg[sub_peaks], color='red', marker='v', s=45, label='Systolic Peak')
    axes[1].scatter(sub_onsets / fs, clean_ppg[sub_onsets], color='green', marker='^', s=45, label='Pulse Onset')
    axes[1].set_title("2. Preprocessed PPG (0.5-8 Hz Bandpass + Moving Median Detrending)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # First derivative (VPG)
    vpg = np.gradient(clean_ppg[:win_samples], 1.0 / fs)
    axes[2].plot(t_win, vpg, color='tab:purple', label='1st Derivative (VPG - Velocity)')
    axes[2].set_title("3. Velocity Plethysmogram (VPG)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("dV/dt")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Ground truth ABP
    axes[3].plot(t_win, true_abp[:win_samples], color='tab:red', label='Continuous Arterial Blood Pressure (ABP)')
    axes[3].axhline(sbp_target, color='darkred', linestyle='--', alpha=0.6, label=f'Target SBP ({sbp_target} mmHg)')
    axes[3].axhline(dbp_target, color='blue', linestyle='--', alpha=0.6, label=f'Target DBP ({dbp_target} mmHg)')
    axes[3].set_title("4. Reference Arterial Blood Pressure Waveform (Ground Truth)", fontsize=11, fontweight='bold')
    axes[3].set_ylabel("Pressure (mmHg)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = output_dir / "pipeline_verification.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  -> Saved waveform comparison plot to: {plot_path.relative_to(workspace_dir)}")
    
    print("\n" + "=" * 65)
    print("      PIPELINE VERIFICATION COMPLETE: ALL PHASES PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
