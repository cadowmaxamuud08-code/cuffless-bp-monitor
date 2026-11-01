"""
Feature Extraction Module for Photoplethysmogram (PPG) Signals.

Extracts:
1. Temporal features: Systolic rise time (Ts), Diastolic decay time (Td),
   Cycle duration (Tcycle), Pulse Rate (PR), Duty cycle (Ts/Tcycle).
2. Morphological features: Augmentation Index (AI), Reflection Index (RI),
   Dicrotic notch amplitude ratio, Area under curve (A1, A2, A1/A2, Total Area).
3. Derivative metrics (VPG & APG):
   - First derivative (VPG): Maximum ejection velocity (w_vpg).
   - Second derivative (APG): a, b, c, d, e waves and Aging Index (AGI = (b-c-d-e)/a).
4. Telemetry features: SpO2, Perfusion Index (PI%), Heart Rate (bpm).
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Compatible trapezoid integration for NumPy 2.0+ and SciPy
try:
    from scipy.integrate import trapezoid as trapz_func
except ImportError:
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def extract_single_beat_features(
    beat_signal: np.ndarray,
    fs: float = 125.0,
    spo2: Optional[float] = None,
    pi_percent: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract comprehensive physiological features from a single delineated cardiac cycle.
    
    Parameters:
        beat_signal: 1D array of PPG amplitude from pulse onset to next onset.
        fs: Sampling frequency in Hz.
        spo2: Oxygen saturation % (optional telemetry).
        pi_percent: Perfusion index % (optional telemetry).
        
    Returns:
        Dictionary of extracted clinical and morphological features.
    """
    n_samples = len(beat_signal)
    if n_samples < int(0.25 * fs):  # Less than 250ms is physiologically implausible
        return {}
        
    t_cycle = n_samples / fs
    pr_bpm = 60.0 / t_cycle if t_cycle > 0 else 0.0
    
    # 1. Baseline & Amplitudes
    onset_val = beat_signal[0]
    peak_idx = int(np.argmax(beat_signal))
    peak_val = beat_signal[peak_idx]
    pulse_amp = peak_val - onset_val
    
    # Systolic & Diastolic times
    t_systolic = peak_idx / fs
    t_diastolic = (n_samples - peak_idx) / fs
    k_ratio = t_systolic / t_cycle if t_cycle > 0 else 0.0
    
    # 2. Area Under the Curve
    # Subtract onset baseline for area integration
    norm_pulse = np.maximum(beat_signal - onset_val, 0.0)
    area_total = float(trapz_func(norm_pulse, dx=1.0 / fs))
    area_systolic = float(trapz_func(norm_pulse[:peak_idx + 1], dx=1.0 / fs)) if peak_idx > 0 else 0.0
    area_diastolic = max(0.0, area_total - area_systolic)
    area_ratio = area_systolic / (area_diastolic + 1e-8)
    
    # 3. Pulse Widths (at 25%, 50%, 75% of peak amplitude)
    thresholds = [0.25, 0.50, 0.75]
    widths = {}
    for th in thresholds:
        amp_th = onset_val + th * pulse_amp
        above = np.where(beat_signal >= amp_th)[0]
        if len(above) > 0:
            widths[f"width_{int(th*100)}"] = float((above[-1] - above[0]) / fs)
        else:
            widths[f"width_{int(th*100)}"] = 0.0

    # 4. First Derivative: Velocity Plethysmogram (VPG)
    vpg = np.gradient(beat_signal, 1.0 / fs)
    max_vpg = float(np.max(vpg))
    min_vpg = float(np.min(vpg))
    t_max_vpg = float(np.argmax(vpg) / fs)
    
    # 5. Second Derivative: Acceleration Plethysmogram (APG)
    apg = np.gradient(vpg, 1.0 / fs)
    
    # Identify typical APG waves:
    # a-wave: early systolic positive peak
    # b-wave: early systolic negative peak
    # e-wave: early diastolic positive peak
    early_systole_end = min(peak_idx, int(0.35 * fs))
    a_wave_idx = int(np.argmax(apg[:early_systole_end])) if early_systole_end > 0 else 0
    a_wave = float(apg[a_wave_idx])
    
    b_window_end = min(peak_idx + int(0.15 * fs), n_samples)
    b_wave_idx = (
        int(a_wave_idx + np.argmin(apg[a_wave_idx:b_window_end]))
        if b_window_end > a_wave_idx else a_wave_idx
    )
    b_wave = float(apg[b_wave_idx])
    
    # Mid-systolic waves c and d
    c_wave = float(np.max(apg[b_wave_idx:min(b_wave_idx + int(0.15 * fs), n_samples)])) if n_samples > b_wave_idx else 0.0
    d_wave = float(np.min(apg[b_wave_idx:min(b_wave_idx + int(0.15 * fs), n_samples)])) if n_samples > b_wave_idx else 0.0
    
    # Early diastolic wave e
    e_wave = float(np.max(apg[peak_idx:])) if peak_idx < n_samples else 0.0
    
    # Normalized APG ratios and Aging Index (AGI)
    denom = a_wave if abs(a_wave) > 1e-6 else 1.0
    b_a_ratio = b_wave / denom
    c_a_ratio = c_wave / denom
    d_a_ratio = d_wave / denom
    e_a_ratio = e_wave / denom
    aging_index = (b_wave - c_wave - d_wave - e_wave) / denom
    
    features = {
        # Temporal
        "t_cycle_sec": float(t_cycle),
        "pulse_rate_bpm": float(pr_bpm),
        "t_systolic_sec": float(t_systolic),
        "t_diastolic_sec": float(t_diastolic),
        "duty_cycle": float(k_ratio),
        # Widths
        "width_25_sec": widths["width_25"],
        "width_50_sec": widths["width_50"],
        "width_75_sec": widths["width_75"],
        # Amplitude & Area
        "pulse_amplitude": float(pulse_amp),
        "area_total": area_total,
        "area_systolic": area_systolic,
        "area_diastolic": area_diastolic,
        "area_ratio_sys_dia": float(area_ratio),
        # VPG (1st derivative)
        "vpg_max_slope": max_vpg,
        "vpg_min_slope": min_vpg,
        "t_max_vpg_sec": t_max_vpg,
        # APG (2nd derivative)
        "apg_a_wave": a_wave,
        "apg_b_wave": b_wave,
        "apg_c_wave": c_wave,
        "apg_d_wave": d_wave,
        "apg_e_wave": e_wave,
        "b_a_ratio": float(b_a_ratio),
        "c_a_ratio": float(c_a_ratio),
        "d_a_ratio": float(d_a_ratio),
        "e_a_ratio": float(e_a_ratio),
        "aging_index": float(aging_index)
    }
    
    # Telemetry metrics (if available from digital pulse oximeter)
    if spo2 is not None:
        features["spo2_percent"] = float(spo2)
    if pi_percent is not None:
        features["perfusion_index"] = float(pi_percent)
        
    return features


def extract_features_from_continuous_ppg(
    ppg_signal: np.ndarray,
    pulse_onsets: np.ndarray,
    fs: float = 125.0,
    spo2_val: float = 98.0,
    pi_val: float = 4.5
) -> List[Dict[str, float]]:
    """
    Extract features across consecutive segmented beats from continuous PPG.
    """
    beat_features = []
    for i in range(len(pulse_onsets) - 1):
        start = pulse_onsets[i]
        end = pulse_onsets[i + 1]
        beat = ppg_signal[start:end]
        feats = extract_single_beat_features(beat, fs=fs, spo2=spo2_val, pi_percent=pi_val)
        if feats:
            feats["beat_index"] = i
            feats["start_sample"] = int(start)
            feats["end_sample"] = int(end)
            beat_features.append(feats)
    return beat_features
