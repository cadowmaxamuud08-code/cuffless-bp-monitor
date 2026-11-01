"""
Preprocessing Module for Pulse Oximeter & PPG Signal Processing.

Provides zero-phase bandpass filtering, baseline wander removal,
fiducial point detection (systolic peaks, onsets, dicrotic notches),
and Signal Quality Index (SQI) screening.
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, List, Optional


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    fs: float = 125.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase 4th-order forward-backward Butterworth bandpass filter.
    
    Parameters:
        data: 1D raw PPG signal.
        lowcut: Lower cutoff frequency in Hz (default 0.5 Hz to cut respiration).
        highcut: Upper cutoff frequency in Hz (default 8.0 Hz to cut high-freq hiss).
        fs: Sampling frequency in Hz (default 125 Hz).
        order: Filter order (effective order is 2*order due to filtfilt).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    # Zero-phase forward-backward filter
    filtered = signal.filtfilt(b, a, data)
    return filtered


def remove_baseline_wander(
    data: np.ndarray,
    fs: float = 125.0,
    window_sec: float = 0.8
) -> np.ndarray:
    """
    Remove residual baseline wander using a moving median subtraction.
    """
    win_len = int(window_sec * fs)
    if win_len % 2 == 0:
        win_len += 1
    baseline = signal.medfilt(data, kernel_size=win_len)
    return data - baseline


def normalize_ppg(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal amplitude: 'zscore' or 'minmax'.
    """
    if method == 'zscore':
        std = np.std(data)
        return (data - np.mean(data)) / (std if std > 1e-8 else 1.0)
    elif method == 'minmax':
        ptp = np.ptp(data)
        return (data - np.min(data)) / (ptp if ptp > 1e-8 else 1.0)
    return data


def calculate_sqi(pulse_window: np.ndarray, fs: float = 125.0) -> Dict[str, float]:
    """
    Compute Signal Quality Indices (SQI) for a candidate pulse window:
    - Skewness (S_sqi): positive for normal arterial pulses.
    - Kurtosis (K_sqi): peakiness metric.
    - Relative Power (P_sqi): ratio of cardiac band (0.5-4 Hz) to total power (0-8 Hz).
    """
    # Skewness
    mean_val = np.mean(pulse_window)
    std_val = np.std(pulse_window)
    if std_val < 1e-8:
        return {'skewness': 0.0, 'kurtosis': 0.0, 'power_sqi': 0.0, 'is_valid': False}
    
    skew = np.mean(((pulse_window - mean_val) / std_val) ** 3)
    kurt = np.mean(((pulse_window - mean_val) / std_val) ** 4)
    
    # Power spectral density
    freqs, psd = signal.welch(pulse_window, fs=fs, nperseg=min(len(pulse_window), int(2 * fs)))
    cardiac_band = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_band = np.sum(psd[(freqs >= 0.0) & (freqs <= 8.0)]) + 1e-12
    p_sqi = float(cardiac_band / total_band)
    
    # Simple quality gating
    is_valid = bool(skew > 0.0 and p_sqi > 0.5)
    
    return {
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'power_sqi': float(p_sqi),
        'is_valid': is_valid
    }


def detect_fiducial_points(
    ppg_signal: np.ndarray,
    fs: float = 125.0
) -> Dict[str, np.ndarray]:
    """
    Detect pulse onsets and systolic peaks from clean PPG signal.
    """
    min_dist = int(0.4 * fs)  # Max heart rate 150 bpm -> min distance 0.4s
    
    # Systolic peaks (local maxima)
    peaks, _ = signal.find_peaks(ppg_signal, distance=min_dist, prominence=0.2)
    
    # Pulse onsets/feet (local minima preceding each systolic peak)
    onsets = []
    for i in range(len(peaks)):
        start_idx = 0 if i == 0 else peaks[i - 1]
        end_idx = peaks[i]
        if end_idx > start_idx:
            valley = start_idx + np.argmin(ppg_signal[start_idx:end_idx])
            onsets.append(valley)
            
    return {
        'systolic_peaks': np.array(peaks, dtype=int),
        'pulse_onsets': np.array(onsets, dtype=int)
    }


def generate_synthetic_ppg(
    duration_sec: float = 60.0,
    fs: float = 125.0,
    heart_rate_bpm: float = 72.0,
    sbp_target: float = 120.0,
    dbp_target: float = 80.0,
    noise_level: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic realistic PPG, ECG, and ABP ground truth waveforms
    for pipeline verification and unit testing.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)
    hr_freq = heart_rate_bpm / 60.0
    
    # Realistic PPG morphology: fundamental + dicrotic wave + reflections
    cycle = 2 * np.pi * hr_freq * t
    ppg = (
        0.6 * np.sin(cycle - 0.2)
        + 0.3 * np.sin(2 * cycle - 0.8)
        + 0.15 * np.sin(3 * cycle - 1.4)
    )
    # Add slight baseline wander (respiration ~0.25 Hz)
    baseline = 0.2 * np.sin(2 * np.pi * 0.25 * t)
    noise = np.random.normal(0, noise_level, size=len(t))
    raw_ppg = ppg + baseline + noise
    
    # Ground truth ABP waveform (pulsating between SBP and DBP)
    abp_pulse = (np.sin(cycle - 0.5) + 1.0) / 2.0
    abp = dbp_target + (sbp_target - dbp_target) * (abp_pulse ** 1.3)
    
    # Synthetic simplified ECG lead II (QRS spikes)
    ecg = np.zeros_like(t)
    r_peaks = np.arange(0, len(t), int(fs / hr_freq))
    for r in r_peaks:
        if r < len(t):
            ecg[max(0, r-2):min(len(t), r+3)] = 1.0
            
    return raw_ppg, abp, ecg
