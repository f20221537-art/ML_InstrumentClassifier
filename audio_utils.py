"""
audio_utils.py
Feature extraction utilities for instrument detection.
All features are normalized to [0, 1] range for consistent scoring.
"""

import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract a comprehensive set of acoustic features from an audio signal.
    All returned values are normalized to [0, 1].
    """
    features = {}

    # ── Harmonic / Percussive separation ───────────────────────────────────
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic ** 2) + 1e-10
    percussive_energy = np.sum(y_percussive ** 2) + 1e-10
    total_energy = harmonic_energy + percussive_energy

    features["harmonic_ratio"] = float(harmonic_energy / total_energy)
    features["percussive_ratio"] = float(percussive_energy / total_energy)

    # ── Spectral centroid (normalized 0-1 over Nyquist) ─────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    nyquist = sr / 2.0
    features["centroid_norm"] = float(np.mean(centroid) / nyquist)

    # ── Spectral bandwidth ──────────────────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth"] = float(np.clip(np.mean(bandwidth) / nyquist, 0, 1))

    # ── Spectral flatness (0 = tonal, 1 = noise-like) ──────────────────────
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness"] = float(np.clip(np.mean(flatness), 0, 1))

    # ── Spectral contrast ───────────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast)
    features["spectral_contrast_mean"] = float(
        np.clip((contrast_mean + 50) / 100.0, 0, 1)
    )

    # ── Zero crossing rate ──────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zero_crossing_rate"] = float(np.clip(np.mean(zcr) * 10, 0, 1))

    # ── Onset density (onsets per second, normalized) ───────────────────────
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    onsets_per_sec = len(onset_frames) / max(duration, 1e-3)
    features["onset_density"] = float(np.clip(onsets_per_sec / 10.0, 0, 1))

    # ── Frequency band energy ratios ────────────────────────────────────────
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    total_fft_energy = np.sum(fft ** 2) + 1e-10

    low_mask  = freqs < 300
    mid_mask  = (freqs >= 300) & (freqs < 2000)
    high_mask = freqs >= 2000

    features["low_freq_energy_ratio"]  = float(np.sum(fft[low_mask]  ** 2) / total_fft_energy)
    features["mid_freq_energy_ratio"]  = float(np.sum(fft[mid_mask]  ** 2) / total_fft_energy)
    features["high_freq_energy_ratio"] = float(np.sum(fft[high_mask] ** 2) / total_fft_energy)

    # ── Vibrato / pitch modulation ─────────────────────────────────────────
    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else np.array([])

    if len(voiced_f0) > 10:
        f0_smooth = np.convolve(voiced_f0, np.ones(5) / 5, mode='valid')
        modulation = np.std(f0_smooth) / (np.mean(f0_smooth) + 1e-6)
        features["vibrato_presence"] = float(np.clip(modulation * 5, 0, 1))
    else:
        features["vibrato_presence"] = 0.0

    # ── Formant-like presence (vocal indicator via MFCC spread) ────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_std = np.std(mfcc, axis=1)
    formant_score = np.mean(mfcc_std[2:6]) / (np.mean(mfcc_std) + 1e-6)
    features["formant_presence"] = float(np.clip(formant_score / 3.0, 0, 1))

    # ── MFCC means (raw, for reference) ────────────────────────────────────
    mfcc_means = np.mean(mfcc, axis=1)
    for i, val in enumerate(mfcc_means[:5]):
        features[f"mfcc_{i+1}"] = float(val)

    return features


def get_spectral_features(y: np.ndarray, sr: int) -> dict:
    """
    Returns a lighter set of spectral features for quick display.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr      = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "Spectral Centroid (Hz)": float(np.mean(centroid)),
        "Spectral Rolloff (Hz)":  float(np.mean(rolloff)),
        "Zero Crossing Rate":     float(np.mean(zcr)),
    }
