"""
instrument_classifier.py
Rule-based + spectral heuristic instrument classifier using librosa features.
This approach works without training data and gives interpretable results.
"""

import numpy as np
import librosa
from audio_utils import extract_features


class InstrumentClassifier:
    """
    Classifies instruments from audio using spectral and rhythmic features.
    Uses a heuristic / rule-based system based on known acoustic properties
    of each instrument family.
    """

    INSTRUMENT_PROFILES = {
        "Drums": {
            "percussive_ratio":     (0.3, 1.0, 1.0),  # (min, max, weight)
            "onset_density":        (0.5, 1.0, 0.8),
            "spectral_flatness":    (0.01, 0.5, 0.6),
            "zero_crossing_rate":   (0.05, 0.5, 0.4),
        },
        "Bass Guitar": {
            "centroid_norm":        (0.0, 0.15, 1.0),
            "low_freq_energy_ratio":(0.4, 1.0, 1.0),
            "harmonic_ratio":       (0.3, 1.0, 0.7),
            "spectral_bandwidth":   (0.0, 0.25, 0.6),
        },
        "Piano": {
            "harmonic_ratio":       (0.4, 1.0, 1.0),
            "centroid_norm":        (0.1, 0.5, 0.8),
            "onset_density":        (0.1, 0.8, 0.5),
            "spectral_contrast_mean":(0.3, 1.0, 0.7),
        },
        "Guitar": {
            "harmonic_ratio":       (0.35, 0.9, 1.0),
            "centroid_norm":        (0.1, 0.4, 0.8),
            "spectral_flatness":    (0.005, 0.15, 0.6),
            "mid_freq_energy_ratio":(0.2, 0.8, 0.7),
        },
        "Violin": {
            "harmonic_ratio":       (0.5, 1.0, 1.0),
            "centroid_norm":        (0.25, 0.65, 1.0),
            "spectral_bandwidth":   (0.1, 0.4, 0.7),
            "vibrato_presence":     (0.1, 1.0, 0.8),
        },
        "Cello": {
            "harmonic_ratio":       (0.5, 1.0, 1.0),
            "centroid_norm":        (0.05, 0.3, 1.0),
            "spectral_bandwidth":   (0.05, 0.35, 0.8),
            "vibrato_presence":     (0.05, 1.0, 0.6),
        },
        "Trumpet": {
            "harmonic_ratio":       (0.4, 1.0, 0.9),
            "centroid_norm":        (0.3, 0.7, 1.0),
            "high_freq_energy_ratio":(0.15, 0.7, 0.8),
            "spectral_contrast_mean":(0.4, 1.0, 0.7),
        },
        "Saxophone": {
            "harmonic_ratio":       (0.35, 0.9, 0.9),
            "centroid_norm":        (0.15, 0.55, 1.0),
            "spectral_flatness":    (0.002, 0.08, 0.7),
            "mid_freq_energy_ratio":(0.3, 0.85, 0.8),
        },
        "Flute": {
            "harmonic_ratio":       (0.3, 0.85, 0.8),
            "centroid_norm":        (0.4, 0.85, 1.0),
            "spectral_flatness":    (0.01, 0.2, 0.7),
            "high_freq_energy_ratio":(0.2, 0.8, 0.8),
        },
        "Vocals": {
            "centroid_norm":        (0.1, 0.5, 0.8),
            "harmonic_ratio":       (0.4, 1.0, 1.0),
            "formant_presence":     (0.2, 1.0, 1.0),
            "vibrato_presence":     (0.05, 1.0, 0.6),
        },
        "Synthesizer": {
            "spectral_flatness":    (0.05, 0.6, 1.0),
            "harmonic_ratio":       (0.2, 0.8, 0.6),
            "centroid_norm":        (0.05, 0.8, 0.4),
            "onset_density":        (0.0, 0.5, 0.5),
        },
        "Trombone": {
            "harmonic_ratio":       (0.4, 1.0, 0.9),
            "centroid_norm":        (0.1, 0.45, 1.0),
            "low_freq_energy_ratio":(0.2, 0.75, 0.8),
            "spectral_bandwidth":   (0.1, 0.45, 0.7),
        },
    }

    def __init__(self, confidence_threshold=0.3, max_instruments=6, mode="Standard"):
        self.confidence_threshold = confidence_threshold
        self.max_instruments = max_instruments
        self.mode = mode

    def classify(self, y: np.ndarray, sr: int):
        """
        Returns a sorted list of (instrument_name, confidence) tuples.
        """
        features = extract_features(y, sr)
        scores = {}

        for instrument, profile in self.INSTRUMENT_PROFILES.items():
            score = self._score_instrument(features, profile)
            if score >= self.confidence_threshold:
                scores[instrument] = score

        # Sort by confidence descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:self.max_instruments]

    def _score_instrument(self, features: dict, profile: dict) -> float:
        """
        Compute a weighted match score for an instrument profile.
        Each feature contributes proportionally based on how well the
        extracted value falls within the expected range.
        """
        total_weight = 0.0
        weighted_score = 0.0

        for feat_name, (low, high, weight) in profile.items():
            if feat_name not in features:
                continue

            val = features[feat_name]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            total_weight += weight

            if low <= val <= high:
                # Full match: score by how centrally it falls
                center = (low + high) / 2
                spread = (high - low) / 2
                proximity = 1.0 - abs(val - center) / max(spread, 1e-6)
                weighted_score += weight * (0.7 + 0.3 * proximity)
            else:
                # Partial match: decay by distance outside range
                dist = min(abs(val - low), abs(val - high))
                span = max(high - low, 1e-6)
                decay = max(0.0, 1.0 - (dist / span) * 2.5)
                weighted_score += weight * decay * 0.3

        if total_weight == 0:
            return 0.0

        raw = weighted_score / total_weight

        # Apply mode-based scaling
        if self.mode == "Deep (Slower)":
            raw = min(raw * 1.15, 1.0)
        elif self.mode == "Fast (Less Accurate)":
            raw = raw * 0.9

        return float(np.clip(raw, 0.0, 1.0))
