import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv
import io
import urllib.request

# Instrument-related class labels from AudioSet (YAMNet)
INSTRUMENT_KEYWORDS = [
    "guitar", "piano", "drum", "violin", "flute", "saxophone", "trumpet",
    "bass", "cello", "harp", "organ", "synthesizer", "banjo", "mandolin",
    "ukulele", "harmonica", "accordion", "trombone", "clarinet", "oboe",
    "tabla", "sitar", "mridangam", "percussion", "keyboard", "xylophone",
    "marimba", "steel guitar", "electric guitar", "acoustic guitar"
]

YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

_model = None
_class_names = None


def load_model():
    global _model, _class_names
    if _model is None:
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    if _class_names is None:
        with urllib.request.urlopen(YAMNET_CLASS_MAP_URL) as f:
            reader = csv.DictReader(io.TextIOWrapper(f))
            _class_names = [row["display_name"] for row in reader]
    return _model, _class_names


def preprocess_audio(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16kHz mono float32 as required by YAMNet."""
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    audio_data = audio_data.astype(np.float32)
    # Normalize
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data


def detect_instruments(audio_data: np.ndarray, sr: int, top_k: int = 10):
    """
    Run YAMNet on audio and return instrument detections.

    Returns:
        List of (instrument_name, confidence) tuples sorted by confidence.
    """
    model, class_names = load_model()
    waveform = preprocess_audio(audio_data, sr)

    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Get all predictions
    top_indices = np.argsort(mean_scores)[::-1]

    instruments_found = []
    seen = set()

    for idx in top_indices:
        label = class_names[idx].lower()
        score = float(mean_scores[idx])

        if score < 0.05:
            break

        for keyword in INSTRUMENT_KEYWORDS:
            if keyword in label and label not in seen:
                seen.add(label)
                instruments_found.append({
                    "instrument": class_names[idx],
                    "confidence": round(score * 100, 2),
                    "raw_label": label
                })
                break

        if len(instruments_found) >= top_k:
            break

    return instruments_found


def get_audio_features(audio_data: np.ndarray, sr: int) -> dict:
    """Extract musical features for display."""
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    rms = np.mean(librosa.feature.rms(y=audio_data))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

    return {
        "tempo_bpm": round(float(tempo), 1),
        "spectral_centroid_hz": round(float(spectral_centroid), 1),
        "rms_energy": round(float(rms), 4),
        "zero_crossing_rate": round(float(zcr), 4),
    }
