import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import time
import threading
import queue
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from instrument_detector import detect_instruments, get_audio_features, load_model

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Instrument Detector",
    page_icon="🎸",
    layout="wide",
)

st.title("🎵 Real-Time Instrument Detector")
st.markdown(
    "Upload an audio file **or** record live audio — this app will identify "
    "which instruments are being played using Google's YAMNet model."
)

# ─── Load model once ────────────────────────────────────────────────────────────
with st.spinner("Loading YAMNet model (first run may take ~30s)..."):
    load_model()
st.success("✅ Model loaded!")

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["📂 Upload File", "🎤 Live Microphone"])
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", min_value=1, max_value=50, value=5
)
top_k = st.sidebar.slider("Max Instruments to Show", min_value=3, max_value=15, value=8)

# ─── Helper: Plot waveform ───────────────────────────────────────────────────────
def plot_waveform(audio_data, sr):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), facecolor="#0e1117")
    ax1, ax2 = axes

    # Waveform
    times = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax1.plot(times, audio_data, color="#1DB954", linewidth=0.5)
    ax1.set_facecolor("#0e1117")
    ax1.set_ylabel("Amplitude", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title("Waveform", color="white")

    # Spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data.astype(np.float32))), ref=np.max
    )
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax2, cmap="magma")
    ax2.set_facecolor("#0e1117")
    ax2.set_ylabel("Frequency (Hz)", color="white")
    ax2.set_xlabel("Time (s)", color="white")
    ax2.tick_params(colors="white")
    ax2.set_title("Spectrogram", color="white")

    fig.tight_layout()
    return fig


# ─── Helper: Display results ─────────────────────────────────────────────────────
def display_results(instruments, features, audio_data, sr):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎼 Detected Instruments")
        filtered = [i for i in instruments if i["confidence"] >= confidence_threshold]

        if not filtered:
            st.warning("No instruments detected above the confidence threshold.")
        else:
            for item in filtered[:top_k]:
                conf = item["confidence"]
                bar_color = (
                    "🟢" if conf > 30 else "🟡" if conf > 15 else "🔴"
                )
                st.markdown(f"**{bar_color} {item['instrument']}**")
                st.progress(min(int(conf), 100), text=f"{conf}% confidence")

    with col2:
        st.subheader("📊 Audio Features")
        st.metric("🥁 Tempo", f"{features['tempo_bpm']} BPM")
        st.metric("🌊 Spectral Centroid", f"{features['spectral_centroid_hz']} Hz")
        st.metric("⚡ RMS Energy", str(features["rms_energy"]))
        st.metric("〰️ Zero Crossing Rate", str(features["zero_crossing_rate"]))

    st.subheader("📈 Waveform & Spectrogram")
    fig = plot_waveform(
        audio_data if audio_data.ndim == 1 else np.mean(audio_data, axis=1), sr
    )
    st.pyplot(fig)
    plt.close(fig)


# ─── Mode 1: Upload File ─────────────────────────────────────────────────────────
if mode == "📂 Upload File":
    st.header("📂 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with st.spinner("🔍 Analyzing instruments..."):
            audio_bytes = uploaded_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)

            # Analyze in 30-second chunks if long
            if audio_data.ndim > 1:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data

            max_samples = 30 * sr
            clip = audio_mono[:max_samples]

            instruments = detect_instruments(clip, sr, top_k=top_k)
            features = get_audio_features(clip, sr)

        display_results(instruments, features, clip, sr)


# ─── Mode 2: Live Microphone ─────────────────────────────────────────────────────
elif mode == "🎤 Live Microphone":
    st.header("🎤 Live Microphone Recording")

    duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=30, value=10)

    if st.button("🔴 Start Recording", type="primary"):
        progress_bar = st.progress(0, text="Recording...")
        status = st.empty()

        SAMPLE_RATE = 16000
        recording = []

        def audio_callback(indata, frames, time_info, status_flags):
            recording.append(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            for i in range(duration):
                time.sleep(1)
                progress_bar.progress((i + 1) / duration, text=f"Recording... {i+1}/{duration}s")

        status.info("⏳ Processing recording...")

        audio_data = np.concatenate(recording, axis=0).flatten()

        with st.spinner("🔍 Analyzing instruments..."):
            instruments = detect_instruments(audio_data, SAMPLE_RATE, top_k=top_k)
            features = get_audio_features(audio_data, SAMPLE_RATE)

        status.success("✅ Analysis complete!")
        display_results(instruments, features, audio_data, SAMPLE_RATE)

        # Offer download of recording
        buf = io.BytesIO()
        sf.write(buf, audio_data, SAMPLE_RATE, format="WAV")
        st.download_button(
            "💾 Download Recording",
            data=buf.getvalue(),
            file_name="recording.wav",
            mime="audio/wav",
        )

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built with [YAMNet](https://tfhub.dev/google/yamnet/1) · "
    "[TensorFlow Hub](https://www.tensorflow.org/hub) · "
    "[librosa](https://librosa.org) · "
    "[Streamlit](https://streamlit.io)"
)
