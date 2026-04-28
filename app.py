import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import librosa
import queue

# --- UI Configuration ---
st.set_page_config(page_title="Instrument Detector", page_icon="🎸")
st.title("🎸 Real-Time Instrument Classifier")
st.markdown("Launch the mic and play an instrument to see the detection!")

# --- Mock Prediction Logic ---
# In a production app, you'd load a .h5 or .pkl model here
def predict_instrument(audio_data, sr):
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    # This is a placeholder for your model's prediction logic
    # For now, we'll simulate a result
    instruments = ["Guitar", "Piano", "Violin", "Drums", "Flute"]
    return np.random.choice(instruments)

# --- Audio Processing Class ---
class InstrumentProcessor(AudioProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()

    def recv(self, frame):
        # Convert PyAV frame to numpy array
        audio = frame.to_ndarray().flatten().astype(np.float32)
        
        # Simple processing every ~1 second of audio
        if len(audio) > 0:
            label = predict_instrument(audio, 16000)
            self.result_queue.put(label)
        
        return frame

# --- Streamlit Layout ---
ctx = webrtc_streamer(
    key="instrument-detect",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=InstrumentProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Visualizing the Output
if ctx.audio_processor:
    result_placeholder = st.empty()
    while True:
        try:
            label = ctx.audio_processor.result_queue.get(timeout=1.0)
            # Add a bit of "animation" style via markdown and metrics
            result_placeholder.markdown(f"### Detected: `{label}`")
            st.balloons() if label == "Guitar" else None # Fun trigger
        except queue.Empty:
            break
