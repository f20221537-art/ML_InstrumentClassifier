import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import time
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from instrument_classifier import InstrumentClassifier
from audio_utils import extract_features, get_spectral_features

st.set_page_config(
    page_title="🎵 Instrument Detector",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f7971e, #ffd200, #f7971e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .instrument-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .instrument-card:hover {
        border-color: #f7971e;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(247, 151, 30, 0.15);
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        margin-top: 8px;
        transition: width 0.5s ease;
    }

    .stButton button {
        background: linear-gradient(135deg, #f7971e, #ffd200) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(247, 151, 30, 0.4) !important;
    }

    .metric-box {
        background: #0f0f23;
        border: 1px solid #1e1e3f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .detected-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffd200;
    }

    .confidence-pct {
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: #f7971e;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #0f3460;
        border-radius: 12px;
        padding: 1rem;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffd200;
        border-bottom: 1px solid #1e1e3f;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-tag {
        display: inline-block;
        background: #0f3460;
        color: #ffd200;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


def render_instrument_results(instruments):
    """Render detected instruments as cards."""
    if not instruments:
        st.info("🎵 No instruments detected. Try uploading an audio file!")
        return

    st.markdown('<div class="section-header">🎼 Detected Instruments</div>', unsafe_allow_html=True)

    cols = st.columns(min(len(instruments), 3))
    for i, (instrument, confidence) in enumerate(instruments):
        with cols[i % 3]:
            bar_width = int(confidence * 100)
            emoji = get_instrument_emoji(instrument)
            st.markdown(f"""
            <div class="instrument-card">
                <div class="detected-label">{emoji} {instrument}</div>
                <div class="confidence-pct">{confidence*100:.1f}% confidence</div>
                <div style="background:#1e1e3f; border-radius:4px; margin-top:8px; height:8px;">
                    <div class="confidence-bar" style="width:{bar_width}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def get_instrument_emoji(instrument):
    emojis = {
        "Guitar": "🎸", "Piano": "🎹", "Drums": "🥁",
        "Bass Guitar": "🎸", "Violin": "🎻", "Flute": "🎵",
        "Trumpet": "🎺", "Saxophone": "🎷", "Vocals": "🎤",
        "Synthesizer": "🎛️", "Cello": "🎻", "Clarinet": "🎵",
        "Harp": "🪗", "Organ": "🎹", "Banjo": "🪕",
        "Mandolin": "🎸", "Ukulele": "🪗", "Trombone": "🎺",
    }
    return emojis.get(instrument, "🎵")


def plot_waveform(y, sr):
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times[::max(1, len(times)//2000)],
        y=y[::max(1, len(y)//2000)],
        mode='lines',
        line=dict(color='#f7971e', width=1),
        fill='tozeroy',
        fillcolor='rgba(247,151,30,0.08)'
    ))
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'),
        title_font=dict(color='#ffd200'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    fig.update_xaxes(gridcolor='#1e1e3f', zerolinecolor='#1e1e3f')
    fig.update_yaxes(gridcolor='#1e1e3f', zerolinecolor='#1e1e3f')
    return fig


def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = px.imshow(
        D,
        aspect='auto',
        color_continuous_scale='Hot',
        origin='lower',
        labels=dict(x="Time Frame", y="Frequency Bin", color="dB")
    )
    fig.update_layout(
        title="Spectrogram",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'),
        title_font=dict(color='#ffd200'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        coloraxis_showscale=False
    )
    return fig


def plot_confidence_radar(instruments):
    if not instruments:
        return None

    categories = [i[0] for i in instruments]
    values = [i[1] * 100 for i in instruments]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=values,
            colorscale=[[0, '#0f3460'], [0.5, '#f7971e'], [1, '#ffd200']],
            showscale=False
        )
    ))
    fig.update_layout(
        title="Confidence Scores",
        yaxis_title="Confidence (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'),
        title_font=dict(color='#ffd200'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        yaxis=dict(range=[0, 105])
    )
    fig.update_xaxes(gridcolor='#1e1e3f')
    fig.update_yaxes(gridcolor='#1e1e3f')
    return fig


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎵 Instrument Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time audio analysis · Powered by librosa + ML</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                                     help="Only show instruments above this confidence level")
    max_instruments = st.slider("Max Instruments Shown", 1, 10, 6,
                                 help="Maximum number of instruments to display")
    analysis_mode = st.selectbox("Analysis Mode", ["Standard", "Deep (Slower)", "Fast (Less Accurate)"])

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app analyzes audio files and detects which instruments are being played using:
    - **Spectral analysis** via librosa
    - **Harmonic/percussive separation**
    - **Feature-based ML classification**
    
    Supports: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
    """)

    st.markdown("---")
    st.markdown("### 🎼 Detectable Instruments")
    instruments_list = [
        "🎸 Guitar", "🎹 Piano", "🥁 Drums", "🎻 Violin",
        "🎺 Trumpet", "🎷 Saxophone", "🎵 Flute", "🎤 Vocals",
        "🎛️ Synthesizer", "🎸 Bass Guitar", "🎺 Trombone", "🎻 Cello"
    ]
    for inst in instruments_list:
        st.markdown(f'<span class="info-tag">{inst}</span>', unsafe_allow_html=True)

# Main content
uploaded_file = st.file_uploader(
    "🎧 Upload an audio file to analyze",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supports WAV, MP3, FLAC, OGG, M4A"
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        st.audio(uploaded_file)

        col1, col2, col3 = st.columns(3)
        with col1:
            analyze_btn = st.button("🔍 Analyze Instruments", use_container_width=True)
        with col2:
            show_viz = st.checkbox("Show Visualizations", value=True)
        with col3:
            st.markdown(f'<div class="metric-box"><small>📁 File</small><br><b style="color:#ffd200">{uploaded_file.name}</b></div>', unsafe_allow_html=True)

        if analyze_btn:
            with st.spinner("🎵 Loading and analyzing audio..."):
                progress = st.progress(0)

                # Load audio
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                progress.progress(20)

                # Display basic info
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("Duration", f"{duration:.1f}s")
                with info_cols[1]:
                    st.metric("Sample Rate", f"{sr:,} Hz")
                with info_cols[2]:
                    st.metric("Samples", f"{len(y):,}")
                with info_cols[3]:
                    st.metric("Channels", "Mono")

                progress.progress(40)

                # Visualizations
                if show_viz:
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                    with viz_col2:
                        st.plotly_chart(plot_spectrogram(y, sr), use_container_width=True)

                progress.progress(60)

                # Instrument classification
                classifier = InstrumentClassifier(
                    confidence_threshold=confidence_threshold,
                    max_instruments=max_instruments,
                    mode=analysis_mode
                )
                instruments = classifier.classify(y, sr)
                progress.progress(90)

                # Results
                render_instrument_results(instruments)

                if instruments and show_viz:
                    radar_fig = plot_confidence_radar(instruments)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)

                progress.progress(100)
                st.success(f"✅ Analysis complete! Found {len(instruments)} instrument(s).")

                # Detailed breakdown
                if instruments:
                    with st.expander("📊 Detailed Analysis Report"):
                        report_data = {
                            "Instrument": [i[0] for i in instruments],
                            "Confidence (%)": [f"{i[1]*100:.1f}" for i in instruments],
                            "Likelihood": ["High" if i[1] > 0.7 else "Medium" if i[1] > 0.5 else "Low" for i in instruments]
                        }
                        st.table(report_data)

                        # Feature info
                        features = extract_features(y, sr)
                        st.markdown("**Spectral Features Extracted:**")
                        feat_cols = st.columns(3)
                        feature_names = list(features.keys())[:9]
                        for idx, fname in enumerate(feature_names):
                            with feat_cols[idx % 3]:
                                val = features[fname]
                                if isinstance(val, (np.floating, float)):
                                    st.metric(fname, f"{val:.4f}")

    except Exception as e:
        st.error(f"❌ Error analyzing file: {str(e)}")
        st.info("Make sure the file is a valid audio file and not corrupted.")
    finally:
        os.unlink(tmp_path)

else:
    # Landing state
    st.markdown("---")
    demo_cols = st.columns(3)
    with demo_cols[0]:
        st.markdown("""
        <div class="instrument-card" style="text-align:center;">
            <div style="font-size:2rem">🎸</div>
            <div class="detected-label">Upload Audio</div>
            <div style="color:#666; font-size:0.85rem; margin-top:0.5rem">Drag & drop any audio file above</div>
        </div>
        """, unsafe_allow_html=True)
    with demo_cols[1]:
        st.markdown("""
        <div class="instrument-card" style="text-align:center;">
            <div style="font-size:2rem">🔍</div>
            <div class="detected-label">AI Analysis</div>
            <div style="color:#666; font-size:0.85rem; margin-top:0.5rem">Spectral + harmonic feature extraction</div>
        </div>
        """, unsafe_allow_html=True)
    with demo_cols[2]:
        st.markdown("""
        <div class="instrument-card" style="text-align:center;">
            <div style="font-size:2rem">🎼</div>
            <div class="detected-label">Get Results</div>
            <div style="color:#666; font-size:0.85rem; margin-top:0.5rem">See instruments with confidence scores</div>
        </div>
        """, unsafe_allow_html=True)
