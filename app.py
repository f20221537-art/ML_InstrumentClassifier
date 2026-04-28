import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="🎵 Instrument Detector",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .main-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #f7971e, #ffd200, #f7971e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #888;
        font-family: 'Space Mono', monospace; font-size: 0.85rem;
        margin-bottom: 2rem; letter-spacing: 2px; text-transform: uppercase;
    }
    .instrument-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 12px;
        padding: 1.2rem; margin: 0.5rem 0;
    }
    .confidence-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #f7971e, #ffd200); margin-top: 8px; }
    .stButton button {
        background: linear-gradient(135deg, #f7971e, #ffd200) !important;
        color: #000 !important; font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; font-family: 'Space Mono', monospace !important;
        letter-spacing: 1px !important; text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
    }
    .detected-label { font-size: 1.1rem; font-weight: 700; color: #ffd200; }
    .confidence-pct { font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #f7971e; }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #ffd200;
        border-bottom: 1px solid #1e1e3f; padding-bottom: 0.5rem;
        margin-bottom: 1rem; font-family: 'Space Mono', monospace;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-box { background: #0f0f23; border: 1px solid #1e1e3f; border-radius: 10px; padding: 1rem; text-align: center; }
    .info-tag { display: inline-block; background: #0f3460; color: #ffd200; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-family: 'Space Mono', monospace; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_features(y, sr):
    features = {}

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy   = np.sum(y_harmonic ** 2) + 1e-10
    percussive_energy = np.sum(y_percussive ** 2) + 1e-10
    total_energy      = harmonic_energy + percussive_energy

    features["harmonic_ratio"]   = float(harmonic_energy / total_energy)
    features["percussive_ratio"] = float(percussive_energy / total_energy)

    nyquist  = sr / 2.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["centroid_norm"] = float(np.mean(centroid) / nyquist)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth"] = float(np.clip(np.mean(bandwidth) / nyquist, 0, 1))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness"] = float(np.clip(np.mean(flatness), 0, 1))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = float(np.clip((np.mean(contrast) + 50) / 100.0, 0, 1))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zero_crossing_rate"] = float(np.clip(np.mean(zcr) * 10, 0, 1))

    onset_frames   = librosa.onset.onset_detect(y=y, sr=sr)
    duration       = librosa.get_duration(y=y, sr=sr)
    onsets_per_sec = len(onset_frames) / max(duration, 1e-3)
    features["onset_density"] = float(np.clip(onsets_per_sec / 10.0, 0, 1))

    fft   = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    total_fft = np.sum(fft ** 2) + 1e-10
    features["low_freq_energy_ratio"]  = float(np.sum(fft[freqs < 300]  ** 2) / total_fft)
    features["mid_freq_energy_ratio"]  = float(np.sum(fft[(freqs >= 300) & (freqs < 2000)] ** 2) / total_fft)
    features["high_freq_energy_ratio"] = float(np.sum(fft[freqs >= 2000] ** 2) / total_fft)

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_harmonic,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else np.array([])
        if len(voiced_f0) > 10:
            f0_smooth  = np.convolve(voiced_f0, np.ones(5) / 5, mode='valid')
            modulation = np.std(f0_smooth) / (np.mean(f0_smooth) + 1e-6)
            features["vibrato_presence"] = float(np.clip(modulation * 5, 0, 1))
        else:
            features["vibrato_presence"] = 0.0
    except Exception:
        features["vibrato_presence"] = 0.0

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_std = np.std(mfcc, axis=1)
    formant_score = np.mean(mfcc_std[2:6]) / (np.mean(mfcc_std) + 1e-6)
    features["formant_presence"] = float(np.clip(formant_score / 3.0, 0, 1))

    return features


# ── Instrument Profiles & Scoring ───────────────────────────────────────────

INSTRUMENT_PROFILES = {
    "Drums":       {"percussive_ratio":(0.3,1.0,1.0), "onset_density":(0.5,1.0,0.8), "spectral_flatness":(0.01,0.5,0.6), "zero_crossing_rate":(0.05,0.5,0.4)},
    "Bass Guitar": {"centroid_norm":(0.0,0.15,1.0), "low_freq_energy_ratio":(0.4,1.0,1.0), "harmonic_ratio":(0.3,1.0,0.7), "spectral_bandwidth":(0.0,0.25,0.6)},
    "Piano":       {"harmonic_ratio":(0.4,1.0,1.0), "centroid_norm":(0.1,0.5,0.8), "onset_density":(0.1,0.8,0.5), "spectral_contrast_mean":(0.3,1.0,0.7)},
    "Guitar":      {"harmonic_ratio":(0.35,0.9,1.0), "centroid_norm":(0.1,0.4,0.8), "spectral_flatness":(0.005,0.15,0.6), "mid_freq_energy_ratio":(0.2,0.8,0.7)},
    "Violin":      {"harmonic_ratio":(0.5,1.0,1.0), "centroid_norm":(0.25,0.65,1.0), "spectral_bandwidth":(0.1,0.4,0.7), "vibrato_presence":(0.1,1.0,0.8)},
    "Cello":       {"harmonic_ratio":(0.5,1.0,1.0), "centroid_norm":(0.05,0.3,1.0), "spectral_bandwidth":(0.05,0.35,0.8), "vibrato_presence":(0.05,1.0,0.6)},
    "Trumpet":     {"harmonic_ratio":(0.4,1.0,0.9), "centroid_norm":(0.3,0.7,1.0), "high_freq_energy_ratio":(0.15,0.7,0.8), "spectral_contrast_mean":(0.4,1.0,0.7)},
    "Saxophone":   {"harmonic_ratio":(0.35,0.9,0.9), "centroid_norm":(0.15,0.55,1.0), "spectral_flatness":(0.002,0.08,0.7), "mid_freq_energy_ratio":(0.3,0.85,0.8)},
    "Flute":       {"harmonic_ratio":(0.3,0.85,0.8), "centroid_norm":(0.4,0.85,1.0), "spectral_flatness":(0.01,0.2,0.7), "high_freq_energy_ratio":(0.2,0.8,0.8)},
    "Vocals":      {"centroid_norm":(0.1,0.5,0.8), "harmonic_ratio":(0.4,1.0,1.0), "formant_presence":(0.2,1.0,1.0), "vibrato_presence":(0.05,1.0,0.6)},
    "Synthesizer": {"spectral_flatness":(0.05,0.6,1.0), "harmonic_ratio":(0.2,0.8,0.6), "centroid_norm":(0.05,0.8,0.4), "onset_density":(0.0,0.5,0.5)},
    "Trombone":    {"harmonic_ratio":(0.4,1.0,0.9), "centroid_norm":(0.1,0.45,1.0), "low_freq_energy_ratio":(0.2,0.75,0.8), "spectral_bandwidth":(0.1,0.45,0.7)},
}


def score_instrument(features, profile):
    total_weight, weighted_score = 0.0, 0.0
    for feat_name, (low, high, weight) in profile.items():
        if feat_name not in features:
            continue
        val = features[feat_name]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        total_weight += weight
        if low <= val <= high:
            center    = (low + high) / 2
            spread    = max((high - low) / 2, 1e-6)
            proximity = 1.0 - abs(val - center) / spread
            weighted_score += weight * (0.7 + 0.3 * proximity)
        else:
            dist  = min(abs(val - low), abs(val - high))
            span  = max(high - low, 1e-6)
            decay = max(0.0, 1.0 - (dist / span) * 2.5)
            weighted_score += weight * decay * 0.3
    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_score / total_weight, 0.0, 1.0))


def classify_instruments(y, sr, confidence_threshold=0.3, max_instruments=6, mode="Standard"):
    features = extract_features(y, sr)
    scores = {}
    for instrument, profile in INSTRUMENT_PROFILES.items():
        s = score_instrument(features, profile)
        if mode == "Deep (Slower)":
            s = min(s * 1.15, 1.0)
        elif mode == "Fast (Less Accurate)":
            s = s * 0.9
        if s >= confidence_threshold:
            scores[instrument] = s
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_instruments]


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_waveform(y, sr):
    times = np.linspace(0, len(y) / sr, num=len(y))
    step  = max(1, len(times) // 2000)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=times[::step], y=y[::step], mode='lines',
        line=dict(color='#f7971e', width=1),
        fill='tozeroy', fillcolor='rgba(247,151,30,0.08)'
    ))
    fig.update_layout(
        title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'), title_font=dict(color='#ffd200'),
        margin=dict(l=20,r=20,t=40,b=20), height=200,
    )
    fig.update_xaxes(gridcolor='#1e1e3f', zerolinecolor='#1e1e3f')
    fig.update_yaxes(gridcolor='#1e1e3f', zerolinecolor='#1e1e3f')
    return fig


def plot_spectrogram(y, sr):
    D   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = px.imshow(D, aspect='auto', color_continuous_scale='Hot',
                    origin='lower', labels=dict(x="Time Frame", y="Frequency Bin", color="dB"))
    fig.update_layout(
        title="Spectrogram", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'), title_font=dict(color='#ffd200'),
        margin=dict(l=20,r=20,t=40,b=20), height=250, coloraxis_showscale=False,
    )
    return fig


def plot_confidence_bar(instruments):
    if not instruments:
        return None
    categories = [i[0] for i in instruments]
    values     = [round(i[1] * 100, 1) for i in instruments]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker=dict(color=values, colorscale=[[0,'#0f3460'],[0.5,'#f7971e'],[1,'#ffd200']], showscale=False)
    ))
    fig.update_layout(
        title="Confidence Scores", yaxis_title="Confidence (%)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,15,35,0.8)',
        font=dict(color='#888', family='Space Mono'), title_font=dict(color='#ffd200'),
        margin=dict(l=20,r=20,t=40,b=20), height=280, yaxis=dict(range=[0,105]),
    )
    fig.update_xaxes(gridcolor='#1e1e3f')
    fig.update_yaxes(gridcolor='#1e1e3f')
    return fig


EMOJIS = {
    "Guitar":"🎸","Piano":"🎹","Drums":"🥁","Bass Guitar":"🎸",
    "Violin":"🎻","Flute":"🎵","Trumpet":"🎺","Saxophone":"🎷",
    "Vocals":"🎤","Synthesizer":"🎛️","Cello":"🎻","Trombone":"🎺",
}


def render_instrument_results(instruments):
    if not instruments:
        st.info("🎵 No instruments detected above the threshold. Try lowering the confidence slider.")
        return
    st.markdown('<div class="section-header">🎼 Detected Instruments</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(instruments), 3))
    for i, (instrument, confidence) in enumerate(instruments):
        with cols[i % 3]:
            emoji = EMOJIS.get(instrument, "🎵")
            bar_w = int(confidence * 100)
            st.markdown(f"""
            <div class="instrument-card">
                <div class="detected-label">{emoji} {instrument}</div>
                <div class="confidence-pct">{confidence*100:.1f}% confidence</div>
                <div style="background:#1e1e3f;border-radius:4px;margin-top:8px;height:8px;">
                    <div class="confidence-bar" style="width:{bar_w}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ── App Layout ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🎵 Instrument Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time audio analysis · Powered by librosa</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    max_instruments      = st.slider("Max Instruments Shown", 1, 10, 6)
    analysis_mode        = st.selectbox("Analysis Mode", ["Standard", "Deep (Slower)", "Fast (Less Accurate)"])
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("Upload any audio file and the app detects instruments using spectral analysis.\n\nSupports: `.wav` `.mp3` `.flac` `.ogg` `.m4a`")
    st.markdown("---")
    st.markdown("### 🎼 Detectable Instruments")
    for label in ["🎸 Guitar","🎹 Piano","🥁 Drums","🎸 Bass Guitar","🎻 Violin","🎻 Cello","🎺 Trumpet","🎺 Trombone","🎷 Saxophone","🎵 Flute","🎤 Vocals","🎛️ Synthesizer"]:
        st.markdown(f'<span class="info-tag">{label}</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("🎧 Upload an audio file to analyze", type=["wav","mp3","flac","ogg","m4a"])

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
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
            with st.spinner("🎵 Analyzing audio..."):
                progress = st.progress(0)

                y, sr    = librosa.load(tmp_path, sr=None, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                progress.progress(20)

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Duration", f"{duration:.1f}s")
                with c2: st.metric("Sample Rate", f"{sr:,} Hz")
                with c3: st.metric("Samples", f"{len(y):,}")
                with c4: st.metric("Channels", "Mono")

                progress.progress(40)

                if show_viz:
                    v1, v2 = st.columns(2)
                    with v1: st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
                    with v2: st.plotly_chart(plot_spectrogram(y, sr), use_container_width=True)

                progress.progress(60)

                instruments = classify_instruments(
                    y, sr,
                    confidence_threshold=confidence_threshold,
                    max_instruments=max_instruments,
                    mode=analysis_mode
                )
                progress.progress(90)

                render_instrument_results(instruments)

                if instruments and show_viz:
                    fig = plot_confidence_bar(instruments)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                progress.progress(100)
                st.success(f"✅ Analysis complete! Found {len(instruments)} instrument(s).")

                if instruments:
                    with st.expander("📊 Detailed Analysis Report"):
                        st.table({
                            "Instrument":     [i[0] for i in instruments],
                            "Confidence (%)": [f"{i[1]*100:.1f}" for i in instruments],
                            "Likelihood":     ["High" if i[1]>0.7 else "Medium" if i[1]>0.5 else "Low" for i in instruments],
                        })

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure the file is a valid, non-corrupted audio file.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    cards = [("🎸","Upload Audio","Drag & drop any audio file above"),
             ("🔍","AI Analysis","Spectral + harmonic feature extraction"),
             ("🎼","Get Results","Instruments with confidence scores")]
    for col, (icon, title, desc) in zip([c1,c2,c3], cards):
        with col:
            st.markdown(f"""
            <div class="instrument-card" style="text-align:center;">
                <div style="font-size:2rem">{icon}</div>
                <div class="detected-label">{title}</div>
                <div style="color:#666;font-size:0.85rem;margin-top:0.5rem">{desc}</div>
            </div>""", unsafe_allow_html=True)
