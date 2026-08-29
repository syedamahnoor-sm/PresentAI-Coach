import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PresentAI Coach",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# ENHANCED MODERN CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Root Palette & Font Variables */
    :root {
        --bg-main: #070913;
        --card-bg: rgba(22, 27, 46, 0.55);
        --card-border: rgba(255, 255, 255, 0.08);
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --primary-glow: rgba(99, 102, 241, 0.25);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Global Application Reset */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 90% 10%, rgba(168, 85, 247, 0.10) 0px, transparent 50%),
            radial-gradient(at 50% 90%, rgba(14, 165, 233, 0.08) 0px, transparent 50%);
        background-attachment: fixed;
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hide Default Chrome */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Glass Container Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        border: 1px solid var(--card-border) !important;
        background: var(--card-bg) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(168, 85, 247, 0.3) !important;
    }

    /* Typography Upgrades */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-secondary);
    text-align: center;
    width: 100%;             /* Allow full width centering */
    max-width: 100%;         /* Remove narrow container constraint */
    margin: 0 auto 2rem auto;
    line-height: 1.6;
    }

    .card-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Feature Badges */
    .badge-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.6rem;
        margin: 1rem 0;
    }

    .badge-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 0.88rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom Metrics */
    div[data-testid="stMetric"] {
        padding: 14px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
    }

    /* Button Styling */
    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: var(--primary-gradient);
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 20px var(--primary-glow);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.4);
        color: white;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: none;
        color: var(--text-primary);
    }

    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }

    /* File Uploader Customization */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 1.5px dashed rgba(168, 85, 247, 0.4);
        background: rgba(15, 23, 42, 0.6);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

if "mode" not in st.session_state:
    st.session_state.mode = None

# =========================================================
# NAVBAR
# =========================================================

nav_col1, nav_col2 = st.columns([4, 1])

with nav_col1:
    st.markdown(
        "<span style='font-size: 1.25rem; font-weight: 700; color: #fff;'>"
        "✨ Present<span style='color: #a855f7;'>AI</span> Coach</span>",
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        "<div style='text-align: right;'>"
        "<span style='background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); "
        "color: #c084fc; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;'>"
        "v2.0 ACTIVE</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<hr style='margin: 0.8rem 0 2rem 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);'>",
    unsafe_allow_html=True,
)

# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.mode is None:

    # Hero Section
    st.markdown("<h1 class='hero-title'>Master Your Presentation Skills</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-subtitle'>"
        "Get instant, actionable feedback on visual delivery, speech pace, posture, and facial expressions powered by real-time computer vision."
        "</p>",
        unsafe_allow_html=True,
    )

    # Mode Cards Section
    live_col, upload_col = st.columns(2, gap="large")

    with live_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>🎥 Real-Time Practice</div>", unsafe_allow_html=True)
            st.write(
                "Connect your camera and mic to receive instant visual feedback while practicing live."
            )
            
            st.markdown(
                """
                <div class="badge-grid">
                    <div class="badge-item">🧍 Posture Tracking</div>
                    <div class="badge-item">👋 Gesture Analysis</div>
                    <div class="badge-item">👁️ Eye Contact</div>
                    <div class="badge-item">🎙️ Pace & Pauses</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.caption("🔒 On-device processing • Complete privacy")
            st.write("")

            if st.button("Start Live Session →", key="live_home", type="primary"):
                st.session_state.mode = "live"
                st.rerun()

    with upload_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>☁️ Video Recording Analysis</div>", unsafe_allow_html=True)
            st.write(
                "Upload a pre-recorded presentation video to generate a comprehensive diagnostic report."
            )
            
            st.markdown(
                """
                <div class="badge-grid">
                    <div class="badge-item">🎬 Full Video Audit</div>
                    <div class="badge-item">🎧 Audio Diagnostics</div>
                    <div class="badge-item">💬 Filler Detection</div>
                    <div class="badge-item">📊 Overall Scoring</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("🔒 Secure processing • Encrypted local upload")
            st.write("")

            if st.button("Upload Recording →", key="upload_home", type="primary"):
                st.session_state.mode = "upload"
                st.rerun()

    st.write("")
    st.write("")

    # Feature Highlights Grid
    b1, b2, b3, b4 = st.columns(4)

    features = [
        ("⚡ Multi-Modal", "Synchronized visual and audio intelligence processing."),
        ("🚀 Instant Metrics", "Immediate key performance breakdown after every run."),
        ("🔐 Privacy First", "Your video and speech recordings are never saved remotely."),
        ("🤖 AI Insights", "Turn raw analytical metrics into practical action steps."),
    ]

    for col, (title, desc) in zip([b1, b2, b3, b4], features):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)

# =========================================================
# LIVE PRESENTATION PAGE
# =========================================================

elif st.session_state.mode == "live":

    title_col, back_col = st.columns([5, 1])

    with title_col:
        st.markdown("<h2 style='margin:0;'>🎥 Live Presentation Mode</h2>", unsafe_allow_html=True)
        st.caption("Align your frame properly before initiating real-time feedback.")

    with back_col:
        if st.button("← Back", key="back_live", type="secondary"):
            st.session_state.mode = None
            st.rerun()

    st.write("")

    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("### Setup Guidelines")
            st.write(
                "Position your web camera at eye level, ensure well-lit surrounding space, "
                "and speak clearly into your primary microphone."
            )
            st.info("💡 Keep your upper body and hands within frame to enable posture evaluation.")

        with right:
            st.markdown("### Active Trackers")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Visual Indicators", "3 Metrics")
                st.caption("Posture, Gestures, Gaze")
            with m2:
                st.metric("Speech Indicators", "4 Metrics")
                st.caption("Pace, Fillers, Tone, Clarity")

    st.write("")

    if st.button("Initialize Camera & Begin →", key="start_live", type="primary"):
        st.info("Live streaming pipeline target connected.")

# =========================================================
# UPLOAD PRESENTATION PAGE
# =========================================================

elif st.session_state.mode == "upload":

    title_col, back_col = st.columns([5, 1])

    with title_col:
        st.markdown("<h2 style='margin:0;'>☁️ Upload Recording</h2>", unsafe_allow_html=True)
        st.caption("Upload your recorded presentation for comprehensive delivery processing.")

    with back_col:
        if st.button("← Back", key="back_upload", type="secondary"):
            st.session_state.mode = None
            st.rerun()

    st.write("")

    uploaded_file = st.file_uploader(
        "Select presentation video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported Formats: MP4, MOV, AVI, MKV (Max 200MB)",
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully. Ready for evaluation.")
        st.write("")

        preview_col, info_col = st.columns([3, 2], gap="large")

        with preview_col:
            st.video(uploaded_file)

        with info_col:
            with st.container(border=True):
                st.markdown("### Analysis Pipeline")
                st.write("Your recording will be processed through the following modules:")
                st.markdown(
                    """
                    - **Posture & Frame Alignment**
                    - **Gesture Frequency Tracking**
                    - **Eye Contact Duration**
                    - **Speech Tempo & Filler Word Audit**
                    """
                )

        st.write("")

        if st.button("Run Full Diagnostic Analysis →", key="analyze_upload", type="primary"):
            st.info("Processing video framework pipeline...")

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<hr style='margin: 3rem 0 1rem 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);'>",
    unsafe_allow_html=True,
)

footer_l, footer_r = st.columns([4, 1])

with footer_l:
    st.caption("PresentAI Coach • Powered by Real-Time Vision & Speech Analytics")

with footer_r:
    st.markdown(
        "<div style='text-align: right;'><span style='color: #64748b; font-size: 0.8rem;'>"
        "Designed for Streamlit</span></div>",
        unsafe_allow_html=True,
    )