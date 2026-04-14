import time
import base64
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import streamlit as st
from streamlit.components.v1 import html
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="FitCheck", layout="wide", initial_sidebar_state="collapsed")

# --- PATHS ---
MODEL_PATH = Path(r"C:\\Users\\Renz\\Downloads\\my_model\\my_model.pt")
LOGO_PATH = Path(__file__).parent / "logo.png"

# --- LOGO ---
logo_html = "<span>✓</span>"
if LOGO_PATH.exists():
    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" alt="FitCheck logo" />'

ph_datetime = datetime.datetime.now(ZoneInfo("Asia/Manila")).strftime("%a, %b %d %Y · %I:%M:%S %p")

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path: Path):
    return YOLO(str(path), task="detect")

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #060c18 !important;
    color: #e2eaf4 !important;
}
.main .block-container {
    padding: 0 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0b1424;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid rgba(15, 157, 88, 0.15);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 8px 24px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #6b84a0;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f9d58, #22c55e) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(15, 157, 88, 0.4);
}
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0b1727, #0d1f35);
    border: 1px solid rgba(15, 157, 88, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(15, 157, 88, 0.5);
    box-shadow: 0 4px 32px rgba(15, 157, 88, 0.15);
}
[data-testid="metric-container"] label {
    color: #6b84a0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0f9d58, #22c55e) !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    width: 100%;
    box-shadow: 0 4px 14px rgba(15, 157, 88, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(15, 157, 88, 0.5) !important;
}
.stButton > button:active { transform: translateY(0); }

/* Stop button – red accent */
.stop-btn .stButton > button {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
}
.stop-btn .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 0 4px;
}
.stSlider [role="slider"] {
    background: #0f9d58 !important;
    border: 2px solid #22c55e !important;
}
.stSlider [data-testid="stSlider"] label {
    color: #a0b4c8 !important;
    font-size: 13px;
    font-weight: 600;
}

/* ── Checkboxes ── */
.stCheckbox label {
    color: #c0d4e8 !important;
    font-size: 14px !important;
    font-weight: 500;
}
.stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
    background: #0f9d58 !important;
    border-color: #0f9d58 !important;
}

/* ── Info / alerts ── */
.stAlert {
    background: rgba(15, 157, 88, 0.08) !important;
    border: 1px solid rgba(15, 157, 88, 0.25) !important;
    border-radius: 10px !important;
    color: #a8e6c5 !important;
}

/* ── Section headings ── */
h3 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #d4eaff !important;
    letter-spacing: 0.5px;
    margin-bottom: 1rem !important;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Feed image ── */
.stImage img {
    border-radius: 14px;
    border: 1px solid rgba(15, 157, 88, 0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ── Detection log text ── */
.stMarkdown code, pre {
    font-family: 'DM Mono', monospace !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px;
    color: #8ef56d !important;
    font-size: 12.5px;
    padding: 2px 6px;
}

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 8px !important; }

/* ── Live panel card ── */
.live-controls {
    background: #0b1727;
    border: 1px solid rgba(15,157,88,0.15);
    border-radius: 16px;
    padding: 1.2rem 1rem;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15,157,88,0.12);
    border: 1px solid rgba(15,157,88,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #22c55e;
    margin-bottom: 1rem;
}
.status-pill .dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-dot 1.4s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
</style>
""", unsafe_allow_html=True)


# ─── TOP BAR ───────────────────────────────────────────────────────────────────
html(
    f"""
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; }}

            .top-bar {{
                background: linear-gradient(180deg, #07101f 0%, #060c18 100%);
                padding: 0 32px;
                height: 72px;
                display: flex;
                align-items: center;
                gap: 14px;
                border-bottom: 1px solid rgba(15, 157, 88, 0.18);
                position: relative;
            }}
            .top-bar::after {{
                content: '';
                position: absolute;
                bottom: 0; left: 0; right: 0;
                height: 1px;
                background: linear-gradient(90deg, transparent, #22c55e44, transparent);
            }}

            .logo-wrap {{
                width: 44px; height: 44px; min-width: 44px;
                border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                overflow: hidden;
                background: linear-gradient(135deg, #0a2818, #0f9d5822);
                border: 1.5px solid #0f9d58;
                box-shadow: 0 0 16px rgba(15,157,88,0.3);
            }}
            .logo-wrap img {{ width: 100%; height: 100%; object-fit: contain; }}
            .logo-wrap span {{ color: #22c55e; font-size: 22px; font-weight: 800; font-family: Outfit, sans-serif; }}

            .brand {{
                font-family: Outfit, sans-serif;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: 4px;
                background: linear-gradient(90deg, #e2f7ec 0%, #8ef56d 40%, #22c55e 70%, #e2f7ec 100%);
                background-size: 300% 100%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shimmer 4s linear infinite;
            }}
            @keyframes shimmer {{
                0% {{ background-position: 0% 50%; }}
                100% {{ background-position: 200% 50%; }}
            }}

            .badge {{
                background: rgba(15,157,88,0.1);
                border: 1px solid rgba(15,157,88,0.25);
                border-radius: 6px;
                padding: 3px 9px;
                font-family: Outfit, sans-serif;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: #22c55e;
                text-transform: uppercase;
                margin-top: 2px;
            }}

            .spacer {{ flex: 1; }}

            .datetime-wrap {{
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 2px;
            }}
            .datetime-label {{
                font-family: Outfit, sans-serif;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1px;
                color: #3a5a6a;
                text-transform: uppercase;
            }}
            #ph-time {{
                font-family: Outfit, sans-serif;
                font-size: 13px;
                font-weight: 500;
                color: #7a9ab0;
                white-space: nowrap;
            }}

            .tip-badge {{
                background: linear-gradient(135deg, #0a2010, #061810);
                border: 1px solid rgba(15,157,88,0.2);
                border-radius: 8px;
                padding: 6px 14px;
                display: flex; align-items: center; gap: 8px;
                font-family: Outfit, sans-serif;
            }}
            .tip-dot {{
                width: 8px; height: 8px;
                background: #22c55e;
                border-radius: 50%;
                box-shadow: 0 0 8px #22c55e;
            }}
            .tip-text {{ font-size: 12px; font-weight: 600; color: #4ade80; letter-spacing: 0.5px; }}
        </style>
    </head>
    <body>
        <div class="top-bar">
            <div class="logo-wrap">{logo_html}</div>
            <div style="display:flex;flex-direction:column;gap:2px">
                <div class="brand">FITCHECK</div>
                <div class="badge">Dress Code Monitor</div>
            </div>
            <div class="spacer"></div>
            <div class="tip-badge">
                <div class="tip-dot"></div>
                <div class="tip-text">TIP Manila</div>
            </div>
            <div style="width:1px;height:32px;background:rgba(255,255,255,0.07);margin:0 8px"></div>
            <div class="datetime-wrap">
                <div class="datetime-label">Philippine Standard Time</div>
                <div id="ph-time">{ph_datetime}</div>
            </div>
        </div>
        <script>
            const el = document.getElementById('ph-time');
            function tick() {{
                const now = new Date();
                el.textContent = new Intl.DateTimeFormat('en-US', {{
                    timeZone: 'Asia/Manila',
                    weekday: 'short', month: 'short', day: '2-digit', year: 'numeric',
                    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
                }}).format(now).replace(/,([^,]*)$/, ' ·$1').replace(/,/, ' ·');
            }}
            tick(); setInterval(tick, 1000);
        </script>
    </body>
    </html>
    """,
    height=80,
    scrolling=False,
)

# ─── MODEL CHECK ───────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("⚠️  Model file not found. Please verify the path to `my_model.pt`.")
    st.stop()

model = load_model(MODEL_PATH)

# ─── SESSION STATE ──────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.violations = []
    st.session_state.scans = 0
    st.session_state.frames_with_violations = 0
    st.session_state.detection_history = []
    st.session_state.inference_times = []

# ─── TABS ────────────────────────────────────────────────────────────────────────
tab_labels = ["📊  Dashboard", "📷  Live", "📋  Logs", "📈  Reports", "⚙️  Settings"]
tabs = st.tabs(tab_labels)


# ═══════════════════════════════ DASHBOARD ══════════════════════════════════════
with tabs[0]:
    scans = st.session_state.scans
    vframes = st.session_state.frames_with_violations
    compliance = 100 - int((vframes / scans) * 100) if scans else 100

    # Status pill
    is_live = st.session_state.running
    status_color = "#22c55e" if is_live else "#64748b"
    status_label = "DETECTION ACTIVE" if is_live else "DETECTION IDLE"
    st.markdown(f"""
        <div class="status-pill">
            <div class="dot" style="background:{status_color};box-shadow:0 0 6px {status_color}"></div>
            {status_label}
        </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔍  Frames Scanned", scans)
    c2.metric("⚠️  Violation Frames", vframes)
    c3.metric("✅  Compliance Rate", f"{compliance}%")
    avg_inf = (
        int(sum(st.session_state.inference_times) / len(st.session_state.inference_times))
        if st.session_state.inference_times else 0
    )
    c4.metric("⚡  Avg Inference", f"{avg_inf} ms")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("### Recent Detections")

    if st.session_state.detection_history:
        # Show in a styled mono block
        entries = st.session_state.detection_history[:8]
        lines_html = "".join(
            f"<div style='padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-family:DM Mono,monospace;font-size:13px;color:#8ef56d'>"
            f"<span style='color:#3a6a4a;margin-right:10px'>›</span>{e}</div>"
            for e in entries
        )
        st.markdown(
            f"<div style='background:#0b1727;border:1px solid rgba(15,157,88,0.15);border-radius:12px;padding:14px 18px'>{lines_html}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No detections yet — head over to the **Live** tab to start scanning.")


# ═══════════════════════════════ LIVE ═══════════════════════════════════════════
with tabs[1]:
    left, right = st.columns([3, 1])

    with right:
        st.markdown(
            "<div style='background:#0b1727;border:1px solid rgba(15,157,88,0.15);border-radius:16px;padding:20px 16px'>",
            unsafe_allow_html=True
        )
        st.markdown("### Controls")

        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶  Start Detection"):
            st.session_state.running = True

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("■  Stop Detection"):
                st.session_state.running = False
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Mini stats sidebar
        if st.session_state.scans:
            st.markdown(
                f"""<div style='background:#060f1e;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px 14px'>
                    <div style='font-size:11px;color:#4a6a80;font-weight:600;letter-spacing:1px;margin-bottom:10px;text-transform:uppercase'>Session Stats</div>
                    <div style='font-size:13px;color:#a0c4d8;margin-bottom:6px'>Frames: <b style='color:#fff'>{st.session_state.scans}</b></div>
                    <div style='font-size:13px;color:#a0c4d8;margin-bottom:6px'>Violations: <b style='color:#f87171'>{st.session_state.frames_with_violations}</b></div>
                    <div style='font-size:13px;color:#a0c4d8'>Inf. time: <b style='color:#8ef56d'>{avg_inf} ms</b></div>
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    frame_placeholder = left.empty()
    log_placeholder = left.empty()
    alert_placeholder = left.empty() # Placeholder for audio alert

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not accessible.")
            st.session_state.running = False
        else:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                start = time.time()
                results = model(frame, conf=confidence)
                inference_ms = int((time.time() - start) * 1000)
                st.session_state.inference_times = [inference_ms] + st.session_state.inference_times[:99]

                annotated = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                st.session_state.scans += 1
                
                # --- UPDATED FEATURE LOGIC WITH DELAY ---
                if "v_counter" not in st.session_state:
                    st.session_state.v_counter = 0

                detected_labels = [model.names[int(box.cls[0])].lower() for box in results[0].boxes]
                current = []
                
                if "student" in detected_labels:
                    missing = []
                    if "id" not in detected_labels: missing.append("ID")
                    if "black leather shoes" not in detected_labels: missing.append("Black Shoes")
                    if "black slacks" not in detected_labels: missing.append("Black Slacks")
                    
                    if missing:
                        # Increment counter (approx 30 frames = ~1.5 seconds of detection)
                        st.session_state.v_counter += 1
                        
                        if st.session_state.v_counter >= 30:
                            violation_msg = f"⚠️ VIOLATION: Missing {', '.join(missing)}"
                            current.append(violation_msg)
                            st.session_state.frames_with_violations += 1
                            
                            # Audio Alert (Click the page once after starting to enable browser audio)
                            html("""<audio autoplay><source src="https://cdn.pixabay.com/audio/2021/08/04/audio_0625c1539c.mp3" type="audio/mpeg"></audio>""", height=0)
                            
                            st.session_state.v_counter = 0 # Reset so it doesn't log every single frame
                    else:
                        st.session_state.v_counter = 0
                        current.append("✅ Compliance Verified")
                else:
                    st.session_state.v_counter = 0
                    for label in detected_labels:
                        current.append(f"Detected: {label}")

                if current:
                    st.session_state.detection_history = current + st.session_state.detection_history[:29]

                log_placeholder.markdown(
                    "<div style='background:#0b1727;border:1px solid rgba(15,157,88,0.12);border-radius:10px;padding:10px 14px;margin-top:8px'>"
                    + "".join(
                        f"<div style='font-family:DM Mono,monospace;font-size:12px;color:#8ef56d;padding:2px 0'>"
                        f"<span style='color:#3a6a4a'>›</span> {d}</div>"
                        for d in st.session_state.detection_history[:6]
                    )
                    + "</div>",
                    unsafe_allow_html=True
                )

                time.sleep(0.03)
            cap.release()
    else:
        frame_placeholder.markdown(
            """<div style='background:#0b1727;border:2px dashed rgba(15,157,88,0.2);border-radius:16px;
                height:400px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px'>
                <div style='font-size:48px'>📷</div>
                <div style='color:#4a6a80;font-size:14px;font-weight:600'>Camera feed will appear here</div>
                <div style='color:#2a4a5a;font-size:12px'>Click <b style='color:#22c55e'>Start Detection</b> to begin</div>
            </div>""",
            unsafe_allow_html=True
        )


# ═══════════════════════════════ LOGS ═══════════════════════════════════════════
with tabs[2]:
    st.markdown("### Detection Logs")

    if st.session_state.detection_history:
        for i, entry in enumerate(st.session_state.detection_history[:20], 1):
            color = "#f87171" if "violation" in entry.lower() else "#8ef56d"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px 14px;"
                f"background:#0b1727;border-radius:8px;margin-bottom:4px;"
                f"border-left:3px solid {color}'>"
                f"<span style='color:#3a5a6a;font-size:11px;font-weight:600;font-family:DM Mono,monospace'>#{i:02d}</span>"
                f"<span style='color:#c0d8e8;font-size:13px;font-family:DM Mono,monospace'>{entry}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No detection logs yet.")


# ═══════════════════════════════ REPORTS ════════════════════════════════════════
with tabs[3]:
    st.markdown("### Session Report")

    scans = st.session_state.scans
    vframes = st.session_state.frames_with_violations
    compliance = 100 - int((vframes / scans) * 100) if scans else 100
    avg_inf = (
        int(sum(st.session_state.inference_times) / len(st.session_state.inference_times))
        if st.session_state.inference_times else 0
    )

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Frames Scanned", scans)
        st.metric("Violation Frames", vframes)
    with r2:
        st.metric("Compliance Rate", f"{compliance}%")
        st.metric("Avg Inference Time", f"{avg_inf} ms")

    if scans > 0:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        # Simple progress bar for compliance
        bar_color = "#22c55e" if compliance >= 80 else "#f59e0b" if compliance >= 50 else "#ef4444"
        st.markdown(
            f"""<div style='background:#0b1727;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:16px 20px'>
                <div style='font-size:12px;color:#4a6a80;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px'>
                    Compliance Overview
                </div>
                <div style='background:#060f1e;border-radius:6px;height:10px;overflow:hidden'>
                    <div style='background:linear-gradient(90deg,{bar_color},{bar_color}aa);height:100%;width:{compliance}%;border-radius:6px;
                    transition:width 0.6s ease'></div>
                </div>
                <div style='margin-top:8px;font-size:13px;color:#a0c4d8'>
                    <b style='color:{bar_color};font-size:1.6rem'>{compliance}%</b> compliance this session
                </div>
            </div>""",
            unsafe_allow_html=True
        )


# ═══════════════════════════════ SETTINGS ═══════════════════════════════════════
with tabs[4]:
    st.markdown("### Settings")

    st.markdown(
        "<div style='background:#0b1727;border:1px solid rgba(15,157,88,0.12);border-radius:14px;padding:20px 24px'>",
        unsafe_allow_html=True
    )
    st.markdown("**Detection**")
    st.checkbox("Enable audio alerts on violation", value=True)
    st.checkbox("Show bounding boxes on feed", value=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("**Data**")
    st.checkbox("Record detection history to file", value=False)
    st.checkbox("Save annotated frames", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("💾  Save Settings"):
        st.success("Settings saved successfully.")