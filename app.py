import time
import base64
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="FitCheck", layout="wide")

# --- PATHS ---
MODEL_PATH = Path(r"C:\Users\ferdi\Downloads\my_model\my_model.pt")
LOGO_PATH = Path(__file__).parent / "logo.png"

# --- LOGO ---
logo_html = "<span>✓</span>"
if LOGO_PATH.exists():
    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" alt="FitCheck logo" />'

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path: Path):
    return YOLO(str(path), task="detect")

# --- TOP BAR + CSS (FIXED) ---
st.markdown(
    f"""
    <style>
    .top-bar {{
        background: #050811;
        padding: 20px 30px;
        display: flex;
        align-items: center;
        gap: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}

    .top-bar h1 {{
        margin: 0;
        font-size: 32px;
        letter-spacing: 3px;
        background: linear-gradient(90deg, #ffffff, #8ef56d, #ffffff);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
    }}

    .top-bar .logo {{
        width: 52px;
        height: 52px;
        min-width: 52px;
        min-height: 52px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border: 1px solid #0f9d58;
        background: #0f172a;
    }}

    .top-bar .logo img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    .top-bar .logo span {{
        color: #22c55e;
        font-size: 26px;
        font-weight: 800;
    }}

    @keyframes shine {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    </style>

    <div class="top-bar">
        <div class="logo">{logo_html}</div>
        <h1>FITCHECK</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- CHECK MODEL ---
if not MODEL_PATH.exists():
    st.error("Missing model file. Fix path to my_model.pt")
    st.stop()

model = load_model(MODEL_PATH)

# --- SESSION STATE ---
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.violations = []
    st.session_state.scans = 0
    st.session_state.frames_with_violations = 0
    st.session_state.detection_history = []
    st.session_state.inference_times = []

# --- TABS ---
tabs = st.tabs(["Dashboard", "Live", "Logs", "Reports", "Settings"])

# ================= DASHBOARD =================
with tabs[0]:
    st.markdown("### Dashboard")

    scans = st.session_state.scans
    violation_frames = st.session_state.frames_with_violations
    compliance_rate = 100 - int((violation_frames / scans) * 100) if scans else 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Scanned", scans)
    col2.metric("Violation frames", violation_frames)
    col3.metric("Compliance", f"{compliance_rate}%")

    st.markdown("### Live Status")

    if st.session_state.detection_history:
        st.write("Recent detections:")
        st.write("\n".join(st.session_state.detection_history[:5]))
    else:
        st.info("No detections yet. Go to LIVE tab.")

# ================= LIVE =================
with tabs[1]:
    st.markdown("### Live Detection")

    left, right = st.columns([3, 1])

    with right:
        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)

        if st.button("Start detection"):
            st.session_state.running = True

        if st.button("Stop detection"):
            st.session_state.running = False

    frame_placeholder = left.empty()
    log_placeholder = left.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible")
            st.session_state.running = False
        else:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                start = time.time()
                results = model(frame, conf=confidence)
                inference_ms = int((time.time() - start) * 1000)

                st.session_state.inference_times = (
                    [inference_ms] + st.session_state.inference_times[:99]
                )

                annotated = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width=720)

                st.session_state.scans += 1

                current = []
                has_violation = False

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    conf = float(box.conf[0])

                    current.append(f"{label}: {conf:.2f}")
                    has_violation = True

                if has_violation:
                    st.session_state.frames_with_violations += 1

                if current:
                    st.session_state.detection_history = (
                        current + st.session_state.detection_history[:29]
                    )

                log_placeholder.write("\n".join(st.session_state.detection_history[:10]))

                time.sleep(0.03)

            cap.release()
    else:
        frame_placeholder.info("Click START detection")

# ================= LOGS =================
with tabs[2]:
    st.markdown("### Logs")

    if st.session_state.violations:
        for i, v in enumerate(st.session_state.violations[:20], 1):
            st.write(f"{i}. {v}")
    else:
        st.info("No logs yet")

# ================= REPORTS =================
with tabs[3]:
    st.markdown("### Reports")

    scans = st.session_state.scans
    violation_frames = st.session_state.frames_with_violations

    avg_time = (
        int(sum(st.session_state.inference_times) / len(st.session_state.inference_times))
        if st.session_state.inference_times
        else 0
    )

    st.metric("Frames scanned", scans)
    st.metric("Violation frames", violation_frames)
    st.metric("Avg inference time", f"{avg_time} ms")

# ================= SETTINGS =================
with tabs[4]:
    st.markdown("### Settings")

    st.checkbox("Enable audio alert", value=True)
    st.checkbox("Record history", value=False)
    st.checkbox("Show bounding boxes", value=True)

    st.button("Save settings")