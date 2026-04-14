import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Debug Mode", layout="wide")

@st.cache_resource
def load_model():
    # Loading model with the path you provided
    return YOLO(r"C:\Users\Renz\Downloads\my_model\my_model.pt")

model = load_model()

st.title("Inference Debugger")
frame_placeholder = st.empty()
stop_btn = st.button("Stop App")

# --- 2. CAMERA ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot open camera. Check if another app is using it.")

while cap.isOpened() and not stop_btn:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. INFERENCE & PRINT DEBUGGING ---
    # We set verbose=True so YOLO itself logs to your terminal
    results = model.predict(frame, conf=0.3, verbose=True)
    
    # Custom Manual Print Debugging
    detections = results[0].boxes
    if len(detections) > 0:
        print(f"\n--- DETECTED {len(detections)} OBJECT(S) ---")
        for box in detections:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            # This will print in your VS Code terminal
            print(f"CLASS: {label} | CONF: {confidence:.2f}")
    else:
        # Uncomment the line below if you want to see "Nothing" every frame
        # print("No objects found...")
        pass

    # --- 4. DISPLAY ---
    # Use .plot() to get the image with boxes drawn by the model
    annotated_frame = results[0].plot()
    
    # Convert for Streamlit
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(display_frame, channels="RGB")

cap.release()
print("Camera released.")