import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Canteen Queue AI",
    page_icon="🍽️",
    layout="wide"
)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")
model_path = "best.pt"  # Make sure this file is in the same folder
confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.4, 0.05)
service_time = st.sidebar.number_input("Avg Service Time (min)", value=2.0)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- MAIN INTERFACE ---
st.title("🍽️ Live Canteen Queue Forecaster")
st.markdown("---")

# Layout: 2 Columns
col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.header("Live Analytics")
    kpi_queue = st.empty()
    kpi_wait = st.empty()
    status_text = st.empty()

# --- VIDEO SOURCE SELECTION ---
source_radio = st.sidebar.radio("Select Source", ["Upload Video", "Live Webcam"])

cap = None
temp_file_path = None

if source_radio == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video (mp4/avi)", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        # Save to temp file because OpenCV needs a path
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        cap = cv2.VideoCapture(temp_file_path)

elif source_radio == "Live Webcam":
    # 0 is the default webcam ID
    cap = cv2.VideoCapture(0)

# --- PROCESSING LOOP ---
with col1:
    st_frame = st.empty()  # Placeholder for the video

    if cap:
        stop_button = st.sidebar.button("Stop Processing")
        
        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            
            if not success:
                # Loop video if it's a file
                if source_radio == "Upload Video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    st.warning("Camera disconnected or stream ended.")
                    break

            # 1. Inference
            # Resize to 640 for standard speed, or 320 for faster
            results = model(frame, conf=confidence, imgsz=640)
            
            # 2. Extract Data
            person_count = 0
            annotated_frame = frame
            
            for r in results:
                annotated_frame = r.plot()
                person_count = len(r.boxes)
            
            wait_time = person_count * service_time

            # 3. Update Stats (Right Column)
            kpi_queue.metric("Students in Queue", f"{person_count}")
            
            # Change color logic for wait time
            if wait_time > 10:
                kpi_wait.metric("Est. Wait Time", f"{wait_time:.1f} min", delta="- High Traffic", delta_color="inverse")
            else:
                kpi_wait.metric("Est. Wait Time", f"{wait_time:.1f} min", delta="Normal", delta_color="normal")

            status_text.caption(f"Mode: {source_radio}")

            # 4. Display Video
            # OpenCV is BGR, Streamlit needs RGB
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
    else:
        st.info("Please upload a video or select Live Webcam to start.")
