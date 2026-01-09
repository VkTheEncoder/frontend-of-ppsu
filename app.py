import streamlit as st
import cv2
import tempfile
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Canteen Queue AI", page_icon="🍽️", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")
confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.4, 0.05)
service_time = st.sidebar.number_input("Avg Service Time (min)", value=2.0)
source_radio = st.sidebar.radio("Select Source", ["Live Webcam", "Upload Video"])

st.title("🍽️ Live Canteen Queue Forecaster")
st.markdown("---")

# Layout
col1, col2 = st.columns([0.7, 0.3])

# --- GLOBAL VARIABLES FOR STATS ---
# We use st.session_state to pass data from the video processor to the UI
if 'person_count' not in st.session_state:
    st.session_state.person_count = 0

# --- WEBCAM PROCESSOR CLASS ---
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Run YOLO
        results = model(img, conf=confidence, imgsz=320)
        
        # 2. Draw & Count
        count = 0
        for r in results:
            img = r.plot()
            count = len(r.boxes)
        
        # Store count in a global variable (hacky but works for simple apps)
        st.session_state.person_count = count

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN LOGIC ---
with col1:
    if source_radio == "Live Webcam":
        st.write("Click 'START' to use your camera.")
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
    
    elif source_radio == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=['mp4', 'avi'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            st_frame = st.empty()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Inference
                results = model(frame, conf=confidence, imgsz=320)
                annotated_frame = frame
                count = 0
                for r in results:
                    annotated_frame = r.plot()
                    count = len(r.boxes)
                
                # Update State
                st.session_state.person_count = count
                
                # Display
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

# --- STATS DISPLAY (Updates Automatically) ---
with col2:
    st.header("Live Analytics")
    
    # Simple logic: If using Webrtc, we might need to rely on the last known count
    # Note: Real-time stat updates from Webrtc to Streamlit UI are tricky.
    # This basic version updates when you interact with the page.
    
    count = st.session_state.person_count
    wait_time = count * service_time
    
    st.metric("Students in Queue", f"{count}")
    
    if wait_time > 10:
        st.metric("Est. Wait Time", f"{wait_time:.1f} min", delta="- High Traffic", delta_color="inverse")
    else:
        st.metric("Est. Wait Time", f"{wait_time:.1f} min", delta="Normal")
