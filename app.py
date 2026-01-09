import streamlit as st
import cv2
import tempfile
import av
import numpy as np
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

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

col1, col2 = st.columns([0.7, 0.3])

# --- WEBCAM PROCESSOR CLASS ---
class VideoProcessor:
    def __init__(self):
        # Initialize a variable to store the count inside the processor
        self.person_count = 0  # <--- CHANGED: New instance variable

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Run YOLO
        results = model(img, conf=confidence, imgsz=320)
        
        # 2. Draw & Count
        count = 0
        for r in results:
            img = r.plot()
            count = len(r.boxes)
        
        # 3. Save count to SELF, not session_state
        self.person_count = count  # <--- CHANGED: Updating self instead of session_state

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN LOGIC ---
with col1:
    ctx = None
    if source_radio == "Live Webcam":
        st.write("Click 'START' to use your camera.")
        # We assign the streamer to 'ctx' so we can access it later
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
    
    elif source_radio == "Upload Video":
        # ... (Your existing upload logic remains here) ...
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=['mp4', 'avi'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            st_frame = st.empty()
            # Loop for uploaded video
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                results = model(frame, conf=confidence, imgsz=320)
                annotated_frame = frame
                count = 0
                for r in results:
                    annotated_frame = r.plot()
                    count = len(r.boxes)
                
                # Hack to update the side stats for Upload Mode
                st.session_state.person_count = count 
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)


# --- STATS DISPLAY (THE FIX) ---
with col2:
    st.header("Live Analytics")
    
    # Create placeholders so we can update them inside a loop
    kpi_count = st.empty()
    kpi_wait = st.empty()

    # <--- CHANGED: This loop pulls data from the processor in real-time
    if ctx and ctx.state.playing:
        while True:
            # Check if the processor exists
            if ctx.video_processor:
                # Get the count from the processor
                live_count = ctx.video_processor.person_count 
                
                # Calculate Wait Time
                wait = live_count * service_time
                
                # Update Metrics
                kpi_count.metric("Students in Queue", f"{live_count}")
                if wait > 10:
                    kpi_wait.metric("Est. Wait Time", f"{wait:.1f} min", delta="- High", delta_color="inverse")
                else:
                    kpi_wait.metric("Est. Wait Time", f"{wait:.1f} min", delta="Normal")
            
            # Sleep briefly to save CPU
            time.sleep(0.5) 
    
    # Fallback for Upload Mode or Stopped Camera
    elif source_radio == "Upload Video" and 'person_count' in st.session_state:
         # This part is just for the file upload static display
         c = st.session_state.person_count
         kpi_count.metric("Students in Queue", f"{c}")
         kpi_wait.metric("Est. Wait Time", f"{c * service_time:.1f} min")
