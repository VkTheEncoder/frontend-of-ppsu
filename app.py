import os
import cv2
import time
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "best.pt" 
model = YOLO(MODEL_PATH)

# Global Variables
# Default source is 0 (Webcam). 
# If you have a CCTV URL, you can change this default.
video_source = 0 
is_file = False # To track if we are playing a file (to loop it)

current_stats = {
    "person_count": 0,
    "wait_time": 0,
    "mode": "Live Camera"
}

# Service time per person (minutes)
AVG_SERVICE_TIME = 2.0 

def generate_frames():
    global video_source, is_file
    
    # Open the camera or video file
    cap = cv2.VideoCapture(video_source)
    
    while True:
        success, frame = cap.read()
        
        # If video finishes and it's a file, loop it
        if not success:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break # If camera disconnects, stop

        # 1. Run YOLO Inference
        results = model(frame, stream=True, conf=0.4) 
        
        person_count = 0
        
        for r in results:
            annotated_frame = r.plot()
            person_count = len(r.boxes)

        # 2. Update Stats
        current_stats["person_count"] = person_count
        current_stats["wait_time"] = person_count * AVG_SERVICE_TIME

        # 3. Encode for Web
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(current_stats)

# --- NEW: Route to Upload Video ---
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_source, is_file
    
    if 'file' not in request.files:
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # UPDATE SOURCE TO FILE
        video_source = filepath
        is_file = True
        current_stats["mode"] = f"File: {filename}"
        
        return jsonify({"status": "success", "message": "Video uploaded and playing!"})

# --- NEW: Route to Switch Back to Live ---
@app.route('/switch_live', methods=['POST'])
def switch_live():
    global video_source, is_file
    video_source = 0 # Reset to Webcam
    is_file = False
    current_stats["mode"] = "Live Camera"
    return jsonify({"status": "success", "message": "Switched to Live Camera"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
