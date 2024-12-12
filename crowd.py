from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import winsound
from deepface import DeepFace
import os
from PIL import Image
import threading

app = Flask(__name__)
# Global variables
camera = None
crowd_threshold = 2
criminal_database = {}
missing_person_data = None
missing_object_image = None
alert_active = False
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        self.current_frame = None
        self.tracker = None  # No tracker at start
        self.tracking = False
        self.bbox = None  # To store the bounding box of the tracked person

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        self.current_frame = frame

        if self.tracking and self.bbox is not None:
            # Update the tracker on the current frame
            success, bbox = self.tracker.update(frame)
            if success:
                # If tracking was successful, draw the bounding box
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                self.tracking = False  # If tracking fails, stop tracking

        return frame

    def start_tracking(self, frame, bbox):
        # Initialize the tracker with the detected person's bounding box
        self.bbox = bbox
        self.tracker = cv2.TrackerCSRT_create()  # CSRT tracker for better accuracy
        self.tracker.init(frame, self.bbox)
        self.tracking = True

def process_person_tracking(frame):
    results = yolo_model(frame)
    
    # Iterate over detected results
    for result in results:
        for bbox in result.boxes:
            if bbox.cls[0] == 0:  # Class 0 is for 'person' in YOLO
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Get the bounding box coordinates
                person_bbox = (x1, y1, x2 - x1, y2 - y1)  # Define bbox for tracking

                # Start tracking the person if not already tracking
                if not camera.tracking:
                    camera.start_tracking(frame, person_bbox)

    return frame

def process_crowd_monitoring(frame):
    try:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Count the number of faces
        face_count = len(faces)

        # Color logic and alerts based on the number of faces
        alert_message = ""
        if face_count > 3:
            color = (0, 0, 255)  # Red for more than 3 faces
            alert_message = "Crowded"
        elif face_count == 3:
            color = (0, 165, 255)  # Orange for exactly 3 faces
            alert_message = "Full"
        else:
            color = (0, 255, 0)  # Green for fewer than 3 faces
            alert_message = "Normal"

        # Draw bounding boxes around detected faces and show face count
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)  # Draw a colored rectangle around each face

        # Display the face count and alert message on the frame
        cv2.putText(frame, f"Face Count: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, alert_message, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    except Exception as e:
        print(f"Error in crowd monitoring: {e}")
        cv2.putText(frame, "Error in Face Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def gen_frames(mode):
    global camera
    if camera is None:
        camera = VideoCamera()

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # Process the frame based on the selected mode
        if mode == "crowd":
            frame = process_crowd_monitoring(frame)
        elif mode == "person":
            frame = process_person_tracking(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    return Response(gen_frames(mode),
                   mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

