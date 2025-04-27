import streamlit as st
import cv2
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
import numpy as np
import time
from PIL import Image
from gpiozero import LED, Buzzer
from signal import pause
from threading import Thread
from datetime import datetime
import pandas as pd
from queue import Queue

# Thread class for capturing frames
class VideoCapture:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.q = Queue(maxsize=2)  # Small queue size to ensure fresh frames
        self.stopped = False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                
                # If queue is not empty, clear it to always get the latest frame
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except:
                        pass
                
                self.q.put(frame)
            else:
                # Small sleep to prevent CPU overuse
                time.sleep(0.001)
                
    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# Set page configuration
st.set_page_config(
    page_title="Smart Imaging Dashboard",
    page_icon="🎥",
    layout="wide"
)

# GPIO setup
def setup_gpio():
    try:
        buzzer = Buzzer(18)
        led = LED(23)
        return buzzer, led
    except:
        st.warning("GPIO pins not available. Running in simulation mode.")
        return None, None

# Function to trigger alarm
def trigger_alarm(buzzer_pin, led_pin, duration=2):
    if buzzer_pin and led_pin:
        # Turn on both buzzer and LED
        buzzer_pin.on()
        led_pin.on()
        # Keep them on for the specified duration
        time.sleep(duration)
        buzzer_pin.off()
        led_pin.off()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Filter for people only (class 0)
    return model

# Modify the process_frame function to separate detection from visualization
def process_frame(frame, model, restricted_area, people_counter, detection_records, last_detections=None):
    # Get current timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Always draw the restricted area regardless of processing
    display_frame = frame.copy()
    cv2.rectangle(display_frame, 
                 (restricted_area[0], restricted_area[1]),
                 (restricted_area[2], restricted_area[3]), 
                 (0, 0, 255), 2)
    
    # Add timestamp to frame
    cv2.putText(display_frame, timestamp, (display_frame.shape[1] - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # If we're skipping detection for this frame, use the last known detections
    if last_detections is not None:
        # Draw previous detections on the current frame
        for detection in last_detections:
            bbox = detection.get("bbox")
            if bbox is not None:
                # Draw bounding box
                cv2.rectangle(
                    display_frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0) if detection.get("status") != "INTRUSION" else (0, 0, 255),
                    2
                )
                
                # Add label
                label = f"{detection.get('class', 'Person')}: {detection.get('confidence', '0.00')}"
                cv2.putText(
                    display_frame,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if detection.get("status") != "INTRUSION" else (0, 0, 255),
                    2
                )
        
        # Add people count to frame
        cv2.putText(display_frame, f"People Count: {people_counter['count']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        return display_frame, False, []
    
    # If we're processing this frame, do the full detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]
    people_detections = detections[detections['class'] == 0]
    
    # Count people
    people_count = len(people_detections)
    people_counter['count'] = people_count
    
    # Process detections
    intrusion_detected = False
    current_detections = []
    new_entries = []
    
    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        # Create detection record
        det_record = {
            "bbox": [x1, y1, x2, y2],
            "class": "Person",
            "confidence": f"{conf:.2f}",
            "timestamp": timestamp
        }
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Person: {conf:.2f}"
        cv2.putText(display_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if person is in restricted area
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        
        in_restricted_area = (restricted_area[0] < person_center_x < restricted_area[2] and
                             restricted_area[1] < person_center_y < restricted_area[3])
        
        if in_restricted_area:
            intrusion_detected = True
            # Highlight the intruding person with a red box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            det_record["status"] = "INTRUSION"
            
            # Check if this is a new intrusion (not in last_detections)
            is_new_intrusion = True
            if last_detections:
                for last_det in last_detections:
                    if last_det.get("status") == "INTRUSION":
                        # Check if it's roughly the same person (simple IOU check)
                        last_bbox = last_det.get("bbox")
                        if last_bbox:
                            # Skip logging if this appears to be the same person
                            if is_same_person([x1, y1, x2, y2], last_bbox):
                                is_new_intrusion = False
                                break
            
            if is_new_intrusion:
                new_entries.append({
                    "timestamp": timestamp,
                    "location": f"({person_center_x}, {person_center_y})",
                    "confidence": f"{conf:.2f}",
                    "status": "INTRUSION"
                })
        else:
            det_record["status"] = "Detected"
        
        current_detections.append(det_record)
    
    # Add people count to frame
    cv2.putText(display_frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add new entries to detection records
    if new_entries:
        detection_records.extend(new_entries)
    
    return display_frame, intrusion_detected, current_detections

# Helper function to check if two bounding boxes represent the same person
def is_same_person(bbox1, bbox2, iou_threshold=0.5):
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return False  # No intersection
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate IoU
    iou = intersection / (area1 + area2 - intersection)
    
    return iou >= iou_threshold


def alarm_thread_function(buzzer_pin, led_pin):
    trigger_alarm(buzzer_pin, led_pin, 2)

def main():
    st.title("Smart Imaging Dashboard")

    # Sidebar configuration
    st.sidebar.title("Settings")

    # Camera selection
    camera_option = st.sidebar.selectbox(
        "Select Camera Source",
        options=["Raspberry Pi Camera", "USB Webcam"]
    )

    # Performance settings
    st.sidebar.subheader("Performance Settings")
    skip_frames = st.sidebar.slider("Frame Skip Rate", 0, 5, 1,
                                  help="Higher values improve performance but reduce smoothness")

    # Define restricted area (default values)
    st.sidebar.subheader("Restricted Area Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1", value=100, min_value=0, max_value=1000)
        y1 = st.number_input("Y1", value=100, min_value=0, max_value=1000)
    with col2:
        x2 = st.number_input("X2", value=400, min_value=0, max_value=1000)
        y2 = st.number_input("Y2", value=400, min_value=0, max_value=1000)

    restricted_area = [x1, y1, x2, y2]

    # Alarm settings
    alarm_enabled = st.sidebar.checkbox("Enable Alarm", value=True)

    # Load model
    with st.spinner("Loading YOLOv5 model..."):
        model = load_model()
        st.sidebar.success("Model loaded successfully!")

    # Set up GPIO
    buzzer_pin, led_pin = setup_gpio()

    # Initialize people counter and detection records
    if 'people_counter' not in st.session_state:
        st.session_state.people_counter = {'count': 0}
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []

    people_counter = st.session_state.people_counter
    detection_records = st.session_state.detection_records

    # Create placeholder for video feed
    video_placeholder = st.empty()

    # Create metrics placeholders
    col1, col2 = st.columns(2)
    people_count_metric = col1.metric("People Count", 0)
    status_metric = col2.metric("Status", "Safe")

    # Detection records table
    table_placeholder = st.empty()

    # Initialize camera with threading
    camera_id = 0 if camera_option == "Raspberry Pi Camera" else 1
    video_capture = VideoCapture(camera_id).start()

    # For frame skipping
    frame_count = 0

    # Start streaming
    # In the main function
    try:
        last_detections = None
        while True:
            frame = video_capture.read()

            if frame is None:
                continue

            frame_count += 1

        # Process every nth frame for detection, but always draw the UI elements
            if frame_count % (skip_frames + 1) == 0:
                processed_frame, intrusion_detected, current_detections = process_frame(
                    frame, model, restricted_area, people_counter, detection_records
                    )
                last_detections = current_detections
            else:
            # Just update visuals using last detections
                processed_frame, intrusion_detected, _ = process_frame(
                    frame, model, restricted_area, people_counter, 
                    detection_records, last_detections
                    )

        # Update metrics
            people_count_metric.metric("People Count", people_counter['count'])

        # Check for intrusion and trigger alarm if enabled
            if intrusion_detected:
                status_metric.metric("Status", "⚠️ INTRUSION DETECTED", delta="Alert")

                if alarm_enabled:
                # Start alarm in a separate thread to avoid blocking
                    alarm_thread = Thread(target=alarm_thread_function, args=(buzzer_pin, led_pin))
                    alarm_thread.daemon = True
                    alarm_thread.start()
            else:
                status_metric.metric("Status", "Safe", delta=None)

        # Display the frame
            video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                channels="RGB", use_container_width=True)

        # Update detection records table (only show last 10 entries)
            if detection_records:
                df = pd.DataFrame(detection_records[-10:])
                table_placeholder.dataframe(df)

        # Small delay to reduce CPU usage
            time.sleep(0.01)

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Release resources
        video_capture.stop()

        # Save detection records
        if detection_records:
            df = pd.DataFrame(detection_records)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Detection Records",
                data=csv,
                file_name=f"detection_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

