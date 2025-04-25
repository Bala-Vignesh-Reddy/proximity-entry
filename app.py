import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
from gpiozero import Device, LED, Buzzer
from threading import Thread
import atexit

# Video Stream Class for threaded capture
class VideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stopped = False
        self.frame = None
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            self.frame = frame
            time.sleep(0.01)  # Small delay to reduce CPU usage
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# Set page configuration
st.set_page_config(
    page_title="Smart Imaging Dashboard",
    page_icon="🎥",
    layout="wide"
)

# Global variables for GPIO devices
buzzer = None
led = None

# GPIO setup
def setup_gpio():
    global buzzer, led
    
    # Close any existing instances
    if buzzer is not None:
        buzzer.close()
    if led is not None:
        led.close()
    
    # Create new instances
    buzzer = Buzzer(18)
    led = LED(23)
    
    # Register cleanup function
    atexit.register(cleanup_gpio)
    
    return buzzer, led

# Cleanup function
def cleanup_gpio():
    global buzzer, led
    if buzzer is not None:
        buzzer.close()
    if led is not None:
        led.close()

# Function to trigger alarm
def trigger_alarm(buzzer_pin, led_pin, duration=1):
    try:
        buzzer_pin.on()
        led_pin.on()
        time.sleep(duration)
    finally:
        buzzer_pin.off()
        led_pin.off()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Filter for people only (class 0)
    return model

# Function to process frame with YOLOv5
def process_frame(frame, model, restricted_area, people_counter):
    if frame is None:
        return None, False
        
    # Convert frame to RGB for YOLOv5
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(rgb_frame)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Filter for people (class 0)
    people_detections = detections[detections['class'] == 0]
    
    # Count people
    people_count = len(people_detections)
    people_counter['count'] = people_count
    
    # Draw bounding boxes and check for restricted area intrusion
    intrusion_detected = False
    
    # Draw restricted area
    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]), 
                 (restricted_area[2], restricted_area[3]), (0, 0, 255), 2)
    
    for _, detection in people_detections.iterrows():
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        conf = detection['confidence']
        label = f"Person: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if person is in restricted area
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        
        if (restricted_area[0] < person_center_x < restricted_area[2] and 
            restricted_area[1] < person_center_y < restricted_area[3]):
            intrusion_detected = True
            # Highlight the intruding person with a red box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Add people count to frame
    cv2.putText(frame, f"People Count: {people_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, intrusion_detected

# Alarm thread function
def alarm_thread_function(buzzer_pin, led_pin):
    trigger_alarm(buzzer_pin, led_pin, 1)

def main():
    st.title("Smart Imaging Dashboard")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Camera selection
    camera_option = st.sidebar.selectbox(
        "Select Camera Source",
        options=["Raspberry Pi Camera", "USB Webcam"]
    )
    
    # USB camera index selection
    usb_camera_index = st.sidebar.number_input(
        "USB Camera Index (try different numbers if camera not found)",
        min_value=0,
        max_value=10,
        value=1 if camera_option == "USB Webcam" else 0,
        step=1
    )
    
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
    try:
        buzzer_pin, led_pin = setup_gpio()
    except Exception as e:
        st.error(f"GPIO setup error: {e}")
        st.warning("Running without GPIO functionality")
        buzzer_pin = None
        led_pin = None
    
    # Initialize people counter
    people_counter = {'count': 0}
    
    # Create placeholder for video feed
    video_placeholder = st.empty()
    
    # Create metrics placeholders
    col1, col2 = st.columns(2)
    people_count_metric = col1.metric("People Count", 0)
    status_metric = col2.metric("Status", "Safe")
    
    # Initialize camera using threaded video capture
    camera_index = 0 if camera_option == "Raspberry Pi Camera" else usb_camera_index
    video_stream = VideoStream(src=camera_index, resolution=(640, 480)).start()
    
    # Give the camera sensor time to warm up
    time.sleep(2.0)
    
    # Start streaming
    try:
        while True:
            # Get frame from threaded video stream
            frame = video_stream.read()
            
            if frame is None:
                st.error(f"Error: Failed to capture image from camera (index: {camera_index}).")
                st.info("Try a different camera index if using USB webcam.")
                break
            
            # Process frame
            processed_frame, intrusion_detected = process_frame(frame, model, restricted_area, people_counter)
            
            # Update metrics
            people_count_metric.metric("People Count", people_counter['count'])
            
            # Check for intrusion and trigger alarm if enabled
            if intrusion_detected:
                status_metric.metric("Status", "⚠️ INTRUSION DETECTED", delta="Alert")
                
                if alarm_enabled and buzzer_pin is not None and led_pin is not None:
                    # Start alarm in a separate thread to avoid blocking the main thread
                    alarm_thread = Thread(target=alarm_thread_function, args=(buzzer_pin, led_pin))
                    alarm_thread.daemon = True
                    alarm_thread.start()
            else:
                status_metric.metric("Status", "Safe", delta=None)
            
            # Convert to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        # Release resources
        video_stream.stop()
        cleanup_gpio()

if __name__ == "__main__":
    main()
