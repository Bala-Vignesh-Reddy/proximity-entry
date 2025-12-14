import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
from gpiozero import Device, LED, Buzzer
from threading import Thread, Lock
import atexit
import pickle
import os
import face_recognition

class VideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.src = src
        self.resolution = resolution
        self.stream = None
        self.stopped = False
        self.frame = None
        self.lock = Lock()
        self.initialize_stream()

    def initialize_stream(self):
        """Initialize or reinitialize the video stream"""
        if self.stream is not None:
            self.stream.release()

        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        time.sleep(0.5)

    def start(self):
        self.stopped = False
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is not None and self.stream.isOpened():
                grabbed, frame = self.stream.read()
                if grabbed:
                    with self.lock:
                        self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.stream is not None:
            self.stream.release()

class GPIOManager:
    """Manages GPIO pins with proper cleanup and reinitialization"""
    def __init__(self):
        self.buzzer = None
        self.led = None
        self.lock = Lock()

    def setup(self, buzzer_pin=18, led_pin=23):
        with self.lock:
            try:
                if self.buzzer is not None:
                    self.buzzer.close()
                if self.led is not None:
                    self.led.close()

                self.buzzer = Buzzer(buzzer_pin)
                self.led = LED(led_pin)
                return True
            except Exception as e:
                st.warning(f"GPIO setup warning: {e}")
                return False

    def cleanup(self):
        with self.lock:
            if self.buzzer is not None:
                try:
                    self.buzzer.off()
                    self.buzzer.close()
                except:
                    pass
            if self.led is not None:
                try:
                    self.led.off()
                    self.led.close()
                except:
                    pass

    def trigger_alarm(self, duration=1):
        with self.lock:
            if self.buzzer is not None and self.led is not None:
                try:
                    self.buzzer.on()
                    self.led.on()
                    time.sleep(duration)
                    self.buzzer.off()
                    self.led.off()
                except:
                    pass

class FaceRecognizer:
    """Accurate face recognition using face_recognition library (dlib + deep learning)"""
    def __init__(self):
        self.authorized_faces = {}  # {name: [encoding1, encoding2, ...]}
        self.face_encodings_file = "authorized_faces.pkl"
        self.recognition_cache = {}  # Cache recent recognitions for stability
        self.cache_timeout = 2.0  # seconds to keep cache
        self.load_authorized_faces()

    def load_authorized_faces(self):
        """Load authorized face encodings from file"""
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    self.authorized_faces = pickle.load(f)
                st.success(f"✅ Loaded {len(self.authorized_faces)} authorized persons")
            except Exception as e:
                st.warning(f"Could not load saved faces: {e}")
                self.authorized_faces = {}

    def save_authorized_faces(self):
        """Save authorized face encodings to file"""
        try:
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(self.authorized_faces, f)
        except Exception as e:
            st.error(f"Could not save faces: {e}")

    def add_authorized_person(self, name, frame, person_bbox):
        """Add a person to authorized list using their face"""
        x1, y1, x2, y2 = person_bbox

        # Expand bbox slightly to ensure we capture the whole face
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return False, "Empty crop"

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Detect faces in the crop
        face_locations = face_recognition.face_locations(rgb_crop, model="hog")

        if len(face_locations) == 0:
            return False, "No face detected"

        # Get face encodings (128-dimensional vector)
        face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

        if len(face_encodings) == 0:
            return False, "Could not encode face"

        # Use the first (largest) face detected
        encoding = face_encodings[0]

        # Initialize or append to existing person
        if name not in self.authorized_faces:
            self.authorized_faces[name] = []

        self.authorized_faces[name].append(encoding)

        # Keep only last 5 samples for better accuracy and memory efficiency
        if len(self.authorized_faces[name]) > 5:
            self.authorized_faces[name] = self.authorized_faces[name][-5:]

        self.save_authorized_faces()
        return True, f"Captured ({len(self.authorized_faces[name])} samples)"

    def is_authorized(self, frame, person_bbox, tolerance=0.5):
        """
        Check if person is authorized
        tolerance: Lower = stricter matching (default 0.6, we use 0.5 for better accuracy)
        """
        if not self.authorized_faces:
            return False, None, 0.0

        # Check cache first for stability
        cache_key = f"{person_bbox[0]}-{person_bbox[1]}"
        current_time = time.time()

        if cache_key in self.recognition_cache:
            cached_result, cached_time = self.recognition_cache[cache_key]
            if current_time - cached_time < self.cache_timeout:
                return cached_result

        x1, y1, x2, y2 = person_bbox

        # Expand bbox slightly
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return False, None, 0.0

        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_crop, model="hog")

        if len(face_locations) == 0:
            return False, None, 0.0

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

        if len(face_encodings) == 0:
            return False, None, 0.0

        # Use the first face detected
        unknown_encoding = face_encodings[0]

        best_match_name = None
        best_match_distance = float('inf')

        # Compare with all authorized faces
        for name, auth_encodings in self.authorized_faces.items():
            # Compare with all samples of this person
            distances = face_recognition.face_distance(auth_encodings, unknown_encoding)
            min_distance = np.min(distances)

            if min_distance < best_match_distance:
                best_match_distance = min_distance
                best_match_name = name

        # Check if best match is within tolerance
        is_auth = best_match_distance <= tolerance
        confidence = 1.0 - best_match_distance  # Convert distance to confidence

        result = (is_auth, best_match_name if is_auth else None, confidence)

        # Cache the result for stability
        self.recognition_cache[cache_key] = (result, current_time)

        # Clean old cache entries
        self.recognition_cache = {
            k: v for k, v in self.recognition_cache.items()
            if current_time - v[1] < self.cache_timeout
        }

        return result

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]
    return model

@st.cache_resource
def get_face_recognizer():
    return FaceRecognizer()

@st.cache_resource
def get_gpio_manager():
    gpio_mgr = GPIOManager()
    gpio_mgr.setup()
    atexit.register(gpio_mgr.cleanup)
    return gpio_mgr

@st.cache_resource
def get_video_stream(_camera_index):
    vs = VideoStream(src=_camera_index, resolution=(640, 480))
    vs.start()
    time.sleep(1.0)
    return vs

def process_frame(frame, model, face_recognizer, restricted_area, tolerance=0.5):
    if frame is None:
        return None, False, [], 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]
    people_detections = detections[detections['class'] == 0]

    people_count = len(people_detections)
    intrusion_detected = False
    person_details = []

    # Draw restricted area
    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]),
                 (restricted_area[2], restricted_area[3]), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED AREA", (restricted_area[0] + 5, restricted_area[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        in_restricted = (restricted_area[0] < person_center_x < restricted_area[2] and
                        restricted_area[1] < person_center_y < restricted_area[3])

        # Check if person is authorized (only if in restricted area)
        is_authorized = False
        person_name = "Unknown"
        confidence = 0.0

        if in_restricted:
            is_auth, name, conf_score = face_recognizer.is_authorized(
                frame, (x1, y1, x2, y2), tolerance
            )
            if is_auth:
                is_authorized = True
                person_name = name
                confidence = conf_score

        # Set color based on authorization and location
        if in_restricted:
            if is_authorized:
                color = (0, 255, 0)  # Green for authorized
                label = f"{person_name} - AUTHORIZED ({confidence:.2%})"
                thickness = 3
            else:
                color = (0, 0, 255)  # Red for unauthorized intrusion
                label = f"UNAUTHORIZED INTRUDER!"
                intrusion_detected = True
                thickness = 3
        else:
            color = (0, 255, 0)  # Green outside restricted area
            label = f"Person: {conf:.2f}"
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add label background for better visibility
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        person_details.append({
            'bbox': (x1, y1, x2, y2),
            'in_restricted': in_restricted,
            'is_authorized': is_authorized,
            'name': person_name,
            'confidence': confidence
        })

    cv2.putText(frame, f"Total People: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, intrusion_detected, person_details, people_count

def main():
    st.set_page_config(page_title="Smart Imaging Dashboard", page_icon="🎥", layout="wide")
    st.title("🎥 Smart Proximity Detection with Face Recognition")

    # Initialize session state
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    if 'restricted_area' not in st.session_state:
        st.session_state.restricted_area = [100, 100, 400, 400]
    if 'capture_count' not in st.session_state:
        st.session_state.capture_count = 0
    if 'last_alarm_time' not in st.session_state:
        st.session_state.last_alarm_time = 0
    if 'running' not in st.session_state:
        st.session_state.running = False

    # Sidebar settings
    st.sidebar.title("⚙️ Settings")

    # Camera settings
    st.sidebar.subheader("📹 Camera Settings")
    camera_option = st.sidebar.selectbox(
        "Select Camera Source",
        options=["Raspberry Pi Camera", "USB Webcam"]
    )

    new_camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=1 if camera_option == "USB Webcam" else 0,
        step=1
    )

    # Restricted area settings
    st.sidebar.subheader("🚫 Restricted Area")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1", value=st.session_state.restricted_area[0], min_value=0, max_value=640)
        y1 = st.number_input("Y1", value=st.session_state.restricted_area[1], min_value=0, max_value=480)
    with col2:
        x2 = st.number_input("X2", value=st.session_state.restricted_area[2], min_value=0, max_value=640)
        y2 = st.number_input("Y2", value=st.session_state.restricted_area[3], min_value=0, max_value=480)

    st.session_state.restricted_area = [x1, y1, x2, y2]

    # Alarm settings
    alarm_enabled = st.sidebar.checkbox("Enable Alarm", value=True)

    # Authorization management
    st.sidebar.subheader("👤 Authorization Management")
    face_recognizer = get_face_recognizer()

    st.sidebar.info("💡 To authorize: Enter name, click 'Capture Sample', person enters restricted area. Capture 3-5 samples from different angles.")

    person_name = st.sidebar.text_input("Person's Name:", key="person_name_input")

    col_a, col_b = st.sidebar.columns(2)

    if col_a.button("📸 Capture Sample"):
        if person_name:
            st.session_state.capture_mode = True
            st.session_state.capture_name = person_name
        else:
            st.sidebar.warning("Please enter a name first!")

    if col_b.button("✅ Done"):
        if 'capture_mode' in st.session_state:
            del st.session_state.capture_mode
            if 'capture_name' in st.session_state:
                del st.session_state.capture_name
            st.session_state.capture_count = 0

    # Show authorized persons
    if face_recognizer.authorized_faces:
        st.sidebar.write("**✅ Authorized Persons:**")
        for name in list(face_recognizer.authorized_faces.keys()):
            col_a, col_b = st.sidebar.columns([3, 1])
            samples = len(face_recognizer.authorized_faces[name])
            col_a.write(f"👤 {name} ({samples} samples)")
            if col_b.button("❌", key=f"del_{name}"):
                del face_recognizer.authorized_faces[name]
                face_recognizer.save_authorized_faces()
                st.rerun()

    st.sidebar.markdown("---")
    recognition_tolerance = st.sidebar.slider(
        "Recognition Strictness",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Lower = stricter (0.4-0.5 recommended), Higher = more lenient"
    )

    # Load models
    with st.spinner("Loading models..."):
        model = load_model()
        gpio_mgr = get_gpio_manager()
        st.sidebar.success("✅ Models loaded!")

    # Main display
    video_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    people_count_metric = col1.empty()
    status_metric = col2.empty()
    auth_metric = col3.empty()

    # Info messages placeholder
    info_placeholder = st.empty()

    # Control buttons
    col_start, col_stop = st.columns(2)

    if col_start.button("▶️ Start Detection", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()

    if col_stop.button("⏹️ Stop Detection", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

    # Get video stream
    if st.session_state.running:
        video_stream = get_video_stream(new_camera_index)
        alarm_cooldown = 2  # seconds

        # Process frames
        for frame_count in range(100):  # Process 100 frames then rerun
            if not st.session_state.running:
                break

            frame = video_stream.read()

            if frame is None:
                st.error(f"❌ Camera error (index: {new_camera_index})")
                time.sleep(0.1)
                continue

            processed_frame, intrusion_detected, person_details, people_count = process_frame(
                frame, model, face_recognizer, st.session_state.restricted_area, recognition_tolerance
            )

            # Handle authorization capture
            if 'capture_mode' in st.session_state and st.session_state.capture_mode:
                # Auto-capture when person is in restricted area
                for person in person_details:
                    if person['in_restricted']:
                        success, message = face_recognizer.add_authorized_person(
                            st.session_state.capture_name, frame, person['bbox']
                        )
                        if success:
                            st.session_state.capture_count += 1
                            info_placeholder.success(f"✅ {message} for {st.session_state.capture_name}")
                            time.sleep(0.5)
                        else:
                            info_placeholder.warning(f"⚠️ {message}")
                        break
                else:
                    info_placeholder.info(f"📸 CAPTURE MODE: {st.session_state.capture_name} - Stand in restricted area to capture")

            # Update metrics
            people_count_metric.metric("👥 People Count", people_count)

            current_time = time.time()
            if intrusion_detected:
                status_metric.metric("🚨 Status", "INTRUSION!", delta="Alert")
                if alarm_enabled and (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
                    Thread(target=gpio_mgr.trigger_alarm, args=(0.5,), daemon=True).start()
                    st.session_state.last_alarm_time = current_time
            else:
                status_metric.metric("✅ Status", "Safe")

            # Show authorized count
            auth_count = sum(1 for p in person_details if p['is_authorized'] and p['in_restricted'])
            auth_metric.metric("🔓 Authorized in Area", auth_count)

            # Display frame
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        # Auto-rerun to continue processing
        if st.session_state.running:
            st.rerun()
    else:
        st.info("👆 Click 'Start Detection' to begin monitoring")

if __name__ == "__main__":
    main()
