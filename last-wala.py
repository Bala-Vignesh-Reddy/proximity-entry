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

class AggressiveTracker:
    """
    Aggressive tracker with immediate face verification
    - Check face IMMEDIATELY when entering restricted area
    - Very short grace period (0.5-1 second max)
    - Quick decision: authorized or intruder
    """
    def __init__(self, max_lost=40, iou_threshold=0.25, grace_attempts=3):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.grace_attempts = grace_attempts  # Number of failed face checks before marking unauthorized

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_centroid(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def compute_distance(self, point1, point2):
        """Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def update(self, detections):
        """
        Update tracks with immediate authorization check
        """
        tracked_objects = []

        # Create new tracks
        if not self.tracks:
            for det in detections:
                track_id = self.next_id
                self.next_id += 1

                # Determine initial status
                if det['in_restricted']:
                    if det['face_checked'] and det['authorized']:
                        status = 'authorized'
                    else:
                        status = 'checking'
                else:
                    status = 'outside'

                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'lost': 0,
                    'authorized': det['authorized'],
                    'name': det['name'],
                    'confidence': det['confidence'],
                    'in_restricted': det['in_restricted'],
                    'status': status,
                    'check_attempts': 0 if det['in_restricted'] and not det['authorized'] else -1
                }

                det_copy = det.copy()
                det_copy['track_id'] = track_id
                det_copy['status'] = status
                tracked_objects.append(det_copy)
            return tracked_objects

        # Match detections to tracks
        matched_tracks = set()
        matched_detections = set()

        for det_idx, det in enumerate(detections):
            best_score = 0
            best_track_id = None
            det_centroid = self.get_centroid(det['bbox'])

            for track_id, track in self.tracks.items():
                if track['lost'] > self.max_lost:
                    continue

                iou = self.compute_iou(det['bbox'], track['bbox'])
                track_centroid = self.get_centroid(track['bbox'])
                distance = self.compute_distance(det_centroid, track_centroid)
                normalized_distance = 1.0 - min(distance / (640 * np.sqrt(2)), 1.0)

                score = 0.6 * iou + 0.4 * normalized_distance

                if score > best_score and iou > self.iou_threshold:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)

                track = self.tracks[best_track_id]

                # CRITICAL: Authorization logic
                if track['authorized']:
                    # Already authorized - permanent
                    det['authorized'] = True
                    det['name'] = track['name']
                    det['confidence'] = track['confidence']
                    track['status'] = 'authorized'
                else:
                    # Not yet authorized
                    if det['face_checked'] and det['authorized']:
                        # Just got authorized!
                        track['authorized'] = True
                        track['name'] = det['name']
                        track['confidence'] = det['confidence']
                        track['status'] = 'authorized'
                        track['check_attempts'] = -1  # Stop checking
                    else:
                        # Still not authorized
                        if det['in_restricted']:
                            if track['check_attempts'] >= 0:
                                track['check_attempts'] += 1

                            if track['check_attempts'] >= self.grace_attempts:
                                # Out of attempts - INTRUDER
                                track['status'] = 'unauthorized'
                            else:
                                # Still checking
                                track['status'] = 'checking'
                        else:
                            # Left restricted area
                            track['status'] = 'outside'

                # Update track position
                track['bbox'] = det['bbox']
                track['lost'] = 0
                track['in_restricted'] = det['in_restricted']

                det_copy = det.copy()
                det_copy['track_id'] = best_track_id
                det_copy['status'] = track['status']
                det_copy['authorized'] = track['authorized']
                det_copy['name'] = track['name']
                tracked_objects.append(det_copy)

        # Create new tracks for unmatched
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = self.next_id
                self.next_id += 1

                if det['in_restricted']:
                    if det['face_checked'] and det['authorized']:
                        status = 'authorized'
                        check_attempts = -1
                    else:
                        status = 'checking'
                        check_attempts = 0
                else:
                    status = 'outside'
                    check_attempts = -1

                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'lost': 0,
                    'authorized': det['authorized'],
                    'name': det['name'],
                    'confidence': det['confidence'],
                    'in_restricted': det['in_restricted'],
                    'status': status,
                    'check_attempts': check_attempts
                }

                det_copy = det.copy()
                det_copy['track_id'] = track_id
                det_copy['status'] = status
                tracked_objects.append(det_copy)

        # Update lost tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['lost'] += 1

        # Remove old tracks
        tracks_to_remove = [tid for tid, track in self.tracks.items() if track['lost'] > self.max_lost]
        for tid in tracks_to_remove:
            del self.tracks[tid]

        return tracked_objects

    def reset(self):
        """Reset all tracks"""
        self.tracks = {}
        self.next_id = 1

class GPIOManager:
    """Manages GPIO pins"""
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
    """Fast face recognition"""
    def __init__(self):
        self.authorized_faces = {}
        self.face_encodings_file = "authorized_faces.pkl"
        self.load_authorized_faces()

    def load_authorized_faces(self):
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    self.authorized_faces = pickle.load(f)
                st.success(f"✅ Loaded {len(self.authorized_faces)} authorized persons")
            except Exception as e:
                st.warning(f"Could not load saved faces: {e}")
                self.authorized_faces = {}

    def save_authorized_faces(self):
        try:
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(self.authorized_faces, f)
        except Exception as e:
            st.error(f"Could not save faces: {e}")

    def add_authorized_person(self, name, frame, person_bbox):
        try:
            x1, y1, x2, y2 = person_bbox
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return False, "Crop too small"

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_crop, model="hog")
            if len(face_locations) == 0:
                return False, "No face detected"

            face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
            if len(face_encodings) == 0:
                return False, "Could not encode"

            encoding = face_encodings[0]

            if name not in self.authorized_faces:
                self.authorized_faces[name] = []

            self.authorized_faces[name].append(encoding)

            if len(self.authorized_faces[name]) > 5:
                self.authorized_faces[name] = self.authorized_faces[name][-5:]

            self.save_authorized_faces()
            return True, f"✓ {len(self.authorized_faces[name])} samples"

        except Exception as e:
            return False, "Error"

    def is_authorized(self, frame, person_bbox, tolerance=0.5):
        """Quick face check"""
        if not self.authorized_faces:
            return False, None, 0.0

        try:
            x1, y1, x2, y2 = person_bbox
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return False, None, 0.0

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_crop, model="hog")
            if len(face_locations) == 0:
                return False, None, 0.0

            face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
            if len(face_encodings) == 0:
                return False, None, 0.0

            unknown_encoding = face_encodings[0]

            best_match_name = None
            best_match_distance = float('inf')

            for name, auth_encodings in self.authorized_faces.items():
                valid_encodings = [enc for enc in auth_encodings if isinstance(enc, np.ndarray) and enc.shape == (128,)]
                if not valid_encodings:
                    continue

                distances = face_recognition.face_distance(valid_encodings, unknown_encoding)
                min_distance = np.min(distances)

                if min_distance < best_match_distance:
                    best_match_distance = min_distance
                    best_match_name = name

            is_auth = best_match_distance <= tolerance
            confidence = max(0.0, min(1.0, 1.0 - best_match_distance))

            return (is_auth, best_match_name if is_auth else None, confidence)

        except Exception:
            return False, None, 0.0

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

@st.cache_resource
def get_tracker(_grace_attempts):
    return AggressiveTracker(max_lost=40, iou_threshold=0.25, grace_attempts=_grace_attempts)

def process_frame(frame, model, face_recognizer, tracker, restricted_area, tolerance, frame_count):
    """
    AGGRESSIVE face checking:
    - Check face EVERY time when in restricted area
    - Track keeps count of failed checks
    - After N failed checks -> INTRUDER
    """
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

    # Process detections
    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        in_restricted = (restricted_area[0] < person_center_x < restricted_area[2] and
                        restricted_area[1] < person_center_y < restricted_area[3])

        # ALWAYS check face when in restricted area
        face_checked = False
        is_authorized = False
        person_name = "Unknown"
        confidence = 0.0

        if in_restricted:
            # Check face IMMEDIATELY
            is_auth, name, conf_score = face_recognizer.is_authorized(
                frame, (x1, y1, x2, y2), tolerance
            )
            face_checked = True
            if is_auth:
                is_authorized = True
                person_name = name
                confidence = conf_score

        person_details.append({
            'bbox': (x1, y1, x2, y2),
            'in_restricted': in_restricted,
            'authorized': is_authorized,
            'name': person_name,
            'confidence': confidence,
            'face_checked': face_checked
        })

    # Update tracker
    tracked_objects = tracker.update(person_details)

    # Draw tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['bbox']
        track_id = obj['track_id']
        status = obj['status']

        # Color coding based on status
        if status == 'authorized':
            color = (0, 255, 0)  # GREEN - authorized
            label = f"ID:{track_id} ✓ {obj['name']}"
            thickness = 3
        elif status == 'checking':
            color = (0, 255, 255)  # YELLOW - checking (very brief)
            label = f"ID:{track_id} - Verifying..."
            thickness = 3
        elif status == 'unauthorized':
            color = (0, 0, 255)  # RED - intruder
            label = f"ID:{track_id} - INTRUDER!"
            intrusion_detected = True
            thickness = 3
        else:  # outside
            color = (255, 165, 0)  # ORANGE - outside
            label = f"ID:{track_id}"
            thickness = 2

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"People: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, intrusion_detected, tracked_objects, people_count

def main():
    st.set_page_config(page_title="Smart Access Control", page_icon="🎥", layout="wide")
    st.title("🎥 Immediate Face Verification System")

    # Session state
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    if 'restricted_area' not in st.session_state:
        st.session_state.restricted_area = [100, 100, 400, 400]
    if 'last_alarm_time' not in st.session_state:
        st.session_state.last_alarm_time = 0
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Camera
    st.sidebar.subheader("📹 Camera")
    camera_option = st.sidebar.selectbox(
        "Source",
        options=["Raspberry Pi Camera", "USB Webcam"]
    )

    new_camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=1 if camera_option == "USB Webcam" else 0,
        step=1
    )

    # Restricted area
    st.sidebar.subheader("🚫 Restricted Area")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1", value=st.session_state.restricted_area[0], min_value=0, max_value=640)
        y1 = st.number_input("Y1", value=st.session_state.restricted_area[1], min_value=0, max_value=480)
    with col2:
        x2 = st.number_input("X2", value=st.session_state.restricted_area[2], min_value=0, max_value=640)
        y2 = st.number_input("Y2", value=st.session_state.restricted_area[3], min_value=0, max_value=480)

    st.session_state.restricted_area = [x1, y1, x2, y2]

    # Security settings
    st.sidebar.subheader("🎯 Security")

    grace_attempts = st.sidebar.slider(
        "Verification Attempts",
        min_value=1,
        max_value=10,
        value=3,
        help="Failed face checks before marking as intruder (3 = ~1 second)"
    )

    alarm_enabled = st.sidebar.checkbox("Enable Alarm", value=True)

    # Authorization
    st.sidebar.subheader("👤 Authorization")
    face_recognizer = get_face_recognizer()

    st.sidebar.info("**Status:**\n"
                   "🟡 Yellow = Checking (~1s)\n"
                   "🟢 Green = Authorized\n"
                   "🔴 Red = INTRUDER")

    person_name = st.sidebar.text_input("Name:", key="person_name_input")

    col_a, col_b = st.sidebar.columns(2)

    if col_a.button("📸 Capture"):
        if person_name:
            st.session_state.capture_mode = True
            st.session_state.capture_name = person_name
        else:
            st.sidebar.warning("Enter name first!")

    if col_b.button("✅ Done"):
        if 'capture_mode' in st.session_state:
            del st.session_state.capture_mode
            if 'capture_name' in st.session_state:
                del st.session_state.capture_name

    # Authorized list
    if face_recognizer.authorized_faces:
        st.sidebar.write("**Authorized:**")
        for name in list(face_recognizer.authorized_faces.keys()):
            col_a, col_b = st.sidebar.columns([3, 1])
            samples = len(face_recognizer.authorized_faces[name])
            col_a.write(f"👤 {name} ({samples})")
            if col_b.button("❌", key=f"del_{name}"):
                del face_recognizer.authorized_faces[name]
                face_recognizer.save_authorized_faces()
                st.rerun()

    st.sidebar.markdown("---")
    recognition_tolerance = st.sidebar.slider(
        "Strictness",
        min_value=0.35,
        max_value=0.65,
        value=0.48,
        step=0.02,
        help="Lower = stricter"
    )

    if st.sidebar.button("🔄 Reset Tracks"):
        st.cache_resource.clear()
        st.sidebar.success("Reset! Restart detection.")

    # Load
    with st.spinner("Loading..."):
        model = load_model()
        gpio_mgr = get_gpio_manager()
        tracker = get_tracker(grace_attempts)
        st.sidebar.success("✅ Ready!")

    # Display
    video_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)
    people_metric = col1.empty()
    status_metric = col2.empty()
    auth_metric = col3.empty()
    tracks_metric = col4.empty()

    info_placeholder = st.empty()

    # Controls
    col_start, col_stop = st.columns(2)

    if col_start.button("▶️ Start", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.rerun()

    if col_stop.button("⏹️ Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

    # Main loop
    if st.session_state.running:
        video_stream = get_video_stream(new_camera_index)
        alarm_cooldown = 2

        for loop_count in range(100):
            if not st.session_state.running:
                break

            frame = video_stream.read()
            if frame is None:
                st.error(f"❌ Camera error")
                time.sleep(0.1)
                continue

            st.session_state.frame_count += 1

            processed_frame, intrusion_detected, tracked_objects, people_count = process_frame(
                frame, model, face_recognizer, tracker, st.session_state.restricted_area,
                recognition_tolerance, st.session_state.frame_count
            )

            # Capture mode
            if 'capture_mode' in st.session_state and st.session_state.capture_mode:
                for person in tracked_objects:
                    if person['in_restricted']:
                        success, message = face_recognizer.add_authorized_person(
                            st.session_state.capture_name, frame, person['bbox']
                        )
                        if success:
                            info_placeholder.success(f"✅ {message} - {st.session_state.capture_name}")
                            time.sleep(0.3)
                        else:
                            info_placeholder.warning(f"⚠️ {message}")
                        break
                else:
                    info_placeholder.info(f"📸 Stand in restricted area")

            # Metrics
            people_metric.metric("👥 People", people_count)

            current_time = time.time()
            if intrusion_detected:
                status_metric.metric("🚨 Status", "INTRUSION!")
                if alarm_enabled and (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
                    Thread(target=gpio_mgr.trigger_alarm, args=(0.5,), daemon=True).start()
                    st.session_state.last_alarm_time = current_time
            else:
                status_metric.metric("✅ Status", "Safe")

            auth_count = sum(1 for p in tracked_objects if p['authorized'] and p['in_restricted'])
            auth_metric.metric("🔓 Auth", auth_count)

            tracks_metric.metric("🎯 Tracks", len(tracker.tracks))

            # Display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        if st.session_state.running:
            st.rerun()
    else:
        st.info("👆 Click 'Start' to begin")

if __name__ == "__main__":
    main()
