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
from collections import defaultdict
from scipy.spatial import distance

class PersonTracker:
    """Tracks individuals across frames with persistent authorization"""
    def __init__(self):
        self.next_id = 1
        self.tracked_persons = {}  # {person_id: PersonInfo}
        self.max_disappeared = 30  # Frames before considering person as "left"
        self.iou_threshold = 0.3  # IOU threshold for matching

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        intersect_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersect_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersect = intersect_w * intersect_h

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersect

        return intersect / union if union > 0 else 0

    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of dicts with 'bbox', 'in_restricted', 'confidence'
        Returns: list of dicts with added 'person_id', 'is_authorized', 'name'
        """
        current_time = time.time()

        # If no tracked persons yet, assign new IDs
        if not self.tracked_persons:
            results = []
            for det in detections:
                person_id = self.next_id
                self.next_id += 1
                self.tracked_persons[person_id] = {
                    'bbox': det['bbox'],
                    'last_seen': current_time,
                    'disappeared_count': 0,
                    'is_authorized': None,  # Not yet checked
                    'name': None,
                    'needs_face_check': True,
                    'in_restricted': det['in_restricted']
                }
                results.append({
                    **det,
                    'person_id': person_id,
                    'is_authorized': None,
                    'name': None,
                    'needs_face_check': True
                })
            return results

        # Match current detections with tracked persons
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracked_persons.keys())

        # Calculate IOU matrix
        if detections and unmatched_tracks:
            iou_matrix = np.zeros((len(detections), len(unmatched_tracks)))
            for i, det in enumerate(detections):
                for j, track_id in enumerate(unmatched_tracks):
                    track = self.tracked_persons[track_id]
                    iou_matrix[i, j] = self.calculate_iou(det['bbox'], track['bbox'])

            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det_idx = unmatched_detections[i]
                track_id = unmatched_tracks[j]

                matched_pairs.append((det_idx, track_id))

                # Remove matched items
                iou_matrix = np.delete(iou_matrix, i, axis=0)
                iou_matrix = np.delete(iou_matrix, j, axis=1)
                unmatched_detections.pop(i)
                unmatched_tracks.pop(j)

        results = []

        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            det = detections[det_idx]
            track = self.tracked_persons[track_id]

            # Update bbox
            track['bbox'] = det['bbox']
            track['last_seen'] = current_time
            track['disappeared_count'] = 0
            track['in_restricted'] = det['in_restricted']

            # If already authorized, keep the status
            # Only need face check if authorization is None (first time in restricted area)
            if track['is_authorized'] is not None:
                needs_face_check = False
            else:
                # Check face only if in restricted area
                needs_face_check = det['in_restricted']

            results.append({
                **det,
                'person_id': track_id,
                'is_authorized': track['is_authorized'],
                'name': track['name'],
                'needs_face_check': needs_face_check
            })

        # Handle unmatched detections (new persons)
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            person_id = self.next_id
            self.next_id += 1

            self.tracked_persons[person_id] = {
                'bbox': det['bbox'],
                'last_seen': current_time,
                'disappeared_count': 0,
                'is_authorized': None,
                'name': None,
                'needs_face_check': det['in_restricted'],
                'in_restricted': det['in_restricted']
            }

            results.append({
                **det,
                'person_id': person_id,
                'is_authorized': None,
                'name': None,
                'needs_face_check': det['in_restricted']
            })

        # Handle unmatched tracks (disappeared persons)
        for track_id in unmatched_tracks:
            track = self.tracked_persons[track_id]
            track['disappeared_count'] += 1

            # Remove if disappeared for too long
            if track['disappeared_count'] > self.max_disappeared:
                del self.tracked_persons[track_id]

        return results

    def update_authorization(self, person_id, is_authorized, name):
        """Update authorization status for a person"""
        if person_id in self.tracked_persons:
            self.tracked_persons[person_id]['is_authorized'] = is_authorized
            self.tracked_persons[person_id]['name'] = name
            self.tracked_persons[person_id]['needs_face_check'] = False

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
                return False, "Invalid crop"

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            try:
                face_locations = face_recognition.face_locations(rgb_crop, model="hog", number_of_times_to_upsample=1)
            except Exception as e:
                return False, f"Face detection error"

            if len(face_locations) == 0:
                return False, "No face detected"

            try:
                face_encodings = face_recognition.face_encodings(rgb_crop, face_locations, num_jitters=1)
            except Exception as e:
                return False, f"Encoding error"

            if len(face_encodings) == 0:
                return False, "Could not encode face"

            encoding = face_encodings[0]

            if not isinstance(encoding, np.ndarray) or encoding.shape != (128,):
                return False, f"Invalid encoding"

            if name not in self.authorized_faces:
                self.authorized_faces[name] = []

            self.authorized_faces[name].append(encoding)

            if len(self.authorized_faces[name]) > 5:
                self.authorized_faces[name] = self.authorized_faces[name][-5:]

            self.save_authorized_faces()
            return True, f"Captured ({len(self.authorized_faces[name])} samples)"

        except Exception as e:
            return False, f"Error: {str(e)[:50]}"

    def is_authorized(self, frame, person_bbox, tolerance=0.5):
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

            try:
                face_locations = face_recognition.face_locations(rgb_crop, model="hog", number_of_times_to_upsample=1)
            except Exception:
                return False, None, 0.0

            if len(face_locations) == 0:
                return False, None, 0.0

            try:
                face_encodings = face_recognition.face_encodings(rgb_crop, face_locations, num_jitters=1)
            except Exception:
                return False, None, 0.0

            if len(face_encodings) == 0:
                return False, None, 0.0

            unknown_encoding = face_encodings[0]

            if not isinstance(unknown_encoding, np.ndarray) or unknown_encoding.shape != (128,):
                return False, None, 0.0

            best_match_name = None
            best_match_distance = float('inf')

            for name, auth_encodings in self.authorized_faces.items():
                valid_encodings = [enc for enc in auth_encodings
                                 if isinstance(enc, np.ndarray) and enc.shape == (128,)]

                if not valid_encodings:
                    continue

                try:
                    distances = face_recognition.face_distance(valid_encodings, unknown_encoding)
                    min_distance = np.min(distances)

                    if min_distance < best_match_distance:
                        best_match_distance = min_distance
                        best_match_name = name
                except Exception:
                    continue

            is_auth = best_match_distance <= tolerance
            confidence = max(0.0, min(1.0, 1.0 - best_match_distance))

            return (is_auth, best_match_name if is_auth else None, confidence)

        except Exception as e:
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
def get_person_tracker():
    return PersonTracker()

def process_frame(frame, model, face_recognizer, tracker, restricted_area, tolerance=0.5):
    if frame is None:
        return None, False, [], 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]
    people_detections = detections[detections['class'] == 0]

    # Draw restricted area
    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]),
                 (restricted_area[2], restricted_area[3]), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED AREA", (restricted_area[0] + 5, restricted_area[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Prepare detections for tracker
    detection_list = []
    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        in_restricted = (restricted_area[0] < person_center_x < restricted_area[2] and
                        restricted_area[1] < person_center_y < restricted_area[3])

        detection_list.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': conf,
            'in_restricted': in_restricted
        })

    # Update tracker
    tracked_results = tracker.update(detection_list)

    intrusion_detected = False
    person_details = []

    # Process tracked persons
    for person in tracked_results:
        x1, y1, x2, y2 = person['bbox']
        person_id = person['person_id']
        needs_face_check = person['needs_face_check']
        is_authorized = person['is_authorized']
        name = person['name']
        in_restricted = person['in_restricted']

        # Perform face recognition only if needed
        if needs_face_check and in_restricted:
            is_auth, detected_name, confidence = face_recognizer.is_authorized(
                frame, (x1, y1, x2, y2), tolerance
            )
            # Update tracker with recognition result
            tracker.update_authorization(person_id, is_auth, detected_name)
            is_authorized = is_auth
            name = detected_name

        # Determine color and label based on authorization
        if in_restricted:
            if is_authorized is None:
                # Still checking...
                color = (255, 255, 0)  # Yellow - checking
                label = f"ID:{person_id} - CHECKING..."
                thickness = 2
            elif is_authorized:
                color = (0, 255, 0)  # Green - authorized
                label = f"ID:{person_id} - {name} ✓"
                thickness = 3
            else:
                color = (0, 0, 255)  # Red - unauthorized
                label = f"ID:{person_id} - UNAUTHORIZED!"
                intrusion_detected = True
                thickness = 3
        else:
            color = (0, 255, 0)  # Green outside restricted area
            label = f"ID:{person_id}"
            thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        person_details.append({
            'person_id': person_id,
            'bbox': (x1, y1, x2, y2),
            'in_restricted': in_restricted,
            'is_authorized': is_authorized,
            'name': name
        })

    people_count = len(tracked_results)
    cv2.putText(frame, f"Total People: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, intrusion_detected, person_details, people_count

def main():
    st.set_page_config(page_title="Smart Imaging Dashboard", page_icon="🎥", layout="wide")
    st.title("🎥 Smart Tracking with Persistent Authorization")

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

    st.sidebar.title("⚙️ Settings")

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

    st.sidebar.subheader("🚫 Restricted Area")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1", value=st.session_state.restricted_area[0], min_value=0, max_value=640)
        y1 = st.number_input("Y1", value=st.session_state.restricted_area[1], min_value=0, max_value=480)
    with col2:
        x2 = st.number_input("X2", value=st.session_state.restricted_area[2], min_value=0, max_value=640)
        y2 = st.number_input("Y2", value=st.session_state.restricted_area[3], min_value=0, max_value=480)

    st.session_state.restricted_area = [x1, y1, x2, y2]

    alarm_enabled = st.sidebar.checkbox("Enable Alarm", value=True)

    st.sidebar.subheader("👤 Authorization Management")
    face_recognizer = get_face_recognizer()

    st.sidebar.info("💡 Capture 3-5 samples from different angles for best results")

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
        help="Lower = stricter (0.4-0.5 recommended)"
    )

    with st.spinner("Loading models..."):
        model = load_model()
        gpio_mgr = get_gpio_manager()
        tracker = get_person_tracker()
        st.sidebar.success("✅ Models loaded!")

    video_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    people_count_metric = col1.empty()
    status_metric = col2.empty()
    auth_metric = col3.empty()

    info_placeholder = st.empty()

    col_start, col_stop = st.columns(2)

    if col_start.button("▶️ Start Detection", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()

    if col_stop.button("⏹️ Stop Detection", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

    if st.session_state.running:
        video_stream = get_video_stream(new_camera_index)
        alarm_cooldown = 3

        for frame_count in range(100):
            if not st.session_state.running:
                break

            frame = video_stream.read()

            if frame is None:
                st.error(f"❌ Camera error (index: {new_camera_index})")
                time.sleep(0.1)
                continue

            processed_frame, intrusion_detected, person_details, people_count = process_frame(
                frame, model, face_recognizer, tracker, st.session_state.restricted_area, recognition_tolerance
            )

            if 'capture_mode' in st.session_state and st.session_state.capture_mode:
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
                    info_placeholder.info(f"📸 CAPTURE MODE: {st.session_state.capture_name} - Stand in restricted area")

            people_count_metric.metric("👥 People Count", people_count)

            current_time = time.time()
            if intrusion_detected:
                status_metric.metric("🚨 Status", "INTRUSION!", delta="Alert")
                if alarm_enabled and (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
                    Thread(target=gpio_mgr.trigger_alarm, args=(0.5,), daemon=True).start()
                    st.session_state.last_alarm_time = current_time
            else:
                status_metric.metric("✅ Status", "Safe")

            auth_count = sum(1 for p in person_details if p['is_authorized'] and p['in_restricted'])
            auth_metric.metric("🔓 Authorized in Area", auth_count)

            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        if st.session_state.running:
            st.rerun()
    else:
        st.info("👆 Click 'Start Detection' to begin monitoring")

if __name__ == "__main__":
    main()
