"""
Flask Backend Server for Online Proctoring System
Provides REST API and WebSocket for real-time proctoring
"""

print(">>> Server starting...")

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from ultralytics import YOLO
from face_spoofing import detect_spoof, reset_spoof_state
import base64
import io
from PIL import Image
import time
from datetime import datetime

print(">>> Imports done")

# ================= FLASK INITIALIZATION =================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ================= AI MODELS INITIALIZATION =================
print(">>> Loading AI models...")
embedder = FaceNet()
print(">>> FaceNet loaded")

# MediaPipe Face Detection and Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print(">>> MediaPipe models loaded")

yolo_model = YOLO('yolov8n.pt')
print(">>> YOLO model loaded")

# ================= STATIC 3D FACE MODEL =================
model_3d = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

NOSE_TIP = 1
CHIN = 152
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

LANDMARK_IDS = [NOSE_TIP, CHIN, LEFT_EYE, RIGHT_EYE, LEFT_MOUTH, RIGHT_MOUTH]

# Eye landmarks
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_IRIS = 468

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_IRIS = 473

# Research-grade gaze tracking constants
GAZE_CALIBRATION_FRAMES = 20
GAZE_HISTORY_WINDOW = 10
GAZE_CENTER_DELTA = 0.08
GAZE_DEVIATION_SECONDS = 2.0
GAZE_COOLDOWN_SECONDS = 3.0
GAZE_MIN_DETECTION_CONFIDENCE = 0.7
GAZE_PROCESS_EVERY_N = 2
SPOOF_EVENT_COOLDOWN_SECONDS = 2.0

# Head-pose stability constants
POSE_HISTORY_WINDOW = 10
POSE_YAW_THRESHOLD = 15.0
POSE_PITCH_THRESHOLD = 10.0
POSE_ROLL_THRESHOLD = 10.0
POSE_DEVIATION_SECONDS = 2.0
POSE_DEVIATION_SECONDS_THRESHOLD = 5

# Research-grade object/person detection robustness constants
PERSON_CONFIDENCE_THRESHOLD = 0.6
OBJECT_CONFIDENCE_THRESHOLD = 0.5
YOLO_INFERENCE_CONFIDENCE = 0.5
YOLO_INFERENCE_IOU = 0.45
OBJECT_CONTINUOUS_FRAMES_THRESHOLD = 5
MULTIPLE_PERSON_CONTINUOUS_FRAMES_THRESHOLD = 5
SPOOF_CONTINUOUS_FRAMES_THRESHOLD = 5
DETECTION_COOLDOWN_SECONDS = 3.0
MIN_BOX_AREA = 5000
DETECTION_HISTORY_WINDOW = 5
DETECTION_PROCESS_EVERY_N = 2
ABSENCE_SECONDS_THRESHOLD = 3
VIOLATION_FRAMES_THRESHOLD = 15

# YOLO classes
PERSON_CLASS = 0
CELL_PHONE_CLASS = 67
BOOK_CLASS = 73
LAPTOP_CLASS = 63

PROHIBITED_CLASSES = {
    CELL_PHONE_CLASS: 'Cell Phone',
    BOOK_CLASS: 'Book',
    LAPTOP_CLASS: 'Laptop'
}
PROHIBITED_CLASSES.update({
    77: 'Remote',
    84: 'Tablet'
})


def create_detection_tracking_state():
    """Create default state for robust person/object temporal validation."""
    return {
        'person_continuous_frames': 0,
        'object_continuous_frames': 0,
        'spoof_continuous_frames': 0,
        'person_history': deque(maxlen=DETECTION_HISTORY_WINDOW),
        'object_history': deque(maxlen=DETECTION_HISTORY_WINDOW),
        'multiple_person_violation': False,
        'prohibited_object_violation': False,
        'no_person_start_time': None,
        'absence_triggered': False,
        'pose_deviation_start_time': None,
        'pose_violation_triggered': False,
        'unauthorized_person_frames': 0,
        'object_frames': 0,
        'pose_deviation_frames': 0,
        'warning_active': False,
        'warning_start_time': None,
        'warning_violation_type': None,
        'warning_last_emitted_second': None,
        'termination_triggered': False,
        'critical_violation_type': None,
        'last_detection_alert_time': {
            'multiple_person': None,
            'prohibited_object': None
        },
        'last_detection_result': None,
        'true_positive_person': 0,
        'false_positive_person': 0,
        'true_negative_person': 0,
        'false_negative_person': 0,
        'true_positive_object': 0,
        'false_positive_object': 0,
        'true_negative_object': 0,
        'false_negative_object': 0
    }


def create_gaze_tracking_state():
    """Create default state for adaptive geometric gaze tracking."""
    return {
        'gaze_baseline': None,
        'gaze_calibrated': False,
        'gaze_calibration_values': [],
        'gaze_history': deque(maxlen=GAZE_HISTORY_WINDOW),
        'gaze_deviation_start_time': None,
        'gaze_last_alert_time': None,
        'last_gaze_result': None,
        'gaze_true_positive': 0,
        'gaze_false_positive': 0,
        'gaze_true_negative': 0,
        'gaze_false_negative': 0
    }


def create_pose_tracking_state():
    """Create default state for pose smoothing and temporal persistence."""
    return {
        'pose_yaw_history': deque(maxlen=POSE_HISTORY_WINDOW),
        'pose_pitch_history': deque(maxlen=POSE_HISTORY_WINDOW),
        'pose_roll_history': deque(maxlen=POSE_HISTORY_WINDOW),
        'pose_stabilization_start_time': None
    }

# ================= SESSION STATE =================
session_state = {
    'reference_embedding': None,
    'baseline_yaw': None,
    'baseline_pitch': None,
    'baseline_roll': None,
    'baseline_calibrated': False,
    'calibration_yaws': [],
    'calibration_pitches': [],
    'calibration_rolls': [],
    'frame_count': 0,
    'same_person_frames': 0,
    'different_person_frames': 0,
    'deviation_frames': 0,
    'gaze_deviation_frames': 0,
    'multiple_person_frames': 0,
    'prohibited_object_frames': 0,
    'spoof_frames': 0,
    'total_frames': 0,
    'last_spoof_detected': False,
    'last_spoof_event_time': None,
    'prohibited_objects_detected': {},
    'is_active': False,
    'CALIBRATION_FRAMES': 20,
    **create_pose_tracking_state(),
    **create_gaze_tracking_state(),
    **create_detection_tracking_state()
}

# ================= HELPER FUNCTIONS =================

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def get_face_embedding(frame, bbox):
    """Generate face embedding from frame and bounding box"""
    x, y, w, h = bbox
    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (160, 160))
    return embedder.embeddings([face_crop])[0]

def get_head_pose(frame, landmarks):
    """Calculate head pose angles"""
    h, w = frame.shape[:2]

    image_2d = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = float(w)
    center_x = w / 2.0
    center_y = h / 2.0
    cam_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_3d,
        image_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)

    pitch = np.degrees(np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
    roll  = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    
    return yaw, pitch, roll

def get_eye_gaze(frame, landmarks):
    """Calculate eye gaze direction"""
    h, w = frame.shape[:2]
    
    # Left eye
    left_iris = landmarks[LEFT_IRIS]
    left_outer = landmarks[LEFT_EYE_LEFT]
    left_inner = landmarks[LEFT_EYE_RIGHT]
    
    left_iris_x = left_iris.x * w
    left_inner_x = left_inner.x * w
    left_outer_x = left_outer.x * w
    
    left_eye_width = abs(left_inner_x - left_outer_x)
    left_gaze_ratio = (left_iris_x - left_outer_x) / left_eye_width if left_eye_width > 0 else 0.5
    
    # Right eye
    right_iris = landmarks[RIGHT_IRIS]
    right_inner = landmarks[RIGHT_EYE_LEFT]
    right_outer = landmarks[RIGHT_EYE_RIGHT]
    
    right_iris_x = right_iris.x * w
    right_inner_x = right_inner.x * w
    right_outer_x = right_outer.x * w
    
    right_eye_width = abs(right_inner_x - right_outer_x)
    right_gaze_ratio = (right_iris_x - right_outer_x) / right_eye_width if right_eye_width > 0 else 0.5
    
    avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
    
    if avg_gaze_ratio < 0.35:
        gaze_direction = "Looking Right"
    elif avg_gaze_ratio > 0.65:
        gaze_direction = "Looking Left"
    else:
        gaze_direction = "Looking Center"
    
    return left_gaze_ratio, right_gaze_ratio, gaze_direction

def detect_persons_and_objects(frame, frame_count):
    """
    Detect persons and prohibited objects with confidence/area filtering.
    Uses YOLO built-in NMS through conf/iou settings and caches results for skipped frames.
    """
    if (
        frame_count % DETECTION_PROCESS_EVERY_N != 0
        and session_state.get('last_detection_result') is not None
    ):
        cached = dict(session_state['last_detection_result'])
        cached['reused'] = True
        return (
            cached['person_count'],
            dict(cached['prohibited_objects']),
            cached
        )

    results = yolo_model(
        frame,
        conf=YOLO_INFERENCE_CONFIDENCE,
        iou=YOLO_INFERENCE_IOU,
        verbose=False
    )

    person_count = 0
    prohibited_objects = {}
    person_confidence_max = 0.0
    object_confidence_max = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            if box_area < MIN_BOX_AREA:
                continue

            if cls == PERSON_CLASS and conf >= PERSON_CONFIDENCE_THRESHOLD:
                person_count += 1
                person_confidence_max = max(person_confidence_max, conf)

            if cls in PROHIBITED_CLASSES and conf >= OBJECT_CONFIDENCE_THRESHOLD:
                obj_name = PROHIBITED_CLASSES[cls]
                prohibited_objects[obj_name] = prohibited_objects.get(obj_name, 0) + 1
                object_confidence_max = max(object_confidence_max, conf)

    detection_meta = {
        'person_count': person_count,
        'prohibited_objects': prohibited_objects,
        'person_confidence_max': float(person_confidence_max),
        'object_confidence_max': float(object_confidence_max),
        'frame_count': frame_count,
        'reused': False
    }
    session_state['last_detection_result'] = detection_meta

    return person_count, prohibited_objects, detection_meta


def get_detection_confidence(detection):
    """Safely extract detection confidence across MediaPipe versions."""
    try:
        if hasattr(detection, 'score') and detection.score:
            return float(detection.score[0])
    except Exception:
        pass
    return 0.0


def reset_gaze_tracking_state():
    """Reset gaze tracking state for a new session/reference."""
    gaze_state = create_gaze_tracking_state()
    session_state.update(gaze_state)


def reset_pose_tracking_state():
    """Reset pose smoothing/timing state for a new session/reference."""
    pose_state = create_pose_tracking_state()
    session_state.update(pose_state)


def add_overlay_text(frame, relative_yaw, relative_pitch, relative_roll, gaze_direction):
    """Draw lightweight debug overlay directly on frame."""
    text_rows = [
        f"Yaw: {relative_yaw:+.2f}",
        f"Pitch: {relative_pitch:+.2f}",
        f"Roll: {relative_roll:+.2f}",
        f"Gaze: {gaze_direction}"
    ]

    y_start = 30
    for idx, text in enumerate(text_rows):
        y = y_start + (idx * 26)
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


def reset_detection_tracking_state():
    """Reset detection robustness state for a new session/reference."""
    detection_state = create_detection_tracking_state()
    session_state.update(detection_state)


def check_absence_termination(person_count, frame_number):
    """Track continuous no-person duration and trigger session termination when threshold is crossed."""
    current_time = time.time()

    if person_count == 0:
        if session_state['no_person_start_time'] is None:
            session_state['no_person_start_time'] = current_time

        elapsed_time = current_time - session_state['no_person_start_time']

        if elapsed_time >= ABSENCE_SECONDS_THRESHOLD and not session_state['absence_triggered']:
            session_state['absence_triggered'] = True
            session_state['is_active'] = False
            session_state['termination_triggered'] = True
            session_state['critical_violation_type'] = 'no_person_detected'

            print(
                f"SCREEN EMPTY DETECTED\n"
                f"elapsed_time: {elapsed_time:.2f} sec"
            )

            print(
                f"TERMINATION TRIGGERED\n"
                f"termination_reason: no_person_detected\n"
                f"frame_number: {frame_number}"
            )

            critical_payload = {
                'type': 'no_person_detected',
                'frame': frame_number,
                'action': 'session_terminated',
                'reason': 'Screen empty for 3 seconds'
            }
            socketio.emit('critical_violation', critical_payload)
            return True, critical_payload, elapsed_time

        return False, None, elapsed_time

    session_state['no_person_start_time'] = None
    session_state['absence_triggered'] = False
    return False, None, 0.0


def check_pose_deviation_termination(pose_deviated, frame_number):
    """Track continuous pose deviation duration and terminate when threshold is crossed."""
    current_time = time.time()

    if pose_deviated:
        if session_state['pose_deviation_start_time'] is None:
            session_state['pose_deviation_start_time'] = current_time

        elapsed_time = current_time - session_state['pose_deviation_start_time']

        if elapsed_time >= POSE_DEVIATION_SECONDS_THRESHOLD and not session_state['pose_violation_triggered']:
            session_state['pose_violation_triggered'] = True
            session_state['is_active'] = False
            session_state['termination_triggered'] = True
            session_state['critical_violation_type'] = 'head_pose_deviation'

            print(
                f"HEAD POSE DEVIATION DETECTED\n"
                f"elapsed_time: {elapsed_time:.2f} sec"
            )

            print(
                f"TERMINATION TRIGGERED\n"
                f"termination_reason: head_pose_deviation\n"
                f"frame_number: {frame_number}"
            )

            critical_payload = {
                'type': 'head_pose_deviation',
                'frame': frame_number,
                'action': 'session_terminated',
                'reason': 'Continuous head pose deviation for 5 seconds'
            }
            socketio.emit('critical_violation', critical_payload)
            return True, critical_payload, elapsed_time

        return False, None, elapsed_time

    session_state['pose_deviation_start_time'] = None
    session_state['pose_violation_triggered'] = False
    return False, None, 0.0


def check_unified_continuous_termination(frame_number):
    """Terminate session when any tracked violation persists across the unified frame threshold."""
    termination_reason = None

    if session_state['unauthorized_person_frames'] >= VIOLATION_FRAMES_THRESHOLD:
        termination_reason = 'unauthorized_person'

    if session_state['multiple_person_frames'] >= VIOLATION_FRAMES_THRESHOLD:
        termination_reason = 'multiple_person'

    if session_state['object_frames'] >= VIOLATION_FRAMES_THRESHOLD:
        termination_reason = 'prohibited_object'

    if session_state['spoof_continuous_frames'] >= VIOLATION_FRAMES_THRESHOLD:
        termination_reason = 'spoof_detected'

    if session_state['gaze_deviation_frames'] >= VIOLATION_FRAMES_THRESHOLD:
        termination_reason = 'gaze_deviation'

    if termination_reason is None or session_state['termination_triggered']:
        return False, None

    session_state['is_active'] = False
    session_state['termination_triggered'] = True
    session_state['critical_violation_type'] = termination_reason

    print(
        f"TERMINATION TRIGGERED\n"
        f"termination_reason: {termination_reason}\n"
        f"frame_number: {frame_number}"
    )

    critical_payload = {
        'type': termination_reason,
        'frame': frame_number,
        'action': 'session_terminated',
        'message': 'Stopped due to unwanted movement'
    }
    socketio.emit('critical_violation', critical_payload)
    print("CRITICAL VIOLATION EMITTED")
    return True, critical_payload


def cap_violation_counters():
    """Clamp unified violation counters to the global threshold."""
    session_state['unauthorized_person_frames'] = min(
        session_state['unauthorized_person_frames'],
        VIOLATION_FRAMES_THRESHOLD
    )
    session_state['multiple_person_frames'] = min(
        session_state['multiple_person_frames'],
        VIOLATION_FRAMES_THRESHOLD
    )
    session_state['object_frames'] = min(
        session_state['object_frames'],
        VIOLATION_FRAMES_THRESHOLD
    )
    session_state['spoof_continuous_frames'] = min(
        session_state['spoof_continuous_frames'],
        VIOLATION_FRAMES_THRESHOLD
    )
    session_state['gaze_deviation_frames'] = min(
        session_state['gaze_deviation_frames'],
        VIOLATION_FRAMES_THRESHOLD
    )


def safe_div(numerator, denominator):
    """Safe division helper for metric calculations."""
    return (numerator / denominator) if denominator else 0.0


def calculate_binary_metrics(tp, fp, tn, fn):
    """Calculate standard binary classification metrics."""
    total = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    return {
        'accuracy': safe_div(tp + tn, total),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': safe_div(fp, fp + tn)
    }


def calculate_gaze_research_metrics():
    """Compute research metrics for gaze classification."""
    tp = session_state['gaze_true_positive']
    fp = session_state['gaze_false_positive']
    tn = session_state['gaze_true_negative']
    fn = session_state['gaze_false_negative']

    total = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)

    return {
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'accuracy': safe_div(tp + tn, total),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': safe_div(fp, fp + tn)
    }


def update_gaze_research_metrics(predicted_positive, ground_truth_positive):
    """Update TP/FP/TN/FN counters using optional ground-truth labels."""
    if ground_truth_positive and predicted_positive:
        session_state['gaze_true_positive'] += 1
    elif (not ground_truth_positive) and predicted_positive:
        session_state['gaze_false_positive'] += 1
    elif (not ground_truth_positive) and (not predicted_positive):
        session_state['gaze_true_negative'] += 1
    else:
        session_state['gaze_false_negative'] += 1


def update_detection_research_metrics(
    predicted_multiple_person,
    predicted_prohibited_object,
    expected_multiple_person=None,
    expected_prohibited_object=None
):
    """Update detection TP/FP/TN/FN counters when optional labels are provided."""
    if isinstance(expected_multiple_person, bool):
        if expected_multiple_person and predicted_multiple_person:
            session_state['true_positive_person'] += 1
        elif (not expected_multiple_person) and predicted_multiple_person:
            session_state['false_positive_person'] += 1
        elif (not expected_multiple_person) and (not predicted_multiple_person):
            session_state['true_negative_person'] += 1
        else:
            session_state['false_negative_person'] += 1

    if isinstance(expected_prohibited_object, bool):
        if expected_prohibited_object and predicted_prohibited_object:
            session_state['true_positive_object'] += 1
        elif (not expected_prohibited_object) and predicted_prohibited_object:
            session_state['false_positive_object'] += 1
        elif (not expected_prohibited_object) and (not predicted_prohibited_object):
            session_state['true_negative_object'] += 1
        else:
            session_state['false_negative_object'] += 1


def calculate_detection_research_metrics():
    """Compute research metrics for person/object detection robustness."""
    person_tp = session_state['true_positive_person']
    person_fp = session_state['false_positive_person']
    person_tn = session_state['true_negative_person']
    person_fn = session_state['false_negative_person']

    object_tp = session_state['true_positive_object']
    object_fp = session_state['false_positive_object']
    object_tn = session_state['true_negative_object']
    object_fn = session_state['false_negative_object']

    return {
        'person': {
            'true_positive': person_tp,
            'false_positive': person_fp,
            'true_negative': person_tn,
            'false_negative': person_fn,
            **calculate_binary_metrics(person_tp, person_fp, person_tn, person_fn)
        },
        'object': {
            'true_positive': object_tp,
            'false_positive': object_fp,
            'true_negative': object_tn,
            'false_negative': object_fn,
            **calculate_binary_metrics(object_tp, object_fp, object_tn, object_fn)
        }
    }


def normalize(value, low, high):
    """Normalize value into [0, 1] with clipping."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def build_invalid_gaze_result(reason, detection_confidence, frame_count):
    """Return a consistent gaze payload for ignored/invalid frames."""
    session_state['gaze_deviation_start_time'] = None
    last = session_state.get('last_gaze_result')

    result = {
        'valid': False,
        'reason': reason,
        'reused': False,
        'detection_confidence': float(detection_confidence),
        'gaze_confidence': 0.0,
        'gaze_ratio_left': float(last['gaze_ratio_left']) if last and last.get('gaze_ratio_left') is not None else None,
        'gaze_ratio_right': float(last['gaze_ratio_right']) if last and last.get('gaze_ratio_right') is not None else None,
        'gaze_ratio': float(last['gaze_ratio']) if last and last.get('gaze_ratio') is not None else None,
        'smoothed_gaze': float(last['smoothed_gaze']) if last and last.get('smoothed_gaze') is not None else None,
        'baseline_gaze': float(session_state['gaze_baseline']) if session_state['gaze_baseline'] is not None else None,
        'center_min': float(session_state['gaze_baseline'] - GAZE_CENTER_DELTA) if session_state['gaze_baseline'] is not None else None,
        'center_max': float(session_state['gaze_baseline'] + GAZE_CENTER_DELTA) if session_state['gaze_baseline'] is not None else None,
        'deviation_duration': 0.0,
        'outside_center': False,
        'looking_away': False,
        'sustained_deviation': False,
        'head_pose_deviated': False,
        'violation': False,
        'alert_triggered': False,
        'suspicious': False,
        'direction': last['direction'] if last and last.get('direction') else 'Unknown',
        'calibrated': session_state['gaze_calibrated'],
        'calibration_progress': len(session_state['gaze_calibration_values']),
        'frame_count': frame_count
    }

    session_state['last_gaze_result'] = result
    return result


def get_adaptive_gaze_result(landmarks, frame_shape, detection_confidence, head_pose_deviated, frame_count):
    """
    Adaptive geometric gaze estimation with temporal filtering and adaptive thresholds.
    """
    if frame_count % GAZE_PROCESS_EVERY_N != 0 and session_state['last_gaze_result'] is not None:
        reused = dict(session_state['last_gaze_result'])
        reused['reused'] = True
        reused['alert_triggered'] = False
        reused['suspicious'] = False
        return reused

    if detection_confidence < GAZE_MIN_DETECTION_CONFIDENCE:
        return build_invalid_gaze_result('low_detection_confidence', detection_confidence, frame_count)

    h, w = frame_shape[:2]

    required_ids = [LEFT_EYE_LEFT, LEFT_EYE_RIGHT, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, LEFT_IRIS, RIGHT_IRIS]
    for idx in required_ids:
        lm = landmarks[idx]
        if lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
            return build_invalid_gaze_result('unreliable_iris_landmarks', detection_confidence, frame_count)

    left_corner_x = landmarks[LEFT_EYE_LEFT].x * w
    right_corner_x = landmarks[LEFT_EYE_RIGHT].x * w
    left_iris_x = landmarks[LEFT_IRIS].x * w

    right_left_corner_x = landmarks[RIGHT_EYE_LEFT].x * w
    right_right_corner_x = landmarks[RIGHT_EYE_RIGHT].x * w
    right_iris_x = landmarks[RIGHT_IRIS].x * w

    left_eye_width = right_corner_x - left_corner_x
    right_eye_width = right_right_corner_x - right_left_corner_x

    if left_eye_width <= 1.0 or right_eye_width <= 1.0:
        return build_invalid_gaze_result('eye_geometry_invalid', detection_confidence, frame_count)

    gaze_ratio_left = (left_iris_x - left_corner_x) / left_eye_width
    gaze_ratio_right = (right_iris_x - right_left_corner_x) / right_eye_width

    if not np.isfinite(gaze_ratio_left) or not np.isfinite(gaze_ratio_right):
        return build_invalid_gaze_result('gaze_ratio_invalid', detection_confidence, frame_count)

    raw_gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2.0

    width_score = normalize((left_eye_width + right_eye_width) / 2.0, 6.0, 30.0)
    ratio_score = 1.0 if (-0.2 <= gaze_ratio_left <= 1.2 and -0.2 <= gaze_ratio_right <= 1.2) else 0.4
    gaze_confidence = min(float(detection_confidence), 0.5 * width_score + 0.5 * ratio_score)

    if gaze_confidence < GAZE_MIN_DETECTION_CONFIDENCE:
        return build_invalid_gaze_result('gaze_confidence_below_threshold', detection_confidence, frame_count)

    session_state['gaze_calibration_values'].append(raw_gaze_ratio)
    if (not session_state['gaze_calibrated']) and len(session_state['gaze_calibration_values']) >= GAZE_CALIBRATION_FRAMES:
        session_state['gaze_baseline'] = float(np.mean(session_state['gaze_calibration_values']))
        session_state['gaze_calibrated'] = True

    session_state['gaze_history'].append(raw_gaze_ratio)
    smoothed_gaze = float(np.mean(session_state['gaze_history']))

    baseline = session_state['gaze_baseline'] if session_state['gaze_baseline'] is not None else smoothed_gaze
    center_min = baseline - GAZE_CENTER_DELTA
    center_max = baseline + GAZE_CENTER_DELTA

    outside_center = session_state['gaze_calibrated'] and (smoothed_gaze < center_min or smoothed_gaze > center_max)

    now = time.time()
    if outside_center:
        if session_state['gaze_deviation_start_time'] is None:
            session_state['gaze_deviation_start_time'] = now
        deviation_duration = now - session_state['gaze_deviation_start_time']
    else:
        session_state['gaze_deviation_start_time'] = None
        deviation_duration = 0.0

    sustained_deviation = outside_center and deviation_duration >= GAZE_DEVIATION_SECONDS
    violation = sustained_deviation

    alert_triggered = False
    if violation:
        last_alert = session_state['gaze_last_alert_time']
        if last_alert is None or (now - last_alert) >= GAZE_COOLDOWN_SECONDS:
            alert_triggered = True
            session_state['gaze_last_alert_time'] = now

    if smoothed_gaze < center_min:
        direction = 'Looking Right'
    elif smoothed_gaze > center_max:
        direction = 'Looking Left'
    else:
        direction = 'Looking Center'

    result = {
        'valid': True,
        'reason': 'ok',
        'reused': False,
        'detection_confidence': float(detection_confidence),
        'gaze_confidence': float(gaze_confidence),
        'gaze_ratio_left': float(gaze_ratio_left),
        'gaze_ratio_right': float(gaze_ratio_right),
        'gaze_ratio': float(raw_gaze_ratio),
        'smoothed_gaze': float(smoothed_gaze),
        'baseline_gaze': float(baseline),
        'center_min': float(center_min),
        'center_max': float(center_max),
        'deviation_duration': float(deviation_duration),
        'outside_center': bool(outside_center),
        'looking_away': bool(outside_center),
        'sustained_deviation': bool(sustained_deviation),
        'head_pose_deviated': bool(head_pose_deviated),
        'violation': bool(violation),
        'alert_triggered': bool(alert_triggered),
        'suspicious': bool(alert_triggered),
        'direction': direction,
        'calibrated': session_state['gaze_calibrated'],
        'calibration_progress': len(session_state['gaze_calibration_values']),
        'frame_count': frame_count
    }

    session_state['last_gaze_result'] = result
    return result


def calculate_session_verdict():
    """Calculate current verdict with spoof and robust detection rules."""
    analyzed_frames = max(0, session_state['total_frames'] - session_state['CALIBRATION_FRAMES'])
    spoof_ratio = (session_state['spoof_frames'] / analyzed_frames) if analyzed_frames > 0 else 0.0
    spoof_invalid = analyzed_frames > 0 and spoof_ratio >= 0.10
    critical_type = session_state.get('critical_violation_type') if session_state.get('termination_triggered') else None
    person_invalid = critical_type in ('multiple_person', 'multiple_people')
    object_invalid = critical_type == 'prohibited_object'
    spoof_critical = critical_type == 'spoof_detected'

    if person_invalid:
        status = 'EXAM INVALID'
        reason = 'Persistent multiple-person detection threshold reached'
    elif object_invalid:
        status = 'EXAM INVALID'
        reason = 'Persistent prohibited-object detection threshold reached'
    elif spoof_critical:
        status = 'EXAM INVALID'
        reason = 'Persistent spoof detection threshold reached'
    elif spoof_invalid:
        status = 'EXAM INVALID'
        reason = 'Face spoofing detected in >= 10% of analyzed frames'
    else:
        status = 'PENDING'
        reason = 'No spoofing threshold breach'

    return {
        'status': status,
        'reason': reason,
        'analyzed_frames': analyzed_frames,
        'spoof_ratio': spoof_ratio,
        'spoof_invalid': spoof_invalid,
        'person_invalid': person_invalid,
        'object_invalid': object_invalid,
        'spoof_critical': spoof_critical
    }

# ================= API ENDPOINTS =================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'timestamp': time.time()
    })

@app.route('/api/set-reference', methods=['POST'])
def set_reference():
    """Set reference image for face verification"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        frame = base64_to_image(image_data)
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)
        
        if not result.detections:
            return jsonify({'error': 'No face detected in reference image'}), 400
        
        detection = result.detections[0]
        d = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        bbox = (
            max(0, int(d.xmin * w)),
            max(0, int(d.ymin * h)),
            int(d.width * w),
            int(d.height * h)
        )
        
        embedding = get_face_embedding(frame, bbox)
        if embedding is None:
            return jsonify({'error': 'Failed to generate face embedding'}), 400
        
        session_state['reference_embedding'] = embedding
        
        # Reset session
        session_state['baseline_calibrated'] = False
        session_state['calibration_yaws'] = []
        session_state['calibration_pitches'] = []
        session_state['calibration_rolls'] = []
        session_state['frame_count'] = 0
        session_state['same_person_frames'] = 0
        session_state['different_person_frames'] = 0
        session_state['deviation_frames'] = 0
        session_state['gaze_deviation_frames'] = 0
        session_state['multiple_person_frames'] = 0
        session_state['prohibited_object_frames'] = 0
        session_state['spoof_frames'] = 0
        session_state['total_frames'] = 0
        session_state['last_spoof_detected'] = False
        session_state['last_spoof_event_time'] = None
        session_state['prohibited_objects_detected'] = {}
        reset_spoof_state()
        reset_pose_tracking_state()
        reset_gaze_tracking_state()
        reset_detection_tracking_state()
        
        return jsonify({
            'success': True,
            'message': 'Reference image set successfully'
        })
    
    except Exception as e:
        print(f"Error in set_reference: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start proctoring session"""
    if session_state['reference_embedding'] is None:
        return jsonify({'error': 'Reference image not set'}), 400
    
    reset_spoof_state()
    session_state['frame_count'] = 0
    session_state['same_person_frames'] = 0
    session_state['different_person_frames'] = 0
    session_state['deviation_frames'] = 0
    session_state['gaze_deviation_frames'] = 0
    session_state['multiple_person_frames'] = 0
    session_state['prohibited_object_frames'] = 0
    session_state['prohibited_objects_detected'] = {}
    session_state['spoof_frames'] = 0
    session_state['total_frames'] = 0
    session_state['last_spoof_detected'] = False
    session_state['last_spoof_event_time'] = None
    reset_pose_tracking_state()
    reset_gaze_tracking_state()
    reset_detection_tracking_state()
    session_state['unauthorized_person_frames'] = 0
    session_state['multiple_person_frames'] = 0
    session_state['object_frames'] = 0
    session_state['pose_deviation_frames'] = 0
    session_state['object_continuous_frames'] = 0
    session_state['person_continuous_frames'] = 0
    session_state['spoof_continuous_frames'] = 0

    session_state['is_active'] = True
    return jsonify({'success': True, 'message': 'Session started'})

@app.route('/api/stop-session', methods=['POST'])
def stop_session():
    """Stop proctoring session"""
    session_state['is_active'] = False
    verdict = calculate_session_verdict()
    return jsonify({
        'success': True,
        'message': 'Session stopped',
        'verdict': {
            'status': verdict['status'],
            'reason': verdict['reason']
        }
    })

@app.route('/api/get-stats', methods=['GET'])
def get_stats():
    """Get current session statistics"""
    total_frames = session_state['total_frames']
    verdict = calculate_session_verdict()
    analyzed_frames = max(1, verdict['analyzed_frames'])
    gaze_metrics = calculate_gaze_research_metrics()
    detection_metrics = calculate_detection_research_metrics()
    
    return jsonify({
        'total_frames': total_frames,
        'analyzed_frames': analyzed_frames,
        'calibrated': session_state['baseline_calibrated'],
        'baseline': {
            'yaw': session_state['baseline_yaw'],
            'pitch': session_state['baseline_pitch'],
            'roll': session_state['baseline_roll']
        } if session_state['baseline_calibrated'] else None,
        'stats': {
            'same_person': session_state['same_person_frames'],
            'different_person': session_state['different_person_frames'],
            'deviation': session_state['deviation_frames'],
            'gaze_deviation': session_state['gaze_deviation_frames'],
            'multiple_person': session_state['multiple_person_frames'],
            'prohibited_object': session_state['prohibited_object_frames'],
            'spoof': session_state['spoof_frames'],
            'person_continuous_frames': session_state['person_continuous_frames'],
            'object_continuous_frames': session_state['object_continuous_frames'],
            'spoof_continuous_frames': session_state['spoof_continuous_frames']
        },
        'prohibited_objects': session_state['prohibited_objects_detected'],
        'spoof': {
            'spoof_frames': session_state['spoof_frames'],
            'confidence_threshold': 0.10,
            'ratio': float(verdict['spoof_ratio']),
            'invalid': verdict['spoof_invalid']
        },
        'gaze': {
            'calibrated': session_state['gaze_calibrated'],
            'calibration_progress': len(session_state['gaze_calibration_values']),
            'calibration_frames': GAZE_CALIBRATION_FRAMES,
            'baseline': float(session_state['gaze_baseline']) if session_state['gaze_baseline'] is not None else None,
            'center_min': float(session_state['gaze_baseline'] - GAZE_CENTER_DELTA) if session_state['gaze_baseline'] is not None else None,
            'center_max': float(session_state['gaze_baseline'] + GAZE_CENTER_DELTA) if session_state['gaze_baseline'] is not None else None,
            'deviation_seconds_threshold': GAZE_DEVIATION_SECONDS,
            'cooldown_seconds': GAZE_COOLDOWN_SECONDS,
            'metrics': gaze_metrics
        },
        'detection': {
            'person_confidence_threshold': PERSON_CONFIDENCE_THRESHOLD,
            'object_confidence_threshold': OBJECT_CONFIDENCE_THRESHOLD,
            'object_continuous_frames_threshold': OBJECT_CONTINUOUS_FRAMES_THRESHOLD,
            'multiple_person_continuous_frames_threshold': MULTIPLE_PERSON_CONTINUOUS_FRAMES_THRESHOLD,
            'spoof_continuous_frames_threshold': SPOOF_CONTINUOUS_FRAMES_THRESHOLD,
            'unified_continuous_frames_threshold': VIOLATION_FRAMES_THRESHOLD,
            'absence_seconds_threshold': ABSENCE_SECONDS_THRESHOLD,
            'pose_deviation_seconds_threshold': POSE_DEVIATION_SECONDS_THRESHOLD,
            'cooldown_seconds': DETECTION_COOLDOWN_SECONDS,
            'min_box_area': MIN_BOX_AREA,
            'history_window': DETECTION_HISTORY_WINDOW,
            'process_every_n': DETECTION_PROCESS_EVERY_N,
            'multiple_person_violation': session_state['multiple_person_violation'],
            'prohibited_object_violation': session_state['prohibited_object_violation'],
            'unauthorized_person_frames': session_state['unauthorized_person_frames'],
            'multiple_person_frames': session_state['multiple_person_frames'],
            'object_frames': session_state['object_frames'],
            'spoof_continuous_frames': session_state['spoof_continuous_frames'],
            'pose_deviation_frames': session_state['pose_deviation_frames'],
            'absence_triggered': session_state['absence_triggered'],
            'pose_violation_triggered': session_state['pose_violation_triggered'],
            'warning_active': session_state['warning_active'],
            'warning_violation_type': session_state['warning_violation_type'],
            'termination_triggered': session_state['termination_triggered'],
            'critical_violation_type': session_state['critical_violation_type'],
            'metrics': detection_metrics
        },
        'verdict': {
            'status': verdict['status'],
            'reason': verdict['reason']
        }
    })

# ================= WEBSOCKET EVENTS =================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to proctoring server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('process_frame')
def handle_frame(data):
    """Process a single frame from client"""
    try:
        if not session_state['is_active']:
            emit('frame_result', {'error': 'Session not active'})
            return
        
        if session_state['reference_embedding'] is None:
            emit('frame_result', {'error': 'Reference not set'})
            return
        
        image_data = data.get('image')
        frame = base64_to_image(image_data)
        
        if frame is None:
            emit('frame_result', {'error': 'Invalid frame data'})
            return
        
        session_state['frame_count'] += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_result = face_detection.process(rgb)
        
        if not det_result.detections:
            session_state['unauthorized_person_frames'] = 0
            session_state['person_continuous_frames'] = 0
            session_state['multiple_person_frames'] = 0
            session_state['object_continuous_frames'] = 0
            session_state['object_frames'] = 0
            session_state['spoof_continuous_frames'] = 0
            session_state['pose_deviation_frames'] = 0
            session_state['pose_deviation_start_time'] = None
            session_state['pose_violation_triggered'] = False

            emit('frame_result', {
                'no_face': True,
                'message': 'No face detected',
                'frame_count': session_state['frame_count']
            })
            return
        
        # Get bounding box and detection confidence
        detection = det_result.detections[0]
        d = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        bbox = (
            max(0, int(d.xmin * w)),
            max(0, int(d.ymin * h)),
            int(d.width * w),
            int(d.height * h)
        )
        detection_confidence = get_detection_confidence(detection)

        # Spoof detection before face verification
        session_state['total_frames'] += 1
        is_real, spoof_confidence = detect_spoof(frame)

        if not is_real:
            session_state['spoof_frames'] += 1
            session_state['spoof_continuous_frames'] += 1
            session_state['unauthorized_person_frames'] = 0
            session_state['pose_deviation_start_time'] = None
            session_state['pose_violation_triggered'] = False
            session_state['pose_deviation_frames'] = 0
            session_state['person_continuous_frames'] = 0
            session_state['multiple_person_frames'] = 0
            session_state['object_continuous_frames'] = 0
            session_state['object_frames'] = 0
            session_state['multiple_person_violation'] = False
            session_state['prohibited_object_violation'] = False
            verdict = calculate_session_verdict()
            current_time = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
            now = time.time()
            last_spoof_event_time = session_state.get('last_spoof_event_time')

            # Emit only when spoof status changes to keep realtime traffic lightweight.
            should_emit_spoof_event = (
                (not session_state['last_spoof_detected']) and
                (
                    last_spoof_event_time is None or
                    (now - last_spoof_event_time) >= SPOOF_EVENT_COOLDOWN_SECONDS
                )
            )

            if should_emit_spoof_event:
                socketio.emit('spoof_event', {
                    'spoof_detected': True,
                    'confidence': float(spoof_confidence),
                    'frame': session_state['frame_count'],
                    'timestamp': current_time
                })
                session_state['last_spoof_event_time'] = now
            session_state['last_spoof_detected'] = True

            print(
                f"Frame {session_state['frame_count']}: "
                f"Spoof: SPOOF DETECTED ({spoof_confidence:.2f})"
            )

            cap_violation_counters()

            print(
                "Unauthorized:", session_state['unauthorized_person_frames'],
                "Multiple:", session_state['multiple_person_frames'],
                "Object:", session_state['object_frames'],
                "Spoof:", session_state['spoof_continuous_frames'],
                "Gaze:", session_state['gaze_deviation_frames']
            )

            termination_triggered, critical_payload = check_unified_continuous_termination(
                frame_number=session_state['frame_count']
            )

            if termination_triggered:

                emit('frame_result', {
                    'success': True,
                    'frame_count': session_state['frame_count'],
                    'status': 'SPOOF DETECTED',
                    'spoof': {
                        'spoof_detected': True,
                        'confidence': float(spoof_confidence)
                    },
                    'critical_violation': critical_payload,
                    'session_terminated': True,
                    'calibrated': session_state['baseline_calibrated'],
                    'calibration_progress': len(session_state['calibration_yaws']) if not session_state['baseline_calibrated'] else session_state['CALIBRATION_FRAMES'],
                    'verdict': {
                        'status': 'EXAM INVALID',
                        'reason': 'Persistent spoof detection threshold reached'
                    }
                })
                return

            emit('frame_result', {
                'success': True,
                'frame_count': session_state['frame_count'],
                'status': 'SPOOF DETECTED',
                'spoof': {
                    'spoof_detected': True,
                    'confidence': float(spoof_confidence)
                },
                'calibrated': session_state['baseline_calibrated'],
                'calibration_progress': len(session_state['calibration_yaws']) if not session_state['baseline_calibrated'] else session_state['CALIBRATION_FRAMES'],
                'verdict': {
                    'status': verdict['status'],
                    'reason': verdict['reason']
                }
            })
            return

        if session_state['last_spoof_detected']:
            session_state['last_spoof_detected'] = False
        session_state['spoof_continuous_frames'] = 0
        
        # Face verification
        embedding = get_face_embedding(frame, bbox)
        if embedding is None:
            emit('frame_result', {'error': 'Failed to generate embedding'})
            return
        
        distance = np.linalg.norm(embedding - session_state['reference_embedding'])
        
        if distance < 1.0:
            session_state['same_person_frames'] += 1
            identity = "Authorized"
            session_state['unauthorized_person_frames'] = 0
        else:
            session_state['different_person_frames'] += 1
            identity = "Unauthorized"
            session_state['unauthorized_person_frames'] += 1
        
        # Head pose
        mesh_result = face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            # Non-fatal: face mesh may fail transiently due motion/blur.
            session_state['pose_deviation_start_time'] = None
            session_state['pose_violation_triggered'] = False
            session_state['pose_deviation_frames'] = 0
            session_state['person_continuous_frames'] = 0
            session_state['multiple_person_frames'] = 0
            session_state['object_continuous_frames'] = 0
            session_state['object_frames'] = 0

            termination_triggered, critical_payload = check_unified_continuous_termination(
                frame_number=session_state['frame_count']
            )

            if termination_triggered:
                emit('frame_result', {
                    'success': True,
                    'frame_count': session_state['frame_count'],
                    'identity': identity,
                    'distance': float(distance),
                    'spoof': {
                        'spoof_detected': False,
                        'confidence': float(spoof_confidence)
                    },
                    'critical_violation': critical_payload,
                    'session_terminated': True,
                    'calibrated': session_state['baseline_calibrated']
                })
                # No explicit loop exists in this handler; return is the immediate stop equivalent.
                return

            print(f"Frame {session_state['frame_count']}: Face mesh unavailable; frame skipped")
            emit('frame_result', {
                'success': True,
                'frame_count': session_state['frame_count'],
                'identity': identity,
                'distance': float(distance),
                'spoof': {
                    'spoof_detected': False,
                    'confidence': float(spoof_confidence)
                },
                'gaze': {
                    'valid': False,
                    'reason': 'no_face_mesh',
                    'direction': 'Unknown',
                    'suspicious': False,
                    'status': 'Unavailable'
                },
                'mesh_missing': True,
                'calibrated': session_state['baseline_calibrated']
            })
            return
        
        landmarks = mesh_result.multi_face_landmarks[0].landmark
        yaw, pitch, roll = get_head_pose(frame, landmarks)
        
        # Baseline calibration
        if not session_state['baseline_calibrated']:
            session_state['calibration_yaws'].append(yaw)
            session_state['calibration_pitches'].append(pitch)
            session_state['calibration_rolls'].append(roll)
            
            if len(session_state['calibration_yaws']) >= session_state['CALIBRATION_FRAMES']:
                session_state['baseline_yaw'] = np.mean(session_state['calibration_yaws'])
                session_state['baseline_pitch'] = np.mean(session_state['calibration_pitches'])
                session_state['baseline_roll'] = np.mean(session_state['calibration_rolls'])
                session_state['baseline_calibrated'] = True
                session_state['pose_yaw_history'].clear()
                session_state['pose_pitch_history'].clear()
                session_state['pose_roll_history'].clear()
                session_state['pose_stabilization_start_time'] = None
                
                emit('calibration_complete', {
                    'baseline': {
                        'yaw': float(session_state['baseline_yaw']),
                        'pitch': float(session_state['baseline_pitch']),
                        'roll': float(session_state['baseline_roll'])
                    }
                })
            
            pose_status = "Calibrating"
            relative_yaw = 0
            relative_pitch = 0
            relative_roll = 0
            pose_deviation_duration = 0.0
            pose_deviated_for_timer = False
            head_pose_deviated = False
        else:
            raw_relative_yaw = yaw - session_state['baseline_yaw']
            raw_relative_pitch = pitch - session_state['baseline_pitch']
            raw_relative_roll = roll - session_state['baseline_roll']

            session_state['pose_yaw_history'].append(raw_relative_yaw)
            session_state['pose_pitch_history'].append(raw_relative_pitch)
            session_state['pose_roll_history'].append(raw_relative_roll)

            relative_yaw = float(np.mean(session_state['pose_yaw_history']))
            relative_pitch = float(np.mean(session_state['pose_pitch_history']))
            relative_roll = float(np.mean(session_state['pose_roll_history']))

            outside_pose_range = (
                abs(relative_yaw) > POSE_YAW_THRESHOLD or
                abs(relative_pitch) > POSE_PITCH_THRESHOLD or
                abs(relative_roll) > POSE_ROLL_THRESHOLD
            )
            pose_deviated_for_timer = outside_pose_range

            pose_now = time.time()
            if outside_pose_range:
                if session_state['pose_stabilization_start_time'] is None:
                    session_state['pose_stabilization_start_time'] = pose_now
                pose_deviation_duration = pose_now - session_state['pose_stabilization_start_time']
            else:
                session_state['pose_stabilization_start_time'] = None
                pose_deviation_duration = 0.0

            head_pose_deviated = outside_pose_range and pose_deviation_duration >= POSE_DEVIATION_SECONDS

            if head_pose_deviated:
                session_state['deviation_frames'] += 1

            if head_pose_deviated:
                pose_status = "Deviating"
            elif outside_pose_range:
                pose_status = "Stabilizing"
            else:
                pose_status = "Normal"

        session_state['pose_deviation_frames'] = 0
        
        # Adaptive geometric gaze with temporal filtering and head-pose fusion
        gaze_result = get_adaptive_gaze_result(
            landmarks=landmarks,
            frame_shape=frame.shape,
            detection_confidence=detection_confidence,
            head_pose_deviated=head_pose_deviated,
            frame_count=session_state['frame_count']
        )

        gaze_direction = gaze_result['direction']
        gaze_suspicious = gaze_result['suspicious']
        if gaze_result['calibrated']:
            if gaze_result['violation']:
                gaze_status = "Violation"
            elif gaze_result['outside_center']:
                gaze_status = "Outside Center"
            else:
                gaze_status = "Normal"
        else:
            gaze_status = "Calibrating"

        add_overlay_text(
            frame,
            relative_yaw=relative_yaw,
            relative_pitch=relative_pitch,
            relative_roll=relative_roll,
            gaze_direction=gaze_direction
        )

        print(
            f"Frame {session_state['frame_count']}:\n"
            f"Yaw: {relative_yaw:.2f}\n"
            f"Pitch: {relative_pitch:.2f}\n"
            f"Roll: {relative_roll:.2f}\n"
            f"Gaze: {gaze_direction}"
        )

        if gaze_result['violation'] and session_state['baseline_calibrated']:
            session_state['gaze_deviation_frames'] += 1
        else:
            session_state['gaze_deviation_frames'] = 0

        # Optional ground-truth label for research metrics from client payload.
        expected_gaze_deviation = data.get('expected_gaze_deviation')
        if isinstance(expected_gaze_deviation, bool):
            update_gaze_research_metrics(
                predicted_positive=gaze_result['violation'],
                ground_truth_positive=expected_gaze_deviation
            )
        
        # YOLO detection with temporal persistence and smoothing
        person_count, detected_objects, detection_meta = detect_persons_and_objects(
            frame,
            session_state['frame_count']
        )

        multiple_person_detected = person_count > 1
        prohibited_object_detected = bool(detected_objects)

        session_state['person_history'].append(1 if multiple_person_detected else 0)
        session_state['object_history'].append(1 if prohibited_object_detected else 0)

        person_majority = sum(session_state['person_history']) >= (len(session_state['person_history']) // 2 + 1)
        object_majority = sum(session_state['object_history']) >= (len(session_state['object_history']) // 2 + 1)

        if multiple_person_detected:
            session_state['person_continuous_frames'] += 1
            session_state['multiple_person_frames'] += 1
        else:
            session_state['person_continuous_frames'] = 0
            session_state['multiple_person_frames'] = 0

        if prohibited_object_detected:
            session_state['object_continuous_frames'] += 1
            session_state['object_frames'] += 1
        else:
            session_state['object_continuous_frames'] = 0
            session_state['object_frames'] = 0

        cap_violation_counters()

        session_state['multiple_person_violation'] = (
            session_state['person_continuous_frames'] >= MULTIPLE_PERSON_CONTINUOUS_FRAMES_THRESHOLD
        )
        session_state['prohibited_object_violation'] = (
            session_state['object_continuous_frames'] >= OBJECT_CONTINUOUS_FRAMES_THRESHOLD
        )

        if person_count > 1:
            person_status = f"Multiple ({person_count})"
        elif person_count == 1:
            person_status = "Single"
        else:
            person_status = "None"

        if detected_objects:
            object_status = f"Found: {', '.join([f'{k}({v})' for k, v in detected_objects.items()])}"
        else:
            object_status = "None"

        if session_state['multiple_person_violation']:
            person_status = f"Violation ({person_count})"
        if session_state['prohibited_object_violation']:
            object_status = "Violation"

        if session_state['prohibited_object_violation'] and session_state['baseline_calibrated']:
            session_state['prohibited_object_frames'] += 1
            tracked_objects = detected_objects or detection_meta.get('prohibited_objects', {})
            for obj, count in tracked_objects.items():
                if obj not in session_state['prohibited_objects_detected']:
                    session_state['prohibited_objects_detected'][obj] = 0
                session_state['prohibited_objects_detected'][obj] += count

        # Optional ground-truth labels for detection research metrics.
        expected_multiple_person = data.get('expected_multiple_person')
        expected_prohibited_object = data.get('expected_prohibited_object')
        update_detection_research_metrics(
            predicted_multiple_person=person_majority,
            predicted_prohibited_object=object_majority,
            expected_multiple_person=expected_multiple_person,
            expected_prohibited_object=expected_prohibited_object
        )

        detected_object_names = ', '.join(detected_objects.keys()) if detected_objects else 'None'
        confidence_for_log = max(
            float(detection_meta.get('person_confidence_max', 0.0)),
            float(detection_meta.get('object_confidence_max', 0.0))
        )
        continuous_for_log = max(
            session_state['person_continuous_frames'],
            session_state['object_continuous_frames']
        )
        detection_violation = (
            session_state['multiple_person_violation'] or
            session_state['prohibited_object_violation']
        )

        print(
            f"Frame {session_state['frame_count']}:\n"
            f"Persons Detected: {person_count}\n"
            f"Prohibited Objects: {detected_object_names}\n"
            f"Confidence: {confidence_for_log:.2f}\n"
            f"Continuous Frames: {continuous_for_log}\n"
            f"Violation: {detection_violation}"
        )

        print(
            "Unauthorized:", session_state['unauthorized_person_frames'],
            "Multiple:", session_state['multiple_person_frames'],
            "Object:", session_state['object_frames'],
            "Spoof:", session_state['spoof_continuous_frames'],
            "Gaze:", session_state['gaze_deviation_frames']
        )

        termination_triggered, critical_payload = check_unified_continuous_termination(
            frame_number=session_state['frame_count']
        )

        if termination_triggered:

            emit('frame_result', {
                'success': True,
                'frame_count': session_state['frame_count'],
                'identity': identity,
                'distance': float(distance),
                'spoof': {
                    'spoof_detected': False,
                    'confidence': float(spoof_confidence)
                },
                'pose': {
                    'status': pose_status,
                    'yaw': float(yaw),
                    'pitch': float(pitch),
                    'roll': float(roll),
                    'relative_yaw': float(relative_yaw),
                    'relative_pitch': float(relative_pitch),
                    'relative_roll': float(relative_roll),
                    'deviation_duration': float(pose_deviation_duration),
                    'deviation_seconds_threshold': POSE_DEVIATION_SECONDS
                },
                'gaze': {
                    'direction': gaze_direction,
                    'suspicious': gaze_suspicious,
                    'status': gaze_status
                },
                'detection': {
                    'person_count': person_count,
                    'objects': detected_objects,
                    'person_confidence_max': float(detection_meta.get('person_confidence_max', 0.0)),
                    'object_confidence_max': float(detection_meta.get('object_confidence_max', 0.0)),
                    'person_continuous_frames': session_state['person_continuous_frames'],
                    'object_continuous_frames': session_state['object_continuous_frames'],
                    'spoof_continuous_frames': session_state['spoof_continuous_frames'],
                    'unauthorized_person_frames': session_state['unauthorized_person_frames'],
                    'object_frames': session_state['object_frames'],
                    'pose_deviation_frames': session_state['pose_deviation_frames'],
                    'person_majority': person_majority,
                    'object_majority': object_majority,
                    'multiple_person_violation': session_state['multiple_person_violation'],
                    'prohibited_object_violation': session_state['prohibited_object_violation']
                },
                'critical_violation': critical_payload,
                'session_terminated': True,
                'calibrated': session_state['baseline_calibrated']
            })
            # Stop frame processing immediately after termination.
            return

        if session_state['baseline_calibrated']:
            pose_log = (
                f"{pose_status} (ΔY={relative_yaw:+.1f}°, "
                f"ΔP={relative_pitch:+.1f}°, ΔR={relative_roll:+.1f}°)"
            )
        else:
            pose_log = f"{pose_status} ({len(session_state['calibration_yaws'])}/{session_state['CALIBRATION_FRAMES']})"

        print(
            f"Frame {session_state['frame_count']}: {identity} | "
            f"Spoof: REAL ({spoof_confidence:.2f}) | "
            f"Pose: {pose_log} | "
            f"Gaze: {gaze_status} ({gaze_direction}) | "
            f"Persons: {person_status} | Objects: {object_status}"
        )

        if gaze_result['valid']:
            print(
                f"Frame {session_state['frame_count']}:\n"
                f"Gaze Ratio: {gaze_result['gaze_ratio']:.2f}\n"
                f"Smoothed Gaze: {gaze_result['smoothed_gaze']:.2f}\n"
                f"Baseline: {gaze_result['baseline_gaze']:.2f}\n"
                f"Deviation Duration: {gaze_result['deviation_duration']:.1f} sec\n"
                f"Head Pose: {'Deviated' if head_pose_deviated else 'Normal'}\n"
                f"Violation: {gaze_result['violation']}"
            )
        else:
            print(
                f"Frame {session_state['frame_count']}:\n"
                f"Gaze Ratio: N/A\n"
                f"Smoothed Gaze: N/A\n"
                f"Baseline: {session_state['gaze_baseline'] if session_state['gaze_baseline'] is not None else 'N/A'}\n"
                f"Deviation Duration: 0.0 sec\n"
                f"Head Pose: {'Deviated' if head_pose_deviated else 'Normal'}\n"
                f"Violation: False"
            )

        verdict = calculate_session_verdict()
        gaze_metrics = calculate_gaze_research_metrics()
        
        # Send result
        result = {
            'success': True,
            'frame_count': session_state['frame_count'],
            'identity': identity,
            'distance': float(distance),
            'spoof': {
                'spoof_detected': False,
                'confidence': float(spoof_confidence)
            },
            'pose': {
                'status': pose_status,
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'relative_yaw': float(relative_yaw),
                'relative_pitch': float(relative_pitch),
                'relative_roll': float(relative_roll),
                'deviation_duration': float(pose_deviation_duration),
                'deviation_seconds_threshold': POSE_DEVIATION_SECONDS
            },
            'gaze': {
                'direction': gaze_direction,
                'suspicious': gaze_suspicious,
                'status': gaze_status,
                'valid': gaze_result['valid'],
                'reason': gaze_result['reason'],
                'confidence': float(gaze_result['gaze_confidence']),
                'left_ratio': float(gaze_result['gaze_ratio_left']) if gaze_result['gaze_ratio_left'] is not None else None,
                'right_ratio': float(gaze_result['gaze_ratio_right']) if gaze_result['gaze_ratio_right'] is not None else None,
                'ratio': float(gaze_result['gaze_ratio']) if gaze_result['gaze_ratio'] is not None else None,
                'smoothed_ratio': float(gaze_result['smoothed_gaze']) if gaze_result['smoothed_gaze'] is not None else None,
                'baseline': float(gaze_result['baseline_gaze']) if gaze_result['baseline_gaze'] is not None else None,
                'center_min': float(gaze_result['center_min']) if gaze_result['center_min'] is not None else None,
                'center_max': float(gaze_result['center_max']) if gaze_result['center_max'] is not None else None,
                'deviation_duration': float(gaze_result['deviation_duration']),
                'outside_center': bool(gaze_result['outside_center']),
                'looking_away': bool(gaze_result.get('looking_away', False)),
                'sustained_deviation': bool(gaze_result['sustained_deviation']),
                'head_pose_fused': bool(gaze_result['head_pose_deviated']),
                'violation': bool(gaze_result['violation']),
                'alert_triggered': bool(gaze_result['alert_triggered']),
                'calibrated': bool(gaze_result['calibrated']),
                'calibration_progress': int(gaze_result['calibration_progress']),
                'process_every_n': GAZE_PROCESS_EVERY_N
            },
            'detection': {
                'person_count': person_count,
                'objects': detected_objects,
                'person_confidence_max': float(detection_meta.get('person_confidence_max', 0.0)),
                'object_confidence_max': float(detection_meta.get('object_confidence_max', 0.0)),
                'reused': bool(detection_meta.get('reused', False)),
                'person_majority': bool(person_majority),
                'object_majority': bool(object_majority),
                'person_continuous_frames': session_state['person_continuous_frames'],
                'object_continuous_frames': session_state['object_continuous_frames'],
                'spoof_continuous_frames': session_state['spoof_continuous_frames'],
                'unauthorized_person_frames': session_state['unauthorized_person_frames'],
                'object_frames': session_state['object_frames'],
                'pose_deviation_frames': session_state['pose_deviation_frames'],
                'unified_continuous_frames_threshold': VIOLATION_FRAMES_THRESHOLD,
                'object_continuous_frames_threshold': OBJECT_CONTINUOUS_FRAMES_THRESHOLD,
                'multiple_person_continuous_frames_threshold': MULTIPLE_PERSON_CONTINUOUS_FRAMES_THRESHOLD,
                'spoof_continuous_frames_threshold': SPOOF_CONTINUOUS_FRAMES_THRESHOLD,
                'multiple_person_violation': session_state['multiple_person_violation'],
                'prohibited_object_violation': session_state['prohibited_object_violation']
            },
            'calibrated': session_state['baseline_calibrated'],
            'calibration_progress': len(session_state['calibration_yaws']) if not session_state['baseline_calibrated'] else session_state['CALIBRATION_FRAMES'],
            'gaze_metrics': gaze_metrics,
            'verdict': {
                'status': verdict['status'],
                'reason': verdict['reason']
            }
        }
        
        emit('frame_result', result)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('frame_result', {'error': str(e)})

# ================= MAIN =================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Online Proctoring Server Ready!")
    print("="*50)
    print("Server running on http://localhost:5000")
    print("WebSocket ready for real-time proctoring")
    print("="*50 + "\n")
    
    socketio.run(
        app,
        debug=False,
        use_reloader=False,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )
