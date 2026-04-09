"""
Lightweight face anti-spoofing for real-time webcam frames.

This module uses a heuristic fallback approach based on:
- Blink / eye activity estimation (EAR variability)
- Face motion estimation (inter-frame ROI difference)
- Texture analysis (Laplacian variance)
- Depth variation proxy (landmark z-value spread)

Public API:
    detect_spoof(frame) -> (is_real: bool, confidence: float)
"""

from collections import deque

import cv2
import mediapipe as mp
import numpy as np


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5,
)

_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]

PROCESS_EVERY_N = 3
EAR_CLOSED_THRESHOLD = 0.18


_state = {
    "frame_index": 0,
    "last_is_real": True,
    "last_confidence": 0.5,
    "prev_face_gray": None,
    "ear_history": deque(maxlen=20),
    "blink_frames": deque(maxlen=50),
}


def _normalize(value, low, high):
    """Normalize to [0, 1] with clipping."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _safe_bbox(relative_bbox, width, height):
    """Convert MediaPipe relative bbox to clipped integer bbox."""
    x = max(0, int(relative_bbox.xmin * width))
    y = max(0, int(relative_bbox.ymin * height))
    w = int(relative_bbox.width * width)
    h = int(relative_bbox.height * height)

    if w <= 0 or h <= 0:
        return None

    w = min(w, width - x)
    h = min(h, height - y)

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def _points_from_landmarks(landmarks, eye_ids, width, height):
    pts = []
    for idx in eye_ids:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * width, lm.y * height], dtype=np.float32))
    return pts


def _eye_aspect_ratio(points):
    """
    EAR formula for 6 eye points:
    (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    a = np.linalg.norm(points[1] - points[5])
    b = np.linalg.norm(points[2] - points[4])
    c = np.linalg.norm(points[0] - points[3])
    if c == 0:
        return 0.0
    return float((a + b) / (2.0 * c))


def reset_spoof_state():
    """Reset temporal state when a new session starts."""
    _state["frame_index"] = 0
    _state["last_is_real"] = True
    _state["last_confidence"] = 0.5
    _state["prev_face_gray"] = None
    _state["ear_history"].clear()
    _state["blink_frames"].clear()


def detect_spoof(frame):
    """
    Input: OpenCV frame
    Output:
    is_real (bool)
    confidence (float)
    """
    _state["frame_index"] += 1

    # Keep cost low: reuse the last decision on skipped frames.
    if _state["frame_index"] % PROCESS_EVERY_N != 0:
        return _state["last_is_real"], float(_state["last_confidence"])

    if frame is None or frame.size == 0:
        _state["last_is_real"] = False
        _state["last_confidence"] = 0.0
        return False, 0.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_result = _detector.process(rgb)

    # If the face detector misses this frame, keep the previous stable value.
    if not det_result.detections:
        return _state["last_is_real"], float(_state["last_confidence"])

    h, w = frame.shape[:2]
    rel_bbox = det_result.detections[0].location_data.relative_bounding_box
    bbox = _safe_bbox(rel_bbox, w, h)

    if bbox is None:
        return _state["last_is_real"], float(_state["last_confidence"])

    x, y, bw, bh = bbox
    face_roi = frame[y:y + bh, x:x + bw]

    if face_roi.size == 0:
        return _state["last_is_real"], float(_state["last_confidence"])

    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (96, 96))

    # 1) Motion score
    motion_score = 0.5
    if _state["prev_face_gray"] is not None:
        diff = cv2.absdiff(face_gray, _state["prev_face_gray"])
        mean_diff = float(np.mean(diff))
        motion_score = _normalize(mean_diff, 1.5, 16.0)
    _state["prev_face_gray"] = face_gray

    # 2) Texture score (print/replay often has flatter texture)
    lap_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    texture_score = _normalize(lap_var, 35.0, 180.0)

    # 3) Depth variation + 4) Eye activity score
    depth_score = 0.35
    eye_score = 0.35

    mesh_result = _mesh.process(rgb)
    if mesh_result.multi_face_landmarks:
        landmarks = mesh_result.multi_face_landmarks[0].landmark

        z_vals = np.array([lm.z for lm in landmarks], dtype=np.float32)
        z_std = float(np.std(z_vals))
        depth_score = _normalize(z_std, 0.007, 0.028)

        left_eye_pts = _points_from_landmarks(landmarks, LEFT_EYE_IDS, w, h)
        right_eye_pts = _points_from_landmarks(landmarks, RIGHT_EYE_IDS, w, h)
        left_ear = _eye_aspect_ratio(left_eye_pts)
        right_ear = _eye_aspect_ratio(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        _state["ear_history"].append(avg_ear)

        if len(_state["ear_history"]) >= 5:
            ear_min = float(min(_state["ear_history"]))
            ear_max = float(max(_state["ear_history"]))
            ear_std = float(np.std(_state["ear_history"]))

            blink_like = ear_min < EAR_CLOSED_THRESHOLD and ear_max > (EAR_CLOSED_THRESHOLD + 0.06)
            if blink_like:
                _state["blink_frames"].append(_state["frame_index"])

            recent_blink = bool(
                _state["blink_frames"]
                and (_state["frame_index"] - _state["blink_frames"][-1] <= 45)
            )

            variability = _normalize(ear_std, 0.008, 0.03)
            eye_score = 0.6 * variability + (0.4 if recent_blink else 0.0)

    confidence = (
        0.30 * texture_score
        + 0.25 * motion_score
        + 0.25 * depth_score
        + 0.20 * eye_score
    )

    # Threshold selected for robustness on webcam streams without GPU.
    is_real = confidence >= 0.45

    confidence = float(np.clip(confidence, 0.0, 1.0))
    _state["last_is_real"] = bool(is_real)
    _state["last_confidence"] = confidence

    return bool(is_real), confidence
