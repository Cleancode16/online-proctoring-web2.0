print(">>> Script started")

import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from ultralytics import YOLO
from face_spoofing import detect_spoof, reset_spoof_state

print(">>> Imports done")

# ================= INITIALIZATION =================
embedder = FaceNet()
print(">>> FaceNet loaded")

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print(">>> MediaPipe models loaded")

# ================= YOLO INITIALIZATION =================
yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model for speed
print(">>> YOLO model loaded")

# COCO dataset class IDs
PERSON_CLASS = 0
CELL_PHONE_CLASS = 67
BOOK_CLASS = 73
LAPTOP_CLASS = 63

PROHIBITED_CLASSES = {
    CELL_PHONE_CLASS: 'Cell Phone',
    BOOK_CLASS: 'Book',
    LAPTOP_CLASS: 'Laptop'
}

# ================= STATIC 3D FACE MODEL =================
model_3d = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye corner
    (43.3, 32.7, -26.0),      # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1)      # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices
# nose, chin, left eye, right eye, left mouth, right mouth
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Eye landmarks for gaze tracking
# Left eye: corners and iris center
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_IRIS = 468  # Left iris center

# Right eye: corners and iris center
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS = 473  # Right iris center

# ================= FACE EMBEDDING =================
def get_face_embedding(frame, bbox):
    x, y, w, h = bbox
    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (160, 160))
    return embedder.embeddings([face_crop])[0]

# ================= HEAD POSE =================
def get_head_pose(frame, landmarks):
    h, w = frame.shape[:2]

    image_2d = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    cam_matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    _, rvec, _ = cv2.solvePnP(
        model_3d,
        image_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)

    pitch = np.degrees(np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
    roll  = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    
    return yaw, pitch, roll

# ================= EYE GAZE TRACKING =================
def get_eye_gaze(frame, landmarks):
    """
    Calculate eye gaze direction based on iris position relative to eye corners.
    Returns (left_gaze_ratio, right_gaze_ratio, gaze_direction)
    """
    h, w = frame.shape[:2]
    
    # Left eye
    left_eye_inner = landmarks[LEFT_EYE_INNER]
    left_eye_outer = landmarks[LEFT_EYE_OUTER]
    left_iris = landmarks[LEFT_IRIS]
    
    left_iris_x = left_iris.x * w
    left_inner_x = left_eye_inner.x * w
    left_outer_x = left_eye_outer.x * w
    
    # Calculate ratio: 0.5 is center, <0.5 is looking right, >0.5 is looking left
    left_eye_width = abs(left_inner_x - left_outer_x)
    if left_eye_width > 0:
        left_gaze_ratio = (left_iris_x - left_outer_x) / left_eye_width
    else:
        left_gaze_ratio = 0.5
    
    # Right eye
    right_eye_inner = landmarks[RIGHT_EYE_INNER]
    right_eye_outer = landmarks[RIGHT_EYE_OUTER]
    right_iris = landmarks[RIGHT_IRIS]
    
    right_iris_x = right_iris.x * w
    right_inner_x = right_eye_inner.x * w
    right_outer_x = right_eye_outer.x * w
    
    right_eye_width = abs(right_inner_x - right_outer_x)
    if right_eye_width > 0:
        right_gaze_ratio = (right_iris_x - right_outer_x) / right_eye_width
    else:
        right_gaze_ratio = 0.5
    
    # Average gaze ratio
    avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
    
    # Determine gaze direction
    if avg_gaze_ratio < 0.35:
        gaze_direction = "Looking Right"
    elif avg_gaze_ratio > 0.65:
        gaze_direction = "Looking Left"
    else:
        gaze_direction = "Looking Center"
    
    return left_gaze_ratio, right_gaze_ratio, gaze_direction

# ================= YOLO DETECTION =================
def detect_persons_and_objects(frame):
    """
    Detect persons and prohibited objects using YOLO.
    Returns (person_count, prohibited_objects_dict)
    """
    results = yolo_model(frame, verbose=False)
    
    person_count = 0
    prohibited_objects = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Count persons (confidence > 0.5)
            if cls == PERSON_CLASS and conf > 0.5:
                person_count += 1
            
            # Detect prohibited objects (confidence > 0.4)
            if cls in PROHIBITED_CLASSES and conf > 0.4:
                obj_name = PROHIBITED_CLASSES[cls]
                if obj_name not in prohibited_objects:
                    prohibited_objects[obj_name] = 0
                prohibited_objects[obj_name] += 1
    
    return person_count, prohibited_objects

# ================= PATHS =================
video_path = r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260207_13_29_05_Pro.mp4"
reference_image = r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260206_21_26_58_Pro.jpg"

# ================= LOAD REFERENCE =================
print(">>> Loading reference image")
ref_img = cv2.imread(reference_image)
ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

ref_result = face_detection.process(ref_rgb)

if not ref_result.detections:
    raise ValueError("No face detected in reference image")

d = ref_result.detections[0].location_data.relative_bounding_box
h, w = ref_img.shape[:2]
ref_bbox = (
    int(d.xmin * w),
    int(d.ymin * h),
    int(d.width * w),
    int(d.height * h)
)

ref_embedding = get_face_embedding(ref_img, ref_bbox)
print(">>> Reference embedding created")

# ================= VIDEO LOOP =================
cap = cv2.VideoCapture(video_path)
print(">>> Video opened")

frame_count = 0
processed = 0

same_person_frames = 0
different_person_frames = 0
deviation_frames = 0
gaze_deviation_frames = 0  # Track eye gaze deviations
multiple_person_frames = 0  # Track frames with multiple persons
prohibited_object_frames = 0  # Track frames with prohibited objects
spoof_frames = 0
total_frames = 0
prohibited_objects_detected = {}  # Track types of prohibited objects

# Baseline head pose angles (calibrated from initial frames)
baseline_yaw = None
baseline_pitch = None
baseline_roll = None
baseline_calibrated = False
CALIBRATION_FRAMES = 3  # Number of initial frames to average for baseline
calibration_yaws = []
calibration_pitches = []
calibration_rolls = []

MAX_FRAMES = 50
reset_spoof_state()

while processed < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 != 0:
        frame_count += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_result = face_detection.process(rgb)

    if not det_result.detections:
        frame_count += 1
        continue

    d = det_result.detections[0].location_data.relative_bounding_box
    h, w = frame.shape[:2]
    bbox = (
        int(d.xmin * w),
        int(d.ymin * h),
        int(d.width * w),
        int(d.height * h)
    )

    # -------- SPOOF DETECTION --------
    total_frames += 1
    is_real, spoof_confidence = detect_spoof(frame)

    if not is_real:
        spoof_frames += 1
        print(f"Frame {frame_count}: Spoof: SPOOF DETECTED ({spoof_confidence:.2f})")
        processed += 1
        frame_count += 1
        continue

    # -------- FACE VERIFICATION --------
    embedding = get_face_embedding(frame, bbox)
    if embedding is None:
        frame_count += 1
        continue

    distance = np.linalg.norm(embedding - ref_embedding)

    if distance < 1.0:
        same_person_frames += 1
        identity = "Authorized"
    else:
        different_person_frames += 1
        identity = "Unauthorized"

    # -------- HEAD POSE --------
    mesh_result = face_mesh.process(rgb)
    if not mesh_result.multi_face_landmarks:
        frame_count += 1
        continue

    landmarks = mesh_result.multi_face_landmarks[0].landmark
    yaw, pitch, roll = get_head_pose(frame, landmarks)

    # -------- BASELINE CALIBRATION --------
    if not baseline_calibrated:
        # Collect calibration data from initial frames
        calibration_yaws.append(yaw)
        calibration_pitches.append(pitch)
        calibration_rolls.append(roll)
        
        if len(calibration_yaws) >= CALIBRATION_FRAMES:
            # Calculate baseline as average of initial frames
            baseline_yaw = np.mean(calibration_yaws)
            baseline_pitch = np.mean(calibration_pitches)
            baseline_roll = np.mean(calibration_rolls)
            baseline_calibrated = True
            print(f"\n>>> Baseline calibrated: Yaw={baseline_yaw:.1f}°, Pitch={baseline_pitch:.1f}°, Roll={baseline_roll:.1f}°\n")
        
        # Skip deviation detection during calibration
        pose_status = "Calibrating"
        relative_yaw = 0
        relative_pitch = 0
        relative_roll = 0
    else:
        # Calculate relative angles from baseline
        relative_yaw = yaw - baseline_yaw
        relative_pitch = pitch - baseline_pitch
        relative_roll = roll - baseline_roll
        
        # Detect deviation based on relative movement
        deviating = abs(relative_yaw) > 15 or abs(relative_pitch) > 10 or abs(relative_roll) > 10
        pose_status = "Deviating" if deviating else "Normal"
        
        if deviating:
            deviation_frames += 1

    # -------- EYE GAZE TRACKING --------
    left_gaze, right_gaze, gaze_direction = get_eye_gaze(frame, landmarks)
    
    # Gaze is suspicious if looking away from center
    gaze_suspicious = gaze_direction != "Looking Center"
    gaze_status = "Suspicious" if gaze_suspicious else "Normal"
    
    if gaze_suspicious:
        gaze_deviation_frames += 1

    # -------- YOLO DETECTION --------
    person_count, detected_objects = detect_persons_and_objects(frame)
    
    # Track multiple persons
    if person_count > 1:
        multiple_person_frames += 1
        person_status = f"Multiple ({person_count})"
    elif person_count == 1:
        person_status = "Single"
    else:
        person_status = "None"
    
    # Track prohibited objects
    if detected_objects:
        prohibited_object_frames += 1
        for obj, count in detected_objects.items():
            if obj not in prohibited_objects_detected:
                prohibited_objects_detected[obj] = 0
            prohibited_objects_detected[obj] += count
        object_status = f"Found: {', '.join([f'{k}({v})' for k, v in detected_objects.items()])}"
    else:
        object_status = "None"

    # Display output
    if baseline_calibrated:
        print(
            f"Frame {frame_count}: {identity} | Spoof: REAL ({spoof_confidence:.2f}) | "
            f"Pose: {pose_status} (ΔY={relative_yaw:+.1f}°, ΔP={relative_pitch:+.1f}°, ΔR={relative_roll:+.1f}°) | "
            f"Gaze: {gaze_status} ({gaze_direction}) | Persons: {person_status} | Objects: {object_status}"
        )
    else:
        print(
            f"Frame {frame_count}: {identity} | Spoof: REAL ({spoof_confidence:.2f}) | "
            f"Pose: {pose_status} ({len(calibration_yaws)}/{CALIBRATION_FRAMES}) | "
            f"Gaze: {gaze_status} ({gaze_direction}) | Persons: {person_status} | Objects: {object_status}"
        )

    processed += 1
    frame_count += 1

cap.release()

# ================= FINAL DECISION =================
print("\n========== FINAL RESULT ==========")

# Adjust processed count to exclude calibration frames
analyzed_frames = total_frames - CALIBRATION_FRAMES if baseline_calibrated else total_frames

final_identity = (
    "AUTHORIZED PERSON"
    if same_person_frames > different_person_frames
    else "UNAUTHORIZED PERSON"
)

# Use analyzed_frames (excluding calibration) for percentage calculations
final_deviation = (
    "DEVIATED"
    if analyzed_frames > 0 and deviation_frames >= (0.25 * analyzed_frames)
    else "NOT DEVIATED"
)

final_gaze = (
    "SUSPICIOUS EYE MOVEMENT"
    if analyzed_frames > 0 and gaze_deviation_frames >= (0.30 * analyzed_frames)
    else "NORMAL EYE MOVEMENT"
)

final_person_check = (
    "MULTIPLE PERSONS DETECTED"
    if analyzed_frames > 0 and multiple_person_frames >= (0.20 * analyzed_frames)
    else "SINGLE PERSON"
)

final_object_check = (
    "PROHIBITED OBJECTS DETECTED"
    if analyzed_frames > 0 and prohibited_object_frames >= (0.15 * analyzed_frames)
    else "NO PROHIBITED OBJECTS"
)

final_spoof_check = (
    "SPOOF DETECTED"
    if analyzed_frames > 0 and spoof_frames >= (0.10 * analyzed_frames)
    else "NO SPOOFING DETECTED"
)

print("\n========== FINAL RESULT ==========")
print("Identity Result     :", final_identity)
print("Spoof Result        :", final_spoof_check)
print("Deviation Result    :", final_deviation)
print("Eye Gaze Result     :", final_gaze)
print("Person Count Result :", final_person_check)
print("Object Detection    :", final_object_check)
print("==================================")
if baseline_calibrated:
    print(f"Baseline Head Pose  : Yaw={baseline_yaw:.1f}°, Pitch={baseline_pitch:.1f}°, Roll={baseline_roll:.1f}°")
    print(f"Detection Thresholds: ΔYaw=±15°, ΔPitch=±10°, ΔRoll=±10°")
else:
    print("WARNING: Baseline calibration not completed!")
print("==================================")
print(f"Total frames processed: {total_frames}")
if baseline_calibrated:
    print(f"Calibration frames: {CALIBRATION_FRAMES}")
    print(f"Analyzed frames: {analyzed_frames}")
print(f"Spoof frames: {spoof_frames} ({spoof_frames/analyzed_frames*100:.1f}%)" if analyzed_frames > 0 else "Spoof frames: 0 (0.0%)")
print(f"Head pose deviation: {deviation_frames} ({deviation_frames/analyzed_frames*100:.1f}%)" if analyzed_frames > 0 else "Head pose deviation: 0 (0.0%)")
print(f"Gaze deviation: {gaze_deviation_frames} ({gaze_deviation_frames/analyzed_frames*100:.1f}%)" if analyzed_frames > 0 else "Gaze deviation: 0 (0.0%)")
print(f"Multiple persons: {multiple_person_frames} ({multiple_person_frames/analyzed_frames*100:.1f}%)" if analyzed_frames > 0 else "Multiple persons: 0 (0.0%)")
print(f"Prohibited objects: {prohibited_object_frames} ({prohibited_object_frames/analyzed_frames*100:.1f}%)" if analyzed_frames > 0 else "Prohibited objects: 0 (0.0%)")
if prohibited_objects_detected:
    print(f"Objects found: {prohibited_objects_detected}")
print("==================================")

# Overall verdict
if final_identity == "UNAUTHORIZED PERSON":
    verdict = "⚠️ EXAM INVALID - Unauthorized person detected"
elif final_spoof_check == "SPOOF DETECTED":
    verdict = "⚠️ EXAM INVALID - Face spoofing detected"
elif final_person_check == "MULTIPLE PERSONS DETECTED":
    verdict = "⚠️ EXAM INVALID - Multiple persons detected"
elif final_object_check == "PROHIBITED OBJECTS DETECTED":
    verdict = "⚠️ EXAM INVALID - Prohibited objects found"
elif final_deviation == "DEVIATED" or final_gaze == "SUSPICIOUS EYE MOVEMENT":
    verdict = "⚠️ EXAM SUSPICIOUS - Review required"
else:
    verdict = "✓ EXAM VALID - No violations detected"

print(f"\n{'='*50}")
print(f"FINAL VERDICT: {verdict}")
print(f"{'='*50}")
