# Frontend-Backend Connection Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                              │
│                      http://localhost:5173                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  Webcam      │───▶│ SetupScreen  │───▶│  Proctoring  │        │
│  │  Capture     │    │   Capture    │    │   Session    │        │
│  └──────────────┘    │  Reference   │    │   (Active)   │        │
│                      └──────────────┘    └──────┬───────┘        │
│                                                  │                 │
│                      ┌───────────────────────────┼─────────────┐  │
│                      │                           │             │  │
│                 ┌────▼─────┐              ┌─────▼──────┐   ┌──▼────────┐
│                 │Statistics│              │ AlertPanel │   │StatusBar  │
│                 │Dashboard │              │   (Log)    │   │Indicator  │
│                 └──────────┘              └────────────┘   └───────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  ││
                   ┌──────────────┼┼────────────────┐
                   │              ││                │
         ┌─────────▼──┐     ┌─────▼▼──────┐   ┌────▼──────────┐
         │ REST API   │     │  WebSocket  │   │  Vite Proxy   │
         │ (axios)    │     │ (Socket.IO) │   │  (Dev Only)   │
         └─────────┬──┘     └─────┬───────┘   └────┬──────────┘
                   │              │                │
                   │              │                │
┌──────────────────┴──────────────┴────────────────┴──────────────────┐
│                    FLASK BACKEND SERVER                              │
│                    http://localhost:5000                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FLASK + CORS + SocketIO                   │   │
│  └───────┬──────────────────────────────────┬──────────────────┘   │
│          │                                   │                      │
│  ┌───────▼────────────┐            ┌────────▼─────────────┐        │
│  │   REST Endpoints   │            │  WebSocket Events     │        │
│  ├────────────────────┤            ├──────────────────────┤        │
│  │ /api/set-reference │            │ @socketio.on(        │        │
│  │ /api/start-session │            │   'process_frame')   │        │
│  │ /api/stop-session  │            │                      │        │
│  │ /api/get-stats     │            │ emit('frame_result') │        │
│  └────────────────────┘            └──────────┬───────────┘        │
│                                               │                     │
│                                    ┌──────────▼──────────┐          │
│                                    │  AI Processing      │          │
│                                    │  Pipeline           │          │
│                                    └──────────┬──────────┘          │
│                                               │                     │
│  ┌────────────────────────────────────────────▼──────────────────┐ │
│  │                      AI MODELS LAYER                           │ │
│  ├────────────────┬────────────────┬────────────────┬─────────────┤ │
│  │   FaceNet      │   MediaPipe    │  MediaPipe     │   YOLOv8    │ │
│  │  (Identity)    │ Face Detection │  Face Mesh     │  (Objects)  │ │
│  │                │                │ (Pose + Gaze)  │             │ │
│  │  128-D Vector  │   Face BBox    │  468 Landmarks │  COCO 80    │ │
│  │  Euclidean     │   Confidence   │  3D Projection │  Classes    │ │
│  │  Distance      │                │  Iris Tracking │             │ │
│  └────────────────┴────────────────┴────────────────┴─────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SESSION STATE                             │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ • reference_embedding (128-D FaceNet vector)                │   │
│  │ • baseline_yaw, baseline_pitch, baseline_roll               │   │
│  │ • frame counters (total, analyzed, calibration)             │   │
│  │ • violation counters (deviation, gaze, person, object)      │   │
│  │ • prohibited_objects_detected (dict)                        │   │
│  │ • is_active (boolean)                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence

### 1. Reference Setup Flow
```
Browser                          Backend
   │                                │
   ├─ Webcam Capture ─────────┐    │
   │                          │    │
   │  POST /api/set-reference ├───▶│
   │  { image: "base64..." }  │    │
   │                          │    ├─ base64 → OpenCV Image
   │                          │    ├─ Face Detection (MediaPipe)
   │                          │    ├─ Face Crop (bbox)
   │                          │    ├─ FaceNet Embedding (128-D)
   │                          │    └─ Save to session_state
   │                          │    │
   │  ◄────────────────────────────┤ { success: true }
   │                          │    │
   └─ Navigate to Session ────┘    │
```

### 2. Session Start Flow
```
Browser                          Backend
   │                                │
   ├─ POST /api/start-session ─────▶│
   │                                ├─ Reset statistics
   │                                ├─ Clear baseline
   │                                ├─ is_active = true
   │  ◄────────────────────────────┤ { success: true }
   │                                │
   ├─ WebSocket Connect ───────────▶│ Socket.IO connection
   │  (via Socket.IO client)        │ established
   │                                │
   └─ Start Interval Timer ─────┐  │
      (1 frame/second)          │  │
```

### 3. Frame Processing Flow (Real-time Loop)
```
Browser                                    Backend
   │                                          │
   ├─ Webcam.getScreenshot() ──┐             │
   │  (every 1000ms)            │             │
   │                            │             │
   ├─ socket.emit('process_frame', ──────────▶│
   │    { image: "base64..." })                │
   │                                           ├─ base64 → OpenCV
   │                                           │
   │                          ┌────────────────┤ PARALLEL PROCESSING
   │                          │                │
   │                          │  ┌─────────────┤ 1. Face Detection
   │                          │  │             │    └─ BBox + Landmarks
   │                          │  │             │
   │                          │  ├─────────────┤ 2. Identity (FaceNet)
   │                          │  │             │    └─ Compare embedding
   │                          │  │             │       Distance < 1.0 ?
   │                          │  │             │
   │                          │  ├─────────────┤ 3. Head Pose (MediaPipe)
   │                          │  │             │    ├─ 468 landmarks
   │                          │  │             │    ├─ 3D projection
   │                          │  │             │    ├─ Yaw, Pitch, Roll
   │                          │  │             │    └─ Baseline calibration
   │                          │  │             │       (first 3 frames)
   │                          │  │             │
   │                          │  ├─────────────┤ 4. Eye Gaze (Iris)
   │                          │  │             │    ├─ Left/Right iris
   │                          │  │             │    ├─ Gaze ratios
   │                          │  │             │    └─ Direction detection
   │                          │  │             │
   │                          │  └─────────────┤ 5. YOLO Detection
   │                          │                │    ├─ Person count
   │                          │                │    └─ Prohibited objects
   │                          │                │
   │                          └────────────────┤ AGGREGATE RESULTS
   │                                           │
   │  socket.on('frame_result') ◄──────────────┤ emit('frame_result', {
   │                                           │   identity: "Authorized",
   │  ┌─────────────────┐                     │   distance: 0.453,
   │  │ Update UI       │                     │   pose: { status, angles },
   │  ├─────────────────┤                     │   gaze: { direction, ratios },
   │  │ • Status badges │                     │   detection: { persons, objects },
   │  │ • Statistics    │                     │   calibrated: true
   │  │ • Alert panel   │                     │ })
   │  └─────────────────┘                     │
   │                                           │
   └─ [Loop continues every 1s] ──────────────┘
```

### 4. Statistics Update Flow
```
Browser                          Backend
   │                                │
   ├─ GET /api/get-stats ──────────▶│
   │  (every 2000ms)                ├─ Compile session_state
   │                                │  • total_frames
   │                                │  • analyzed_frames
   │                                │  • baseline values
   │                                │  • violation counts
   │  ◄────────────────────────────┤ { total_frames: 150,
   │                                │   analyzed_frames: 147,
   ├─ Update Statistics UI          │   baseline: {...},
   │                                │   stats: {...} }
   │                                │
   └─ [Loop continues] ─────────────┘
```

### 5. Session Stop Flow
```
Browser                          Backend
   │                                │
   ├─ POST /api/stop-session ──────▶│
   │                                ├─ is_active = false
   │                                ├─ Keep statistics
   │                                ├─ Keep baseline
   │  ◄────────────────────────────┤ { success: true, stats: {...} }
   │                                │
   ├─ Clear interval timer ─────┐  │
   ├─ Stop frame capture        │  │
   └─ Display final stats       │  │
```

## Baseline Calibration Algorithm

```
Frame 0-2: Calibration Phase
───────────────────────────────
For each frame in [0, 1, 2]:
  1. Detect face landmarks (MediaPipe 468 points)
  2. Calculate head pose angles (yaw, pitch, roll)
  3. Store in calibration arrays:
     - session_state['calibration_yaws'].append(yaw)
     - session_state['calibration_pitches'].append(pitch)
     - session_state['calibration_rolls'].append(roll)
  
  4. If len(calibration_yaws) == 3:
     - baseline_yaw = mean(calibration_yaws)
     - baseline_pitch = mean(calibration_pitches)
     - baseline_roll = mean(calibration_rolls)
     - baseline_calibrated = True
     - emit('calibration_complete')

Frame 3+: Active Monitoring
───────────────────────────────
For each subsequent frame:
  1. Calculate current pose angles
  2. Compute relative deviations:
     - relative_yaw = current_yaw - baseline_yaw
     - relative_pitch = current_pitch - baseline_pitch
     - relative_roll = current_roll - baseline_roll
  
  3. Check thresholds:
     - IF abs(relative_yaw) > 15° OR
        abs(relative_pitch) > 10° OR
        abs(relative_roll) > 10°
     - THEN pose_status = "Deviating"
     - ELSE pose_status = "Normal"
```

## Alert Type Mapping

```javascript
Frontend Alert Types → Alert Panel Display
─────────────────────────────────────────

'identity' → 🔴 Critical
  ├─ Icon: User
  ├─ Color: Red
  └─ Example: "Unauthorized person detected"

'pose' → 🟡 Warning
  ├─ Icon: AlertTriangle
  ├─ Color: Orange
  └─ Example: "Head position deviation"

'gaze' → 🟡 Warning
  ├─ Icon: Eye
  ├─ Color: Blue
  └─ Example: "Looking away from screen"

'person' → 🟡 Warning
  ├─ Icon: Users
  ├─ Color: Purple
  └─ Example: "Multiple persons detected"

'object' → 🔴 Critical
  ├─ Icon: Package
  ├─ Color: Red
  └─ Example: "Prohibited objects: Cell Phone"
```

## WebSocket Events

```javascript
Client → Server
───────────────
socket.emit('process_frame', {
  image: "data:image/jpeg;base64,/9j/4AAQSkZJ..."
})

Server → Client
───────────────
socket.emit('frame_result', {
  success: true,
  frame_count: 45,
  identity: "Authorized",
  distance: 0.453,
  pose: {
    status: "Normal",
    yaw: 7.3,
    pitch: 108.5,
    roll: -2.1,
    relative_yaw: 2.1,
    relative_pitch: -1.7,
    relative_roll: -0.3
  },
  gaze: {
    direction: "Looking Center",
    suspicious: false,
    left_ratio: 0.48,
    right_ratio: 0.52
  },
  detection: {
    person_count: 1,
    objects: {}
  },
  calibrated: true,
  calibration_progress: 3
})

socket.emit('calibration_complete', {
  baseline: {
    yaw: 5.2,
    pitch: 110.2,
    roll: -1.8
  }
})
```

## Technology Stack

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND                         │
├─────────────────────────────────────────────────────┤
│ Framework:        React 19.2.0                      │
│ Build Tool:       Vite 7.2.4                        │
│ Styling:          Tailwind CSS 3.4.1                │
│ HTTP Client:      Axios 1.6.5                       │
│ WebSocket:        Socket.IO Client 4.7.2            │
│ Webcam:           react-webcam 7.2.0                │
│ Icons:            lucide-react 0.344.0              │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    BACKEND                          │
├─────────────────────────────────────────────────────┤
│ Framework:        Flask 3.0.0                       │
│ CORS:             Flask-CORS 4.0.0                  │
│ WebSocket:        Flask-SocketIO 5.3.5              │
│ Computer Vision:  OpenCV 4.8.1                      │
│ Face Detection:   MediaPipe 0.10.8                  │
│ Face Recognition: FaceNet (keras-facenet 0.3.2)     │
│ Object Detection: YOLOv8 (ultralytics 8.0+)         │
│ Deep Learning:    TensorFlow 2.15.0                 │
│ Arrays:           NumPy 1.24.3                      │
└─────────────────────────────────────────────────────┘
```

## Network Communication

```
Development Environment
──────────────────────
Frontend: http://localhost:5173
Backend:  http://localhost:5000

Vite Proxy (vite.config.js):
  /api/* → http://localhost:5000/api/*
  /socket.io/* → http://localhost:5000/socket.io/*

CORS Configuration (server.py):
  CORS(app, resources={r"/*": {"origins": "*"}})
  socketio = SocketIO(app, cors_allowed_origins="*")
  
  ⚠️ Development only! Change for production.
```

## Performance Characteristics

```
Frame Processing Latency
────────────────────────
Webcam Capture:        ~10ms
Base64 Encoding:       ~20ms
WebSocket Transfer:    ~5ms (localhost)
Image Decoding:        ~15ms
Face Detection:        ~50ms
FaceNet Embedding:     ~100ms
MediaPipe Landmarks:   ~30ms
YOLO Inference:        ~150ms (CPU) / ~20ms (GPU)
────────────────────────
Total (CPU):          ~380ms per frame
Total (GPU):          ~250ms per frame

Processing Rate: 1 frame/second (configurable)
Network: WebSocket (full-duplex, event-driven)
```
