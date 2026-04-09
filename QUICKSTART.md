# Online Proctoring System - Quick Start

## Prerequisites Check
- Python 3.8+: `python --version`
- Node.js 16+: `node --version`  
- npm: `npm --version`

## 1️⃣ Backend Setup (First Terminal)

```powershell
# Navigate to project root
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies (first time only)
pip install -r requirements.txt

# Start backend server
python server.py
```

**✅ Backend should be running on http://localhost:5000**

---

## 2️⃣ Frontend Setup (Second Terminal - Open New Window)

```powershell
# Navigate to frontend directory
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**✅ Frontend should be running on http://localhost:5173**

---

## 3️⃣ Access Application

Open browser: **http://localhost:5173**

---

## Complete Flow

### Phase 1: Reference Setup
1. Open http://localhost:5173
2. Choose "Use Webcam" or "Upload Image"
3. Capture/upload a clear frontal face photo
4. Click "Confirm & Proceed"
5. Wait for reference to be set on backend

### Phase 2: Baseline Calibration
1. Click "Start Session"
2. **Sit naturally** in your comfortable position
3. Look at the screen normally
4. System calibrates for 3 frames (auto-completed in ~3 seconds)
5. Status indicator turns green when calibrated

### Phase 3: Active Monitoring
- System processes 1 frame per second via WebSocket
- Real-time overlays show:
  - ✅ Identity: Authorized/Unauthorized
  - 📐 Head Pose: Normal/Deviating (relative to YOUR baseline)
  - 👁️ Gaze: Looking Center/Left/Right/Away
  - 👥 Person Count
  - 📦 Detected Objects
- Alerts panel logs violations
- Statistics update every 2 seconds

### Phase 4: End Session
1. Click "Stop Session"
2. Review final statistics
3. Click "Reset" to start new session with different reference

---

## What Happens Behind the Scenes

### Backend (server.py)
```
📡 WebSocket Server (Socket.IO) on :5000
├─ REST API Endpoints
│  ├─ POST /api/set-reference (base64 image → save embedding)
│  ├─ POST /api/start-session (reset stats, enable processing)
│  ├─ POST /api/stop-session (disable processing)
│  └─ GET /api/get-stats (return session statistics)
│
└─ WebSocket Events
   ├─ 'process_frame' (receive) → AI processing pipeline
   ├─ 'frame_result' (emit) → detection results
   └─ 'calibration_complete' (emit) → baseline established
```

### AI Processing Pipeline (per frame)
```
1. Base64 → OpenCV image
2. MediaPipe Face Detection → bbox
3. FaceNet Embedding → 128-D vector
4. Compare with reference → identity
5. MediaPipe 468 landmarks → head pose + gaze
6. YOLOv8 → person count + objects
7. JSON result → WebSocket emit
```

### Frontend (React)
```
🌐 Vite Dev Server on :5173 (proxies to :5000)
├─ App.jsx (router: Setup ↔ Session)
├─ SetupScreen.jsx (POST /api/set-reference)
├─ ProctoringSession.jsx (WebSocket client)
│  ├─ Connects to Socket.IO
│  ├─ Captures webcam frame every 1s
│  ├─ Emits 'process_frame' event
│  ├─ Receives 'frame_result' event
│  └─ Updates UI in real-time
├─ Statistics.jsx (GET /api/get-stats every 2s)
└─ AlertPanel.jsx (displays violations)
```

---

## Troubleshooting

### Backend Issues

**PowerShell execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Module not found:**
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Port 5000 already in use:**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000
# Kill it (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Frontend Issues

**npm ERR! code ENOENT:**
```powershell
# Ensure you're in correct directory
cd client\proctoring
# Clean install
rm -r node_modules, package-lock.json
npm install
```

**Webcam not working:**
- Grant camera permissions in browser
- Check if another app is using webcam
- Try different browser (Chrome recommended)

**Connection refused:**
- Ensure backend is running (check Terminal 1)
- Backend must start BEFORE frontend
- Check http://localhost:5000 is accessible

### Common Errors

**CORS errors:**
- Backend CORS is configured for "*"
- Vite proxy should handle this
- If issue persists, check vite.config.js

**WebSocket disconnects:**
- Check network tab in DevTools
- Ensure Socket.IO versions match (client: 4.7.2)
- Backend uses `allow_unsafe_werkzeug=True` for development

**Calibration stuck:**
- Ensure face is visible and well-lit
- Stay still for first 3 frames
- Check console for errors

---

## Performance Notes

- **Frame Rate**: 1 frame/second (configurable in ProctoringSession.jsx line 127)
- **YOLOv8**: First run downloads model (~6MB)
- **CPU Usage**: High during active session (AI models)
- **Memory**: ~2GB RAM for backend (TensorFlow + MediaPipe)

---

## Key Features Summary

✅ **Adaptive Baseline**: Calibrates to YOUR natural position  
✅ **Relative Detection**: Measures deviation FROM baseline, not absolute angles  
✅ **Real-time Processing**: WebSocket for low-latency communication  
✅ **Multi-modal AI**: FaceNet + MediaPipe + YOLOv8 in parallel  
✅ **Visual Feedback**: Live overlays, statistics, color-coded alerts  
✅ **Session Management**: Start/stop with detailed reporting  

---

## Next Steps

1. Read [README.md](README.md) for feature details
2. Check [flow.txt](flow.txt) for algorithm explanations
3. Customize thresholds in server.py (detection sensitivity)
4. Modify UI components in src/components/

---

## Quick Commands Reference

**Backend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"
.\venv\Scripts\Activate.ps1
python server.py
```

**Frontend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"
npm run dev
```

**Open App:**
```
http://localhost:5173
```
