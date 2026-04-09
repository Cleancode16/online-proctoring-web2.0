# ✅ Frontend-Backend Connection Verification Checklist

## Pre-Flight Checks

### System Requirements
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Node.js 16+ installed (`node --version`)
- [ ] npm installed (`npm --version`)
- [ ] Webcam connected and functional
- [ ] Ports 5000 and 5173 available

---

## Backend Verification

### 1. Installation
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Check:**
- [ ] Virtual environment created in `venv/` folder
- [ ] All packages installed without errors
- [ ] No import errors when running `python -c "import flask, cv2, mediapipe, ultralytics"`

### 2. Server Startup
```powershell
python server.py
```

**Expected Output:**
```
>>> Server starting...
>>> Imports done
>>> Loading AI models...
>>> FaceNet loaded
>>> MediaPipe models loaded
>>> YOLO model loaded

==================================================
Online Proctoring Server Ready!
==================================================
Server running on http://localhost:5000
WebSocket ready for real-time proctoring
==================================================
```

**Check:**
- [ ] Server starts without errors
- [ ] All AI models load successfully
- [ ] Listening on port 5000
- [ ] No "Address already in use" errors

### 3. API Endpoint Test
Open another PowerShell and test:

```powershell
curl http://localhost:5000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": 1675786543.123
}
```

**Check:**
- [ ] Health endpoint responds with 200 OK
- [ ] JSON response is valid
- [ ] `models_loaded` is `true`

---

## Frontend Verification

### 1. Installation
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"
npm install
```

**Check:**
- [ ] `node_modules/` folder created
- [ ] No ERR! messages in output
- [ ] `package-lock.json` generated

### 2. Development Server Startup
```powershell
npm run dev
```

**Expected Output:**
```
  VITE v7.2.4  ready in 1234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**Check:**
- [ ] Vite starts without errors
- [ ] Server listening on http://localhost:5173
- [ ] No compilation errors
- [ ] Hot reload enabled

### 3. Browser Access
Open browser to http://localhost:5173

**Check:**
- [ ] Page loads successfully
- [ ] No console errors (press F12 → Console tab)
- [ ] Webcam permission dialog appears
- [ ] UI renders correctly (Setup Screen visible)

---

## Connection Integration Tests

### Test 1: WebSocket Connection
Open browser DevTools (F12) → Network tab → WS filter

**Check:**
- [ ] WebSocket connection to `ws://localhost:5000/socket.io/` established
- [ ] Connection status shows "101 Switching Protocols"
- [ ] No connection refused errors

### Test 2: Reference Image Upload

**Steps:**
1. Click "Use Webcam" or "Upload Image"
2. Capture/upload a clear face photo
3. Click "Confirm & Proceed"

**Check in Browser Console:**
```javascript
// Should see:
POST http://localhost:5000/api/set-reference 200 OK
```

**Check in Backend Terminal:**
```
Reference image set successfully
Embedding shape: (128,)
```

**Verification:**
- [ ] POST request succeeds (200 status)
- [ ] Reference embedding saved
- [ ] Navigates to Proctoring Session screen
- [ ] No CORS errors

### Test 3: Session Start

**Steps:**
1. Click "Start Session" button

**Check in Browser Console:**
```javascript
POST http://localhost:5000/api/start-session 200 OK
Connected to server
Proctoring session started
```

**Check in Backend Terminal:**
```
Session started
Baseline calibration beginning...
```

**Verification:**
- [ ] Session POST succeeds
- [ ] WebSocket remains connected
- [ ] Frame capture interval starts
- [ ] Status indicator turns green

### Test 4: Frame Processing (Critical!)

**Steps:**
1. Wait 1 second after session starts
2. Look at webcam and stay still

**Check in Browser Console:**
```javascript
// Should see repeating messages:
socket.emit('process_frame', { image: "data:image/jpeg..." })
socket.on('frame_result', { success: true, identity: "Authorized", ... })
```

**Check in Backend Terminal:**
```
Frame 0: Calibrating (1/3)
Frame 1: Calibrating (2/3)
Frame 2: Calibrating (3/3)
>>> Baseline calibrated: Yaw=5.3°, Pitch=110.2°, Roll=-2.1°
Frame 3: Authorized | Normal | Looking Center | 1 Person | No objects
Frame 4: Authorized | Normal | Looking Center | 1 Person | No objects
```

**Verification:**
- [ ] Frames sent every 1 second
- [ ] Backend processes each frame
- [ ] Results emitted back via WebSocket
- [ ] Calibration completes after 3 frames
- [ ] Status badges update in real-time

### Test 5: Real-Time UI Updates

**Check on Proctoring Session Screen:**
- [ ] Video feed displays webcam
- [ ] Status badges show (Authorized, Normal, etc.)
- [ ] Calibration progress shows "Calibrating 1/3, 2/3, 3/3"
- [ ] After calibration, badges turn green
- [ ] Statistics panel shows frame counts
- [ ] Baseline values display after calibration

### Test 6: Violation Detection

**Test Identity Violation:**
1. Cover webcam temporarily
2. Show different person (if available)

**Expected:**
- [ ] Alert appears: "Unauthorized person detected"
- [ ] Alert panel logs the violation
- [ ] Badge turns red

**Test Head Pose Violation:**
1. Turn head significantly left or right (>15°)

**Expected:**
- [ ] Alert: "Head position deviation"
- [ ] Pose status changes to "Deviating"
- [ ] Badge color changes

**Test Gaze Violation:**
1. Look away from screen (left/right/down)

**Expected:**
- [ ] Gaze direction updates: "Looking Left/Right"
- [ ] Alert logged if sustained

**Test Object Detection:**
1. Show cell phone to camera

**Expected:**
- [ ] Alert: "Prohibited objects: Cell Phone"
- [ ] Badge shows object detected
- [ ] Red critical alert

### Test 7: Statistics Update

**Steps:**
1. Wait 2+ seconds during active session
2. Check Statistics panel

**Verification:**
- [ ] Total frames increments
- [ ] Analyzed frames updates
- [ ] Baseline values displayed
- [ ] Violation percentages calculate
- [ ] Progress bars render

### Test 8: Session Stop

**Steps:**
1. Click "Stop Session"

**Check in Browser Console:**
```javascript
POST http://localhost:5000/api/stop-session 200 OK
Proctoring session stopped
```

**Verification:**
- [ ] Frame capture stops
- [ ] WebSocket stays connected
- [ ] Final statistics remain visible
- [ ] Can restart session successfully

---

## Common Issues and Fixes

### ❌ Backend: "Address already in use"
```powershell
# Find process on port 5000
netstat -ano | findstr :5000
# Kill process (replace PID)
taskkill /PID <PID> /F
```

### ❌ Frontend: "Failed to connect to localhost:5000"
**Fix:** Ensure backend started FIRST, then frontend

### ❌ "CORS policy" error
**Fix:** Check `vite.config.js` has proxy configuration:
```javascript
server: {
  proxy: {
    '/api': 'http://localhost:5000',
    '/socket.io': { target: 'http://localhost:5000', ws: true }
  }
}
```

### ❌ WebSocket connection fails
**Fix:** 
1. Check backend is running
2. Clear browser cache
3. Try different browser
4. Check firewall settings

### ❌ "No face detected"
**Fix:**
- Ensure good lighting
- Face camera directly
- Remove glasses/hat if blocking detection
- Try different camera angle

### ❌ Calibration stuck at 0/3
**Fix:**
- Check backend terminal for errors
- Ensure MediaPipe can detect landmarks
- Stay still and face camera during calibration

### ❌ Baseline values not showing
**Fix:** Must complete 3 calibration frames first

---

## Success Criteria Checklist

### ✅ Complete Integration
- [ ] Backend serves on :5000
- [ ] Frontend serves on :5173
- [ ] WebSocket connects successfully
- [ ] Reference image uploads
- [ ] Session starts/stops
- [ ] Frames process at 1/second
- [ ] Baseline calibrates (3 frames)
- [ ] All 5 detection types work:
  - [ ] Identity (FaceNet)
  - [ ] Head Pose (MediaPipe)
  - [ ] Eye Gaze (Iris tracking)
  - [ ] Person Count (YOLO)
  - [ ] Objects (YOLO)
- [ ] Statistics update every 2s
- [ ] Alerts log violations
- [ ] UI updates in real-time
- [ ] No console errors
- [ ] No backend errors

---

## Performance Benchmarks

**Expected Metrics:**
```
Frame Processing:  ~380ms (CPU) / ~250ms (GPU)
Frame Rate:        1 frame/second
WebSocket Latency: <50ms (localhost)
UI Update Latency: <100ms
Memory Usage:      ~2GB (backend), ~300MB (frontend)
CPU Usage:         High during active session (normal)
```

**Check:**
- [ ] Frame processing completes in <1 second
- [ ] No lag in UI updates
- [ ] Smooth video feed
- [ ] Alerts appear immediately

---

## Final Verification

### Complete Workflow Test
1. [ ] Start backend → see models load
2. [ ] Start frontend → see UI render
3. [ ] WebSocket connects → see green status
4. [ ] Upload reference → see confirmation
5. [ ] Start session → see session active
6. [ ] Calibrate (3 frames) → see baseline values
7. [ ] Monitor normally → see green badges
8. [ ] Trigger violations → see red/yellow alerts
9. [ ] Check statistics → see counts update
10. [ ] Stop session → see final report
11. [ ] Reset → return to setup screen

**All steps complete without errors?**
- [ ] ✅ YES → System fully connected and working!
- [ ] ❌ NO → Check specific test that failed above

---

## Quick Debug Commands

**Check Backend Status:**
```powershell
curl http://localhost:5000/api/health
```

**Check Frontend Build:**
```powershell
cd client\proctoring
npm run build  # Should compile without errors
```

**View Backend Logs:**
```powershell
# Backend terminal shows all processing
```

**View Frontend Logs:**
```
Browser DevTools (F12) → Console tab
```

**Test WebSocket:**
```javascript
// In browser console:
const socket = io('http://localhost:5000');
socket.on('connect', () => console.log('Connected!'));
```

---

## Documentation References

- **Run Instructions:** [RUN.md](RUN.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Setup Guide:** [QUICKSTART.md](QUICKSTART.md)
- **Full Docs:** [README.md](README.md)
- **Flow Details:** [flow.txt](flow.txt)

---

## ✅ **CHECKLIST COMPLETE!**

If all checks pass, your frontend and backend are fully connected and the online proctoring system is ready to use! 🎉
