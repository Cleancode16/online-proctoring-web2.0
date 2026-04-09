# 🚀 How to Run the Online Proctoring System

## Quick Start (2 Terminals)

### ⚡ Option 1: Using PowerShell Scripts (Easiest)

**Terminal 1 - Backend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"
.\start-backend.ps1
```

**Terminal 2 - Frontend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"
.\start-frontend.ps1
```

> **Note:** If you get execution policy errors, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### 🔧 Option 2: Manual Commands

**Terminal 1 - Backend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"

# First time only
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Every time
.\venv\Scripts\Activate.ps1
python server.py
```

**Terminal 2 - Frontend:**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"

# First time only
npm install

# Every time
npm run dev
```

---

## 🌐 Access the Application

1. **Wait for both servers to start:**
   - ✅ Backend: http://localhost:5000
   - ✅ Frontend: http://localhost:5173

2. **Open in browser:**
   - Navigate to http://localhost:5173
   - Grant webcam permissions

3. **Set up reference:**
   - Capture or upload a reference image
   - Wait for confirmation

4. **Start proctoring:**
   - Click "Start Session"
   - System calibrates for 3 frames
   - Monitoring begins automatically

---

## 📂 File Structure

```
Online-Proctoring/
├── server.py              # Flask backend (Port 5000)
├── requirements.txt       # Python dependencies
├── start-backend.ps1      # Backend startup script
├── start-frontend.ps1     # Frontend startup script
├── QUICKSTART.md          # Detailed setup guide
├── README.md              # Full documentation
├── flow.txt               # Algorithm details
│
└── client/proctoring/
    ├── package.json       # Node dependencies
    ├── vite.config.js     # Vite config with proxy
    └── src/
        ├── App.jsx
        └── components/    # React components
```

---

## ✅ System Check

Before running, verify:
- [ ] Python 3.8+ installed: `python --version`
- [ ] Node.js 16+ installed: `node --version`
- [ ] Webcam connected and working
- [ ] Ports 5000 and 5173 are available

---

## 🔄 Complete Workflow

### 1. Reference Setup (One-time per session)
- Capture/upload clear frontal face photo
- Backend saves FaceNet embedding

### 2. Baseline Calibration (Auto, first 3 frames)
- Sit naturally in comfortable position
- System measures your natural head pose
- Creates baseline for relative deviation detection

### 3. Active Monitoring (Real-time)
- **Frame Processing:** 1 frame/second via WebSocket
- **Identity:** FaceNet compares with reference
- **Head Pose:** MediaPipe tracks relative deviation from baseline
- **Gaze:** Iris tracking detects looking away
- **YOLO:** Detects multiple persons and prohibited objects

### 4. Violation Detection
- 🔴 **Critical:** Unauthorized person, prohibited objects
- 🟡 **Warning:** Head deviation, gaze deviation, multiple persons

### 5. Results
- Real-time statistics dashboard
- Alert log with timestamps
- Violation percentages and thresholds

---

## 🎯 What to Expect

### During Calibration (3 seconds)
```
Frame 1: Calibrating (1/3)
Frame 2: Calibrating (2/3)  
Frame 3: Calibrating (3/3)
✅ Baseline: Yaw=5.2°, Pitch=10.5°, Roll=-1.8°
```

### During Monitoring
```
✅ Identity: Authorized (0.453)
✅ Head Pose: Normal (ΔY=+2.1°, ΔP=-0.8°)
✅ Gaze: Looking Center
✅ Persons: 1
✅ Objects: None
```

### When Violations Occur
```
🔴 Unauthorized person detected!
🟡 Head deviation: ΔY=+18.5°
🟡 Looking Right
🔴 Multiple persons: 2
🔴 Prohibited: Cell Phone (1)
```

---

## 🛠️ Troubleshooting

**Backend won't start:**
```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000
# Kill process if needed
taskkill /PID <PID> /F
```

**Frontend won't start:**
```powershell
# Clean install
cd client\proctoring
Remove-Item node_modules, package-lock.json -Recurse -Force
npm install
```

**Webcam not working:**
- Check browser permissions (chrome://settings/content/camera)
- Close other apps using webcam (Zoom, Teams, etc.)
- Try different browser

**WebSocket errors:**
- Backend must run before frontend
- Check console logs in DevTools
- Verify CORS is not blocking (should use proxy)

---

## 📊 Performance Tips

- **CPU Usage:** High during active monitoring (AI models)
- **Frame Rate:** Adjust in ProctoringSession.jsx (line 127, default 1000ms)
- **Alert Limit:** Keeps last 100 alerts to prevent memory issues
- **Statistics Update:** Every 2 seconds to reduce load

---

## 🔐 Security Notes

- Reference images stored in backend memory only
- No persistent storage of video frames
- Session resets on stop/reset
- All processing happens locally (no cloud)

---

## 📚 Additional Resources

- **Full Documentation:** [README.md](README.md)
- **Algorithm Details:** [flow.txt](flow.txt)
- **Setup Guide:** [QUICKSTART.md](QUICKSTART.md)

---

## 🐛 Need Help?

1. Check console logs (Browser DevTools + Terminal)
2. Verify all dependencies installed correctly
3. Ensure webcam permissions granted
4. Try restarting both servers
5. Check firewall/antivirus not blocking ports

---

## 💡 Quick Tips

- **Good Lighting:** Ensures better face detection
- **Stable Position:** Calibrate in your natural sitting position
- **Look at Screen:** During calibration, look straight ahead
- **Minimize Movement:** After calibration, stay in similar position
- **Check Alerts:** Review violations in real-time panel

---

**Ready to Start?**
1. Open 2 PowerShell terminals
2. Run `.\start-backend.ps1` in Terminal 1
3. Run `.\start-frontend.ps1` in Terminal 2
4. Open http://localhost:5173 in browser
5. Begin proctoring! 🎓
