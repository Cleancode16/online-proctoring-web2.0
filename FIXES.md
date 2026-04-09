# 🔧 Quick Fix Guide - Missing Dependencies

## ✅ **FASTEST FIX - Copy & Run:**

```powershell
pip install protobuf==3.20.3 tensorflow==2.15.0 keras-facenet==0.3.2 mediapipe==0.10.9 flask flask-cors flask-socketio opencv-python numpy pillow ultralytics python-socketio && python server.py
```

---

## 📋 **Or Install from requirements.txt:**

```powershell
pip install -r requirements.txt
python server.py
```

---

## ⚡ **Or Use the Fix Script:**

```powershell
.\fix-protobuf.ps1
python server.py
```

---

## ✅ Expected Success Output:

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
```

---

## 🔄 Alternative: Complete Clean Reinstall

If the above doesn't work, do a complete clean install:

```powershell
# Step 1: Deactivate and delete venv
deactivate
Remove-Item -Recurse -Force venv

# Step 2: Create fresh venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Step 3: Upgrade pip
python -m pip install --upgrade pip

# Step 4: Install from requirements.txt
pip install -r requirements.txt

# Step 5: Start server
python server.py
```

---

## 🔧 Frontend: Package Installation

### If you see npm install errors:

**Option 1: Clean Install**
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"

# Delete node_modules and lock file
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json

# Fresh install
npm install
```

**Option 2: Use npm ci (faster)**
```powershell
npm ci
```

**Option 3: Install with legacy peer deps**
```powershell
npm install --legacy-peer-deps
```

---

## 🚀 Quick Start After Fix

### Terminal 1 - Backend
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"

# Activate venv if not already
.\venv\Scripts\Activate.ps1

# Reinstall mediapipe (if needed)
pip install mediapipe==0.10.9 protobuf==3.20.3

# Start server
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
```

### Terminal 2 - Frontend
```powershell
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"

# Clean install if needed
npm install

# Start dev server
npm run dev
```

**Expected Output:**
```
  VITE v7.2.4  ready in 1234 ms

  ➜  Local:   http://localhost:5173/
```

---

## 📋 Verification Checklist

After applying fixes:

### Backend
- [ ] `python server.py` runs without errors
- [ ] See ">>> MediaPipe models loaded"
- [ ] See "Online Proctoring Server Ready!"
- [ ] Server on http://localhost:5000

### Frontend  
- [ ] `npm run dev` runs without errors
- [ ] See "Local: http://localhost:5173/"
- [ ] Browser opens successfully
- [ ] No console errors in DevTools

---

## 🐛 Still Having Issues?

### MediaPipe Import Error Persists

**Try updating pip first:**
```powershell
python -m pip install --upgrade pip
```

**Then reinstall all dependencies:**
```powershell
pip uninstall mediapipe tensorflow
pip install -r requirements.txt
```

### Frontend npm Errors

**Check Node version:**
```powershell
node --version  # Should be 16+
```

**Update npm:**
```powershell
npm install -g npm@latest
```

**Try different package manager:**
```powershell
# Install yarn
npm install -g yarn

# Use yarn instead
yarn install
yarn dev
```

---

## 📦 Updated Dependencies

### Backend (requirements.txt)
- Added `protobuf==3.20.3` (fixes MediaPipe compatibility)
- Updated `mediapipe==0.10.9` (stable version)

### Frontend (package.json)  
Your current versions:
- react: 18.2.0
- axios: 1.13.4
- socket.io-client: 4.8.3
- react-webcam: 7.2.0

All compatible ✅

---

## ✅ What Was Fixed

1. **server.py**: 
   - Added try/except for MediaPipe imports
   - Supports both new and old MediaPipe API
   - Works with MediaPipe 0.10.8, 0.10.9, 0.10.x

2. **requirements.txt**:
   - Added protobuf version constraint
   - Updated mediapipe to 0.10.9

3. This guide created to help resolve issues quickly

---

## 🎯 Final Test

Once both servers start successfully:

1. Open http://localhost:5173
2. Click "Use Webcam"
3. Capture reference image
4. Start session
5. Wait for calibration (3 frames)
6. See live monitoring!

**Everything working?** Delete this file and enjoy your proctoring system! 🎉
