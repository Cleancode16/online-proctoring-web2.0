# Quick Setup Guide - Web Application

This guide will help you quickly set up and run the Online Proctoring System web application.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm (comes with Node.js)
- Webcam

## Step 1: Backend Setup

### 1.1 Navigate to Project Root
```bash
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring"
```

### 1.2 Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# For Linux/Mac:
# source venv/bin/activate
```

### 1.3 Install Python Dependencies
```bash
pip install flask flask-cors flask-socketio opencv-python mediapipe keras-facenet tensorflow ultralytics numpy
```

### 1.4 Start Backend Server
```bash
python server.py
```

✅ Backend should now be running on http://localhost:5000

**Keep this terminal open!**

## Step 2: Frontend Setup

### 2.1 Open a New Terminal

Open a second terminal window and navigate to the frontend directory:
```bash
cd "c:\Users\bhara\Downloads\Mini Project\Online-Proctoring\client\proctoring"
```

### 2.2 Install Node Dependencies
```bash
npm install
```

This will install:
- React 19.2.0
- Vite 7.2.4
- Tailwind CSS 3.4.1
- Socket.IO Client 4.7.2
- Axios 1.6.5
- react-webcam 7.2.0
- lucide-react 0.344.0

### 2.3 Start Development Server
```bash
npm run dev
```

✅ Frontend should now be running on http://localhost:5173

## Step 3: Access the Application

1. Open your browser and navigate to **http://localhost:5173**
2. Grant webcam permissions when prompted
3. You should see the Setup Screen

## Using the Application

### Setup Phase
1. **Capture Reference Image**
   - Click "Capture from Webcam" to take a photo
   - OR click "Upload Image" to use an existing photo
   - Preview the image to ensure it's clear
   - Click "Confirm & Proceed" to continue

### Proctoring Session
1. **Wait for Connection**
   - The system will connect to the backend server
   - Status indicator will show green checkmarks

2. **Baseline Calibration**
   - **Sit naturally in your normal position**
   - Look at the screen normally
   - The system will calibrate for 3 frames (auto-completed)
   - Calibration status will turn green when complete

3. **Active Monitoring**
   - Stay in your natural position
   - Look at the screen
   - The system will display:
     - ✅ Identity verification status
     - Head pose deviation (relative to YOUR baseline)
     - Eye gaze direction
     - Person count
     - Detected objects
   
4. **View Statistics**
   - Real-time violation metrics
   - Baseline head pose angles
   - Current frame analysis
   - Percentage of violations

5. **Monitor Alerts**
   - Alerts appear in real-time
   - Color-coded by severity:
     - 🔴 Red = Critical (identity, prohibited objects)
     - 🟡 Yellow = Warning (pose, gaze, multiple persons)

6. **End Session**
   - Click "Stop Session" when finished
   - Review final statistics

## Troubleshooting

### Backend Issues

**ImportError: No module named 'flask'**
- Make sure virtual environment is activated
- Run: `pip install flask flask-cors flask-socketio`

**YOLO model downloading on first run**
- This is normal, YOLOv8n.pt will download (~6MB)
- Requires internet connection only for first run

### Frontend Issues

**npm ERR! ENOENT: no such file or directory**
- Ensure you're in the correct directory: `client/proctoring`
- Delete `node_modules` and `package-lock.json`, then run `npm install` again

**Webcam not working**
- Grant camera permissions in browser settings
- Check if another application is using the webcam
- Try refreshing the page

**Cannot connect to server**
- Ensure backend is running on port 5000
- Check firewall settings
- Verify both servers are running

**WebSocket connection failed**
- Make sure backend server is running
- Check console for error messages
- Try restarting both servers

### Common Issues

**High CPU usage**
- Frame processing is computationally intensive
- Backend processes frames every 1 second (configurable)
- Close other resource-intensive applications

**Baseline calibration taking too long**
- Ensure your face is visible and well-lit
- Stay still for the first 3 frames
- Check that MediaPipe can detect your face

## System Architecture

```
┌─────────────┐                    ┌──────────────┐
│   Browser   │  WebSocket (1s)    │ Flask Server │
│   (React)   │◄──────────────────►│   (Python)   │
│  Port 5173  │   Base64 Frames    │  Port 5000   │
└─────────────┘                    └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  AI Models   │
                                   ├──────────────┤
                                   │  - FaceNet   │
                                   │  - MediaPipe │
                                   │  - YOLOv8    │
                                   └──────────────┘
```

## Key Features

- **Adaptive Baseline**: Calibrates to YOUR natural sitting position (first 3 frames)
- **Relative Angles**: Detects movement FROM your baseline, not absolute angles
- **Real-time Processing**: WebSocket communication for low latency
- **Multi-modal Detection**: Face + Pose + Gaze + YOLO in parallel
- **Visual Feedback**: Live indicators, statistics, and alerts

## Next Steps

- Read [README.md](../../README.md) for detailed feature explanations
- Check [flow.txt](../../flow.txt) for algorithm details and diagrams
- Customize thresholds in `server.py` (detection sensitivity)
- Review `ProctoringSession.jsx` for frontend customization

## Need Help?

1. Check the [Troubleshooting](#troubleshooting) section
2. Review console logs (browser DevTools + terminal)
3. Ensure all dependencies are installed correctly
4. Verify webcam and network connectivity
