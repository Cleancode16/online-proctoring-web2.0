import { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { io } from 'socket.io-client';
import { Play, Pause, RotateCcw, Users, Eye, Activity, Shield, AlertTriangle } from 'lucide-react';
import StatusIndicator from './StatusIndicator';
import Statistics from './Statistics';
import AlertPanel from './AlertPanel';

const SOCKET_URL = 'http://localhost:5000';

const ProctoringSession = ({ onReset, sessionActive, setSessionActive }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [spoofFrames, setSpoofFrames] = useState(0);
  const [spoofOverlay, setSpoofOverlay] = useState(null);
  const [sessionTerminated, setSessionTerminated] = useState(false);
  const [terminationReason, setTerminationReason] = useState('');
  const [warningActive, setWarningActive] = useState(false);
  const [warningSeconds, setWarningSeconds] = useState(5);
  const [warningMessage, setWarningMessage] = useState('');
  const [warningViolationType, setWarningViolationType] = useState('');
  const [stats, setStats] = useState({
    total_frames: 0,
    analyzed_frames: 0,
    calibrated: false,
    baseline: null,
    stats: {
      same_person: 0,
      different_person: 0,
      deviation: 0,
      gaze_deviation: 0,
      multiple_person: 0,
      prohibited_object: 0,
      spoof: 0
    },
    verdict: {
      status: 'PENDING',
      reason: ''
    }
  });
  
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const spoofOverlayTimerRef = useRef(null);
  const sessionActiveRef = useRef(sessionActive);
  const connectedRef = useRef(connected);
  const socketRef = useRef(null);
  const sessionTerminatedRef = useRef(sessionTerminated);

  useEffect(() => {
    sessionActiveRef.current = sessionActive;
  }, [sessionActive]);

  useEffect(() => {
    connectedRef.current = connected;
  }, [connected]);

  useEffect(() => {
    sessionTerminatedRef.current = sessionTerminated;
  }, [sessionTerminated]);

  const stopCamera = () => {
    const mediaStream = webcamRef.current?.video?.srcObject;
    if (mediaStream?.getTracks) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
  };

  const handleCriticalViolation = (data) => {
    if (sessionTerminatedRef.current) {
      return;
    }

    const reasonByType = {
      unauthorized_person: 'Stopped because an unauthorized person was continuously detected',
      multiple_person: 'Stopped due to unwanted movement',
      multiple_people: 'Stopped due to unwanted movement',
      prohibited_object: 'Stopped due to prohibited object detection',
      spoof: 'Stopped due to spoofing attempt',
      spoof_detected: 'Stopped due to spoofing attempt',
      no_person_detected: 'Stopped because no person was detected for 3 seconds',
      head_pose_deviation: 'Stopped because head pose was continuously deviated for 5 seconds',
    };
    const resolvedReason = reasonByType[data?.type] || 'Stopped due to unwanted movement';

    console.log('Violation received:', data);
    console.log(
      'Session terminated due to critical violation.',
      {
        type: data?.type,
        frame: data?.frame,
        continuous_frames: data?.continuous_frames,
      }
    );

    setTerminationReason(resolvedReason);
    setSessionTerminated(true);
    sessionTerminatedRef.current = true;
    setSessionActive(false);
    setWarningActive(false);
    setWarningMessage('');
    setWarningViolationType('');
    setWarningSeconds(WARNING_COUNTDOWN_FALLBACK);

    addAlert(
      'system',
      `Session terminated: ${data?.type || 'critical_violation'}`,
      `Frame: ${data?.frame ?? 'N/A'} | Continuous: ${data?.continuous_frames ?? 'N/A'}`
    );

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    stopCamera();
  };

  const WARNING_COUNTDOWN_FALLBACK = 5;

  const handleWarningCountdown = (data) => {
    if (sessionTerminatedRef.current) {
      return;
    }

    const seconds = Number(data?.seconds ?? WARNING_COUNTDOWN_FALLBACK);
    setWarningActive(true);
    setWarningSeconds(seconds);
    setWarningMessage(data?.message || 'Unwanted movement detected. Please correct behavior.');
    setWarningViolationType(data?.violation_type || 'violation');
  };

  const handleWarningUpdate = (data) => {
    if (sessionTerminatedRef.current) {
      return;
    }

    const remaining = Number(data?.remaining_seconds ?? WARNING_COUNTDOWN_FALLBACK);
    setWarningActive(true);
    setWarningSeconds(Math.max(0, remaining));
    if (data?.violation_type) {
      setWarningViolationType(data.violation_type);
    }
  };

  const handleWarningCancelled = (data) => {
    setWarningActive(false);
    setWarningSeconds(WARNING_COUNTDOWN_FALLBACK);
    setWarningMessage('');
    setWarningViolationType('');
    if (data?.message) {
      addAlert('system', data.message);
    }
  };

  useEffect(() => {
    if (!sessionTerminated) {
      return;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    stopCamera();
  }, [sessionTerminated]);

  // Initialize socket connection
  useEffect(() => {
    // Use polling transport with Werkzeug dev server to avoid websocket frame-header errors.
    const newSocket = io(SOCKET_URL, {
      transports: ['polling'],
      upgrade: false,
    });
    
    newSocket.on('connect', () => {
      console.log('Connected to server');
      connectedRef.current = true;
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      connectedRef.current = false;
      setConnected(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    });

    newSocket.on('frame_result', (data) => {
      if (sessionTerminatedRef.current) {
        return;
      }
      handleFrameResult(data);
    });

    newSocket.on('calibration_complete', (data) => {
      console.log('Baseline calibration completed:', data.baseline);
    });

    newSocket.on('spoof_event', (data) => {
      console.log('Spoof detected:', data);
      const eventTime = data.timestamp || new Date().toISOString();
      const confidence = Number(data.confidence || 0);

      setSpoofFrames(prev => prev + 1);
      addAlert(
        'spoof',
        'Spoof detected',
        `Frame: ${data.frame} | Confidence: ${confidence.toFixed(2)}`,
        {
          frame: data.frame,
          confidence,
          timestamp: eventTime
        }
      );

      setSpoofOverlay({
        frame: data.frame,
        confidence,
        timestamp: eventTime
      });

      if (spoofOverlayTimerRef.current) {
        clearTimeout(spoofOverlayTimerRef.current);
      }
      spoofOverlayTimerRef.current = setTimeout(() => {
        setSpoofOverlay(null);
      }, 1600);
    });

    const criticalViolationHandler = (data) => {
      handleCriticalViolation(data);
    };
    const warningCountdownHandler = (data) => {
      handleWarningCountdown(data);
    };
    const warningUpdateHandler = (data) => {
      handleWarningUpdate(data);
    };
    const warningCancelledHandler = (data) => {
      handleWarningCancelled(data);
    };

    newSocket.on('critical_violation', criticalViolationHandler);
    newSocket.on('warning_countdown', warningCountdownHandler);
    newSocket.on('warning_update', warningUpdateHandler);
    newSocket.on('warning_cancelled', warningCancelledHandler);

    setSocket(newSocket);
    socketRef.current = newSocket;

    return () => {
      if (spoofOverlayTimerRef.current) {
        clearTimeout(spoofOverlayTimerRef.current);
      }
      newSocket.off('critical_violation', criticalViolationHandler);
      newSocket.off('warning_countdown', warningCountdownHandler);
      newSocket.off('warning_update', warningUpdateHandler);
      newSocket.off('warning_cancelled', warningCancelledHandler);
      socketRef.current = null;
      newSocket.close();
    };
  }, []);

  const handleFrameResult = (data) => {
    if (sessionTerminatedRef.current) {
      return;
    }

    if (data?.session_terminated || data?.critical_violation) {
      handleCriticalViolation(data?.critical_violation || data);
      return;
    }

    if (data.error) {
      console.error('Frame processing error:', data.error);

      if (data.error === 'Session not active') {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        setSessionActive(false);
        addAlert('system', 'Session became inactive on backend. Please start session again.');
      }
      return;
    }

    if (data.no_face) {
      return; // Don't add alert for every frame without face
    }

    if (data.spoof?.spoof_detected) {
      return;
    }

    setCurrentFrame(data);

    // Check for violations
    if (data.identity === 'Unauthorized') {
      addAlert('identity', 'Unauthorized person detected - identity mismatch!', `Distance: ${data.distance?.toFixed(3)}`);
    }

    if (data.pose?.status === 'Deviating') {
      const details = `ΔY: ${data.pose.relative_yaw.toFixed(1)}°, ΔP: ${data.pose.relative_pitch.toFixed(1)}°, ΔR: ${data.pose.relative_roll.toFixed(1)}°`;
      addAlert('pose', `Head position deviation detected`, details);
    }

    if (data.gaze?.suspicious) {
      addAlert('gaze', `Looking away from screen - ${data.gaze.direction}`, `L: ${data.gaze.left_ratio.toFixed(2)}, R: ${data.gaze.right_ratio.toFixed(2)}`);
    }

    if (data.detection?.person_count > 1) {
      addAlert('person', `Multiple persons detected in frame`, `Count: ${data.detection.person_count}`);
    }

    if (data.detection?.objects && Object.keys(data.detection.objects).length > 0) {
      const objectList = Object.entries(data.detection.objects)
        .map(([obj, count]) => `${obj} (${count})`)
        .join(', ');
      addAlert('object', `Prohibited objects detected: ${objectList}`);
    }
  };

  const addAlert = (type, message, details = null, extra = {}) => {
    const alert = {
      type,
      message,
      details,
      timestamp: extra.timestamp || new Date().toISOString(),
      frame: extra.frame,
      confidence: extra.confidence
    };
    
    setAlerts(prev => [...prev, alert].slice(-100)); // Keep last 100 alerts
  };

  const captureAndSendFrame = () => {
    if (sessionTerminatedRef.current) {
      return;
    }

    const active = sessionActiveRef.current;
    const isConnected = connectedRef.current;
    const liveSocket = socketRef.current;

    if (!active || !isConnected || !liveSocket?.connected) {
      return;
    }

    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        liveSocket.emit('process_frame', { image: imageSrc });
      }
    }
  };

  const startSession = async () => {
    if (sessionTerminated) {
      return;
    }

    try {
      const response = await fetch(`${SOCKET_URL}/api/start-session`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        setSessionTerminated(false);
        sessionTerminatedRef.current = false;
        setTerminationReason('');
        setWarningActive(false);
        setWarningSeconds(WARNING_COUNTDOWN_FALLBACK);
        setWarningMessage('');
        setWarningViolationType('');
        setSessionActive(true);
        setSpoofFrames(0);
        setSpoofOverlay(null);
        intervalRef.current = setInterval(captureAndSendFrame, 1000); // Process every second
        console.log('Proctoring session started');
      }
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  const stopSession = async () => {
    if (sessionTerminated) {
      return;
    }

    try {
      const response = await fetch(`${SOCKET_URL}/api/stop-session`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        setSessionActive(false);
        setSpoofOverlay(null);
        setWarningActive(false);
        setWarningSeconds(WARNING_COUNTDOWN_FALLBACK);
        setWarningMessage('');
        setWarningViolationType('');
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        console.log('Proctoring session stopped');
      }
    } catch (error) {
      console.error('Failed to stop session:', error);
    }
  };

  const fetchStats = async () => {
    if (sessionTerminatedRef.current) {
      return;
    }

    try {
      const response = await fetch(`${SOCKET_URL}/api/get-stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  useEffect(() => {
    if (sessionActive) {
      const statsInterval = setInterval(fetchStats, 2000);
      return () => clearInterval(statsInterval);
    }
  }, [sessionActive]);

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Online Exam Proctoring</h1>
            <p className="text-gray-600 mt-1">Real-time monitoring and violation detection</p>
          </div>
          
          <div className="flex items-center gap-4">
            <StatusIndicator connected={connected} calibrated={stats.calibrated} />
            
            {!sessionActive ? (
              <button onClick={startSession} className="btn btn-primary" disabled={sessionTerminated}>
                <Play className="w-5 h-5 mr-2" />
                Start Session
              </button>
            ) : (
              <button onClick={stopSession} className="btn btn-danger" disabled={sessionTerminated}>
                <Pause className="w-5 h-5 mr-2" />
                Stop Session
              </button>
            )}
            
            <button onClick={onReset} className="btn btn-secondary" disabled={sessionTerminated}>
              <RotateCcw className="w-5 h-5 mr-2" />
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Video Feed */}
        <div className="lg:col-span-2 space-y-6">
          {/* Webcam Feed */}
          <div className="card">
            <div className={`relative bg-gray-900 rounded-lg overflow-hidden aspect-video ${spoofOverlay ? 'ring-4 ring-red-500' : ''}`}>
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full h-full object-cover"
                mirrored={true}
              />

              {spoofOverlay && (
                <div className="absolute inset-0 border-4 border-red-500 pointer-events-none animate-pulse" />
              )}
              
              {/* Status Overlays */}
              {sessionActive && currentFrame && (
                <div className="absolute top-4 left-4 right-4 flex flex-wrap gap-2">
                  <div className={`badge ${
                    currentFrame.identity === 'Authorized'
                      ? 'badge-success' 
                      : 'badge-danger'
                  }`}>
                    <Shield className="w-4 h-4 inline mr-1" />
                    {currentFrame.identity || 'Unknown'}
                  </div>
                  
                  {currentFrame.calibrated && currentFrame.pose && currentFrame.gaze && currentFrame.detection ? (
                    <>
                      <div className={`badge ${
                        currentFrame.pose?.status === 'Normal' 
                          ? 'badge-success' 
                          : currentFrame.pose?.status === 'Deviating'
                          ? 'badge-danger'
                          : 'badge-info'
                      }`}>
                        <Activity className="w-4 h-4 inline mr-1" />
                        {currentFrame.pose?.status || 'Unavailable'}
                      </div>
                      
                      <div className={`badge ${
                        !currentFrame.gaze?.suspicious 
                          ? 'badge-success' 
                          : 'badge-warning'
                      }`}>
                        <Eye className="w-4 h-4 inline mr-1" />
                        {currentFrame.gaze?.direction || 'Unknown'}
                      </div>
                      
                      <div className={`badge ${
                        currentFrame.detection?.person_count === 1 
                          ? 'badge-success' 
                          : 'badge-danger'
                      }`}>
                        <Users className="w-4 h-4 inline mr-1" />
                        {currentFrame.detection?.person_count ?? 0} Person(s)
                      </div>
                    </>
                  ) : currentFrame.mesh_missing ? (
                    <div className="badge badge-warning animate-pulse">
                      Face mesh unavailable
                    </div>
                  ) : (
                    <div className="badge badge-info animate-pulse">
                      Calibrating: {currentFrame.calibration_progress}/{stats.calibrated ? 3 : currentFrame.calibration_progress}/3
                    </div>
                  )}
                </div>
              )}

              {spoofOverlay && (
                <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-2 rounded-lg shadow-lg animate-pulse z-20">
                  <div className="flex items-center gap-2 font-bold text-sm">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Spoof Detected</span>
                    <span>|</span>
                    <span>Frame {spoofOverlay.frame}</span>
                    <span>|</span>
                    <span>{new Date(spoofOverlay.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              )}
              
              {!sessionActive && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                  <div className="text-center text-white">
                    <Play className="w-16 h-16 mx-auto mb-4 opacity-70" />
                    <p className="text-lg font-medium">Session Not Active</p>
                    <p className="text-sm opacity-75">Click "Start Session" to begin</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Statistics */}
          <Statistics stats={stats} currentFrame={currentFrame} spoofFrames={spoofFrames} />
        </div>

        {/* Right Column - Alerts */}
        <div className="lg:col-span-1">
          <AlertPanel alerts={alerts} />
        </div>
      </div>

      {sessionTerminated && (
        <div className="fixed inset-0 z-[9999] bg-black bg-opacity-90 flex items-center justify-center">
          <div className="text-center text-white px-6">
            <h1 className="text-4xl font-bold text-red-500">SESSION TERMINATED</h1>
            <p className="mt-4 text-lg">{terminationReason || 'Stopped due to unwanted movement'}</p>
          </div>
        </div>
      )}

      {warningActive && !sessionTerminated && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-40 pointer-events-none">
          <div className="w-full max-w-md mx-4 rounded-xl bg-white shadow-2xl overflow-hidden border-2 border-yellow-500">
            <div className="bg-yellow-500 px-6 py-4">
              <h2 className="text-2xl font-bold text-white">⚠️ WARNING</h2>
            </div>
            <div className="px-6 py-5">
              <p className="text-gray-900 font-medium mb-2">
                {warningMessage || 'Unwanted movement detected. Please correct behavior.'}
              </p>
              <p className="text-gray-700 mb-3">
                Session will terminate in:
              </p>
              <div className="text-4xl font-bold text-red-600 leading-none">
                {warningSeconds}
              </div>
              {warningViolationType && (
                <p className="text-xs text-gray-500 mt-3 uppercase tracking-wide">
                  Violation: {warningViolationType.replace('_', ' ')}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProctoringSession;
