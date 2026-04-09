import { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Camera, CheckCircle, AlertCircle, Upload } from 'lucide-react';

const API_URL = 'http://localhost:5000';

const SetupScreen = ({ onReferenceSet }) => {
  const [captureMode, setCaptureMode] = useState('webcam'); // 'webcam' or 'upload'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleCapture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setCapturedImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!capturedImage) {
      setError('Please capture or upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/api/set-reference`, {
        image: capturedImage
      });

      if (response.data.success) {
        setSuccess(true);
        setTimeout(() => {
          onReferenceSet();
        }, 1500);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to set reference image');
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setError(null);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-4xl w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Online Proctoring System
          </h1>
          <p className="text-gray-600">
            Set up your reference image to begin the proctoring session
          </p>
        </div>

        {/* Main Card */}
        <div className="card">
          {/* Mode Selection */}
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => {
                setCaptureMode('webcam');
                setCapturedImage(null);
              }}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                captureMode === 'webcam'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Camera className="inline-block w-5 h-5 mr-2" />
              Use Webcam
            </button>
            <button
              onClick={() => {
                setCaptureMode('upload');
                setCapturedImage(null);
              }}
              className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                captureMode === 'upload'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Upload className="inline-block w-5 h-5 mr-2" />
              Upload Image
            </button>
          </div>

          {/* Camera/Image Display */}
          <div className="relative bg-gray-900 rounded-lg overflow-hidden mb-6 aspect-video">
            {!capturedImage ? (
              captureMode === 'webcam' ? (
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="w-full h-full object-cover"
                  mirrored={true}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <button
                    onClick={() => fileInputRef.current.click()}
                    className="btn btn-primary"
                  >
                    <Upload className="inline-block w-5 h-5 mr-2" />
                    Choose Image
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>
              )
            ) : (
              <img
                src={capturedImage}
                alt="Captured"
                className="w-full h-full object-cover"
              />
            )}
          </div>

          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-blue-900 mb-2">Instructions:</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Ensure your face is clearly visible and well-lit</li>
              <li>• Look directly at the camera with a neutral expression</li>
              <li>• Remove any obstructions (glasses, masks, etc.) if possible</li>
              <li>• This image will be used to verify your identity during the exam</li>
            </ul>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6 flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <p className="text-green-800">Reference image set successfully! Redirecting...</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            {!capturedImage ? (
              <button
                onClick={handleCapture}
                disabled={captureMode !== 'webcam'}
                className="btn btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Camera className="inline-block w-5 h-5 mr-2" />
                Capture Photo
              </button>
            ) : (
              <>
                <button
                  onClick={handleRetake}
                  className="btn btn-secondary flex-1"
                  disabled={loading}
                >
                  Retake
                </button>
                <button
                  onClick={handleSubmit}
                  className="btn btn-primary flex-1"
                  disabled={loading || success}
                >
                  {loading ? (
                    <>
                      <div className="inline-block w-5 h-5 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="inline-block w-5 h-5 mr-2" />
                      Confirm & Continue
                    </>
                  )}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SetupScreen;
