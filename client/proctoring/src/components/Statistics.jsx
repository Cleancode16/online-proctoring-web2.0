import { TrendingUp, AlertTriangle, Eye, Users, Package } from 'lucide-react';

const Statistics = ({ stats, currentFrame, spoofFrames = 0 }) => {
  const metrics = stats?.stats || {
    same_person: 0,
    different_person: 0,
    deviation: 0,
    gaze_deviation: 0,
    multiple_person: 0,
    prohibited_object: 0,
    spoof: 0,
  };
  const analyzedFrames = stats?.analyzed_frames || 0;

  const getPercentage = (value, total) => {
    if (total === 0) return 0;
    return ((value / total) * 100).toFixed(1);
  };

  const getStatusClass = (percentage, threshold) => {
    return percentage >= threshold ? 'text-red-600' : 'text-green-600';
  };

  const spoofCount = Math.max(spoofFrames, metrics.spoof || 0);
  const spoofDetected = spoofCount > 0 || Boolean(stats?.spoof?.invalid);

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-gray-900 mb-4">Session Statistics</h2>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 font-medium">Total Frames</p>
              <p className="text-2xl font-bold text-blue-900">{stats?.total_frames || 0}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 font-medium">Analyzed</p>
              <p className="text-2xl font-bold text-purple-900">{analyzedFrames}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Baseline Info */}
      {stats?.calibrated && stats?.baseline && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-green-900 mb-2">Baseline Head Pose</h3>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div>
              <span className="text-green-600">Yaw:</span>
              <span className="font-mono ml-1 text-green-900">{stats.baseline.yaw?.toFixed(1)}°</span>
            </div>
            <div>
              <span className="text-green-600">Pitch:</span>
              <span className="font-mono ml-1 text-green-900">{stats.baseline.pitch?.toFixed(1)}°</span>
            </div>
            <div>
              <span className="text-green-600">Roll:</span>
              <span className="font-mono ml-1 text-green-900">{stats.baseline.roll?.toFixed(1)}°</span>
            </div>
          </div>
        </div>
      )}

      {/* Current Frame Details */}
      {currentFrame && stats?.calibrated && (
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <h3 className="font-semibold text-gray-900 mb-3">Current Frame Analysis</h3>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Identity Match:</span>
              <span className={`font-medium ${
                currentFrame.identity === 'Authorized' ? 'text-green-600' : 'text-red-600'
              }`}>
                {currentFrame.distance?.toFixed(3) ?? 'N/A'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">Head Deviation:</span>
              <span className="font-mono text-gray-900">
                ΔY: {currentFrame.pose?.relative_yaw?.toFixed(1) ?? 'N/A'}° 
                ΔP: {currentFrame.pose?.relative_pitch?.toFixed(1) ?? 'N/A'}°
                ΔR: {currentFrame.pose?.relative_roll?.toFixed(1) ?? 'N/A'}°
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">Gaze Ratios:</span>
              <span className="font-mono text-gray-900">
                L: {currentFrame.gaze?.left_ratio?.toFixed(2) ?? 'N/A'} 
                R: {currentFrame.gaze?.right_ratio?.toFixed(2) ?? 'N/A'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Violation Statistics */}
      <div className="space-y-3">
        <h3 className="font-semibold text-gray-900">Violation Metrics</h3>

        {/* Spoof Frames */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-500" />
            <span className="text-sm text-gray-700">Spoof Frames</span>
          </div>
          <div className="text-right">
            <span className={`text-sm font-semibold ${
              getStatusClass(parseFloat(getPercentage(spoofCount, analyzedFrames)), 10)
            }`}>
              {spoofCount} ({getPercentage(spoofCount, analyzedFrames)}%)
            </span>
            <span className="text-xs text-gray-500 ml-1">/ 10%</span>
          </div>
        </div>
        
        {/* Head Pose Deviation */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            <span className="text-sm text-gray-700">Head Deviation</span>
          </div>
          <div className="text-right">
            <span className={`text-sm font-semibold ${
              getStatusClass(parseFloat(getPercentage(metrics.deviation, analyzedFrames)), 25)
            }`}>
              {metrics.deviation} ({getPercentage(metrics.deviation, analyzedFrames)}%)
            </span>
            <span className="text-xs text-gray-500 ml-1">/ 25%</span>
          </div>
        </div>

        {/* Gaze Deviation */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-blue-500" />
            <span className="text-sm text-gray-700">Gaze Deviation</span>
          </div>
          <div className="text-right">
            <span className={`text-sm font-semibold ${
              getStatusClass(parseFloat(getPercentage(metrics.gaze_deviation, analyzedFrames)), 30)
            }`}>
              {metrics.gaze_deviation} ({getPercentage(metrics.gaze_deviation, analyzedFrames)}%)
            </span>
            <span className="text-xs text-gray-500 ml-1">/ 30%</span>
          </div>
        </div>

        {/* Multiple Persons */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-purple-500" />
            <span className="text-sm text-gray-700">Multiple Persons</span>
          </div>
          <div className="text-right">
            <span className={`text-sm font-semibold ${
              getStatusClass(parseFloat(getPercentage(metrics.multiple_person, analyzedFrames)), 20)
            }`}>
              {metrics.multiple_person} ({getPercentage(metrics.multiple_person, analyzedFrames)}%)
            </span>
            <span className="text-xs text-gray-500 ml-1">/ 20%</span>
          </div>
        </div>

        {/* Prohibited Objects */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Package className="w-4 h-4 text-red-500" />
            <span className="text-sm text-gray-700">Prohibited Objects</span>
          </div>
          <div className="text-right">
            <span className={`text-sm font-semibold ${
              getStatusClass(parseFloat(getPercentage(metrics.prohibited_object, analyzedFrames)), 15)
            }`}>
              {metrics.prohibited_object} ({getPercentage(metrics.prohibited_object, analyzedFrames)}%)
            </span>
            <span className="text-xs text-gray-500 ml-1">/ 15%</span>
          </div>
        </div>
      </div>

      {/* Final Summary */}
      <div className={`mt-6 border rounded-lg p-4 ${
        spoofDetected ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
      }`}>
        <h3 className={`font-semibold ${spoofDetected ? 'text-red-900' : 'text-green-900'}`}>
          Spoof Detection Result
        </h3>
        <p className={`text-sm mt-1 ${spoofDetected ? 'text-red-700' : 'text-green-700'}`}>
          {spoofDetected ? 'Spoof Detected' : 'No Spoof Detected'}
        </p>
      </div>

      {/* Progress Bars */}
      <div className="mt-4 space-y-2">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-300 ${
              parseFloat(getPercentage(metrics.deviation, analyzedFrames)) >= 25 
                ? 'bg-red-500' 
                : 'bg-green-500'
            }`}
            style={{ width: `${Math.min(100, parseFloat(getPercentage(metrics.deviation, analyzedFrames)))}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default Statistics;
