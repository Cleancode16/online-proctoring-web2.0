import { Wifi, WifiOff, CheckCircle, Clock } from 'lucide-react';

const StatusIndicator = ({ connected, calibrated }) => {
  return (
    <div className="flex items-center gap-4">
      {/* Connection Status */}
      <div className="flex items-center gap-2">
        {connected ? (
          <>
            <Wifi className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-green-700">Connected</span>
          </>
        ) : (
          <>
            <WifiOff className="w-5 h-5 text-red-600" />
            <span className="text-sm font-medium text-red-700">Disconnected</span>
          </>
        )}
      </div>

      {/* Calibration Status */}
      <div className="h-6 w-px bg-gray-300"></div>
      
      <div className="flex items-center gap-2">
        {calibrated ? (
          <>
            <CheckCircle className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-green-700">Calibrated</span>
          </>
        ) : (
          <>
            <Clock className="w-5 h-5 text-yellow-600 animate-pulse" />
            <span className="text-sm font-medium text-yellow-700">Calibrating...</span>
          </>
        )}
      </div>
    </div>
  );
};

export default StatusIndicator;
