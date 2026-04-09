import { AlertTriangle, Eye, Users, Package, User } from 'lucide-react';

const AlertPanel = ({ alerts }) => {
  const getAlertIcon = (type) => {
    switch (type) {
      case 'identity': return <User className="w-5 h-5" />;
      case 'pose': return <AlertTriangle className="w-5 h-5" />;
      case 'gaze': return <Eye className="w-5 h-5" />;
      case 'person': return <Users className="w-5 h-5" />;
      case 'object': return <Package className="w-5 h-5" />;
      case 'spoof': return <AlertTriangle className="w-5 h-5" />;
      case 'system': return <AlertTriangle className="w-5 h-5" />;
      default: return <AlertTriangle className="w-5 h-5" />;
    }
  };

  const getAlertColor = (type) => {
    switch (type) {
      case 'identity':
      case 'object':
      case 'spoof':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'pose':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'gaze':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'person':
        return 'bg-purple-100 text-purple-800 border-purple-300';
      case 'system':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getSeverityBadge = (type) => {
    const critical = ['identity', 'object', 'spoof'];
    return (
      <span className={`badge ${critical.includes(type) ? 'bg-red-600' : 'bg-yellow-600'}`}>
        {critical.includes(type) ? 'Critical' : 'Warning'}
      </span>
    );
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold mb-4">Alerts Log</h2>

      {alerts.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          <AlertTriangle className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No alerts yet. Monitoring in progress...</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
          {[...alerts].reverse().map((alert, index) => (
            <div
              key={index}
              className={`border rounded-lg p-3 ${getAlertColor(alert.type)}`}
            >
              <div className="flex gap-3">
                {getAlertIcon(alert.type)}

                <div className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="font-semibold text-sm">
                      {{
                        identity: 'Identity Mismatch',
                        pose: 'Head Pose Deviation',
                        gaze: 'Looking Away',
                        person: 'Multiple Persons Detected',
                        object: 'Prohibited Object Detected',
                        spoof: 'Spoof Detected',
                        system: 'System Notice',
                      }[alert.type]}
                    </span>
                    {getSeverityBadge(alert.type)}
                  </div>

                  <p className="text-sm">{alert.message}</p>

                  <div className="flex justify-between mt-2 text-xs opacity-75">
                    <span>{new Date(alert.timestamp).toLocaleTimeString()}</span>
                    {alert.details && <span className="font-mono">{alert.details}</span>}
                  </div>

                  {(alert.frame !== undefined || alert.confidence !== undefined) && (
                    <div className="mt-1 text-xs opacity-80 font-mono">
                      {alert.frame !== undefined && <span>Frame: {alert.frame} </span>}
                      {alert.confidence !== undefined && <span>Confidence: {Number(alert.confidence).toFixed(2)}</span>}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {alerts.length > 0 && (
        <div className="mt-4 pt-4 border-t text-sm flex justify-between">
          <span>Total Alerts:</span>
          <span className="font-semibold">{alerts.length}</span>
        </div>
      )}
    </div>
  );
};

export default AlertPanel;
