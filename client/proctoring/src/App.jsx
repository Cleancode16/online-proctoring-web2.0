import { useState } from 'react';
import SetupScreen from './components/SetupScreen';
import ProctoringSession from './components/ProctoringSession';
import './index.css';

function App() {
  const [referenceSet, setReferenceSet] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);

  const handleReferenceSet = () => {
    setReferenceSet(true);
  };

  const handleReset = () => {
    setReferenceSet(false);
    setSessionActive(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {!referenceSet ? (
        <SetupScreen onReferenceSet={handleReferenceSet} />
      ) : (
        <ProctoringSession 
          onReset={handleReset}
          sessionActive={sessionActive}
          setSessionActive={setSessionActive}
        />
      )}
    </div>
  );
}

export default App;
