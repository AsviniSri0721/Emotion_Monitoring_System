import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import EngagementMeter from '../components/EngagementMeter';
import { useAuth } from '../contexts/AuthContext';
import { useEmotionStream } from '../hooks/useEmotionStream';
import api from '../services/api';
import './LiveSession.css';

const LiveSession: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const webcamRef = useRef<HTMLVideoElement>(null);
  const [session, setSession] = useState<any>(null);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(false);

  // Use the new emotion stream hook
  const {
    emotionResult,
    isDetecting,
    error: emotionError,
    consecutiveLowScores,
    startDetection,
    stopDetection,
  } = useEmotionStream({
    videoElement: webcamRef.current,
    sessionType: 'live',
    sessionId: id || '',
    interval: 2000,
    enabled: emotionDetectionEnabled && !!id,
  });

  useEffect(() => {
    fetchSession();
    joinSession();

    return () => {
      stopDetection();
      leaveSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, stopDetection]);

  const fetchSession = async () => {
    try {
      const response = await api.get(`/sessions/live`);
      const sessions = response.data.sessions;
      const currentSession = sessions.find((s: any) => s.id === id);
      setSession(currentSession);
    } catch (error) {
      console.error('Error fetching session:', error);
    }
  };

  const joinSession = async () => {
    try {
      await api.post(`/sessions/live/${id}/join`);
    } catch (error) {
      console.error('Error joining session:', error);
    }
  };

  const leaveSession = async () => {
    try {
      await api.post(`/sessions/live/${id}/leave`);
      // Generate report
      await api.post(`/reports/generate/live/${id}`, { studentId: user?.id });
    } catch (error) {
      console.error('Error leaving session:', error);
    }
  };

  const startEmotionDetection = async () => {
    if (!webcamRef.current || !id) return;
    setEmotionDetectionEnabled(true);
    await startDetection();
  };

  const stopEmotionDetection = () => {
    setEmotionDetectionEnabled(false);
    stopDetection();
  };

  if (!session) {
    return <div className="loading">Loading session...</div>;
  }

  return (
    <div className="live-session-container">
      <div className="live-session-header">
        <button className="btn btn-secondary" onClick={() => navigate('/student/dashboard')}>
          Leave Session
        </button>
        <h1>{session.title}</h1>
      </div>

      <div className="live-session-content">
        <div className="meet-section">
          <div className="meet-container">
            {session.meet_url ? (
              <iframe
                src={session.meet_url}
                allow="camera; microphone; fullscreen; display-capture"
                style={{
                  width: '100%',
                  height: '600px',
                  border: 'none',
                  borderRadius: '8px',
                }}
                title="Google Meet"
              />
            ) : (
              <div className="meet-placeholder">
                <p>No meeting link provided. Please contact your teacher.</p>
              </div>
            )}
          </div>
        </div>

        <div className="emotion-section">
          <div className="emotion-controls">
            <h3>Emotion Monitoring</h3>
            {!isDetecting ? (
              <button className="btn btn-primary" onClick={startEmotionDetection}>
                Start Monitoring
              </button>
            ) : (
              <button className="btn btn-secondary" onClick={stopEmotionDetection}>
                Stop Monitoring
              </button>
            )}
          </div>

          {webcamRef && (
            <div className="webcam-container">
              <video
                ref={webcamRef}
                autoPlay
                muted
                playsInline
                style={{ width: '100%', maxWidth: '320px', borderRadius: '8px' }}
              />
            </div>
          )}

          {emotionResult && (
            <div className="emotion-display">
              <EngagementMeter
                score={emotionResult.concentrationScore}
                emotion={emotionResult.emotion}
                size={150}
              />
              <div className="emotion-details">
                <p className="confidence">Confidence: {(emotionResult.confidence * 100).toFixed(1)}%</p>
                {consecutiveLowScores > 0 && (
                  <p className="warning">Low concentration: {consecutiveLowScores} consecutive frames</p>
                )}
              </div>
            </div>
          )}

          {emotionError && (
            <div className="error-message">
              <p>{emotionError}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveSession;

