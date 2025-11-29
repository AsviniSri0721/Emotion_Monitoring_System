import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import { emotionDetector, EmotionResult } from '../services/emotionDetection';
import './LiveSession.css';

const LiveSession: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const webcamRef = useRef<HTMLVideoElement>(null);
  const [session, setSession] = useState<any>(null);
  const [emotion, setEmotion] = useState<EmotionResult | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<Array<EmotionResult & { timestamp: number }>>([]);
  const sessionStartTime = useRef<number>(Date.now());
  const lastEmotionSendTime = useRef<number>(0);

  useEffect(() => {
    fetchSession();
    joinSession();

    return () => {
      emotionDetector.stopDetection();
      leaveSession();
    };
  }, [id]);

  useEffect(() => {
    if (emotion) {
      const timestamp = Math.floor((Date.now() - sessionStartTime.current) / 1000);

      // Send emotion data to server every 5 seconds
      if (timestamp - lastEmotionSendTime.current >= 5) {
        sendEmotionData(emotion, timestamp);
        lastEmotionSendTime.current = timestamp;
      }
    }
  }, [emotion]);

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

  // Emotion data is now sent automatically by the detector
  const sendEmotionData = async (emotionData: EmotionResult, timestamp: number) => {
    // Data is sent automatically by emotionDetector
    // This function kept for compatibility but no longer needed
  };

  const startEmotionDetection = async () => {
    if (!webcamRef.current || !id) return;

    try {
      await emotionDetector.startDetection(
        webcamRef.current,
        (result) => {
          setEmotion(result);
          const timestamp = Math.floor((Date.now() - sessionStartTime.current) / 1000);
          setEmotionHistory((prev) => [...prev.slice(-19), { ...result, timestamp }]);
        },
        'live',
        id
      );
    } catch (error) {
      console.error('Error starting emotion detection:', error);
      alert('Could not access webcam. Please allow camera permissions.');
    }
  };

  const stopEmotionDetection = () => {
    emotionDetector.stopDetection();
    setEmotion(null);
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
            {!emotion ? (
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

          {emotion && (
            <div className="emotion-display">
              <div className="emotion-card">
                <h4>Current Emotion</h4>
                <p className="emotion-value">{emotion.emotion}</p>
                <p className="confidence">Confidence: {(emotion.confidence * 100).toFixed(1)}%</p>
                <p className="engagement">Engagement: {(emotion.engagementScore * 100).toFixed(1)}%</p>
              </div>
            </div>
          )}

          {emotionHistory.length > 0 && (
            <div className="emotion-history">
              <h4>Recent Emotions</h4>
              <div className="emotion-timeline">
                {emotionHistory.slice(-10).map((item, index) => (
                  <div key={index} className="emotion-item">
                    <span className="emotion-label">{item.emotion}</span>
                    <span className="emotion-time">{item.timestamp}s</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveSession;

