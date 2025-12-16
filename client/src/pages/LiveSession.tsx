import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { liveSessionsApi } from '../api/liveSessions';
import EngagementMeter from '../components/EngagementMeter';
import { useAuth } from '../contexts/AuthContext';
import { useLiveSessionEmotionStream } from '../hooks/useLiveSessionEmotionStream';
import api from '../services/api';
import './LiveSession.css';

const LiveSession: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const webcamRef = useRef<HTMLVideoElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<any>(null);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(false);

  // Use live session emotion stream hook (NO interventions)
  const {
    emotionResult,
    isDetecting,
    error: emotionError,
    startDetection,
    stopDetection,
  } = useLiveSessionEmotionStream({
    videoElement: webcamRef.current,
    sessionId: id || '',
    interval: 5000, // 5 seconds for live sessions
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

  // Draw bounding box on webcam video
  useEffect(() => {
    if (!overlayCanvasRef.current || !webcamRef.current || !emotionResult?.bbox) {
      // Clear canvas if no bbox
      if (overlayCanvasRef.current) {
        const ctx = overlayCanvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
        }
      }
      return;
    }

    const canvas = overlayCanvasRef.current;
    const video = webcamRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video display size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Get video actual dimensions for scaling
    const videoWidth = video.videoWidth || 640;
    const videoHeight = video.videoHeight || 480;
    const scaleX = canvas.width / videoWidth;
    const scaleY = canvas.height / videoHeight;

    // Clear previous drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scale bounding box coordinates
    const [x1, y1, x2, y2] = emotionResult.bbox;
    const scaledX1 = x1 * scaleX;
    const scaledY1 = y1 * scaleY;
    const scaledX2 = x2 * scaleX;
    const scaledY2 = y2 * scaleY;

    // Draw bounding box
    ctx.strokeStyle = '#00ff00'; // Green color
    ctx.lineWidth = 3;
    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

    // Draw label background
    const labelText = `${emotionResult.emotion} (${(emotionResult.confidence * 100).toFixed(1)}%)`;
    ctx.font = '14px Arial';
    const textMetrics = ctx.measureText(labelText);
    const labelWidth = textMetrics.width + 8;
    const labelHeight = 20;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(scaledX1, scaledY1 - labelHeight - 2, labelWidth, labelHeight);

    // Draw label text
    ctx.fillStyle = '#00ff00';
    ctx.fillText(labelText, scaledX1 + 4, scaledY1 - 6);
  }, [emotionResult]);

  const fetchSession = async () => {
    try {
      const response = await liveSessionsApi.getAvailable();
      const sessions = response.sessions;
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
      // Note: Reports for live sessions are generated on-demand via /live-sessions/:id/report
      // No need to pre-generate engagement_reports for live sessions
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
            <div className="webcam-container" style={{ position: 'relative' }}>
              <video
                ref={webcamRef}
                autoPlay
                muted
                playsInline
                style={{ width: '100%', maxWidth: '320px', borderRadius: '8px' }}
              />
              <canvas
                ref={overlayCanvasRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  maxWidth: '320px',
                  pointerEvents: 'none',
                  borderRadius: '8px',
                }}
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
                {emotionResult.emotion && (
                  <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                    Dominant Emotion: {emotionResult.emotion}
                  </p>
                )}
                <p className="timestamp">Time: {emotionResult.timestamp}s</p>
              </div>
              
              {emotionResult.allEmotions && (
                <div className="all-emotions-display" style={{ marginTop: '1rem', padding: '1rem', background: '#f9f9f9', borderRadius: '8px' }}>
                  <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.9rem', color: '#666' }}>Emotion Levels:</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {Object.entries(emotionResult.allEmotions)
                      .sort(([, a], [, b]) => b - a) // Sort by probability descending
                      .map(([emotion, value]) => {
                        // Ensure value is a number (handle undefined/null)
                        const emotionValue = typeof value === 'number' ? value : 0;
                        return (
                        <div 
                          key={emotion} 
                          className="emotion-row"
                          style={{ 
                            display: 'flex', 
                            alignItems: 'center',
                            gap: '0.5rem',
                            padding: '0.5rem',
                            background: 'white',
                            borderRadius: '4px',
                            border: '1px solid #e0e0e0'
                          }}
                        >
                          <span 
                            className="emotion-label"
                            style={{ 
                              textTransform: 'capitalize', 
                              fontWeight: '500',
                              minWidth: '100px',
                              fontSize: '0.85rem'
                            }}
                          >
                            {emotion}:
                          </span>
                          <progress 
                            value={emotionValue * 100} 
                            max="100" 
                            style={{ 
                              flex: 1,
                              height: '20px',
                              borderRadius: '4px'
                            }}
                          />
                          <span style={{ color: '#666', fontWeight: 'bold', minWidth: '45px', textAlign: 'right', fontSize: '0.85rem' }}>
                            {Math.round(emotionValue * 100)}%
                          </span>
                        </div>
                      );
                      })}
                  </div>
                </div>
              )}
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

