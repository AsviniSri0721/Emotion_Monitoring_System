import React, { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import EngagementMeter from '../components/EngagementMeter';
import { useLiveSessionEmotionStream } from '../hooks/useLiveSessionEmotionStream';
import { liveSessionsApi } from '../api/liveSessions';
import api from '../services/api';
import './LiveMonitoring.css';

const LiveMonitoring: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const webcamRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<any>(null);
  const [cameraPermissionGranted, setCameraPermissionGranted] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Use live session emotion stream hook
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
    enabled: cameraPermissionGranted && !!id,
  });

  // Request camera permission and start monitoring automatically on page load
  useEffect(() => {
    async function startMonitoring() {
      if (!webcamRef.current || !id) return;

      try {
        // Request camera permission
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 },
          audio: false 
        });
        
        // Set video source
        webcamRef.current.srcObject = stream;
        await webcamRef.current.play();
        
        setCameraPermissionGranted(true);
        setCameraError(null);
        
        // Join session in backend
        try {
          await api.post(`/sessions/live/${id}/join`);
        } catch (error) {
          console.error('Error joining session:', error);
          // Continue even if backend call fails
        }
        
        // Start emotion detection
        await startDetection();
      } catch (error: any) {
        console.error('Error accessing camera:', error);
        setCameraError(
          error.name === 'NotAllowedError' 
            ? 'Camera permission denied. Please allow camera access and refresh the page.'
            : 'Could not access camera. Please check your camera settings.'
        );
        setCameraPermissionGranted(false);
      }
    }

    startMonitoring();

    // Cleanup on unmount
    return () => {
      stopDetection();
      if (webcamRef.current?.srcObject) {
        const stream = webcamRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Fetch session details
  useEffect(() => {
    async function fetchSession() {
      try {
        const response = await liveSessionsApi.getAvailable();
        const sessions = response.sessions;
        const currentSession = sessions.find((s: any) => s.id === id);
        setSession(currentSession);
      } catch (error) {
        console.error('Error fetching session:', error);
      }
    }
    fetchSession();
  }, [id]);

  // Draw bounding box on webcam video
  useEffect(() => {
    if (!overlayCanvasRef.current || !webcamRef.current || !emotionResult?.bbox) {
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

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const videoWidth = video.videoWidth || 640;
    const videoHeight = video.videoHeight || 480;
    const scaleX = canvas.width / videoWidth;
    const scaleY = canvas.height / videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [x1, y1, x2, y2] = emotionResult.bbox;
    const scaledX1 = x1 * scaleX;
    const scaledY1 = y1 * scaleY;
    const scaledX2 = x2 * scaleX;
    const scaledY2 = y2 * scaleY;

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

    const labelText = `${emotionResult.emotion} (${(emotionResult.confidence * 100).toFixed(1)}%)`;
    ctx.font = '14px Arial';
    const textMetrics = ctx.measureText(labelText);
    const labelWidth = textMetrics.width + 8;
    const labelHeight = 20;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(scaledX1, scaledY1 - labelHeight - 2, labelWidth, labelHeight);

    ctx.fillStyle = '#00ff00';
    ctx.fillText(labelText, scaledX1 + 4, scaledY1 - 6);
  }, [emotionResult]);

  if (!session) {
    return <div className="loading">Loading session...</div>;
  }

  return (
    <div className="live-monitoring-container">
      <div className="live-monitoring-header">
        <h1>Live Class Monitoring - {session.title}</h1>
        <p style={{ fontSize: '0.9rem', color: '#666' }}>
          Your attention and engagement are being monitored during this live class.
        </p>
      </div>

      <div className="live-monitoring-content">
        {cameraError && (
          <div className="error-message" style={{ 
            padding: '1rem', 
            background: '#fee', 
            border: '1px solid #fcc', 
            borderRadius: '8px',
            marginBottom: '1rem'
          }}>
            <p style={{ color: '#c00', margin: 0 }}>{cameraError}</p>
          </div>
        )}

        {!cameraPermissionGranted && !cameraError && (
          <div className="loading" style={{ padding: '2rem' }}>
            Requesting camera permission...
          </div>
        )}

        {cameraPermissionGranted && (
          <>
            <div className="webcam-section">
              <div className="webcam-container" style={{ position: 'relative' }}>
                <video
                  ref={webcamRef}
                  autoPlay
                  muted
                  playsInline
                  style={{ width: '100%', maxWidth: '640px', borderRadius: '8px' }}
                />
                <canvas
                  ref={overlayCanvasRef}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    maxWidth: '640px',
                    pointerEvents: 'none',
                    borderRadius: '8px',
                  }}
                />
              </div>
            </div>

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
                        .sort(([, a], [, b]) => b - a)
                        .map(([emotion, value]) => {
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

            {isDetecting && (
              <div style={{ marginTop: '1rem', padding: '0.5rem', background: '#e8f5e9', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: '#2e7d32', fontSize: '0.9rem' }}>
                  ✓ Monitoring active
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default LiveMonitoring;

