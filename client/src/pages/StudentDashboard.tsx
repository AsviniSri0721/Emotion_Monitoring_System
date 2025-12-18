import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LiveSession, liveSessionsApi } from '../api/liveSessions';
import { useAuth } from '../contexts/AuthContext';
import { useLiveSessionEmotionStream } from '../hooks/useLiveSessionEmotionStream';
import api from '../services/api';
import './Dashboard.css';

interface Video {
  id: string;
  title: string;
  description: string;
  teacher_name: string;
  created_at: string;
}

const StudentDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [videos, setVideos] = useState<Video[]>([]);
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [activeTab, setActiveTab] = useState<'videos' | 'sessions'>('videos');
  
  // Monitoring state for active session
  const [activeMonitoringSession, setActiveMonitoringSession] = useState<LiveSession | null>(null);
  const [monitoringEnabled, setMonitoringEnabled] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // Use live session emotion stream hook for active monitoring
  // Note: We pass webcamRef.current, but the hook will check the current value when needed
  const {
    emotionResult,
    isDetecting,
    error: emotionError,
    startDetection,
    stopDetection,
  } = useLiveSessionEmotionStream({
    videoElement: webcamRef.current, // May be null initially, hook handles this
    sessionId: activeMonitoringSession?.id || '',
    interval: 5000,
    enabled: monitoringEnabled && !!activeMonitoringSession?.id && !!webcamRef.current,
  });

  useEffect(() => {
    fetchVideos();
    fetchSessions();
  }, []);

  // Poll session status to detect when teacher ends the session
  useEffect(() => {
    if (!activeMonitoringSession) return;

    const checkSessionStatus = async () => {
      try {
        const response = await liveSessionsApi.getAvailable();
        const currentSession = response.sessions.find(s => s.id === activeMonitoringSession.id);
        
        // If session is ended or not found, stop monitoring
        if (!currentSession || currentSession.status === 'ended') {
          console.log('[StudentDashboard] Session ended by teacher, stopping monitoring...');
          stopMonitoringInternal();
        }
      } catch (error) {
        console.error('Error checking session status:', error);
      }
    };

    // Check every 5 seconds
    const intervalId = setInterval(checkSessionStatus, 5000);

    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMonitoringSession]);

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

  const fetchVideos = async () => {
    try {
      const response = await api.get('/videos');
      setVideos(response.data.videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await liveSessionsApi.getAvailable();
      setSessions(response.sessions);
    } catch (error) {
      console.error('Error fetching sessions:', error);
    }
  };

  const joinVideo = async (videoId: string) => {
    try {
      await api.post(`/sessions/recorded/${videoId}/join`);
      navigate(`/video/${videoId}`);
    } catch (error) {
      console.error('Error joining video:', error);
      navigate(`/video/${videoId}`);
    }
  };

  const showMonitoringRequiredPopup = () => {
    alert(
      "Monitoring must be enabled to join the live session. Please allow camera access."
    );
  };

  const handleJoinSession = async (session: LiveSession) => {
    if (!session.meet_url) {
      alert('No Google Meet link available for this session. Please contact your teacher.');
      return;
    }

    try {
      // Step 1: Request camera permission FIRST
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 },
        audio: false 
      });

      // Step 2: Store stream and session FIRST (this triggers video element to render)
      setCameraStream(stream);
      setActiveMonitoringSession(session);
      
      // Step 3: Wait for React to render the video element in the DOM
      await new Promise<void>((resolve) => {
        // Poll for video element to appear
        const checkElement = () => {
          if (webcamRef.current) {
            resolve();
          } else {
            setTimeout(checkElement, 50);
          }
        };
        checkElement();
        // Fallback timeout
        setTimeout(() => resolve(), 2000);
      });

      // Step 4: Now set up video element (it should exist now)
      if (!webcamRef.current) {
        throw new Error('Video element not available after rendering');
      }
      
      webcamRef.current.srcObject = stream;
      await webcamRef.current.play();
      
      // Wait for video to be ready
      await new Promise<void>((resolve) => {
        if (webcamRef.current && webcamRef.current.readyState >= 2) {
          resolve();
        } else if (webcamRef.current) {
          webcamRef.current.addEventListener('loadedmetadata', () => resolve(), { once: true });
          // Fallback timeout
          setTimeout(() => resolve(), 1000);
        } else {
          resolve();
        }
      });
      
      // Step 5: Join session in backend
      try {
        await api.post(`/sessions/live/${session.id}/join`);
      } catch (error) {
        console.error('Error joining session in backend:', error);
        // Continue even if backend call fails
      }

      // Step 6: Wait for React to update hook with new sessionId, then start monitoring
      // The hook needs time to receive the updated sessionId prop
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Verify video element and session are ready
      if (!webcamRef.current || !session.id) {
        throw new Error('Video element or session ID not ready');
      }
      
      // Step 7: Now start monitoring
      setMonitoringEnabled(true);
      
      // Additional small delay to ensure hook has processed the enabled change
      await new Promise(resolve => setTimeout(resolve, 100));
      
      await startDetection();

      // Step 6: ONLY NOW open Google Meet (after monitoring is active)
      window.open(session.meet_url, '_blank', 'noopener,noreferrer');

    } catch (error: any) {
      // Camera permission denied or failed
      console.error('Error accessing camera:', error);
      setMonitoringEnabled(false);
      setActiveMonitoringSession(null);
      showMonitoringRequiredPopup();
      // DO NOT open Google Meet if monitoring fails
    }
  };

  const stopMonitoringInternal = () => {
    // Stop detection
    stopDetection();
    setMonitoringEnabled(false);
    
    // Stop camera stream
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    
    // Clear video element
    if (webcamRef.current) {
      webcamRef.current.srcObject = null;
    }
    
    // Leave session in backend
    if (activeMonitoringSession) {
      api.post(`/sessions/live/${activeMonitoringSession.id}/leave`).catch(console.error);
    }
    
    setActiveMonitoringSession(null);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
      stopDetection();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="dashboard">
      <div className="header">
        <h1>Student Dashboard</h1>
        <div className="header-actions">
          <span>Welcome, {user?.firstName} {user?.lastName}</span>
          <button className="btn btn-secondary" onClick={logout}>Logout</button>
        </div>
      </div>

      <div className="container">
        <div className="tabs">
          <button
            className={activeTab === 'videos' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('videos')}
          >
            Recorded Lectures
          </button>
          <button
            className={activeTab === 'sessions' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('sessions')}
          >
            Live Sessions
          </button>
        </div>

        {activeTab === 'videos' && (
          <div>
            <h2>Available Lectures</h2>
            <div className="grid">
              {videos.map((video) => (
                <div key={video.id} className="card">
                  <h3>{video.title}</h3>
                  <p>{video.description || 'No description'}</p>
                  <p className="text-muted">By: {video.teacher_name}</p>
                  <p className="text-muted">Uploaded: {new Date(video.created_at).toLocaleDateString()}</p>
                  <button className="btn btn-primary" onClick={() => joinVideo(video.id)}>
                    Watch Lecture
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'sessions' && (
          <div>
            <h2>Upcoming & Live Sessions</h2>
            
            {/* Active Monitoring Section - Show when session is set (video element needs to exist) */}
            {activeMonitoringSession && (
              <div className="monitoring-section" style={{
                marginBottom: '2rem',
                padding: '1.5rem',
                background: '#f0f8ff',
                border: '2px solid #4caf50',
                borderRadius: '8px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0, color: '#2e7d32' }}>
                    🟢 Monitoring Active - {activeMonitoringSession.title}
                  </h3>
                  <span style={{ fontSize: '0.85rem', color: '#666' }}>
                    Monitoring will stop when teacher ends the session
                  </span>
                </div>
                
                <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                  {/* Webcam Preview */}
                  <div style={{ position: 'relative' }}>
                    <video
                      ref={webcamRef}
                      autoPlay
                      muted
                      playsInline
                      style={{ width: '320px', height: '240px', borderRadius: '8px', background: '#000' }}
                    />
                    <canvas
                      ref={overlayCanvasRef}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '320px',
                        height: '240px',
                        pointerEvents: 'none',
                        borderRadius: '8px',
                      }}
                    />
                  </div>
                  
                  {/* Emotion Display */}
                  {emotionResult && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4caf50' }}>
                          {Math.round(emotionResult.concentrationScore)}%
                        </div>
                        <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                          Concentration Score
                        </div>
                      </div>
                      <div>
                        <p style={{ fontSize: '0.9rem', margin: '0.5rem 0' }}>
                          <strong>Dominant Emotion:</strong> {emotionResult.emotion}
                        </p>
                        <p style={{ fontSize: '0.9rem', margin: '0.5rem 0' }}>
                          <strong>Confidence:</strong> {(emotionResult.confidence * 100).toFixed(1)}%
                        </p>
                        {emotionResult.allEmotions && (
                          <div style={{ marginTop: '0.5rem' }}>
                            <p style={{ fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>Emotion Levels:</p>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.25rem', fontSize: '0.8rem' }}>
                              {Object.entries(emotionResult.allEmotions)
                                .sort(([, a], [, b]) => b - a)
                                .map(([emotion, value]) => {
                                  const emotionValue = typeof value === 'number' ? value : 0;
                                  return (
                                    <div key={emotion} style={{ display: 'flex', justifyContent: 'space-between' }}>
                                      <span style={{ textTransform: 'capitalize' }}>{emotion}:</span>
                                      <span style={{ fontWeight: 'bold' }}>{Math.round(emotionValue * 100)}%</span>
                                    </div>
                                  );
                                })}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {monitoringEnabled && isDetecting && !emotionResult && (
                    <div style={{ padding: '1rem' }}>
                      <p>Starting monitoring...</p>
                    </div>
                  )}
                  
                  {monitoringEnabled && emotionError && (
                    <div style={{ padding: '1rem', background: '#fee', borderRadius: '4px' }}>
                      <p style={{ color: '#c00', margin: 0 }}>{emotionError}</p>
                    </div>
                  )}
                  
                  {!monitoringEnabled && (
                    <div style={{ padding: '1rem', textAlign: 'center' }}>
                      <p style={{ color: '#666' }}>Setting up monitoring...</p>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <div className="grid">
              {sessions.map((session) => (
                <div key={session.id} className="card">
                  <h3>{session.title}</h3>
                  <p>{session.description || 'No description'}</p>
                  <p className="text-muted">Teacher: {session.teacher_name}</p>
                  <p className="text-muted">
                    Scheduled: {session.scheduled_at ? new Date(session.scheduled_at).toLocaleString() : 'Not scheduled'}
                  </p>
                  <p className="text-muted">Status: {session.status}</p>
                  {(session.status === 'scheduled' || session.status === 'live') && (
                    <div>
                      <p style={{ fontSize: '0.85rem', color: '#666', marginBottom: '0.5rem', fontWeight: '500' }}>
                        Camera monitoring is required to join the live session. No video is recorded.
                      </p>
                      <button 
                        className="btn btn-primary" 
                        onClick={() => handleJoinSession(session)}
                        disabled={activeMonitoringSession?.id === session.id && monitoringEnabled}
                      >
                        {activeMonitoringSession?.id === session.id && monitoringEnabled 
                          ? 'Monitoring Active' 
                          : 'Join Live Session'}
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StudentDashboard;

