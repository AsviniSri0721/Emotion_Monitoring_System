import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import EngagementMeter from '../components/EngagementMeter';
import { useEmotionStream } from '../hooks/useEmotionStream';
import api from '../services/api';
import { getInterventionMessage, getInterventionTitle } from '../utils/learningStates';
import './VideoPlayer.css';

const VideoPlayer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [video, setVideo] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionStartTime, setInterventionStartTime] = useState<number | null>(null);
  const [interventionId, setInterventionId] = useState<string | null>(null);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(false);

  // Use the emotion stream hook (now uses learning states internally)
  const {
    emotionResult,
    isDetecting,
    error: emotionError,
    shouldTriggerIntervention,
    consecutiveLowScores,
    startDetection,
    stopDetection,
  } = useEmotionStream({
    videoElement: webcamRef.current,
    sessionType: 'recorded',
    sessionId: id || '',
    interval: 1000, // 1 second for more real-time updates
    enabled: emotionDetectionEnabled && !!id,
  });

  useEffect(() => {
    fetchVideo();
    joinSession();

    return () => {
      stopDetection();
      leaveSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, stopDetection]);

  // Trigger intervention based on learning state persistence (handled in hook)
  useEffect(() => {
    if (shouldTriggerIntervention && !showIntervention && emotionResult?.learningState) {
      triggerIntervention(emotionResult.learningState);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldTriggerIntervention, showIntervention, emotionResult?.learningState]);

  // Draw bounding box on webcam video
  useEffect(() => {
    console.log('[VideoPlayer] Bbox drawing effect triggered:', {
      hasCanvas: !!overlayCanvasRef.current,
      hasVideo: !!webcamRef.current,
      hasBbox: !!emotionResult?.bbox,
      bbox: emotionResult?.bbox,
      emotion: emotionResult?.emotion,
      confidence: emotionResult?.confidence
    });

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
    // Bbox format from backend: [x1, y1, x2, y2] in original image coordinates
    const [x1, y1, x2, y2] = emotionResult.bbox;
    console.log('[VideoPlayer] Drawing bbox:', {
      original: { x1, y1, x2, y2 },
      videoSize: { width: videoWidth, height: videoHeight },
      canvasSize: { width: canvas.width, height: canvas.height },
      scale: { scaleX, scaleY },
      videoRect: rect,
      videoDisplaySize: { width: video.clientWidth, height: video.clientHeight }
    });
    
    // Scale from original image size (640x480 or videoWidth x videoHeight) to canvas display size
    const scaledX1 = x1 * scaleX;
    const scaledY1 = y1 * scaleY;
    const scaledX2 = x2 * scaleX;
    const scaledY2 = y2 * scaleY;
    
    console.log('[VideoPlayer] Scaled bbox:', {
      scaled: { x1: scaledX1, y1: scaledY1, x2: scaledX2, y2: scaledY2 },
      width: scaledX2 - scaledX1,
      height: scaledY2 - scaledY1
    });

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

  const fetchVideo = async () => {
    try {
      const response = await api.get(`/videos/${id}`);
      setVideo(response.data.video);
    } catch (error) {
      console.error('Error fetching video:', error);
    }
  };

  const joinSession = async () => {
    try {
      await api.post(`/sessions/recorded/${id}/join`);
    } catch (error) {
      console.error('Error joining session:', error);
    }
  };

  const leaveSession = async () => {
    try {
      await api.post(`/sessions/recorded/${id}/leave`);
    } catch (error) {
      console.error('Error leaving session:', error);
    }
  };

  const triggerIntervention = async (triggeredEmotion: string) => {
    if (showIntervention) return; // Already showing intervention

    try {
      const response = await api.post('/interventions/trigger', {
        sessionId: id,
        interventionType: 'mind_game', // Default type for recorded sessions
        triggeredEmotion,
        concentrationScore: emotionResult?.concentrationScore || 0,
      });

      // Store the intervention ID from the response
      const interventionIdFromResponse = response.data?.intervention?.id;
      if (interventionIdFromResponse) {
        setInterventionId(interventionIdFromResponse);
        console.log('[VideoPlayer] Intervention triggered with ID:', interventionIdFromResponse);
      }

      setShowIntervention(true);
      setInterventionStartTime(Date.now());
      
      // Pause video
      if (videoRef.current) {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    } catch (error) {
      console.error('Error triggering intervention:', error);
    }
  };

  const completeIntervention = async () => {
    if (!interventionStartTime) return;

    const duration = Math.floor((Date.now() - interventionStartTime) / 1000);
    
    // Always resume video and close intervention, even if API call fails
    const resumeVideo = () => {
      setShowIntervention(false);
      setInterventionStartTime(null);
      setInterventionId(null);

      // Resume video
      if (videoRef.current) {
        videoRef.current.play();
        setIsPlaying(true);
      }
    };

    // Try to complete intervention in backend if we have an ID
    if (interventionId) {
      try {
        console.log('[VideoPlayer] Completing intervention:', interventionId, 'Duration:', duration);
        await api.post(`/interventions/${interventionId}/complete`, { duration });
        console.log('[VideoPlayer] Intervention completed successfully');
      } catch (error) {
        console.error('[VideoPlayer] Error completing intervention:', error);
        // Continue anyway - resume video even if completion fails
      }
    } else {
      console.warn('[VideoPlayer] No intervention ID stored, skipping completion API call');
    }

    // Always resume video regardless of API call success/failure
    resumeVideo();
  };

  const startEmotionDetection = async (): Promise<boolean> => {
    if (!webcamRef.current || !id) {
      console.log('[VideoPlayer] Cannot start emotion detection:', { 
        hasWebcam: !!webcamRef.current, 
        hasId: !!id 
      });
      return false;
    }

    console.log('[VideoPlayer] Starting emotion detection...');
    setEmotionDetectionEnabled(true);

    try {
      await startDetection();
      return true;
    } catch (error) {
      console.error('[VideoPlayer] Failed to start emotion detection:', error);
      setEmotionDetectionEnabled(false);
      return false;
    }
  };

  const stopEmotionDetection = () => {
    console.log('[VideoPlayer] Stopping emotion detection...');
    setEmotionDetectionEnabled(false);
    stopDetection();
  };

  const handlePlayPause = async () => {
    if (!videoRef.current) return;

    if (isPlaying) {
      // Pause video and stop monitoring
      videoRef.current.pause();
      stopEmotionDetection();
      setIsPlaying(false);
    } else {
      // Try to start monitoring; only play video if camera access succeeds
      const monitoringStarted = await startEmotionDetection();
      if (monitoringStarted) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        console.warn('[VideoPlayer] Monitoring could not start; video will remain paused.');
      }
    }
  };

  const handleVideoPause = () => {
    setIsPlaying(false);
    stopEmotionDetection();
  };

  const handleVideoEnded = () => {
    setIsPlaying(false);
    stopEmotionDetection();
  };

  // Time update handler removed as currentTime is not used

  if (!video) {
    return <div className="loading">Loading video...</div>;
  }

  if (showIntervention) {
    const learningState = emotionResult?.learningState || 'DISENGAGED';
    const interventionTitle = getInterventionTitle(learningState as any);
    const interventionMessage = getInterventionMessage(learningState as any);
    
    return (
      <div className="intervention-container">
        <div className="intervention-content">
          <h2>{interventionTitle}</h2>
          <p>{interventionMessage}</p>
          
          <div className="intervention-activity">
            <a
              href="https://www.youtube.com/watch?v=p7GmO8ewjmw&list=RDp7GmO8ewjmw&start_radio=1"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              style={{ marginBottom: '1rem', display: 'inline-block' }}
            >
              Try Interactive Activity
            </a>
          </div>
          
          <button className="btn btn-secondary" onClick={completeIntervention}>
            Continue Learning
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="video-player-container">
      <div className="video-header">
        <button className="btn btn-secondary" onClick={() => navigate('/student/dashboard')}>
          Back to Dashboard
        </button>
        <h1>{video.title}</h1>
      </div>

      <div className="video-content">
        <div className="video-section">
          <div className="video-wrapper">
            <video
              ref={videoRef}
              src={`${process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000'}/uploads/${video.file_path}`}
              onPlay={() => setIsPlaying(true)}
              onPause={handleVideoPause}
              onEnded={handleVideoEnded}
              controls
            />
          </div>
          <div className="video-controls">
            <button className="btn btn-primary" onClick={handlePlayPause}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>
          </div>
        </div>

        <div className="emotion-section">
          <div className="emotion-controls">
            <h3>Emotion Monitoring</h3>
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
            <div className="emotion-display" key={`emotion-${emotionResult.timestamp}-${emotionResult.concentrationScore}`}>
              <EngagementMeter
                key={`meter-${emotionResult.timestamp}-${emotionResult.concentrationScore}`}
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
                <p style={{ fontSize: '10px', color: '#999' }}>Timestamp: {emotionResult.timestamp}</p>
                {emotionResult.learningState && (
                  <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                    Learning State: {emotionResult.learningState}
                  </p>
                )}
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

export default VideoPlayer;

