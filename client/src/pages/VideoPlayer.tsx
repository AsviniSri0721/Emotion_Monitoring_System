import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import EngagementMeter from '../components/EngagementMeter';
import { useAuth } from '../contexts/AuthContext';
import { useEmotionStream } from '../hooks/useEmotionStream';
import api from '../services/api';
import './VideoPlayer.css';

const VideoPlayer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);
  const [video, setVideo] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionStartTime, setInterventionStartTime] = useState<number | null>(null);
  const [interventionId, setInterventionId] = useState<string | null>(null);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(false);

  // Use the new emotion stream hook
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
    interval: 2000,
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

  // Trigger intervention when concentration is low for 10 consecutive frames
  useEffect(() => {
    if (shouldTriggerIntervention && !showIntervention) {
      triggerIntervention('low_concentration');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldTriggerIntervention, showIntervention]);

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

  const startEmotionDetection = async () => {
    if (!webcamRef.current || !id) {
      console.log('[VideoPlayer] Cannot start emotion detection:', { 
        hasWebcam: !!webcamRef.current, 
        hasId: !!id 
      });
      return;
    }
    console.log('[VideoPlayer] Starting emotion detection...');
    setEmotionDetectionEnabled(true);
    await startDetection();
  };

  const stopEmotionDetection = () => {
    console.log('[VideoPlayer] Stopping emotion detection...');
    setEmotionDetectionEnabled(false);
    stopDetection();
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Time update handler removed as currentTime is not used

  if (!video) {
    return <div className="loading">Loading video...</div>;
  }

  if (showIntervention) {
    return (
      <div className="intervention-container">
        <div className="intervention-content">
          <h2>Low Concentration Detected</h2>
          <p>You lack of concentration so watch and come again</p>
          {emotionResult && (
            <p>Current concentration: {Math.round(emotionResult.concentrationScore)}%</p>
          )}
          
          <div className="intervention-activity">
            <a
              href="https://www.youtube.com/watch?v=p7GmO8ewjmw&list=RDp7GmO8ewjmw&start_radio=1"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
              style={{ marginBottom: '1rem', display: 'inline-block' }}
            >
              Watch on YouTube
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
              onPause={() => setIsPlaying(false)}
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

export default VideoPlayer;

