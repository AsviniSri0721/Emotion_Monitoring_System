import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import { emotionDetector, EmotionResult } from '../services/emotionDetection';
import './VideoPlayer.css';

const VideoPlayer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);
  const [video, setVideo] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [emotion, setEmotion] = useState<EmotionResult | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<Array<EmotionResult & { timestamp: number }>>([]);
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionType, setInterventionType] = useState<'mind_game' | 'music_therapy' | 'micro_learning' | null>(null);
  const [interventionStartTime, setInterventionStartTime] = useState<number | null>(null);
  const sessionStartTime = useRef<number>(Date.now());
  const lastEmotionSendTime = useRef<number>(0);

  useEffect(() => {
    fetchVideo();
    joinSession();

    return () => {
      emotionDetector.stopDetection();
      leaveSession();
    };
  }, [id]);

  useEffect(() => {
    if (emotion && videoRef.current) {
      const timestamp = Math.floor((Date.now() - sessionStartTime.current) / 1000);
      const videoTime = Math.floor(videoRef.current.currentTime);

      // Send emotion data to server every 5 seconds
      if (timestamp - lastEmotionSendTime.current >= 5) {
        sendEmotionData(emotion, videoTime);
        lastEmotionSendTime.current = timestamp;
      }

      // Check for disengagement (only for recorded sessions)
      checkDisengagement(emotion);
    }
  }, [emotion]);

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
      // Generate report
      await api.post(`/reports/generate/recorded/${id}`, { studentId: user?.id });
    } catch (error) {
      console.error('Error leaving session:', error);
    }
  };

  // Emotion data is now sent automatically by the detector
  const sendEmotionData = async (emotionData: EmotionResult, timestamp: number) => {
    // Data is sent automatically by emotionDetector
    // This function kept for compatibility but no longer needed
  };

  const checkDisengagement = (emotionData: EmotionResult) => {
    const disengagedEmotions = ['bored', 'confused', 'sleepy'];
    if (disengagedEmotions.includes(emotionData.emotion) && emotionData.confidence > 0.6) {
      triggerIntervention(emotionData.emotion);
    }
  };

  const triggerIntervention = async (triggeredEmotion: string) => {
    if (showIntervention) return; // Already showing intervention

    const types: Array<'mind_game' | 'music_therapy' | 'micro_learning'> = ['mind_game', 'music_therapy', 'micro_learning'];
    const randomType = types[Math.floor(Math.random() * types.length)];

    try {
      await api.post('/interventions/trigger', {
        sessionId: id,
        interventionType: randomType,
        triggeredEmotion,
      });

      setInterventionType(randomType);
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
    if (!interventionType || !interventionStartTime) return;

    const duration = Math.floor((Date.now() - interventionStartTime) / 1000);
    
    try {
      // Get intervention ID from the last trigger
      const interventions = await api.get(`/interventions/session/${id}`);
      const lastIntervention = interventions.data.interventions[0];
      
      if (lastIntervention) {
        await api.post(`/interventions/${lastIntervention.id}/complete`, { duration });
      }

      setShowIntervention(false);
      setInterventionType(null);
      setInterventionStartTime(null);

      // Resume video
      if (videoRef.current) {
        videoRef.current.play();
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Error completing intervention:', error);
    }
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
        'recorded',
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

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  if (!video) {
    return <div className="loading">Loading video...</div>;
  }

  if (showIntervention) {
    return (
      <div className="intervention-container">
        <div className="intervention-content">
          <h2>Take a Break!</h2>
          <p>We noticed you might be feeling {emotion?.emotion}. Let's take a short break to re-energize.</p>
          
          {interventionType === 'mind_game' && (
            <div className="intervention-activity">
              <h3>Mind Game</h3>
              <p>Try to solve this: What comes next? 2, 4, 8, 16, ?</p>
              <p className="answer">Answer: 32 (each number is doubled)</p>
            </div>
          )}
          
          {interventionType === 'music_therapy' && (
            <div className="intervention-activity">
              <h3>Music Therapy</h3>
              <p>Take a moment to breathe deeply and relax. Close your eyes and focus on your breathing.</p>
              <p>Inhale for 4 counts, hold for 4, exhale for 4.</p>
            </div>
          )}
          
          {interventionType === 'micro_learning' && (
            <div className="intervention-activity">
              <h3>Quick Learning Tip</h3>
              <p>When learning, try the Feynman Technique:</p>
              <ol>
                <li>Choose a concept you want to understand</li>
                <li>Explain it in simple terms</li>
                <li>Identify gaps in your understanding</li>
                <li>Review and simplify further</li>
              </ol>
            </div>
          )}
          
          <button className="btn btn-primary" onClick={completeIntervention}>
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
              onTimeUpdate={handleTimeUpdate}
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

export default VideoPlayer;

