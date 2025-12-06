import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../services/api';

export interface EmotionStreamResult {
  emotion: string;
  confidence: number;
  concentrationScore: number;
  dominantEmotion?: string;
  allEmotions?: Record<string, number>;
  timestamp: number;
}

interface UseEmotionStreamOptions {
  videoElement: HTMLVideoElement | null;
  sessionType: 'recorded' | 'live';
  sessionId: string;
  interval?: number; // Detection interval in milliseconds (default: 2000)
  enabled?: boolean; // Whether detection is enabled
}

export const useEmotionStream = ({
  videoElement,
  sessionType,
  sessionId,
  interval = 2000,
  enabled = true,
}: UseEmotionStreamOptions) => {
  const [emotionResult, setEmotionResult] = useState<EmotionStreamResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [consecutiveLowScores, setConsecutiveLowScores] = useState(0);
  const [shouldTriggerIntervention, setShouldTriggerIntervention] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<number | null>(null);
  const timestampRef = useRef<number>(0);
  const lowScoreCountRef = useRef<number>(0);
  const isDetectingRef = useRef<boolean>(false);

  // Check if intervention should be triggered
  const checkIntervention = useCallback(async () => {
    try {
      const response = await api.get(`/interventions/check/${sessionType}/${sessionId}`);
      if (response.data.should_trigger) {
        setShouldTriggerIntervention(true);
        setConsecutiveLowScores(response.data.consecutive_low);
      }
    } catch (err) {
      console.error('Error checking intervention:', err);
    }
  }, [sessionType, sessionId]);

  // Capture frame and send to backend
  const captureAndDetect = useCallback(async () => {
    if (!videoElement || !sessionId || !isDetectingRef.current) {
      console.log('[EmotionStream] Skipping detection:', { 
        hasVideoElement: !!videoElement, 
        hasSessionId: !!sessionId, 
        isDetecting: isDetectingRef.current 
      });
      return;
    }

    try {
      console.log('[EmotionStream] Capturing frame and sending to backend...');
      // Create canvas if it doesn't exist
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
        canvasRef.current.width = 640;
        canvasRef.current.height = 480;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Capture frame from video
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      // Convert canvas to base64 image
      const imageData = canvas.toDataURL('image/jpeg', 0.8);

      // Send to backend stream endpoint
      const response = await api.post('/emotions/stream', {
        image: imageData,
        sessionType,
        sessionId,
        timestamp: timestampRef.current,
      });

      console.log('[EmotionStream] Detection response:', {
        emotion: response.data.emotion,
        concentration: response.data.concentration_score,
        confidence: response.data.confidence
      });

      if (response.data.emotion) {
        const result: EmotionStreamResult = {
          emotion: response.data.emotion,
          confidence: response.data.confidence,
          concentrationScore: response.data.concentration_score || 50.0,
          dominantEmotion: response.data.dominant_emotion,
          allEmotions: response.data.all_emotions,
          timestamp: response.data.timestamp || timestampRef.current,
        };

        setEmotionResult(result);

        // Track consecutive low concentration scores
        if (result.concentrationScore <= 50) {
          lowScoreCountRef.current += 1;
          if (lowScoreCountRef.current >= 15) {
            setShouldTriggerIntervention(true);
            setConsecutiveLowScores(lowScoreCountRef.current);
          }
        } else {
          lowScoreCountRef.current = 0;
          setShouldTriggerIntervention(false);
        }

        // Check intervention status from backend (every 5 frames)
        if (timestampRef.current % 5 === 0) {
          checkIntervention();
        }

        timestampRef.current += 1;
      }
    } catch (err: any) {
      console.error('Emotion detection error:', err);
      setError(err?.response?.data?.error || 'Failed to detect emotions');
    }
  }, [videoElement, enabled, sessionType, sessionId, checkIntervention]);

  // Start detection
  const startDetection = useCallback(async () => {
    if (!videoElement || isDetecting) {
      console.log('[EmotionStream] Cannot start detection:', { 
        hasVideoElement: !!videoElement, 
        isDetecting 
      });
      return;
    }

    try {
      console.log('[EmotionStream] Starting detection...');
      
      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      streamRef.current = stream;
      videoElement.srcObject = stream;
      await videoElement.play();

      setIsDetecting(true);
      isDetectingRef.current = true; // Set ref immediately
      setError(null);
      timestampRef.current = 0;
      lowScoreCountRef.current = 0;

      console.log('[EmotionStream] Webcam accessed, starting periodic detection...');

      // Start periodic detection
      // Once started, continue regardless of enabled prop
      const detect = () => {
        if (isDetectingRef.current) {
          captureAndDetect();
          intervalRef.current = window.setTimeout(detect, interval);
        } else {
          console.log('[EmotionStream] Detection stopped, ending interval');
        }
      };

      detect();
    } catch (err: any) {
      console.error('[EmotionStream] Error accessing webcam:', err);
      setError('Could not access webcam. Please allow camera permissions.');
      setIsDetecting(false);
      isDetectingRef.current = false;
    }
  }, [videoElement, isDetecting, interval, captureAndDetect]);

  // Stop detection
  const stopDetection = useCallback(() => {
    console.log('[EmotionStream] Stopping detection...');
    setIsDetecting(false);
    isDetectingRef.current = false; // Clear ref immediately
    setShouldTriggerIntervention(false);
    setConsecutiveLowScores(0);
    lowScoreCountRef.current = 0;

    if (intervalRef.current) {
      clearTimeout(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoElement) {
      videoElement.srcObject = null;
    }

    if (canvasRef.current) {
      canvasRef.current = null;
    }

    timestampRef.current = 0;
    console.log('[EmotionStream] Detection stopped');
  }, [videoElement]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, [stopDetection]);

  return {
    emotionResult,
    isDetecting,
    error,
    shouldTriggerIntervention,
    consecutiveLowScores,
    startDetection,
    stopDetection,
  };
};

