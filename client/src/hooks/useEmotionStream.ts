import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../services/api';
import { averageEmotions, calculateConcentration, getMaxEmotion, SMOOTHING_WINDOW } from '../utils/emotionUtils';
import { deriveLearningState, INTERVENTION_STATES, LearningState, REQUIRED_PERSISTENCE_FRAMES } from '../utils/learningStates';

export interface EmotionStreamResult {
  emotion: string;
  confidence: number;
  concentrationScore: number;
  dominantEmotion?: string;
  allEmotions?: Record<string, number>;
  timestamp: number;
  bbox?: number[]; // [x1, y1, x2, y2] bounding box coordinates
  learningState?: LearningState; // Current learning state derived from concentration and emotions
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
  interval = 1000, // 1 second for more real-time updates
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
  const isDetectingRef = useRef<boolean>(false);
  const previousImageHashRef = useRef<string | null>(null);
  const previousEmotionResultRef = useRef<EmotionStreamResult | null>(null);
  const previousVideoTimeRef = useRef<number | null>(null);
  const emotionBufferRef = useRef<Record<string, number>[]>([]);
  
  // Learning state tracking with temporal persistence
  const learningStateRef = useRef<LearningState | null>(null);
  const stateCounterRef = useRef<number>(0);

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

      // Ensure video is playing and has current frame
      if (videoElement.readyState < 2) {
        console.log('[EmotionStream] Video not ready, waiting...');
        return;
      }
      
      // Check if video is actually playing/updating
      if (videoElement.paused) {
        console.warn('[EmotionStream] Video is paused! Attempting to play...');
        try {
          await videoElement.play();
        } catch (e) {
          console.error('[EmotionStream] Cannot play video:', e);
          return;
        }
      }

      // Capture frame from video - ensure we get a fresh frame
      // Get current time BEFORE capture to verify it's changing
      const currentTimeBefore = videoElement.currentTime;
      
      // Use requestVideoFrameCallback if available (better for video frames)
      // Otherwise fall back to requestAnimationFrame
      if ('requestVideoFrameCallback' in videoElement) {
        await new Promise<void>((resolve) => {
          (videoElement as any).requestVideoFrameCallback(() => {
            resolve();
          });
        });
      } else {
        // Fallback: Use double requestAnimationFrame for better frame timing
        await new Promise(resolve => {
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              resolve(undefined);
            });
          });
        });
      }
      
      const currentTimeAfter = videoElement.currentTime;
      const timeChanged = Math.abs(currentTimeAfter - currentTimeBefore) > 0.001;
      
      // Check against previous video time (floating-point safe comparison)
      const isDuplicateFrame = previousVideoTimeRef.current !== null && 
        Math.abs(currentTimeAfter - previousVideoTimeRef.current) < 0.001;
      
      console.log('[EmotionStream] Video time check:', {
        before: currentTimeBefore,
        after: currentTimeAfter,
        changed: timeChanged,
        previousTime: previousVideoTimeRef.current,
        isDuplicate: isDuplicateFrame,
        paused: videoElement.paused,
        readyState: videoElement.readyState
      });
      
      // If video time hasn't changed significantly, log warning but still proceed
      if (isDuplicateFrame) {
        console.warn('[EmotionStream] Duplicate frame detected (small delta).');
      } else if (!timeChanged && currentTimeBefore > 0) {
        console.warn('[EmotionStream] Video time not advancing - may be processing duplicate frames');
      }
      
      // Update previous video time
      previousVideoTimeRef.current = currentTimeAfter;
      
      // Draw the current video frame
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      
      // Add timestamp to ensure uniqueness
      const frameTimestamp = Date.now();
      
      // Get image data hash to verify uniqueness
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      const imageHash = imageData.substring(20, 50); // Use part of base64 as hash
      
      // Check if this is the same frame as before
      const isSameFrame = previousImageHashRef.current === imageHash;
      
      console.log('[EmotionStream] Captured frame:', {
        timestamp: frameTimestamp,
        videoReadyState: videoElement.readyState,
        videoCurrentTime: videoElement.currentTime,
        videoWidth: videoElement.videoWidth,
        videoHeight: videoElement.videoHeight,
        imageDataLength: imageData.length,
        imageHash: imageHash,
        previousHash: previousImageHashRef.current,
        isSameFrame: isSameFrame,
        warning: isSameFrame ? '⚠️ SAME FRAME DETECTED - Video may not be updating!' : '✓ New frame'
      });
      
      // If same frame detected multiple times, log warning but still send
      // (Backend should process it, but we want to know if video isn't updating)
      if (isSameFrame && previousImageHashRef.current !== null) {
        console.warn('[EmotionStream] Duplicate frame detected (small delta).');
        // Still send to backend - it might be a legitimate duplicate or backend can handle it
      }
      
      // Store hash to compare next time
      previousImageHashRef.current = imageHash;

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
        confidence: response.data.confidence,
        bbox: response.data.bbox,
        hasBbox: !!response.data.bbox,
        bboxType: typeof response.data.bbox,
        bboxLength: response.data.bbox?.length
      });

      if (response.data.emotion || response.data.emotions || response.data.probs) {
        // Clear any previous errors on successful detection
        setError(null);
        
        // Get raw emotion probabilities from API
        const rawEmotions = response.data.emotions || response.data.all_emotions || response.data.probs || {};
        
        // Apply temporal smoothing
        emotionBufferRef.current.push(rawEmotions);
        if (emotionBufferRef.current.length > SMOOTHING_WINDOW) {
          emotionBufferRef.current.shift();
        }
        
        // Calculate smoothed emotions
        const smoothedEmotions = averageEmotions(emotionBufferRef.current);
        
        // Compute concentration from smoothed emotions (frontend calculation)
        const concentrationScore = calculateConcentration(smoothedEmotions);
        
        // Get dominant emotion from smoothed emotions
        const dominantEmotion = getMaxEmotion(smoothedEmotions);
        
        // Use original confidence from API
        const confidence = response.data.confidence || 0.0;
        
        // Derive learning state (will be used for persistence tracking)
        const currentState = deriveLearningState(concentrationScore, smoothedEmotions);
        
        const result: EmotionStreamResult = {
          emotion: dominantEmotion,
          confidence: confidence,
          concentrationScore: concentrationScore,
          dominantEmotion: dominantEmotion,
          allEmotions: smoothedEmotions, // Use smoothed emotions for UI
          timestamp: response.data.timestamp || timestampRef.current,
          bbox: response.data.bbox,
          learningState: currentState, // Include learning state in result
        };

        // Log before setting state to verify new data
        console.log('[EmotionStream] Setting new emotion result:', {
          emotion: result.emotion,
          concentration: result.concentrationScore,
          confidence: result.confidence,
          bbox: result.bbox,
          timestamp: result.timestamp,
          smoothedEmotions: smoothedEmotions,
          previousEmotion: previousEmotionResultRef.current?.emotion,
          previousConcentration: previousEmotionResultRef.current?.concentrationScore,
          previousConfidence: previousEmotionResultRef.current?.confidence,
          isDifferent: previousEmotionResultRef.current ? (
            previousEmotionResultRef.current.emotion !== result.emotion ||
            Math.abs(previousEmotionResultRef.current.concentrationScore - result.concentrationScore) > 0.01 ||
            Math.abs(previousEmotionResultRef.current.confidence - result.confidence) > 0.01
          ) : true
        });
        
        setEmotionResult(result);
        // Update ref with new result for next comparison
        previousEmotionResultRef.current = result;

        // Track temporal persistence of learning state
        if (currentState === learningStateRef.current) {
          // State persists, increment counter
          stateCounterRef.current += 1;
        } else {
          // State changed, reset counter
          stateCounterRef.current = 1;
          learningStateRef.current = currentState;
        }
        
        // Trigger intervention only for harmful states that have persisted
        if (
          INTERVENTION_STATES.includes(currentState) &&
          stateCounterRef.current >= REQUIRED_PERSISTENCE_FRAMES
        ) {
          setShouldTriggerIntervention(true);
          setConsecutiveLowScores(stateCounterRef.current);
        } else {
          // Reset intervention trigger if state improves or hasn't persisted long enough
          setShouldTriggerIntervention(false);
          if (!INTERVENTION_STATES.includes(currentState)) {
            setConsecutiveLowScores(0);
          }
        }

        timestampRef.current += 1;
      }
    } catch (err: any) {
      console.error('Emotion detection error:', err);
      setError(err?.response?.data?.error || 'Failed to detect emotions');
    }
  }, [videoElement, sessionType, sessionId]);

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
      
      // Ensure video is set to autoplay and plays inline
      videoElement.autoplay = true;
      videoElement.muted = true;
      videoElement.playsInline = true;
      
      // Wait for video to be ready
      await new Promise((resolve) => {
        if (videoElement.readyState >= 2) {
          resolve(undefined);
        } else {
          videoElement.addEventListener('loadedmetadata', () => resolve(undefined), { once: true });
        }
      });
      
      // Force play and wait for it to start
      await videoElement.play();
      
      // Verify video is actually playing
      if (videoElement.paused) {
        throw new Error('Video failed to play after play() call');
      }
      
      console.log('[EmotionStream] Webcam stream initialized:', {
        playing: !videoElement.paused,
        readyState: videoElement.readyState,
        videoWidth: videoElement.videoWidth,
        videoHeight: videoElement.videoHeight
      });

      setIsDetecting(true);
      isDetectingRef.current = true; // Set ref immediately
      setError(null);
      timestampRef.current = 0;
      previousImageHashRef.current = null; // Reset frame hash

      console.log('[EmotionStream] Webcam accessed, starting periodic detection...');

      // Start periodic detection
      // Once started, continue regardless of enabled prop
      const detect = async () => {
        if (isDetectingRef.current) {
          // Verify video is still playing before capturing
          if (videoElement.paused) {
            console.warn('[EmotionStream] Video paused, attempting to resume...');
            try {
              await videoElement.play();
            } catch (e) {
              console.error('[EmotionStream] Failed to resume video:', e);
            }
          }
          
          // Wait a bit to ensure video has a new frame
          await new Promise(resolve => setTimeout(resolve, 50));
          await captureAndDetect();
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
      // Propagate error so callers can react (e.g., block video playback)
      throw err;
    }
  }, [videoElement, isDetecting, interval, captureAndDetect]);

  // Stop detection
  const stopDetection = useCallback(() => {
    console.log('[EmotionStream] Stopping detection...');
    setIsDetecting(false);
    isDetectingRef.current = false; // Clear ref immediately
    setShouldTriggerIntervention(false);
    setConsecutiveLowScores(0);
    emotionBufferRef.current = []; // Clear emotion buffer
    learningStateRef.current = null; // Reset learning state
    stateCounterRef.current = 0; // Reset state counter

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

