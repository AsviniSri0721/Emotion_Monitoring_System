import { useCallback, useEffect, useRef, useState } from 'react';
import { liveSessionsApi, LiveSessionStreamResponse } from '../api/liveSessions';
import { averageEmotions, calculateConcentration, getMaxEmotion, SMOOTHING_WINDOW } from '../utils/emotionUtils';
import { hashFrame, shouldSendFrame } from '../utils/frameUtils';

export interface LiveEmotionStreamResult {
  emotion: string;
  confidence: number;
  concentrationScore: number;
  engagementScore?: number;
  timestamp: number;
  bbox?: number[] | null;
  allEmotions?: Record<string, number>;
}

interface UseLiveSessionEmotionStreamOptions {
  videoElement: HTMLVideoElement | null;
  sessionId: string;
  interval?: number; // Detection interval in milliseconds (default: 5000 = 5 seconds)
  enabled?: boolean; // Whether detection is enabled
}

export const useLiveSessionEmotionStream = ({
  videoElement,
  sessionId,
  interval = 5000, // 5 seconds for live sessions
  enabled = true,
}: UseLiveSessionEmotionStreamOptions) => {
  const [emotionResult, setEmotionResult] = useState<LiveEmotionStreamResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<number | null>(null);
  const timestampRef = useRef<number>(0);
  const isDetectingRef = useRef<boolean>(false);
  const previousImageHashRef = useRef<string | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const emotionBufferRef = useRef<Record<string, number>[]>([]);
  
  // Track stable frames for boredom reduction
  const stableFrameCountRef = useRef<number>(0);
  const previousEmotionHashRef = useRef<string | null>(null);

  // Capture frame and send to backend (NO intervention checks)
  const captureAndDetect = useCallback(async () => {
    // Get current video element (handle case where it's a ref or direct element)
    const currentVideoElement = (videoElement as any)?.current || videoElement;
    
    if (!currentVideoElement || !sessionId || !isDetectingRef.current) {
      return;
    }

    try {
      // Create canvas if it doesn't exist
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
        canvasRef.current.width = 640;
        canvasRef.current.height = 480;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Ensure video is ready
      if (currentVideoElement.readyState < 2) {
        return;
      }

      if (currentVideoElement.paused) {
        try {
          await currentVideoElement.play();
        } catch (e) {
          console.error('[LiveEmotionStream] Cannot play video:', e);
          return;
        }
      }

      // Wait for next frame
      if ('requestVideoFrameCallback' in currentVideoElement) {
        await new Promise<void>((resolve) => {
          (currentVideoElement as any).requestVideoFrameCallback(() => {
            resolve();
          });
        });
      } else {
        await new Promise(resolve => {
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              resolve(undefined);
            });
          });
        });
      }

      // Draw the current video frame
      ctx.drawImage(currentVideoElement, 0, 0, canvas.width, canvas.height);

      // Get image data for hashing (before converting to base64)
      const frameImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const frameHash = hashFrame(frameImageData);

      // Skip duplicate frames BEFORE sending to backend
      if (!shouldSendFrame(frameHash, previousImageHashRef.current)) {
        console.log('[LiveEmotionStream] Skipping duplicate frame');
        return; // Don't send duplicate frame
      }
      previousImageHashRef.current = frameHash;

      // Convert to base64 for API (only for non-duplicate frames)
      const imageData = canvas.toDataURL('image/jpeg', 0.8);

      // Calculate timestamp (seconds from session start)
      if (!startTimeRef.current) {
        startTimeRef.current = Date.now();
      }
      const currentTimestamp = Math.floor((Date.now() - startTimeRef.current) / 1000);

      // Send to live session stream endpoint (NO interventions)
      const response: LiveSessionStreamResponse = await liveSessionsApi.stream(sessionId, {
        image: imageData,
        timestamp: currentTimestamp,
      });

      // Get raw emotion probabilities from API
      const rawEmotions = (response as any).emotions || (response as any).probs || {};
      
      // Check if emotions are stable (unchanged)
      const emotionHash = JSON.stringify(rawEmotions);
      if (emotionHash === previousEmotionHashRef.current) {
        stableFrameCountRef.current += 1;
      } else {
        stableFrameCountRef.current = 0;
        previousEmotionHashRef.current = emotionHash;
      }
      
      // Apply temporal smoothing
      emotionBufferRef.current.push(rawEmotions);
      if (emotionBufferRef.current.length > SMOOTHING_WINDOW) {
        emotionBufferRef.current.shift();
      }
      
      // Calculate smoothed emotions
      let smoothedEmotions = averageEmotions(emotionBufferRef.current);
      
      // Reduce boredom weight during stable frames (if emotions unchanged for 8+ frames)
      if (stableFrameCountRef.current > 8 && smoothedEmotions.boredom) {
        smoothedEmotions = {
          ...smoothedEmotions,
          boredom: smoothedEmotions.boredom * 0.7
        };
        console.log(`[LiveEmotionStream] Reduced boredom weight (stable frames: ${stableFrameCountRef.current})`);
      }
      
      // Compute concentration from smoothed emotions (frontend calculation)
      const concentrationScore = calculateConcentration(smoothedEmotions);
      
      // Get dominant emotion from smoothed emotions
      const dominantEmotion = getMaxEmotion(smoothedEmotions);
      
      // Update state with result - use smoothed emotions and computed concentration
      setEmotionResult({
        emotion: dominantEmotion,
        confidence: response.confidence || 0.0,
        concentrationScore: concentrationScore,
        engagementScore: response.engagement_score,
        timestamp: response.timestamp,
        bbox: response.bbox,
        allEmotions: smoothedEmotions, // Use smoothed emotions for UI
      });

      timestampRef.current = currentTimestamp;
      setError(null);
    } catch (err: any) {
      console.error('Live emotion detection error:', err);
      setError(err?.response?.data?.error || 'Failed to detect emotions');
    }
  }, [videoElement, sessionId]);

  // Start detection
  const startDetection = useCallback(async () => {
    // Get current video element (handle case where it's a ref or direct element)
    const currentVideoElement = (videoElement as any)?.current || videoElement;
    
    if (!currentVideoElement || !sessionId) {
      setError('Video element or session ID missing');
      console.error('[LiveEmotionStream] Start detection failed:', {
        hasVideoElement: !!currentVideoElement,
        sessionId: sessionId,
        videoElementType: typeof videoElement,
        videoElementValue: videoElement
      });
      return;
    }

    try {
      // Get user media if needed (only if stream not already set)
      if (!streamRef.current) {
        // Check if video element already has a stream
        if (currentVideoElement.srcObject) {
          streamRef.current = currentVideoElement.srcObject as MediaStream;
        } else {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false,
          });
          streamRef.current = stream;
          currentVideoElement.srcObject = stream;
          await currentVideoElement.play();
        }
      }

      // Reset timestamp
      startTimeRef.current = Date.now();
      timestampRef.current = 0;
      isDetectingRef.current = true;
      setIsDetecting(true);
      setError(null);

      // Start interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      intervalRef.current = window.setInterval(() => {
        captureAndDetect();
      }, interval);

      // Initial capture
      setTimeout(() => {
        captureAndDetect();
      }, 1000);
    } catch (err: any) {
      console.error('Error starting live emotion detection:', err);
      setError(err?.message || 'Failed to start detection');
      setIsDetecting(false);
      isDetectingRef.current = false;
    }
  }, [videoElement, sessionId, interval, captureAndDetect]);

  // Stop detection
  const stopDetection = useCallback(() => {
    isDetectingRef.current = false;
    emotionBufferRef.current = []; // Clear emotion buffer
    previousImageHashRef.current = null; // Reset frame hash
    stableFrameCountRef.current = 0; // Reset stable frame count
    previousEmotionHashRef.current = null; // Reset emotion hash
    setIsDetecting(false);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoElement) {
      videoElement.srcObject = null;
    }

    startTimeRef.current = null;
    timestampRef.current = 0;
  }, [videoElement]);

  // Auto-start/stop based on enabled prop
  useEffect(() => {
    if (enabled && videoElement && sessionId) {
      startDetection();
    } else {
      stopDetection();
    }

    return () => {
      stopDetection();
    };
  }, [enabled, videoElement, sessionId, startDetection, stopDetection]);

  return {
    emotionResult,
    isDetecting,
    error,
    startDetection,
    stopDetection,
  };
};







