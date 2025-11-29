import api from './api';

export interface EmotionResult {
  emotion: string;
  confidence: number;
  engagementScore: number;
  dominantEmotion?: string;
  allEmotions?: Record<string, number>;
}

class EmotionDetector {
  private videoElement: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private isDetecting = false;
  private detectionInterval: number | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private sessionType: string = 'recorded';
  private sessionId: string = '';
  private timestamp: number = 0;

  async startDetection(
    videoElement: HTMLVideoElement,
    onEmotionDetected: (result: EmotionResult) => void,
    sessionType: string = 'recorded',
    sessionId: string = ''
  ) {
    if (this.isDetecting) return;

    this.videoElement = videoElement;
    this.sessionType = sessionType;
    this.sessionId = sessionId;

    // Create canvas for frame extraction
    this.canvas = document.createElement('canvas');
    this.canvas.width = 640;
    this.canvas.height = 480;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      videoElement.srcObject = this.stream;
      await videoElement.play();

      this.isDetecting = true;
      this.detectEmotions(onEmotionDetected);
    } catch (error) {
      console.error('Error accessing webcam:', error);
      throw error;
    }
  }

  private detectEmotions(onEmotionDetected: (result: EmotionResult) => void) {
    if (!this.videoElement || !this.isDetecting || !this.canvas) return;

    const detect = async () => {
      if (!this.videoElement || !this.isDetecting || !this.canvas) return;

      try {
        // Capture frame from video
        const ctx = this.canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert canvas to base64 image
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);

        // Send to Python backend for emotion detection
        const response = await api.post('/emotions/detect', {
          image: imageData,
          sessionType: this.sessionType,
          sessionId: this.sessionId,
          timestamp: this.timestamp,
        });

        if (response.data.emotion) {
          onEmotionDetected({
            emotion: response.data.emotion,
            confidence: response.data.confidence,
            engagementScore: response.data.engagement_score,
            dominantEmotion: response.data.dominant_emotion,
            allEmotions: response.data.all_emotions,
          });
        }

        this.timestamp += 1; // Increment timestamp
      } catch (error) {
        console.error('Detection error:', error);
      }

      if (this.isDetecting) {
        this.detectionInterval = window.setTimeout(detect, 1000); // Detect every second
      }
    };

    detect();
  }

  stopDetection() {
    this.isDetecting = false;
    if (this.detectionInterval) {
      clearTimeout(this.detectionInterval);
      this.detectionInterval = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement = null;
    }
    if (this.canvas) {
      this.canvas = null;
    }
    this.timestamp = 0;
  }
}

export const emotionDetector = new EmotionDetector();

