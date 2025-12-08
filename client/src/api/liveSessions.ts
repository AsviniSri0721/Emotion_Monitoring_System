import api from '../services/api';

export interface LiveSession {
  id: string;
  teacher_id: string;
  title: string;
  description?: string;
  meet_url: string;
  scheduled_at: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  status: string;
  created_at: string | null;
  teacher_name?: string;
}

export interface CreateLiveSessionRequest {
  title: string;
  meetUrl: string;
  startTime?: string;
}

export interface LiveSessionStreamRequest {
  image: string; // base64 encoded
  timestamp: number;
}

export interface LiveSessionStreamResponse {
  emotion: string;
  confidence: number;
  concentration_score: number;
  engagement_score?: number;
  timestamp: number;
  bbox?: number[] | null;
  message?: string;
}

export interface LiveSessionReport {
  session: LiveSession;
  total_students?: number;
  total_logs?: number;
  overall_avg_engagement?: number;
  overall_avg_concentration?: number;
  students?: Array<{
    student_id: string;
    student_name: string;
    student_email: string;
    total_logs: number;
    avg_engagement: number;
    avg_concentration: number;
    dominant_emotion: string;
    emotion_counts: Record<string, number>;
    timeline: Array<{
      timestamp: number;
      emotion: string;
      confidence: number;
      engagement_score: number;
      concentration_score: number;
    }>;
  }>;
  student_data?: {
    total_logs: number;
    avg_engagement: number;
    avg_concentration: number;
    dominant_emotion: string;
    emotion_counts: Record<string, number>;
    timeline: Array<{
      timestamp: number;
      emotion: string;
      confidence: number;
      engagement_score: number;
      concentration_score: number;
    }>;
  };
}

export const liveSessionsApi = {
  /**
   * Create a new live session (Teacher only)
   */
  create: async (data: CreateLiveSessionRequest): Promise<{ session: { id: string }; message: string }> => {
    const response = await api.post('/live-sessions/create', data);
    return response.data;
  },

  /**
   * Get all available live sessions
   */
  getAvailable: async (): Promise<{ sessions: LiveSession[] }> => {
    const response = await api.get('/live-sessions/available');
    return response.data;
  },

  /**
   * Stream emotion detection for live session
   * This endpoint does NOT trigger interventions - only records data
   */
  stream: async (sessionId: string, data: LiveSessionStreamRequest): Promise<LiveSessionStreamResponse> => {
    const response = await api.post(`/live-sessions/${sessionId}/stream`, data);
    return response.data;
  },

  /**
   * Get analytics report for a live session
   */
  getReport: async (sessionId: string): Promise<{ report: LiveSessionReport }> => {
    const response = await api.get(`/live-sessions/${sessionId}/report`);
    return response.data;
  },
};

