import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import './Dashboard.css';

interface Video {
  id: string;
  title: string;
  description: string;
  teacher_name: string;
  created_at: string;
}

interface LiveSession {
  id: string;
  title: string;
  description: string;
  meet_url: string;
  scheduled_at: string;
  status: string;
  teacher_name: string;
}

const StudentDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [videos, setVideos] = useState<Video[]>([]);
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [activeTab, setActiveTab] = useState<'videos' | 'sessions'>('videos');

  useEffect(() => {
    fetchVideos();
    fetchSessions();
  }, []);

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
      const response = await api.get('/sessions/live');
      setSessions(response.data.sessions);
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

  const joinLiveSession = async (sessionId: string) => {
    try {
      await api.post(`/sessions/live/${sessionId}/join`);
      navigate(`/live/${sessionId}`);
    } catch (error) {
      console.error('Error joining session:', error);
      navigate(`/live/${sessionId}`);
    }
  };

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
            <div className="grid">
              {sessions.map((session) => (
                <div key={session.id} className="card">
                  <h3>{session.title}</h3>
                  <p>{session.description || 'No description'}</p>
                  <p className="text-muted">Teacher: {session.teacher_name}</p>
                  <p className="text-muted">
                    Scheduled: {new Date(session.scheduled_at).toLocaleString()}
                  </p>
                  <p className="text-muted">Status: {session.status}</p>
                  {(session.status === 'scheduled' || session.status === 'live') && (
                    <button className="btn btn-primary" onClick={() => joinLiveSession(session.id)}>
                      {session.status === 'live' ? 'Join Live Session' : 'Join Session'}
                    </button>
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

