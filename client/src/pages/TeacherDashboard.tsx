import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import './Dashboard.css';

interface Video {
  id: string;
  title: string;
  description: string;
  file_path: string;
  created_at: string;
}

interface LiveSession {
  id: string;
  title: string;
  description: string;
  meet_url: string;
  scheduled_at: string;
  status: string;
}

interface Report {
  id: string;
  student_name: string;
  session_title: string;
  session_type: string;
  session_id: string;
  overall_engagement: number;
  average_emotion: string;
  engagement_drops: number;
  focus_percentage?: number;
  concentration_drops?: number;
  generated_at: string;
}

const TeacherDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [videos, setVideos] = useState<Video[]>([]);
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [activeTab, setActiveTab] = useState<'videos' | 'sessions' | 'reports'>('videos');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [uploadForm, setUploadForm] = useState({ title: '', description: '', file: null as File | null });
  const [sessionForm, setSessionForm] = useState({ title: '', description: '', scheduledAt: '', meetUrl: '' });

  useEffect(() => {
    fetchVideos();
    fetchSessions();
    fetchReports();
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

  const fetchReports = async () => {
    try {
      const response = await api.get('/reports/dashboard/all');
      setReports(response.data.reports);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const handleVideoUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadForm.file) {
      alert('Please select a video file');
      return;
    }

    // Validate file size (500MB)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (uploadForm.file.size > maxSize) {
      alert('File size exceeds 500MB limit');
      return;
    }

    // Validate file type
    const allowedExtensions = ['mp4', 'webm', 'ogg', 'avi', 'mov'];
    const fileExtension = uploadForm.file.name.split('.').pop()?.toLowerCase();
    
    if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
      alert('Invalid file type. Please use: mp4, webm, ogg, avi, or mov');
      return;
    }

    const formData = new FormData();
    formData.append('video', uploadForm.file);
    formData.append('title', uploadForm.title || 'Untitled');
    formData.append('description', uploadForm.description || '');

    try {
      console.log('Uploading video:', {
        title: uploadForm.title,
        filename: uploadForm.file.name,
        size: uploadForm.file.size,
        type: uploadForm.file.type
      });

      const response = await api.post('/videos/upload', formData);
      
      console.log('Upload response:', response.data);
      alert('Video uploaded successfully!');
      setShowUploadModal(false);
      setUploadForm({ title: '', description: '', file: null });
      fetchVideos();
    } catch (error: any) {
      console.error('Error uploading video:', error);
      console.error('Error response:', error.response);
      
      let errorMessage = 'Failed to upload video';
      if (error.response) {
        errorMessage = error.response.data?.error || error.response.data?.details || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = 'No response from server. Check if backend is running on port 5000';
      } else {
        errorMessage = error.message;
      }
      
      alert(`Upload failed: ${errorMessage}\n\nCheck:\n1. Backend is running on port 5000\n2. Database connection is working\n3. File size is under 500MB\n4. File format is supported (mp4, webm, ogg, avi, mov)`);
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.post('/sessions/live', {
        title: sessionForm.title,
        description: sessionForm.description,
        scheduledAt: sessionForm.scheduledAt,
        meetUrl: sessionForm.meetUrl,
      });
      setShowSessionModal(false);
      setSessionForm({ title: '', description: '', scheduledAt: '', meetUrl: '' });
      fetchSessions();
    } catch (error) {
      console.error('Error creating session:', error);
      alert('Failed to create session');
    }
  };

  const startSession = async (sessionId: string) => {
    try {
      await api.post(`/sessions/live/${sessionId}/start`);
      fetchSessions();
    } catch (error) {
      console.error('Error starting session:', error);
    }
  };

  const endSession = async (sessionId: string) => {
    try {
      await api.post(`/sessions/live/${sessionId}/end`);
      fetchSessions();
    } catch (error) {
      console.error('Error ending session:', error);
    }
  };

  return (
    <div className="dashboard">
      <div className="header">
        <h1>Teacher Dashboard</h1>
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
            Videos
          </button>
          <button
            className={activeTab === 'sessions' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('sessions')}
          >
            Live Sessions
          </button>
          <button
            className={activeTab === 'reports' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('reports')}
          >
            Reports
          </button>
        </div>

        {activeTab === 'videos' && (
          <div>
            <div className="section-header">
              <h2>Recorded Lectures</h2>
              <button className="btn btn-primary" onClick={() => setShowUploadModal(true)}>
                Upload Video
              </button>
            </div>
            <div className="grid">
              {videos.map((video) => (
                <div key={video.id} className="card">
                  <h3>{video.title}</h3>
                  <p>{video.description || 'No description'}</p>
                  <p className="text-muted">Uploaded: {new Date(video.created_at).toLocaleDateString()}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'sessions' && (
          <div>
            <div className="section-header">
              <h2>Live Sessions</h2>
              <button className="btn btn-primary" onClick={() => setShowSessionModal(true)}>
                Create Session
              </button>
            </div>
            <div className="grid">
              {sessions.map((session) => (
                <div key={session.id} className="card">
                  <h3>{session.title}</h3>
                  <p>{session.description || 'No description'}</p>
                  <p className="text-muted">
                    Scheduled: {new Date(session.scheduled_at).toLocaleString()}
                  </p>
                  <p className="text-muted">Status: {session.status}</p>
                  {session.meet_url && (
                    <a href={session.meet_url} target="_blank" rel="noopener noreferrer">
                      Join Google Meet
                    </a>
                  )}
                  <div className="card-actions">
                    {session.status === 'scheduled' && (
                      <button className="btn btn-primary" onClick={() => startSession(session.id)}>
                        Start Session
                      </button>
                    )}
                    {session.status === 'live' && (
                      <button className="btn btn-danger" onClick={() => endSession(session.id)}>
                        End Session
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'reports' && (
          <div>
            <h2>Engagement Reports</h2>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Student</th>
                    <th>Session</th>
                    <th>Engagement</th>
                    <th>Concentration</th>
                    <th>Emotion</th>
                    <th>Drops</th>
                    <th>Date</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr key={report.id}>
                      <td>{report.student_name}</td>
                      <td>{report.session_title}</td>
                      <td>{(report.overall_engagement * 100).toFixed(1)}%</td>
                      <td>
                        {report.focus_percentage !== undefined
                          ? `${report.focus_percentage.toFixed(1)}%`
                          : 'N/A'}
                      </td>
                      <td>{report.average_emotion}</td>
                      <td>{report.engagement_drops}</td>
                      <td>{new Date(report.generated_at).toLocaleDateString()}</td>
                      <td>
                        <button
                          className="btn btn-sm btn-primary"
                          onClick={() =>
                            navigate(`/report/${report.session_type}/${report.session_id}`)
                          }
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {showUploadModal && (
        <div className="modal-overlay" onClick={() => setShowUploadModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Upload Video</h2>
            <form onSubmit={handleVideoUpload}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={uploadForm.title}
                  onChange={(e) => setUploadForm({ ...uploadForm, title: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={uploadForm.description}
                  onChange={(e) => setUploadForm({ ...uploadForm, description: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Video File</label>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setUploadForm({ ...uploadForm, file: e.target.files?.[0] || null })}
                  required
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowUploadModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">Upload</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showSessionModal && (
        <div className="modal-overlay" onClick={() => setShowSessionModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Create Live Session</h2>
            <form onSubmit={handleCreateSession}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={sessionForm.title}
                  onChange={(e) => setSessionForm({ ...sessionForm, title: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={sessionForm.description}
                  onChange={(e) => setSessionForm({ ...sessionForm, description: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Scheduled At</label>
                <input
                  type="datetime-local"
                  value={sessionForm.scheduledAt}
                  onChange={(e) => setSessionForm({ ...sessionForm, scheduledAt: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Google Meet URL (optional)</label>
                <input
                  type="url"
                  value={sessionForm.meetUrl}
                  onChange={(e) => setSessionForm({ ...sessionForm, meetUrl: e.target.value })}
                  placeholder="https://meet.google.com/..."
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowSessionModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeacherDashboard;

