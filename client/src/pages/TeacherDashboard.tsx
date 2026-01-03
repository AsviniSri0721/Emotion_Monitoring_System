import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LiveSession, liveSessionsApi } from '../api/liveSessions';
import { InterventionVideo, interventionVideosApi } from '../api/interventionVideos';
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
  const [interventionVideos, setInterventionVideos] = useState<InterventionVideo[]>([]);
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [activeTab, setActiveTab] = useState<'videos' | 'sessions' | 'reports' | 'interventions'>('videos');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showInterventionUploadModal, setShowInterventionUploadModal] = useState(false);
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [showEditVideoModal, setShowEditVideoModal] = useState(false);
  const [showEditSessionModal, setShowEditSessionModal] = useState(false);
  const [editingVideo, setEditingVideo] = useState<Video | null>(null);
  const [editingSession, setEditingSession] = useState<LiveSession | null>(null);
  const [uploadForm, setUploadForm] = useState({ title: '', description: '', file: null as File | null });
  const [interventionUploadForm, setInterventionUploadForm] = useState({ title: '', description: '', duration: 60, file: null as File | null });
  const [sessionForm, setSessionForm] = useState({ title: '', description: '', scheduledAt: '', meetUrl: '' });
  const [editVideoForm, setEditVideoForm] = useState({ title: '', description: '' });
  const [editSessionForm, setEditSessionForm] = useState({ title: '', description: '', scheduledAt: '', meetUrl: '' });

  useEffect(() => {
    fetchVideos();
    fetchInterventionVideos();
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

  const fetchInterventionVideos = async () => {
    try {
      const response = await interventionVideosApi.getAll();
      setInterventionVideos(response.videos);
    } catch (error) {
      console.error('Error fetching intervention videos:', error);
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await liveSessionsApi.getAvailable();
      setSessions(response.sessions);
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

  const handleInterventionUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!interventionUploadForm.file) {
      alert('Please select a video file');
      return;
    }

    const formData = new FormData();
    formData.append('video', interventionUploadForm.file);
    formData.append('title', interventionUploadForm.title || 'Untitled');
    formData.append('description', interventionUploadForm.description || '');
    formData.append('duration', interventionUploadForm.duration.toString());

    try {
      await interventionVideosApi.upload(formData);
      alert('Intervention video uploaded successfully!');
      setShowInterventionUploadModal(false);
      setInterventionUploadForm({ title: '', description: '', duration: 60, file: null });
      fetchInterventionVideos();
    } catch (error: any) {
      console.error('Error uploading intervention video:', error);
      alert(error.response?.data?.error || 'Failed to upload video');
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionForm.title || !sessionForm.meetUrl) {
      alert('Title and Meet URL are required');
      return;
    }
    try {
      await liveSessionsApi.create({
        title: sessionForm.title,
        meetUrl: sessionForm.meetUrl,
        startTime: sessionForm.scheduledAt || undefined,
      });
      setShowSessionModal(false);
      setSessionForm({ title: '', description: '', scheduledAt: '', meetUrl: '' });
      fetchSessions();
      alert('Live session created successfully!');
    } catch (error: any) {
      console.error('Error creating session:', error);
      alert(error?.response?.data?.error || 'Failed to create session');
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
      const response = await api.post(`/sessions/live/${sessionId}/end`);
      fetchSessions();
      // Refresh reports after ending session to show newly generated reports
      fetchReports();
      // Show success message with report count
      const reportsGenerated = response.data.reports_generated || 0;
      if (reportsGenerated > 0) {
        alert(`Session ended successfully. ${reportsGenerated} engagement report(s) generated.`);
      } else {
        alert('Session ended successfully. No emotion data was recorded for this session.');
      }
    } catch (error) {
      console.error('Error ending session:', error);
      alert('Failed to end session. Please try again.');
    }
  };

  const deleteVideo = async (videoId: string) => {
    if (!window.confirm('Are you sure you want to delete this video? This action cannot be undone.')) {
      return;
    }
    try {
      await api.delete(`/videos/${videoId}`);
      alert('Video deleted successfully!');
      fetchVideos();
    } catch (error: any) {
      console.error('Error deleting video:', error);
      alert(error.response?.data?.error || 'Failed to delete video');
    }
  };

  const editVideo = (video: Video) => {
    setEditingVideo(video);
    setEditVideoForm({ title: video.title, description: video.description || '' });
    setShowEditVideoModal(true);
  };

  const handleUpdateVideo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingVideo) return;
    try {
      await api.put(`/videos/${editingVideo.id}`, {
        title: editVideoForm.title,
        description: editVideoForm.description,
      });
      alert('Video updated successfully!');
      setShowEditVideoModal(false);
      setEditingVideo(null);
      fetchVideos();
    } catch (error: any) {
      console.error('Error updating video:', error);
      alert(error.response?.data?.error || 'Failed to update video');
    }
  };

  const deleteSession = async (sessionId: string) => {
    if (!window.confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
      return;
    }
    try {
      await liveSessionsApi.delete(sessionId);
      alert('Session deleted successfully!');
      fetchSessions();
    } catch (error: any) {
      console.error('Error deleting session:', error);
      alert(error.response?.data?.error || 'Failed to delete session');
    }
  };

  const editSession = (session: LiveSession) => {
    setEditingSession(session);
    setEditSessionForm({
      title: session.title,
      description: session.description || '',
      scheduledAt: session.scheduled_at ? new Date(session.scheduled_at).toISOString().slice(0, 16) : '',
      meetUrl: session.meet_url || '',
    });
    setShowEditSessionModal(true);
  };

  const handleUpdateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingSession) return;
    try {
      await liveSessionsApi.update(editingSession.id, {
        title: editSessionForm.title,
        description: editSessionForm.description,
        scheduledAt: editSessionForm.scheduledAt || undefined,
        meetUrl: editSessionForm.meetUrl || undefined,
      });
      alert('Session updated successfully!');
      setShowEditSessionModal(false);
      setEditingSession(null);
      fetchSessions();
    } catch (error: any) {
      console.error('Error updating session:', error);
      alert(error.response?.data?.error || 'Failed to update session');
    }
  };

  const deleteReport = async (reportId: string) => {
    if (!window.confirm('Are you sure you want to delete this report? This action cannot be undone.')) {
      return;
    }
    try {
      await api.delete(`/reports/${reportId}`);
      alert('Report deleted successfully!');
      fetchReports();
    } catch (error: any) {
      console.error('Error deleting report:', error);
      alert(error.response?.data?.error || 'Failed to delete report');
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
          <button
            className={activeTab === 'interventions' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('interventions')}
          >
            Interventions
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
                  <div className="card-actions">
                    <button className="btn btn-primary" onClick={() => editVideo(video)}>
                      Edit
                    </button>
                    <button className="btn btn-danger" onClick={() => deleteVideo(video.id)}>
                      Delete
                    </button>
                  </div>
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
                    Scheduled: {session.scheduled_at ? new Date(session.scheduled_at).toLocaleString() : 'Not scheduled'}
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
                    <button className="btn btn-primary" onClick={() => editSession(session)}>
                      Edit
                    </button>
                    <button className="btn btn-danger" onClick={() => deleteSession(session.id)}>
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'reports' && (
          <div>
            <h2>Engagement Reports (Live Sessions Only)</h2>
            <p style={{ color: '#666', marginBottom: '1rem', fontSize: '0.9rem' }}>
              Reports are generated from real-time student monitoring during live classes. Recorded content is excluded to ensure accuracy and ethical data usage.
            </p>
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
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            className="btn btn-sm btn-primary"
                            onClick={() =>
                              navigate(`/report/${report.session_type}/${report.session_id}`)
                            }
                          >
                            View Details
                          </button>
                          <button
                            className="btn btn-sm btn-danger"
                            onClick={() => deleteReport(report.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'interventions' && (
          <div>
            <div className="section-header">
              <h2>Intervention Content</h2>
              <button className="btn btn-primary" onClick={() => setShowInterventionUploadModal(true)}>
                Upload Intervention
              </button>
            </div>
            <div className="grid">
              {interventionVideos.map((video) => (
                <div key={video.id} className="card">
                  <h3>{video.title}</h3>
                  <p>{video.description || 'No description'}</p>
                  <p><strong>Redirect Duration:</strong> {video.duration}s</p>
                  <p className="text-muted">Uploaded: {new Date(video.created_at).toLocaleDateString()}</p>
                </div>
              ))}
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
                <label>Google Meet URL </label>
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

      {showInterventionUploadModal && (
        <div className="modal-overlay" onClick={() => setShowInterventionUploadModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Upload Intervention Video</h2>
            <form onSubmit={handleInterventionUpload}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={interventionUploadForm.title}
                  onChange={(e) => setInterventionUploadForm({ ...interventionUploadForm, title: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={interventionUploadForm.description}
                  onChange={(e) => setInterventionUploadForm({ ...interventionUploadForm, description: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Redirect Duration (seconds)</label>
                <input
                  type="number"
                  min="5"
                  max="300"
                  value={interventionUploadForm.duration}
                  onChange={(e) => setInterventionUploadForm({ ...interventionUploadForm, duration: parseInt(e.target.value) || 60 })}
                  required
                />
                <small className="text-muted">How long student must watch before being redirected back.</small>
              </div>
              <div className="form-group">
                <label>Video File</label>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setInterventionUploadForm({ ...interventionUploadForm, file: e.target.files?.[0] || null })}
                  required
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowInterventionUploadModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">Upload</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showEditVideoModal && editingVideo && (
        <div className="modal-overlay" onClick={() => setShowEditVideoModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Edit Video</h2>
            <form onSubmit={handleUpdateVideo}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={editVideoForm.title}
                  onChange={(e) => setEditVideoForm({ ...editVideoForm, title: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={editVideoForm.description}
                  onChange={(e) => setEditVideoForm({ ...editVideoForm, description: e.target.value })}
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowEditVideoModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">Update</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showEditSessionModal && editingSession && (
        <div className="modal-overlay" onClick={() => setShowEditSessionModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Edit Live Session</h2>
            <form onSubmit={handleUpdateSession}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  value={editSessionForm.title}
                  onChange={(e) => setEditSessionForm({ ...editSessionForm, title: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  value={editSessionForm.description}
                  onChange={(e) => setEditSessionForm({ ...editSessionForm, description: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Scheduled At</label>
                <input
                  type="datetime-local"
                  value={editSessionForm.scheduledAt}
                  onChange={(e) => setEditSessionForm({ ...editSessionForm, scheduledAt: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>Google Meet URL</label>
                <input
                  type="url"
                  value={editSessionForm.meetUrl}
                  onChange={(e) => setEditSessionForm({ ...editSessionForm, meetUrl: e.target.value })}
                  placeholder="https://meet.google.com/..."
                />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowEditSessionModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">Update</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeacherDashboard;

