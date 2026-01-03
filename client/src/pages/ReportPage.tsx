import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import './ReportPage.css';

interface ConcentrationEvent {
  type: 'drop';
  start_timestamp: number;
  end_timestamp: number;
  duration_seconds: number;
  start_concentration: number;
  recovery_concentration: number | null;
}

interface ConcentrationAnalysis {
  total_drops: number;
  total_drop_duration_seconds: number;
  average_drop_duration_seconds: number;
  longest_drop: ConcentrationEvent | null;
  events: ConcentrationEvent[];
}

interface EmotionSegment {
  emotion: string;
  start_timestamp: number;
  end_timestamp: number;
  start_time: string;
  end_time: string;
  duration_seconds: number;
}

interface ReportData {
  id: string;
  overall_engagement: number;
  average_concentration?: number;
  average_emotion: string;
  engagement_drops: number;
  concentration_drops?: number;
  focus_percentage: number;
  boredom_percentage: number;
  confusion_percentage: number;
  sleepiness_percentage: number;
  timeline?: Array<{
    emotion: string;
    timestamp: number;
    concentration: number;
    engagement_score: number;
  }>;
  concentration_analysis?: ConcentrationAnalysis;
  emotion_segments?: EmotionSegment[];
  session_id?: string;
  student_id?: string;
}

const ReportPage: React.FC = () => {
  const { sessionType, sessionId } = useParams<{ sessionType: string; sessionId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const deleteReport = async () => {
    if (!report?.id) return;
    if (!window.confirm('Are you sure you want to delete this report? This action cannot be undone.')) {
      return;
    }
    try {
      await api.delete(`/reports/${report.id}`);
      alert('Report deleted successfully!');
      navigate(-1);
    } catch (error: any) {
      console.error('Error deleting report:', error);
      alert(error.response?.data?.error || 'Failed to delete report');
    }
  };

  useEffect(() => {
    if (sessionType && sessionId) {
      fetchReport();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionType, sessionId]);

  const fetchReport = async () => {
    try {
      setLoading(true);
      // Get report directly from engagement_reports table (this has the correct values)
      if (sessionType && sessionId) {
        const response = await api.get(`/reports/session/${sessionType}/${sessionId}`);
        const reportData = response.data.report;
        
        // Calculate concentration analysis from timeline if available
        let concentrationAnalysis: ConcentrationAnalysis | undefined = undefined;
        if (reportData.timeline && reportData.timeline.length > 0) {
          const concentrations = reportData.timeline.map((t: any) => t.concentration || 50.0);
          const lowThreshold = 40;
          const highThreshold = 60;
          
          let inLowConcentration = false;
          let lowStart = 0;
          let lowStartTimestamp = 0;
          const events: ConcentrationEvent[] = [];
          
          for (let i = 0; i < concentrations.length; i++) {
            const conc = concentrations[i];
            const timestamp = reportData.timeline[i].timestamp || i;
            
            if (conc < lowThreshold && !inLowConcentration) {
              // Start of drop
              inLowConcentration = true;
              lowStart = i;
              lowStartTimestamp = timestamp;
            } else if (conc >= highThreshold && inLowConcentration) {
              // Recovery
              inLowConcentration = false;
              events.push({
                type: 'drop',
                start_timestamp: lowStartTimestamp,
                end_timestamp: timestamp,
                duration_seconds: timestamp - lowStartTimestamp,
                start_concentration: concentrations[lowStart] || 50.0,
                recovery_concentration: conc,
              });
            }
          }
          
          // Handle ongoing drop at end
          if (inLowConcentration) {
            const lastTimestamp = reportData.timeline[reportData.timeline.length - 1].timestamp || reportData.timeline.length - 1;
            events.push({
              type: 'drop',
              start_timestamp: lowStartTimestamp,
              end_timestamp: lastTimestamp,
              duration_seconds: lastTimestamp - lowStartTimestamp,
              start_concentration: concentrations[lowStart] || 50.0,
              recovery_concentration: null,
            });
          }
          
          const totalDropDuration = events.reduce((sum, e) => sum + e.duration_seconds, 0);
          const avgDropDuration = events.length > 0 ? totalDropDuration / events.length : 0;
          const longestDrop = events.length > 0 ? events.reduce((longest, e) => 
            e.duration_seconds > longest.duration_seconds ? e : longest
          ) : null;
          
          concentrationAnalysis = {
            total_drops: events.length,
            total_drop_duration_seconds: totalDropDuration,
            average_drop_duration_seconds: avgDropDuration,
            longest_drop: longestDrop,
            events: events,
          };
        }
        
        // Transform to match ReportData interface
        setReport({
          id: reportData.id,
          overall_engagement: reportData.overall_engagement,
          average_concentration: reportData.average_concentration || 0,
          average_emotion: reportData.average_emotion,
          engagement_drops: reportData.engagement_drops || 0,
          focus_percentage: reportData.focus_percentage || 0,
          boredom_percentage: reportData.boredom_percentage || 0,
          confusion_percentage: reportData.confusion_percentage || 0,
          sleepiness_percentage: reportData.sleepiness_percentage || 0,
          timeline: reportData.timeline || [],
          emotion_segments: reportData.emotion_segments || [],
          concentration_analysis: concentrationAnalysis,
        });
      } else {
        setError('Invalid session parameters');
      }
      setError(null);
    } catch (err: any) {
      console.error('Error fetching report:', err);
      setError(err?.response?.data?.error || 'Failed to load report');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading report...</div>;
  }

  if (error || !report) {
    return (
      <div className="report-container">
        <div className="error-message">
          <p>{error || 'Report not found'}</p>
          <button className="btn btn-primary" onClick={() => navigate(-1)}>
            Go Back
          </button>
        </div>
      </div>
    );
  }

  // Prepare data for pie chart (emotion distribution)
  const emotionData = [
    { name: 'Focused', value: report.focus_percentage, color: '#4caf50' },
    { name: 'Bored', value: report.boredom_percentage, color: '#ff9800' },
    { name: 'Confused', value: report.confusion_percentage, color: '#f44336' },
    { name: 'Sleepy', value: report.sleepiness_percentage, color: '#9e9e9e' },
  ].filter((item) => item.value > 0);

  // Format timestamp - use actual time if available, otherwise format relative time
  const formatTimestamp = (point: any): string => {
    // If we have actual time from backend, use it
    if (point.time_display) {
      return point.time_display;
    }
    if (point.actual_time) {
      const date = new Date(point.actual_time);
      return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    // Fallback: format relative timestamp (seconds from start)
    const seconds = point.timestamp || 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  // Prepare data for line chart (concentration over time)
  // Sort by timestamp to ensure proper ordering
  const sortedTimeline = report.timeline ? [...report.timeline].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0)) : [];
  const concentrationData = sortedTimeline.map((point) => ({
    time: formatTimestamp(point),
    timeSeconds: point.timestamp || 0, // Keep for sorting/ordering
    concentration: point.concentration || 50.0,
    engagement: (point.engagement_score || 0.5) * 100,
  }));

  return (
    <div className="report-container">
      <div className="report-header">
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button className="btn btn-secondary" onClick={() => navigate(-1)}>
            Back
          </button>
          <h1 style={{ margin: 0, flex: 1 }}>Engagement Report</h1>
          <button className="btn btn-danger" onClick={deleteReport}>
            Delete Report
          </button>
        </div>
      </div>

      <div className="report-content">
        {/* Summary Statistics */}
        <div className="report-summary">
          <div className="stat-card">
            <h3>Overall Engagement</h3>
            <p className="stat-value">{(report.overall_engagement * 100).toFixed(1)}%</p>
          </div>
          {report.average_concentration !== undefined && (
            <div className="stat-card">
              <h3>Average Concentration</h3>
              <p className="stat-value">{report.average_concentration.toFixed(1)}%</p>
            </div>
          )}
          <div className="stat-card">
            <h3>Engagement Drops</h3>
            <p className="stat-value">{report.engagement_drops}</p>
          </div>
          {report.concentration_drops !== undefined && (
            <div className="stat-card">
              <h3>Concentration Drops</h3>
              <p className="stat-value">{report.concentration_drops}</p>
            </div>
          )}
          <div className="stat-card">
            <h3>Dominant Emotion</h3>
            <p className="stat-value">{report.average_emotion}</p>
          </div>
        </div>

        {/* Charts */}
        <div className="report-charts">
          {/* Emotion Distribution Pie Chart */}
          <div className="chart-container">
            <h2>Emotion Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Concentration Timeline */}
          {concentrationData.length > 0 && (
            <div className="chart-container">
              <h2>Concentration Over Time</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={concentrationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="time" 
                    label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
                    type="category"
                    allowDuplicatedCategory={false}
                  />
                  <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip 
                    formatter={(value: number, name: string) => [`${value.toFixed(2)}%`, name]}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="concentration"
                    stroke="#4caf50"
                    strokeWidth={2}
                    name="Concentration"
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="engagement"
                    stroke="#2196f3"
                    strokeWidth={2}
                    name="Engagement"
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Emotion Percentages - Only show non-zero values */}
        <div className="report-details">
          <h2>Emotion Breakdown</h2>
          <div className="emotion-breakdown">
            {report.focus_percentage > 0 && (
              <div className="emotion-item">
                <span className="emotion-label">Focused:</span>
                <span className="emotion-value">{report.focus_percentage.toFixed(1)}%</span>
              </div>
            )}
            {report.boredom_percentage > 0 && (
              <div className="emotion-item">
                <span className="emotion-label">Bored:</span>
                <span className="emotion-value">{report.boredom_percentage.toFixed(1)}%</span>
              </div>
            )}
            {report.confusion_percentage > 0 && (
              <div className="emotion-item">
                <span className="emotion-label">Confused:</span>
                <span className="emotion-value">{report.confusion_percentage.toFixed(1)}%</span>
              </div>
            )}
            {report.sleepiness_percentage > 0 && (
              <div className="emotion-item">
                <span className="emotion-label">Sleepy:</span>
                <span className="emotion-value">{report.sleepiness_percentage.toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Concentration Drop/Recovery Analysis */}
        {report.concentration_analysis && report.concentration_analysis.total_drops > 0 && (
          <div className="report-details">
            <h2>Concentration Analysis</h2>
            <div className="report-summary" style={{ marginBottom: '2rem' }}>
              <div className="stat-card">
                <h3>Total Concentration Drops</h3>
                <p className="stat-value">{report.concentration_analysis.total_drops}</p>
              </div>
              <div className="stat-card">
                <h3>Total Low Concentration Time</h3>
                <p className="stat-value">
                  {Math.floor(report.concentration_analysis.total_drop_duration_seconds / 60)}m{' '}
                  {Math.floor(report.concentration_analysis.total_drop_duration_seconds % 60)}s
                </p>
              </div>
              <div className="stat-card">
                <h3>Average Drop Duration</h3>
                <p className="stat-value">
                  {Math.floor(report.concentration_analysis.average_drop_duration_seconds / 60)}m{' '}
                  {Math.floor(report.concentration_analysis.average_drop_duration_seconds % 60)}s
                </p>
              </div>
              {report.concentration_analysis.longest_drop && (
                <div className="stat-card">
                  <h3>Longest Drop Duration</h3>
                  <p className="stat-value">
                    {Math.floor(report.concentration_analysis.longest_drop.duration_seconds / 60)}m{' '}
                    {Math.floor(report.concentration_analysis.longest_drop.duration_seconds % 60)}s
                  </p>
                </div>
              )}
            </div>

            <div className="concentration-events">
              <h3>Concentration Drop Events</h3>
              <table style={{ width: '100%', marginTop: '1rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #ddd', backgroundColor: '#f5f5f5' }}>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Start Time</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>End Time</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Duration</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Start Concentration</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Recovery Concentration</th>
                  </tr>
                </thead>
                <tbody>
                  {report.concentration_analysis.events.map((event, index) => (
                    <tr key={index} style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '0.75rem' }}>
                        {Math.floor(event.start_timestamp / 60)}m {event.start_timestamp % 60}s
                      </td>
                      <td style={{ padding: '0.75rem' }}>
                        {event.recovery_concentration !== null
                          ? `${Math.floor(event.end_timestamp / 60)}m ${event.end_timestamp % 60}s`
                          : 'Ongoing'}
                      </td>
                      <td style={{ padding: '0.75rem' }}>
                        {Math.floor(event.duration_seconds / 60)}m {Math.floor(event.duration_seconds % 60)}s
                      </td>
                      <td
                        style={{
                          padding: '0.75rem',
                          color: event.start_concentration < 40 ? '#f44336' : '#666',
                          fontWeight: event.start_concentration < 40 ? 'bold' : 'normal',
                        }}
                      >
                        {event.start_concentration.toFixed(1)}%
                      </td>
                      <td
                        style={{
                          padding: '0.75rem',
                          color:
                            event.recovery_concentration && event.recovery_concentration >= 60 ? '#4caf50' : '#666',
                          fontWeight: event.recovery_concentration && event.recovery_concentration >= 60 ? 'bold' : 'normal',
                        }}
                      >
                        {event.recovery_concentration !== null ? `${event.recovery_concentration.toFixed(1)}%` : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Emotion Segments Timeline */}
        {report.emotion_segments && report.emotion_segments.length > 0 && (
          <div className="report-details">
            <h2>Emotion Timeline Segments</h2>
            <p style={{ color: '#666', marginBottom: '1rem', fontSize: '0.9rem' }}>
              Exact timestamps and durations for each emotion state during the session.
            </p>
            <div className="table-container">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #ddd', backgroundColor: '#f5f5f5' }}>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Emotion</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Start Time</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>End Time</th>
                    <th style={{ padding: '0.75rem', textAlign: 'left' }}>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {report.emotion_segments.map((segment, index) => {
                    const getEmotionColor = (emotion: string) => {
                      switch (emotion.toLowerCase()) {
                        case 'focused':
                          return '#4caf50';
                        case 'bored':
                          return '#ff9800';
                        case 'confused':
                          return '#f44336';
                        case 'sleepy':
                          return '#9e9e9e';
                        case 'neutral':
                          return '#2196f3';
                        case 'frustrated':
                          return '#e91e63';
                        default:
                          return '#666';
                      }
                    };

                    const formatDuration = (seconds: number) => {
                      if (seconds < 60) {
                        return `${seconds}s`;
                      } else if (seconds < 3600) {
                        const mins = Math.floor(seconds / 60);
                        const secs = seconds % 60;
                        return `${mins}m ${secs}s`;
                      } else {
                        const hours = Math.floor(seconds / 3600);
                        const mins = Math.floor((seconds % 3600) / 60);
                        const secs = seconds % 60;
                        return `${hours}h ${mins}m ${secs}s`;
                      }
                    };

                    return (
                      <tr key={index} style={{ borderBottom: '1px solid #eee' }}>
                        <td style={{ padding: '0.75rem' }}>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '0.25rem 0.5rem',
                              borderRadius: '4px',
                              backgroundColor: getEmotionColor(segment.emotion),
                              color: 'white',
                              fontWeight: 'bold',
                              textTransform: 'capitalize',
                            }}
                          >
                            {segment.emotion}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem' }}>{segment.start_time}</td>
                        <td style={{ padding: '0.75rem' }}>{segment.end_time}</td>
                        <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>
                          {formatDuration(segment.duration_seconds)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportPage;

