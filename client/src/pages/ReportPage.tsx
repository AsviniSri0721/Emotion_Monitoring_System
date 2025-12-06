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
}

const ReportPage: React.FC = () => {
  const { sessionType, sessionId } = useParams<{ sessionType: string; sessionId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sessionType && sessionId) {
      fetchReport();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionType, sessionId]);

  const fetchReport = async () => {
    try {
      setLoading(true);
      const response = await api.post(`/reports/generate/${sessionType}/${sessionId}`, {
        studentId: user?.id,
      });
      setReport(response.data.report);
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

  // Prepare data for line chart (concentration over time)
  const concentrationData =
    report.timeline?.map((point) => ({
      time: point.timestamp,
      concentration: point.concentration,
      engagement: point.engagement_score * 100,
    })) || [];

  return (
    <div className="report-container">
      <div className="report-header">
        <button className="btn btn-secondary" onClick={() => navigate(-1)}>
          Back
        </button>
        <h1>Engagement Report</h1>
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
                  <XAxis dataKey="time" label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Concentration (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
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

        {/* Emotion Percentages */}
        <div className="report-details">
          <h2>Emotion Breakdown</h2>
          <div className="emotion-breakdown">
            <div className="emotion-item">
              <span className="emotion-label">Focused:</span>
              <span className="emotion-value">{report.focus_percentage.toFixed(1)}%</span>
            </div>
            <div className="emotion-item">
              <span className="emotion-label">Bored:</span>
              <span className="emotion-value">{report.boredom_percentage.toFixed(1)}%</span>
            </div>
            <div className="emotion-item">
              <span className="emotion-label">Confused:</span>
              <span className="emotion-value">{report.confusion_percentage.toFixed(1)}%</span>
            </div>
            <div className="emotion-item">
              <span className="emotion-label">Sleepy:</span>
              <span className="emotion-value">{report.sleepiness_percentage.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportPage;

