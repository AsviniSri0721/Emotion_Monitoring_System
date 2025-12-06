import React from 'react';
import './EngagementMeter.css';

interface EngagementMeterProps {
  score: number; // 0-100 concentration score
  emotion: string;
  size?: number; // Size of the circular meter in pixels
}

const EngagementMeter: React.FC<EngagementMeterProps> = ({
  score,
  emotion,
  size = 120,
}) => {
  // Determine color based on score
  const getColor = (score: number): string => {
    if (score >= 70) return '#4caf50'; // Green - high concentration
    if (score >= 40) return '#ff9800'; // Orange - medium concentration
    return '#f44336'; // Red - low concentration
  };

  // Map emotion to display text
  const getEmotionLabel = (emotion: string): string => {
    const emotionMap: Record<string, string> = {
      focused: 'Focused',
      happy: 'Happy',
      neutral: 'Neutral',
      confused: 'Confused',
      frustrated: 'Frustrated',
      bored: 'Bored',
      sleepy: 'Sleepy',
    };
    return emotionMap[emotion] || emotion.charAt(0).toUpperCase() + emotion.slice(1);
  };

  const color = getColor(score);
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="engagement-meter" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="engagement-meter-svg">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e0e0e0"
          strokeWidth="8"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="engagement-meter-progress"
        />
      </svg>
      <div className="engagement-meter-content">
        <div className="engagement-meter-score" style={{ color }}>
          {Math.round(score)}%
        </div>
        <div className="engagement-meter-emotion">{getEmotionLabel(emotion)}</div>
      </div>
    </div>
  );
};

export default EngagementMeter;

