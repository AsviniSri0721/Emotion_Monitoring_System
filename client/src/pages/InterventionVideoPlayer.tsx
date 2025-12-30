import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { interventionVideosApi, InterventionVideo } from '../api/interventionVideos';
import api from '../services/api';
import './VideoPlayer.css'; // Reuse video player styles or create new ones

const InterventionVideoPlayer: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const returnUrl = searchParams.get('returnUrl') || '/student/dashboard';

    const [video, setVideo] = useState<InterventionVideo | null>(null);
    const [timeLeft, setTimeLeft] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);
    const [isPlaying, setIsPlaying] = useState(true);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        const fetchVideo = async () => {
            try {
                if (!id) return;
                const response = await interventionVideosApi.getOne(id);
                setVideo(response.video);
                setTimeLeft(response.video.duration);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching intervention video:', error);
                // Fallback or error state
                alert('Failed to load intervention video.');
                navigate(returnUrl);
            }
        };

        fetchVideo();
    }, [id, navigate, returnUrl]);

    // Timer logic
    useEffect(() => {
        if (timeLeft === null || timeLeft <= 0) return;

        timerRef.current = setInterval(() => {
            setTimeLeft((prev) => {
                if (prev === null || prev <= 1) {
                    // Time is up!
                    handleTimeUp();
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [timeLeft]);

    const handleTimeUp = async () => {
        if (timerRef.current) clearInterval(timerRef.current);

        // Retrieve interventionSessionId from URL
        const interventionSessionId = searchParams.get('interventionSessionId');

        if (interventionSessionId) {
            try {
                // Call complete API
                // Assuming we use the raw API or we can add it to interventionVideosApi.
                // Using raw api here for direct access as interventionVideosApi handles uploaded videos, not session records.
                // We need to import 'api' from services.
                const duration = video?.duration || 0;
                await api.post(`/interventions/${interventionSessionId}/complete`, { duration });
                console.log('Intervention record completed:', interventionSessionId);
            } catch (e) {
                console.error('Failed to complete intervention record:', e);
            }
        }

        // Force stop video
        if (videoRef.current) {
            videoRef.current.pause();
        }
        // Redirect immediately
        navigate(returnUrl);
    };

    /* 
     * Requirement: "if the student pause the video or anything else, the timer will not get affact."
     * So we don't need to pause the timer on video pause.
     */

    if (loading || !video) {
        return <div className="loading">Loading intervention...</div>;
    }



    const togglePlay = () => {
        if (videoRef.current) {
            if (videoRef.current.paused) {
                videoRef.current.play();
                setIsPlaying(true);
            } else {
                videoRef.current.pause();
                setIsPlaying(false);
            }
        }
    };

    return (
        <div className="video-player-container intervention-mode" style={{ background: '#000', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div className="intervention-header" style={{ position: 'fixed', top: 0, left: 0, right: 0, padding: '1rem', background: 'rgba(0,0,0,0.8)', zIndex: 100, display: 'flex', justifyContent: 'space-between', color: 'white' }}>
                <h2>Intervention: {video.title}</h2>
                <div className="timer" style={{ fontSize: '1.5rem', fontWeight: 'bold', color: timeLeft && timeLeft < 10 ? '#ff4d4d' : '#4caf50' }}>
                    Time Remaining: {timeLeft}s
                </div>
            </div>

            <div className="video-wrapper" style={{ width: '80%', maxWidth: '1000px' }}>
                <video
                    ref={videoRef}
                    src={`${process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000'}/uploads/${video.file_path}`}
                    autoPlay
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    controls={false}
                    style={{ width: '100%', borderRadius: '8px' }}
                />
                <div className="controls" style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                    <button
                        className="btn btn-primary"
                        onClick={togglePlay}
                        style={{ minWidth: '120px' }}
                    >
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>
                </div>
                <p style={{ color: '#aaa', textAlign: 'center', marginTop: '1rem' }}>
                    You must complete the intervention to continue.
                    Timer runs even if paused.
                </p>
            </div>
        </div>
    );
};

export default InterventionVideoPlayer;
