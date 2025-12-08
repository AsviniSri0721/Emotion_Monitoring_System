"""
Routes for live sessions
"""
from flask import Blueprint
from flask_jwt_extended import jwt_required
from live_sessions.controller import (
    create_live_session,
    get_available_sessions,
    stream_emotion,
    get_session_report
)

bp = Blueprint('live_sessions', __name__)

@bp.route('/create', methods=['POST'])
@jwt_required()
def create_live_session_route():
    """Create a new live session"""
    return create_live_session()

@bp.route('/available', methods=['GET'])
@jwt_required()
def get_available_sessions_route():
    """Get all available live sessions"""
    return get_available_sessions()

@bp.route('/<session_id>/stream', methods=['POST'])
@jwt_required()
def stream_emotion_route(session_id: str):
    """Stream emotion detection for live session"""
    return stream_emotion(session_id)

@bp.route('/<session_id>/report', methods=['GET'])
@jwt_required()
def get_session_report_route(session_id: str):
    """Get analytics report for a live session"""
    return get_session_report(session_id)



