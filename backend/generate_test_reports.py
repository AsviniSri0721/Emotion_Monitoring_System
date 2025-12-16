"""
Script to generate test engagement reports for demonstration purposes.
This creates sample emotion data and generates reports so you can see the Reports tab populated.
"""

import os
import sys
from dotenv import load_dotenv
from services.database import execute_query, init_db
import uuid
from datetime import datetime, timedelta
import json

load_dotenv()

def generate_uuid_str():
    """Generate UUID string for database"""
    return str(uuid.uuid4())

def create_test_reports():
    """Create test emotion data and generate reports"""
    print("=" * 60)
    print("Generating Test Engagement Reports")
    print("=" * 60)
    
    # Initialize database
    try:
        init_db()
        print("✓ Database connection established")
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return
    
    # Get a teacher and student from database
    try:
        teacher = execute_query(
            "SELECT id, email, first_name, last_name FROM users WHERE role = 'teacher' LIMIT 1",
            fetch_one=True
        )
        if not teacher:
            print("✗ No teacher found in database. Please create a teacher account first.")
            return
        
        student = execute_query(
            "SELECT id, email, first_name, last_name FROM users WHERE role = 'student' LIMIT 1",
            fetch_one=True
        )
        if not student:
            print("✗ No student found in database. Please create a student account first.")
            return
        
        print(f"✓ Using teacher: {teacher[2]} {teacher[3]} ({teacher[1]})")
        print(f"✓ Using student: {student[2]} {student[3]} ({student[1]})")
        
    except Exception as e:
        print(f"✗ Error fetching users: {str(e)}")
        return
    
    # Get or create a video
    try:
        video = execute_query(
            "SELECT id, title FROM videos LIMIT 1",
            fetch_one=True
        )
        if not video:
            print("✗ No videos found. Please upload a video first.")
            return
        
        video_id = video[0]
        video_title = video[1]
        print(f"✓ Using video: {video_title} (ID: {video_id})")
        
    except Exception as e:
        print(f"✗ Error fetching video: {str(e)}")
        return
    
    # Create test emotion data for the video session
    print("\nCreating test emotion data...")
    session_id = video_id
    student_id = student[0]
    
    # Generate emotion timeline (simulate 2 minutes of data, every 2 seconds)
    emotions_sequence = ['focused', 'focused', 'neutral', 'focused', 'bored', 'confused', 'focused', 'focused', 'sleepy', 'focused']
    engagement_scores = [0.8, 0.85, 0.6, 0.9, 0.4, 0.3, 0.75, 0.8, 0.35, 0.85]
    
    emotion_data_count = 0
    for i, (emotion, engagement) in enumerate(zip(emotions_sequence, engagement_scores)):
        try:
            emotion_id = generate_uuid_str()
            timestamp = i * 2  # 2 seconds apart
            
            execute_query(
                """INSERT INTO emotion_data 
                   (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (emotion_id, 'recorded', session_id, student_id, emotion, 0.75, timestamp, engagement)
            )
            emotion_data_count += 1
        except Exception as e:
            print(f"  Warning: Could not insert emotion data {i}: {str(e)}")
    
    print(f"✓ Created {emotion_data_count} emotion data points")
    
    # Generate engagement report
    print("\nGenerating engagement report...")
    try:
        # Calculate statistics
        total = len(emotions_sequence)
        emotion_counts = {}
        for emotion in emotions_sequence:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        overall_engagement = sum(engagement_scores) / len(engagement_scores)
        average_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Count engagement drops
        drops = 0
        prev_eng = 1.0
        for eng in engagement_scores:
            if eng < 0.5 and prev_eng >= 0.5:
                drops += 1
            prev_eng = eng
        
        # Calculate percentages
        focus_pct = (emotion_counts.get('focused', 0) / total) * 100
        boredom_pct = (emotion_counts.get('bored', 0) / total) * 100
        confusion_pct = (emotion_counts.get('confused', 0) / total) * 100
        sleepiness_pct = (emotion_counts.get('sleepy', 0) / total) * 100
        
        # Create timeline JSON
        timeline = [{'emotion': emotions_sequence[i], 'timestamp': i * 2} for i in range(len(emotions_sequence))]
        timeline_json = json.dumps(timeline)
        behavior_summary = f"Engagement: {overall_engagement:.2%}, Drops: {drops}"
        
        # Check if report exists
        existing = execute_query(
            """SELECT id FROM engagement_reports 
               WHERE session_type = 'recorded' AND session_id = %s AND student_id = %s""",
            (session_id, student_id),
            fetch_one=True
        )
        
        report_id = existing[0] if existing else generate_uuid_str()
        
        if existing:
            # Update existing report
            execute_query(
                """UPDATE engagement_reports SET
                   overall_engagement = %s,
                   average_emotion = %s,
                   engagement_drops = %s,
                   focus_percentage = %s,
                   boredom_percentage = %s,
                   confusion_percentage = %s,
                   sleepiness_percentage = %s,
                   emotional_timeline = %s,
                   behavior_summary = %s
                   WHERE id = %s""",
                (overall_engagement, average_emotion, drops, focus_pct, boredom_pct,
                 confusion_pct, sleepiness_pct, timeline_json, behavior_summary, report_id)
            )
            print(f"✓ Updated existing report (ID: {report_id})")
        else:
            # Insert new report
            execute_query(
                """INSERT INTO engagement_reports 
                   (id, session_type, session_id, student_id, overall_engagement, average_emotion,
                    engagement_drops, focus_percentage, boredom_percentage, confusion_percentage,
                    sleepiness_percentage, emotional_timeline, behavior_summary)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (report_id, 'recorded', session_id, student_id, overall_engagement, average_emotion,
                 drops, focus_pct, boredom_pct, confusion_pct, sleepiness_pct, timeline_json, behavior_summary)
            )
            print(f"✓ Created new report (ID: {report_id})")
        
        print(f"\nReport Summary:")
        print(f"  Student: {student[2]} {student[3]}")
        print(f"  Session: {video_title}")
        print(f"  Overall Engagement: {overall_engagement:.1%}")
        print(f"  Average Emotion: {average_emotion}")
        print(f"  Engagement Drops: {drops}")
        print(f"  Focus: {focus_pct:.1f}% | Boredom: {boredom_pct:.1f}% | Confusion: {confusion_pct:.1f}% | Sleepiness: {sleepiness_pct:.1f}%")
        
    except Exception as e:
        print(f"✗ Error generating report: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    print("\n" + "=" * 60)
    print("✓ Test report generated successfully!")
    print("=" * 60)
    print("\nRefresh the Reports tab in your browser to see the new report.")
    print("=" * 60)

if __name__ == '__main__':
    create_test_reports()















