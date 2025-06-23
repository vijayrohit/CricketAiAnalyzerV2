import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import uuid
import json
from datetime import datetime

from services.video_processor import VideoProcessor
from services.ball_tracker import BallTracker
from services.pose_analyzer import PoseAnalyzer
from services.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "cricket-analysis-secret-key")

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize services
video_processor = VideoProcessor()
ball_tracker = BallTracker()
pose_analyzer = PoseAnalyzer()
report_generator = ReportGenerator()

# Cache for processed videos
analysis_cache = {}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface."""
    return render_template('index.html')

@app.route('/api-docs')
def api_docs():
    """API documentation page."""
    return render_template('api_docs.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'video_processor': 'active',
            'ball_tracker': 'active',
            'pose_analyzer': 'active',
            'report_generator': 'active'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file for analysis."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Supported formats: MP4, AVI, MOV, MKV'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate video file
        if not video_processor.validate_video(filepath):
            os.remove(filepath)
            return jsonify({'error': 'Invalid video file or corrupted'}), 400
        
        logger.info(f"Video uploaded successfully: {unique_filename}")
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'file_id': file_id,
            'filename': filename,
            'upload_time': datetime.now().isoformat()
        }), 200
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/analyze/bowling/<file_id>', methods=['POST'])
def analyze_bowling(file_id):
    """Analyze bowling performance with ball tracking."""
    try:
        # Check if analysis is cached
        cache_key = f"bowling_{file_id}"
        if cache_key in analysis_cache:
            logger.info(f"Returning cached bowling analysis for {file_id}")
            return jsonify(analysis_cache[cache_key])
        
        # Find video file
        video_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if not video_path:
            return jsonify({'error': 'Video file not found'}), 404
        
        logger.info(f"Starting bowling analysis for {file_id}")
        
        # Process video and track ball
        frames = video_processor.extract_frames(video_path)
        if not frames:
            return jsonify({'error': 'Failed to process video'}), 500
        
        # Track ball throughout the video
        ball_tracking_data = ball_tracker.track_ball(frames)
        
        # Generate bowling report
        bowling_report = report_generator.generate_bowling_report(
            ball_tracking_data, 
            video_processor.get_video_info(video_path)
        )
        
        # Cache the result
        analysis_cache[cache_key] = bowling_report
        
        logger.info(f"Bowling analysis completed for {file_id}")
        return jsonify(bowling_report)
        
    except Exception as e:
        logger.error(f"Bowling analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@app.route('/api/analyze/batting/<file_id>', methods=['POST'])
def analyze_batting(file_id):
    """Analyze batting performance with pose estimation."""
    try:
        # Check if analysis is cached
        cache_key = f"batting_{file_id}"
        if cache_key in analysis_cache:
            logger.info(f"Returning cached batting analysis for {file_id}")
            return jsonify(analysis_cache[cache_key])
        
        # Find video file
        video_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if not video_path:
            return jsonify({'error': 'Video file not found'}), 404
        
        logger.info(f"Starting batting analysis for {file_id}")
        
        # Process video and analyze pose
        frames = video_processor.extract_frames(video_path)
        if not frames:
            return jsonify({'error': 'Failed to process video'}), 500
        
        # Analyze batting pose and technique
        pose_data = pose_analyzer.analyze_batting_pose(frames)
        
        # Generate batting report
        batting_report = report_generator.generate_batting_report(
            pose_data, 
            video_processor.get_video_info(video_path)
        )
        
        # Cache the result
        analysis_cache[cache_key] = batting_report
        
        logger.info(f"Batting analysis completed for {file_id}")
        return jsonify(batting_report)
        
    except Exception as e:
        logger.error(f"Batting analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@app.route('/api/analysis/<file_id>/status', methods=['GET'])
def get_analysis_status(file_id):
    """Get analysis status for a video file."""
    try:
        bowling_cached = f"bowling_{file_id}" in analysis_cache
        batting_cached = f"batting_{file_id}" in analysis_cache
        
        # Check if video file exists
        video_exists = False
        for ext in ALLOWED_EXTENSIONS:
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")):
                video_exists = True
                break
        
        return jsonify({
            'file_id': file_id,
            'video_exists': video_exists,
            'bowling_analysis': {
                'completed': bowling_cached,
                'cached': bowling_cached
            },
            'batting_analysis': {
                'completed': batting_cached,
                'cached': batting_cached
            }
        })
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'error': 'Status check failed'}), 500

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all uploaded videos."""
    try:
        videos = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename != '.gitkeep':
                file_id = filename.rsplit('.', 1)[0]
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_stats = os.stat(file_path)
                
                videos.append({
                    'file_id': file_id,
                    'filename': filename,
                    'size': file_stats.st_size,
                    'upload_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'bowling_analyzed': f"bowling_{file_id}" in analysis_cache,
                    'batting_analyzed': f"batting_{file_id}" in analysis_cache
                })
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        logger.error(f"List videos error: {str(e)}")
        return jsonify({'error': 'Failed to list videos'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
