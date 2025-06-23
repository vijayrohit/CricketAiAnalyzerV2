# Cricket Analysis AI

## Overview

This is a Flask-based web application that provides AI-powered cricket video analysis capabilities. The system analyzes cricket videos to provide detailed insights on both bowling and batting performance using computer vision and machine learning techniques. The application processes uploaded videos to track ball movement, analyze player poses, and generate comprehensive performance reports.

## System Architecture

The application follows a modular Flask architecture with separated concerns:

- **Frontend**: HTML templates with Bootstrap for responsive UI, JavaScript for interactive functionality
- **Backend**: Flask web framework with RESTful API endpoints
- **Processing Pipeline**: Modular service classes for different analysis types
- **Storage**: File-based storage for uploaded and processed videos
- **Deployment**: Gunicorn WSGI server with autoscale deployment target

## Key Components

### Core Application Files
- `main.py`: Application entry point
- `app.py`: Main Flask application with route definitions and service initialization
- `pyproject.toml`: Python project configuration and dependencies

### Service Layer
- `VideoProcessor`: Handles video validation, frame extraction, and metadata collection
- `BallTracker`: Implements ball detection and tracking using OpenCV computer vision
- `PoseAnalyzer`: Uses MediaPipe for human pose estimation and batting technique analysis
- `ReportGenerator`: Creates comprehensive performance reports from analysis data

### Frontend Components
- `templates/index.html`: Main upload interface with drag-and-drop functionality
- `templates/api_docs.html`: API documentation page
- `static/js/main.js`: Client-side JavaScript for file handling and API interactions
- `static/css/style.css`: Custom styling for the application

### Storage Structure
- `uploads/`: Stores uploaded video files
- `processed/`: Stores processed analysis results

## Data Flow

1. **Video Upload**: Users upload cricket videos through the web interface
2. **Video Validation**: System validates format, size, and readability
3. **Analysis Selection**: Users choose between bowling or batting analysis
4. **Processing Pipeline**:
   - Video frames are extracted and preprocessed
   - For bowling: Ball tracking, speed calculation, trajectory analysis
   - For batting: Pose estimation, technique evaluation, stance analysis
5. **Report Generation**: Comprehensive reports with metrics and recommendations
6. **Results Display**: Analysis results presented through web interface

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web framework for API and frontend serving
- **Flask-SQLAlYchemg**: Database ORM (configured but not actively used)
- **Gunicorn**: Production WSGI server
- **Werkzeug**: WSGI utilities and security features

### Computer Vision & AI Libraries
- **OpenCV**: Computer vision library for video processing and ball tracking
- **MediaPipe**: Google's ML framework for pose estimation
- **NumPy**: Numerical computing for data processing

### Infrastructure Dependencies
- **PostgreSQL**: Database system (available but not implemented)
- **OpenSSL**: Security library for encrypted communications

## Deployment Strategy

The application is configured for deployment on Replit with the following setup:

- **Runtime**: Python 3.11 with Nix package manager
- **Server**: Gunicorn with auto-reload for development
- **Scaling**: Autoscale deployment target for production
- **Port Configuration**: Binds to 0.0.0.0:5000 with port reuse
- **File Limits**: 500MB maximum file size for video uploads

The deployment uses parallel workflow execution for better performance and includes health checking capabilities.

## Changelog

```
Changelog:
- June 23, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```