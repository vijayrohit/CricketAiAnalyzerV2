"""Enhanced ball tracking system integrating multiple detection methods and physics modeling."""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from .yolo_detector import YOLOBallDetector
from .trajectory_modeling import BallTrajectoryModel
from .calibration import CameraCalibrator
from .audio_impact_detector import AudioImpactDetector

logger = logging.getLogger(__name__)

class EnhancedBallTracker:
    """
    Comprehensive ball tracking system combining computer vision, physics modeling,
    and audio analysis for professional cricket video analysis.
    """
    
    def __init__(self, video_path: str = None):
        # Initialize detection components
        self.ball_detector = YOLOBallDetector(confidence_threshold=0.4)
        self.trajectory_model = BallTrajectoryModel()
        self.calibrator = CameraCalibrator()
        self.audio_detector = AudioImpactDetector()
        
        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # Tracking state
        self.ball_positions = []
        self.timestamps = []
        self.detection_confidence_history = []
        self.frame_count = 0
        
        # Analysis results
        self.trajectory_prediction = None
        self.hawkeye_data = None
        self.audio_impacts = []
        
        # Video properties
        self.video_fps = 30.0
        self.video_duration = 0.0
        
        if video_path:
            self._initialize_from_video(video_path)
    
    def _initialize_from_video(self, video_path: str):
        """Initialize tracker with video-specific parameters."""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.video_duration = frame_count / self.video_fps
                cap.release()
                
                # Attempt camera calibration from video
                self.camera_matrix, self.dist_coeffs, self.calibrated = (
                    self.calibrator.calibrate_from_video(video_path)
                )
                
                # Extract and analyze audio
                self._analyze_video_audio(video_path)
                
        except Exception as e:
            logger.warning(f"Video initialization failed: {e}")
    
    def _analyze_video_audio(self, video_path: str):
        """Extract and analyze audio for impact detection."""
        try:
            audio_data = self.audio_detector.extract_audio_from_video(video_path)
            if audio_data is not None:
                impacts = self.audio_detector.detect_impacts(audio_data)
                self.audio_impacts = self.audio_detector.synchronize_with_video(
                    impacts, self.video_fps, self.video_duration
                )
                logger.info(f"Detected {len(self.audio_impacts)} audio impact events")
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
    
    def track_ball_in_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Comprehensive ball tracking across multiple frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Complete tracking analysis including trajectory prediction
        """
        self.ball_positions = []
        self.timestamps = []
        self.detection_confidence_history = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / self.video_fps
            
            # Apply camera calibration if available
            if self.calibrated:
                frame = self.calibrator.undistort_frame(frame, self.camera_matrix, self.dist_coeffs)
            
            # Detect ball in current frame
            detections = self.ball_detector.detect_ball(frame)
            
            # Process detection results
            if detections:
                best_detection = detections[0]  # Highest confidence
                ball_position = self._extract_ball_center(best_detection['bbox'])
                confidence = best_detection['confidence']
                
                self.ball_positions.append(ball_position)
                self.detection_confidence_history.append(confidence)
            else:
                # Use trajectory prediction to estimate position
                predicted_pos = self._predict_missing_position(frame_idx)
                self.ball_positions.append(predicted_pos)
                self.detection_confidence_history.append(0.1)  # Low confidence for prediction
            
            self.timestamps.append(timestamp)
            self.frame_count += 1
        
        # Generate comprehensive analysis
        return self._generate_comprehensive_analysis()
    
    def _extract_ball_center(self, bbox: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
        """Extract ball center from bounding box."""
        if bbox:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return (center_x, center_y)
        return None
    
    def _predict_missing_position(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Predict ball position when detection fails."""
        if len(self.ball_positions) < 2:
            return None
        
        # Use recent positions for prediction
        recent_positions = [pos for pos in self.ball_positions[-3:] if pos is not None]
        if len(recent_positions) < 2:
            return None
        
        # Simple linear prediction
        pos1, pos2 = recent_positions[-2], recent_positions[-1]
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        predicted_x = pos2[0] + dx
        predicted_y = pos2[1] + dy
        
        return (predicted_x, predicted_y)
    
    def _generate_comprehensive_analysis(self) -> Dict:
        """Generate complete ball tracking analysis."""
        analysis = {
            'detection_summary': self._generate_detection_summary(),
            'trajectory_analysis': None,
            'hawkeye_prediction': None,
            'physics_analysis': None,
            'audio_correlation': None,
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        # Generate trajectory analysis if sufficient data
        valid_positions = [pos for pos in self.ball_positions if pos is not None]
        if len(valid_positions) >= 3:
            analysis['trajectory_analysis'] = self._analyze_trajectory()
            analysis['hawkeye_prediction'] = self._generate_hawkeye_prediction()
            analysis['physics_analysis'] = self._perform_physics_analysis()
        
        # Correlate with audio impacts
        if self.audio_impacts:
            analysis['audio_correlation'] = self._correlate_with_audio()
        
        return analysis
    
    def _generate_detection_summary(self) -> Dict:
        """Generate summary of ball detection performance."""
        total_frames = len(self.ball_positions)
        valid_detections = sum(1 for pos in self.ball_positions if pos is not None)
        
        if total_frames == 0:
            return {'detection_rate': 0, 'avg_confidence': 0, 'quality': 'No data'}
        
        detection_rate = valid_detections / total_frames
        avg_confidence = (sum(self.detection_confidence_history) / len(self.detection_confidence_history) 
                         if self.detection_confidence_history else 0)
        
        # Assess quality
        if detection_rate > 0.8 and avg_confidence > 0.7:
            quality = 'Excellent'
        elif detection_rate > 0.6 and avg_confidence > 0.5:
            quality = 'Good'
        elif detection_rate > 0.4:
            quality = 'Fair'
        else:
            quality = 'Poor'
        
        return {
            'total_frames': total_frames,
            'valid_detections': valid_detections,
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'quality': quality,
            'confidence_distribution': self._analyze_confidence_distribution()
        }
    
    def _analyze_confidence_distribution(self) -> Dict:
        """Analyze distribution of detection confidence scores."""
        if not self.detection_confidence_history:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high_conf = sum(1 for c in self.detection_confidence_history if c > 0.7)
        medium_conf = sum(1 for c in self.detection_confidence_history if 0.4 <= c <= 0.7)
        low_conf = sum(1 for c in self.detection_confidence_history if c < 0.4)
        
        return {
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf
        }
    
    def _analyze_trajectory(self) -> Dict:
        """Analyze ball trajectory using computer vision methods."""
        valid_positions = [(i, pos) for i, pos in enumerate(self.ball_positions) if pos is not None]
        
        if len(valid_positions) < 3:
            return {'error': 'Insufficient trajectory data'}
        
        # Extract position arrays
        x_coords = [pos[1][0] for pos in valid_positions]
        y_coords = [pos[1][1] for pos in valid_positions]
        frame_indices = [pos[0] for pos in valid_positions]
        
        # Calculate trajectory metrics
        trajectory_length = self._calculate_trajectory_length(x_coords, y_coords)
        trajectory_smoothness = self._calculate_trajectory_smoothness(x_coords, y_coords)
        velocity_profile = self._calculate_velocity_profile(x_coords, y_coords, frame_indices)
        
        # Detect bounce points
        bounce_analysis = self._detect_bounce_points(y_coords, frame_indices)
        
        return {
            'total_length_pixels': trajectory_length,
            'smoothness_score': trajectory_smoothness,
            'velocity_profile': velocity_profile,
            'bounce_analysis': bounce_analysis,
            'trajectory_type': self._classify_trajectory_type(x_coords, y_coords)
        }
    
    def _calculate_trajectory_length(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Calculate total trajectory length in pixels."""
        total_length = 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_length += np.sqrt(dx*dx + dy*dy)
        return total_length
    
    def _calculate_trajectory_smoothness(self, x_coords: List[float], y_coords: List[float]) -> float:
        """Calculate trajectory smoothness score (0-100)."""
        if len(x_coords) < 3:
            return 0
        
        # Calculate curvature at each point
        curvatures = []
        for i in range(1, len(x_coords) - 1):
            # Three consecutive points
            p1 = (x_coords[i-1], y_coords[i-1])
            p2 = (x_coords[i], y_coords[i])
            p3 = (x_coords[i+1], y_coords[i+1])
            
            # Calculate curvature using cross product
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
            v1_mag = np.sqrt(v1[0]**2 + v1[1]**2)
            v2_mag = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if v1_mag > 0 and v2_mag > 0:
                curvature = cross_product / (v1_mag * v2_mag)
                curvatures.append(curvature)
        
        if not curvatures:
            return 50  # Neutral score
        
        # Convert to smoothness score (lower curvature = higher smoothness)
        avg_curvature = np.mean(curvatures)
        smoothness = max(0, 100 - avg_curvature * 100)
        return min(100, smoothness)
    
    def _calculate_velocity_profile(self, x_coords: List[float], y_coords: List[float], 
                                  frame_indices: List[int]) -> Dict:
        """Calculate velocity profile along trajectory."""
        velocities = []
        
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dt = (frame_indices[i] - frame_indices[i-1]) / self.video_fps
            
            if dt > 0:
                velocity = np.sqrt(dx*dx + dy*dy) / dt  # pixels per second
                velocities.append(velocity)
        
        if not velocities:
            return {'avg_velocity': 0, 'max_velocity': 0, 'velocity_consistency': 0}
        
        return {
            'avg_velocity': np.mean(velocities),
            'max_velocity': max(velocities),
            'min_velocity': min(velocities),
            'velocity_consistency': max(0, 100 - (np.std(velocities) / np.mean(velocities) * 100))
        }
    
    def _detect_bounce_points(self, y_coords: List[float], frame_indices: List[int]) -> Dict:
        """Detect potential bounce points in trajectory."""
        if len(y_coords) < 5:
            return {'bounce_detected': False, 'bounce_points': []}
        
        # Look for local minima in y-coordinates (lowest points)
        bounce_points = []
        
        for i in range(2, len(y_coords) - 2):
            # Check if this point is a local minimum
            if (y_coords[i] < y_coords[i-1] and y_coords[i] < y_coords[i+1] and
                y_coords[i] < y_coords[i-2] and y_coords[i] < y_coords[i+2]):
                
                bounce_points.append({
                    'frame_index': frame_indices[i],
                    'y_coordinate': y_coords[i],
                    'time': frame_indices[i] / self.video_fps
                })
        
        return {
            'bounce_detected': len(bounce_points) > 0,
            'bounce_points': bounce_points,
            'num_bounces': len(bounce_points)
        }
    
    def _classify_trajectory_type(self, x_coords: List[float], y_coords: List[float]) -> str:
        """Classify the type of ball trajectory."""
        if len(x_coords) < 3:
            return 'insufficient_data'
        
        # Analyze overall trajectory shape
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Calculate trajectory slope
        start_y, end_y = y_coords[0], y_coords[-1]
        y_change = end_y - start_y
        
        if abs(y_change) < y_range * 0.1:
            return 'horizontal'
        elif y_change > y_range * 0.3:
            return 'descending'
        elif y_change < -y_range * 0.3:
            return 'ascending'
        else:
            return 'parabolic'
    
    def _generate_hawkeye_prediction(self) -> Dict:
        """Generate Hawk-Eye style ball path prediction."""
        try:
            # Use physics-based trajectory model
            hawkeye_data = self.trajectory_model.predict_hawkeye_path(
                [{'bbox': self._position_to_bbox(pos)} for pos in self.ball_positions if pos],
                self.video_fps
            )
            
            if hawkeye_data.get('success'):
                # Enhance with our tracking data
                hawkeye_data['tracking_quality'] = self._generate_detection_summary()['quality']
                hawkeye_data['detection_confidence'] = np.mean(self.detection_confidence_history)
                
                return hawkeye_data
            
        except Exception as e:
            logger.warning(f"Hawk-Eye prediction failed: {e}")
        
        return {'success': False, 'error': 'Prediction failed'}
    
    def _position_to_bbox(self, position: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Convert center position to bounding box for compatibility."""
        if position is None:
            return (0, 0, 0, 0)
        x, y = position
        size = 20  # Assume 20x20 pixel ball
        return (x - size/2, y - size/2, x + size/2, y + size/2)
    
    def _perform_physics_analysis(self) -> Dict:
        """Perform physics-based analysis of ball motion."""
        try:
            # Estimate ball parameters from trajectory
            ball_params = self.trajectory_model.estimate_ball_parameters(
                self.ball_positions, self.timestamps
            )
            
            if not ball_params.get('success'):
                return {'success': False, 'error': 'Physics analysis failed'}
            
            # Predict full trajectory
            trajectory_data = self.trajectory_model.predict_trajectory(
                ball_params['initial_position'],
                ball_params['initial_velocity'],
                ball_params['spin_vector']
            )
            
            return {
                'success': True,
                'estimated_parameters': ball_params,
                'physics_prediction': trajectory_data['analysis'],
                'trajectory_quality': 'high' if trajectory_data['success'] else 'low'
            }
            
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _correlate_with_audio(self) -> Dict:
        """Correlate ball tracking with audio impact events."""
        correlation = {
            'impact_events': len(self.audio_impacts),
            'correlations': [],
            'timing_analysis': {}
        }
        
        # Find video events that correlate with audio impacts
        for audio_impact in self.audio_impacts:
            impact_time = audio_impact['time']
            impact_frame = int(impact_time * self.video_fps)
            
            # Look for tracking anomalies around impact time
            correlation_window = 5  # frames
            start_frame = max(0, impact_frame - correlation_window)
            end_frame = min(len(self.ball_positions), impact_frame + correlation_window)
            
            # Analyze tracking quality in window
            window_confidences = self.detection_confidence_history[start_frame:end_frame]
            avg_confidence = np.mean(window_confidences) if window_confidences else 0
            
            # Check for trajectory changes
            trajectory_change = self._detect_trajectory_change_at_frame(impact_frame)
            
            correlation['correlations'].append({
                'audio_impact_time': impact_time,
                'audio_confidence': audio_impact['confidence'],
                'video_frame': impact_frame,
                'tracking_confidence': avg_confidence,
                'trajectory_change_detected': trajectory_change,
                'correlation_strength': self._calculate_correlation_strength(
                    audio_impact, avg_confidence, trajectory_change
                )
            })
        
        return correlation
    
    def _detect_trajectory_change_at_frame(self, frame_idx: int) -> bool:
        """Detect if trajectory changes significantly at given frame."""
        window = 3
        if frame_idx < window or frame_idx >= len(self.ball_positions) - window:
            return False
        
        # Compare trajectory before and after
        before_positions = self.ball_positions[frame_idx-window:frame_idx]
        after_positions = self.ball_positions[frame_idx:frame_idx+window]
        
        # Calculate direction vectors
        before_valid = [pos for pos in before_positions if pos is not None]
        after_valid = [pos for pos in after_positions if pos is not None]
        
        if len(before_valid) < 2 or len(after_valid) < 2:
            return False
        
        # Calculate average direction before and after
        before_dx = before_valid[-1][0] - before_valid[0][0]
        before_dy = before_valid[-1][1] - before_valid[0][1]
        after_dx = after_valid[-1][0] - after_valid[0][0]
        after_dy = after_valid[-1][1] - after_valid[0][1]
        
        # Calculate angle change
        before_angle = np.arctan2(before_dy, before_dx)
        after_angle = np.arctan2(after_dy, after_dx)
        angle_change = abs(before_angle - after_angle)
        
        # Significant change if angle differs by more than 30 degrees
        return angle_change > np.pi / 6
    
    def _calculate_correlation_strength(self, audio_impact: Dict, 
                                      tracking_confidence: float, trajectory_change: bool) -> float:
        """Calculate correlation strength between audio and video events."""
        correlation = 0.0
        
        # Audio confidence contributes to correlation
        correlation += audio_impact['confidence'] * 0.4
        
        # Lower tracking confidence around impact suggests ball interaction
        if tracking_confidence < 0.5:
            correlation += 0.3
        
        # Trajectory change indicates potential impact
        if trajectory_change:
            correlation += 0.3
        
        return min(1.0, correlation)
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate overall tracking performance metrics."""
        detection_summary = self._generate_detection_summary()
        
        # Calculate temporal consistency
        temporal_consistency = self._calculate_temporal_consistency()
        
        # Calculate spatial accuracy (if reference data available)
        spatial_accuracy = self._estimate_spatial_accuracy()
        
        return {
            'detection_rate': detection_summary['detection_rate'],
            'avg_confidence': detection_summary['avg_confidence'],
            'temporal_consistency': temporal_consistency,
            'spatial_accuracy': spatial_accuracy,
            'overall_quality_score': self._calculate_overall_quality_score(
                detection_summary, temporal_consistency, spatial_accuracy
            )
        }
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of ball tracking."""
        if len(self.ball_positions) < 3:
            return 0.0
        
        # Count frames with valid detections
        valid_frames = sum(1 for pos in self.ball_positions if pos is not None)
        
        # Calculate consistency based on detection gaps
        gaps = []
        gap_length = 0
        
        for pos in self.ball_positions:
            if pos is None:
                gap_length += 1
            else:
                if gap_length > 0:
                    gaps.append(gap_length)
                    gap_length = 0
        
        if gap_length > 0:
            gaps.append(gap_length)
        
        # Penalize long gaps
        gap_penalty = sum(min(gap, 10) for gap in gaps) / len(self.ball_positions)
        consistency = max(0, 1.0 - gap_penalty)
        
        return consistency * 100
    
    def _estimate_spatial_accuracy(self) -> float:
        """Estimate spatial tracking accuracy."""
        # Without ground truth, estimate based on trajectory smoothness
        if len(self.ball_positions) < 3:
            return 50.0  # Neutral score
        
        valid_positions = [pos for pos in self.ball_positions if pos is not None]
        
        if len(valid_positions) < 3:
            return 30.0  # Low score for insufficient data
        
        # Use trajectory smoothness as proxy for accuracy
        x_coords = [pos[0] for pos in valid_positions]
        y_coords = [pos[1] for pos in valid_positions]
        
        smoothness = self._calculate_trajectory_smoothness(x_coords, y_coords)
        
        # Combine with detection confidence
        avg_confidence = np.mean([conf for conf in self.detection_confidence_history if conf > 0])
        
        estimated_accuracy = (smoothness * 0.6 + avg_confidence * 100 * 0.4)
        return min(100, estimated_accuracy)
    
    def _calculate_overall_quality_score(self, detection_summary: Dict, 
                                       temporal_consistency: float, spatial_accuracy: float) -> float:
        """Calculate overall tracking quality score."""
        detection_score = detection_summary['detection_rate'] * 100
        confidence_score = detection_summary['avg_confidence'] * 100
        
        # Weighted average of all metrics
        overall_score = (
            detection_score * 0.3 +
            confidence_score * 0.3 +
            temporal_consistency * 0.2 +
            spatial_accuracy * 0.2
        )
        
        return min(100, overall_score)