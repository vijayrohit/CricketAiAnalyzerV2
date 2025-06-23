import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)

class BallTracker:
    """Handles cricket ball detection and tracking in video frames."""
    
    def __init__(self):
        # Ball detection parameters
        self.min_ball_radius = 3
        self.max_ball_radius = 25
        self.ball_color_ranges = [
            # Red ball (traditional cricket ball)
            {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            # White ball (limited overs)
            {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
        ]
        
        # Tracking parameters
        self.max_disappeared = 10
        self.max_distance = 100
        self.trajectory_smoothing = 5
        
        # Initialize trackers
        self.ball_tracker = cv2.TrackerCSRT_create()
        self.tracking_initialized = False
        
    def detect_ball_candidates(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect potential ball candidates in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, radius) tuples for ball candidates
        """
        try:
            candidates = []
            
            # Convert to HSV for color-based detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Try different color ranges for ball detection
            for color_range in self.ball_color_ranges:
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                
                # Morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 20:  # Too small
                        continue
                    
                    # Get enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    if self.min_ball_radius <= radius <= self.max_ball_radius:
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.5:  # Reasonably circular
                                candidates.append((int(x), int(y), int(radius)))
            
            # Also try Hough Circle detection as backup
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=self.min_ball_radius,
                maxRadius=self.max_ball_radius
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    candidates.append((x, y, r))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Ball candidate detection error: {str(e)}")
            return []
    
    def track_ball(self, frames: List[np.ndarray]) -> Dict:
        """
        Track ball throughout the video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing tracking results and analysis
        """
        try:
            if not frames:
                return {'error': 'No frames provided'}
            
            ball_positions = []
            velocities = []
            frame_data = []
            
            prev_position = None
            fps = 30  # Assume 30 fps if not available
            
            for frame_idx, frame in enumerate(frames):
                frame_height, frame_width = frame.shape[:2]
                
                # Detect ball candidates
                candidates = self.detect_ball_candidates(frame)
                
                if not candidates:
                    frame_data.append({
                        'frame': frame_idx,
                        'ball_detected': False,
                        'position': None,
                        'velocity': None
                    })
                    continue
                
                # Select best candidate (closest to previous position if available)
                if prev_position and candidates:
                    distances = [np.sqrt((x - prev_position[0])**2 + (y - prev_position[1])**2) 
                               for x, y, r in candidates]
                    best_idx = np.argmin(distances)
                    current_position = candidates[best_idx]
                else:
                    # Use the largest candidate
                    current_position = max(candidates, key=lambda x: x[2])
                
                ball_positions.append(current_position)
                
                # Calculate velocity if we have previous position
                velocity = None
                if prev_position:
                    dx = current_position[0] - prev_position[0]
                    dy = current_position[1] - prev_position[1]
                    dt = 1.0 / fps
                    
                    # Convert pixel velocity to approximate real-world velocity
                    # Assuming cricket pitch is about 20 meters and spans roughly 60% of frame width
                    pixels_per_meter = frame_width * 0.6 / 20.0
                    velocity_x = (dx / pixels_per_meter) / dt  # m/s
                    velocity_y = (dy / pixels_per_meter) / dt  # m/s
                    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                    
                    velocity = {
                        'x': velocity_x,
                        'y': velocity_y,
                        'magnitude': velocity_magnitude
                    }
                    velocities.append(velocity_magnitude)
                
                frame_data.append({
                    'frame': frame_idx,
                    'ball_detected': True,
                    'position': {
                        'x': current_position[0],
                        'y': current_position[1],
                        'radius': current_position[2]
                    },
                    'velocity': velocity
                })
                
                prev_position = current_position
            
            # Analyze trajectory
            trajectory_analysis = self._analyze_trajectory(ball_positions, frame_data)
            
            # Calculate delivery metrics
            delivery_metrics = self._calculate_delivery_metrics(ball_positions, velocities, frames[0].shape)
            
            return {
                'frames_analyzed': len(frames),
                'ball_detected_frames': len([f for f in frame_data if f['ball_detected']]),
                'detection_rate': len([f for f in frame_data if f['ball_detected']]) / len(frames),
                'frame_data': frame_data,
                'trajectory_analysis': trajectory_analysis,
                'delivery_metrics': delivery_metrics,
                'average_velocity': np.mean(velocities) if velocities else 0,
                'max_velocity': np.max(velocities) if velocities else 0,
                'velocity_data': velocities
            }
            
        except Exception as e:
            logger.error(f"Ball tracking error: {str(e)}")
            return {'error': f'Ball tracking failed: {str(e)}'}
    
    def _analyze_trajectory(self, positions: List[Tuple], frame_data: List[Dict]) -> Dict:
        """
        Analyze ball trajectory for bowling insights.
        
        Args:
            positions: List of ball positions
            frame_data: Frame-by-frame data
            
        Returns:
            Dictionary with trajectory analysis
        """
        try:
            if len(positions) < 3:
                return {'error': 'Insufficient data for trajectory analysis'}
            
            # Extract x, y coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Fit polynomial to trajectory (parabolic for projectile motion)
            if len(x_coords) >= 3:
                # Fit parabola: y = ax² + bx + c
                z = np.polyfit(x_coords, y_coords, 2)
                trajectory_poly = np.poly1d(z)
                
                # Calculate bounce point (lowest y-coordinate)
                bounce_frame = None
                bounce_position = None
                min_y = float('inf')
                
                for data in frame_data:
                    if data['ball_detected'] and data['position']['y'] < min_y:
                        min_y = data['position']['y']
                        bounce_frame = data['frame']
                        bounce_position = data['position']
                
                # Analyze trajectory shape
                trajectory_type = "Parabolic"
                if z[0] > 0:  # Positive coefficient means upward curve
                    trajectory_type = "Rising"
                elif z[0] < 0:  # Negative coefficient means downward curve
                    trajectory_type = "Falling"
                
                return {
                    'trajectory_type': trajectory_type,
                    'polynomial_coefficients': z.tolist(),
                    'bounce_point': {
                        'frame': bounce_frame,
                        'position': bounce_position
                    },
                    'horizontal_distance': max(x_coords) - min(x_coords),
                    'vertical_distance': max(y_coords) - min(y_coords),
                    'trajectory_smoothness': self._calculate_smoothness(x_coords, y_coords)
                }
            
            return {'error': 'Could not fit trajectory'}
            
        except Exception as e:
            logger.error(f"Trajectory analysis error: {str(e)}")
            return {'error': f'Trajectory analysis failed: {str(e)}'}
    
    def _calculate_delivery_metrics(self, positions: List[Tuple], velocities: List[float], frame_shape: Tuple) -> Dict:
        """
        Calculate bowling delivery metrics.
        
        Args:
            positions: Ball positions
            velocities: Ball velocities
            frame_shape: Shape of video frames
            
        Returns:
            Dictionary with delivery metrics
        """
        try:
            if not positions or not velocities:
                return {'error': 'Insufficient data for metrics calculation'}
            
            height, width = frame_shape[:2]
            
            # Estimate delivery speed
            avg_speed = np.mean(velocities) if velocities else 0
            max_speed = np.max(velocities) if velocities else 0
            
            # Classify delivery speed
            speed_category = "Slow"
            if avg_speed > 25:  # m/s (90 km/h)
                speed_category = "Medium"
            if avg_speed > 35:  # m/s (126 km/h)
                speed_category = "Fast"
            if avg_speed > 42:  # m/s (151 km/h)
                speed_category = "Express"
            
            # Analyze line and length
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Line analysis (horizontal deviation)
            line_variation = np.std(x_coords)
            line_consistency = "Good" if line_variation < width * 0.05 else "Needs Improvement"
            
            # Length analysis (where ball pitches)
            pitch_position = "Unknown"
            if y_coords:
                avg_y = np.mean(y_coords)
                if avg_y < height * 0.3:
                    pitch_position = "Short"
                elif avg_y < height * 0.6:
                    pitch_position = "Good Length"
                elif avg_y < height * 0.8:
                    pitch_position = "Full"
                else:
                    pitch_position = "Yorker"
            
            return {
                'average_speed_ms': avg_speed,
                'average_speed_kmh': avg_speed * 3.6,
                'max_speed_ms': max_speed,
                'max_speed_kmh': max_speed * 3.6,
                'speed_category': speed_category,
                'line_consistency': line_consistency,
                'line_variation_pixels': line_variation,
                'pitch_position': pitch_position,
                'delivery_accuracy': {
                    'line_score': max(0, 100 - (line_variation / width * 100 * 10)),
                    'length_score': 85 if pitch_position == "Good Length" else 60
                }
            }
            
        except Exception as e:
            logger.error(f"Delivery metrics calculation error: {str(e)}")
            return {'error': f'Metrics calculation failed: {str(e)}'}
    
    def _calculate_smoothness(self, x_coords: List[float], y_coords: List[float]) -> float:
        """
        Calculate trajectory smoothness score.
        
        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            
        Returns:
            Smoothness score (0-100, higher is smoother)
        """
        try:
            if len(x_coords) < 3:
                return 0
            
            # Calculate second derivative (acceleration changes)
            x_diff2 = np.diff(np.diff(x_coords))
            y_diff2 = np.diff(np.diff(y_coords))
            
            # Calculate curvature variations
            curvature_var = np.var(np.sqrt(x_diff2**2 + y_diff2**2))
            
            # Convert to smoothness score (lower variation = higher smoothness)
            smoothness = max(0, 100 - curvature_var)
            
            return float(smoothness)
            
        except Exception as e:
            logger.error(f"Smoothness calculation error: {str(e)}")
            return 0
