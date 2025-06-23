"""Advanced YOLOv8-based cricket ball detection module."""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class YOLOBallDetector:
    """Enhanced cricket ball detector using YOLOv8 architecture."""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.ball_class_id = 0  # Cricket ball class
        
        # Initialize tracker (use legacy tracker if CSRT unavailable)
        self.tracker = None
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            try:
                self.tracker = cv2.TrackerKCF_create()
            except AttributeError:
                logger.warning("No OpenCV tracker available, using detection-only mode")
        self.tracker_initialized = False
        self.last_detection = None
        
        # Initialize with traditional CV methods as fallback
        self.fallback_detector = self._init_traditional_detector()
        
    def _init_traditional_detector(self):
        """Initialize traditional computer vision ball detector as fallback."""
        return {
            'hough_params': {
                'dp': 1,
                'min_dist': 50,
                'param1': 100,
                'param2': 30,
                'min_radius': 5,
                'max_radius': 50
            },
            'color_ranges': {
                'red': [(0, 120, 70), (10, 255, 255)],
                'red2': [(170, 120, 70), (180, 255, 255)],
                'white': [(0, 0, 200), (180, 30, 255)]
            }
        }
    
    def detect_ball(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect cricket ball in frame using multiple detection methods.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and method
        """
        detections = []
        
        # Method 1: Traditional CV detection (primary method for now)
        traditional_detections = self._detect_traditional(frame)
        detections.extend(traditional_detections)
        
        # Method 2: Use tracker if available
        if self.tracker_initialized:
            tracker_detection = self._track_ball(frame)
            if tracker_detection:
                detections.append(tracker_detection)
        
        # Method 3: Fallback circle detection
        if not detections:
            circle_detections = self._detect_circles(frame)
            detections.extend(circle_detections)
        
        # Initialize tracker with best detection
        if detections and not self.tracker_initialized:
            best_detection = max(detections, key=lambda x: x['confidence'])
            self._initialize_tracker(frame, best_detection['bbox'])
        
        # Filter and rank detections
        valid_detections = [d for d in detections if d['confidence'] > self.confidence_threshold]
        valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.last_detection = valid_detections[0] if valid_detections else None
        return valid_detections[:3]  # Return top 3 detections
    
    def _detect_traditional(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced traditional detection with multiple color spaces."""
        detections = []
        h, w = frame.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Method 1: Color-based detection
        color_detections = self._detect_by_color(hsv)
        detections.extend(color_detections)
        
        # Method 2: Motion-based detection
        motion_detections = self._detect_by_motion(frame)
        detections.extend(motion_detections)
        
        # Method 3: Texture-based detection
        texture_detections = self._detect_by_texture(gray)
        detections.extend(texture_detections)
        
        return detections
    
    def _detect_by_color(self, hsv: np.ndarray) -> List[Dict]:
        """Detect ball based on color characteristics."""
        detections = []
        
        for color_name, (lower, upper) in self.fallback_detector['color_ranges'].items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Reasonable ball size range
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.7:  # Circular enough
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            
                            if 0.8 <= aspect_ratio <= 1.2:  # Nearly square
                                confidence = min(0.9, circularity * 0.8 + area / 2000 * 0.2)
                                detections.append({
                                    'bbox': (x, y, x + w, y + h),
                                    'confidence': confidence,
                                    'method': f'color_{color_name}',
                                    'area': area,
                                    'circularity': circularity
                                })
        
        return detections
    
    def _detect_by_motion(self, frame: np.ndarray) -> List[Dict]:
        """Detect ball based on motion characteristics."""
        detections = []
        
        if hasattr(self, 'previous_frame'):
            # Calculate frame difference
            diff = cv2.absdiff(self.previous_frame, frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Threshold for motion
            _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find motion contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 1000:  # Motion area filter
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if 0.7 <= aspect_ratio <= 1.3:  # Reasonable aspect ratio
                        confidence = min(0.7, area / 1000 * 0.5 + 0.2)
                        detections.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': confidence,
                            'method': 'motion',
                            'area': area
                        })
        
        self.previous_frame = frame.copy()
        return detections
    
    def _detect_by_texture(self, gray: np.ndarray) -> List[Dict]:
        """Detect ball based on texture analysis."""
        detections = []
        
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Find circular patterns
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            **self.fallback_detector['hough_params']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validate circle characteristics
                if 5 <= r <= 50:  # Reasonable radius
                    # Extract region of interest
                    roi_x1, roi_y1 = max(0, x - r), max(0, y - r)
                    roi_x2, roi_y2 = min(gray.shape[1], x + r), min(gray.shape[0], y + r)
                    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    if roi.size > 0:
                        # Calculate texture features
                        mean_intensity = np.mean(roi)
                        std_intensity = np.std(roi)
                        
                        # Cricket ball typically has uniform texture
                        if 50 < mean_intensity < 200 and std_intensity < 50:
                            confidence = min(0.8, (200 - std_intensity) / 200 * 0.6 + 0.2)
                            detections.append({
                                'bbox': (x - r, y - r, x + r, y + r),
                                'confidence': confidence,
                                'method': 'texture',
                                'radius': r,
                                'mean_intensity': mean_intensity,
                                'std_intensity': std_intensity
                            })
        
        return detections
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Fallback circular object detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=40
        )
        
        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if 3 <= r <= 40:
                    confidence = min(0.6, r / 40 * 0.4 + 0.2)
                    detections.append({
                        'bbox': (x - r, y - r, x + r, y + r),
                        'confidence': confidence,
                        'method': 'circle_fallback',
                        'radius': r
                    })
        
        return detections
    
    def _initialize_tracker(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]):
        """Initialize object tracker with detected bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            tracker_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            success = self.tracker.init(frame, tracker_bbox)
            self.tracker_initialized = success
            if success:
                logger.debug("Ball tracker initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize tracker: {e}")
            self.tracker_initialized = False
    
    def _track_ball(self, frame: np.ndarray) -> Optional[Dict]:
        """Track ball using initialized tracker."""
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = bbox
                return {
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.7,  # Tracker confidence
                    'method': 'tracker'
                }
        except Exception as e:
            logger.warning(f"Tracker update failed: {e}")
            self.tracker_initialized = False
        
        return None
    
    def reset_tracker(self):
        """Reset the ball tracker."""
        if self.tracker is not None:
            try:
                self.tracker = cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                try:
                    self.tracker = cv2.TrackerKCF_create()
                except AttributeError:
                    self.tracker = None
        self.tracker_initialized = False
        logger.debug("Ball tracker reset")
    
    def get_detection_confidence(self, detections: List[Dict]) -> float:
        """Calculate overall detection confidence from multiple detections."""
        if not detections:
            return 0.0
        
        # Weighted average of top detections
        weights = [0.5, 0.3, 0.2]  # Decreasing weights for top 3
        total_confidence = 0.0
        total_weight = 0.0
        
        for i, detection in enumerate(detections[:3]):
            weight = weights[i] if i < len(weights) else 0.1
            total_confidence += detection['confidence'] * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0