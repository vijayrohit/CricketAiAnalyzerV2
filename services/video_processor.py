import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video file processing and frame extraction."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate if the video file is readable and has valid content.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if video is valid, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.error(f"Could not read frame from video: {video_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation error: {str(e)}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Extract basic information about the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            dict: Video information including fps, duration, resolution
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'resolution': f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    def extract_frames(self, video_path: str, max_frames: int = 300) -> List[np.ndarray]:
        """
        Extract frames from video for analysis.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of numpy arrays representing frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate step size to get evenly distributed frames
            step = max(1, frame_count // max_frames)
            
            frame_index = 0
            while len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame is not None:
                    frames.append(frame)
                
                frame_index += step
                
                if frame_index >= frame_count:
                    break
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction error: {str(e)}")
            return []
    
    def extract_key_frames(self, video_path: str, threshold: float = 0.3) -> List[Tuple[int, np.ndarray]]:
        """
        Extract key frames based on motion detection.
        
        Args:
            video_path: Path to the video file
            threshold: Motion threshold for key frame detection
            
        Returns:
            List of tuples (frame_number, frame)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            key_frames = []
            prev_frame = None
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff) / 255.0
                    
                    # If significant motion detected, add as key frame
                    if motion_score > threshold:
                        key_frames.append((frame_number, frame))
                
                prev_frame = gray
                frame_number += 1
            
            cap.release()
            logger.info(f"Extracted {len(key_frames)} key frames")
            return key_frames
            
        except Exception as e:
            logger.error(f"Key frame extraction error: {str(e)}")
            return []
    
    def resize_frame(self, frame: np.ndarray, max_width: int = 640) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            max_width: Maximum width of output frame
            
        Returns:
            Resized frame
        """
        try:
            height, width = frame.shape[:2]
            
            if width <= max_width:
                return frame
            
            # Calculate new dimensions
            ratio = max_width / width
            new_width = max_width
            new_height = int(height * ratio)
            
            # Resize frame
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
            
        except Exception as e:
            logger.error(f"Frame resize error: {str(e)}")
            return frame
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply basic enhancement to improve analysis quality.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Slight sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Frame enhancement error: {str(e)}")
            return frame
