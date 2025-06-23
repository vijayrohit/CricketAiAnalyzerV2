"""Camera calibration module for cricket video analysis."""
import cv2
import numpy as np
import glob
import os
import logging

logger = logging.getLogger(__name__)

class CameraCalibrator:
    """Handles camera calibration for accurate cricket ball tracking."""
    
    def __init__(self, chessboard_size=(9, 6)):
        self.chessboard_size = chessboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    def calibrate_from_images(self, images_dir='calibration_images'):
        """
        Calibrate camera using chessboard pattern images.
        
        Args:
            images_dir: Directory containing calibration images
            
        Returns:
            tuple: (camera_matrix, distortion_coefficients, success)
        """
        # Prepare object points
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        if not os.path.exists(images_dir):
            logger.warning(f"Calibration directory {images_dir} not found")
            return self._get_default_calibration()
        
        # Find calibration images
        image_patterns = [
            os.path.join(images_dir, '*.jpg'),
            os.path.join(images_dir, '*.jpeg'),
            os.path.join(images_dir, '*.png')
        ]
        
        images = []
        for pattern in image_patterns:
            images.extend(glob.glob(pattern))
        
        if len(images) < 10:
            logger.warning(f"Not enough calibration images found ({len(images)}). Using default calibration.")
            return self._get_default_calibration()
        
        successful_images = 0
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                successful_images += 1
        
        if successful_images < 10:
            logger.warning(f"Only {successful_images} valid calibration images found. Using default calibration.")
            return self._get_default_calibration()
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            logger.info(f"Camera calibration successful using {successful_images} images")
            return mtx, dist, True
        else:
            logger.error("Camera calibration failed")
            return self._get_default_calibration()
    
    def calibrate_from_video(self, video_path, sample_interval=30):
        """
        Extract calibration images from video and calibrate.
        
        Args:
            video_path: Path to video file
            sample_interval: Frame interval for sampling
            
        Returns:
            tuple: (camera_matrix, distortion_coefficients, success)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return self._get_default_calibration()
        
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                
                if ret_corners:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    imgpoints.append(corners2)
            
            frame_count += 1
        
        cap.release()
        
        if len(objpoints) < 10:
            logger.warning(f"Not enough calibration patterns found in video ({len(objpoints)})")
            return self._get_default_calibration()
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            logger.info(f"Video-based calibration successful using {len(objpoints)} patterns")
            return mtx, dist, True
        else:
            logger.error("Video-based calibration failed")
            return self._get_default_calibration()
    
    def undistort_frame(self, frame, camera_matrix, dist_coeffs):
        """
        Remove lens distortion from frame.
        
        Args:
            frame: Input frame
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            Undistorted frame
        """
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        return cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    
    def _get_default_calibration(self):
        """
        Return default calibration parameters for typical cricket videos.
        
        Returns:
            tuple: (camera_matrix, distortion_coefficients, success)
        """
        # Default parameters for 1920x1080 video
        camera_matrix = np.array([
            [1400.0, 0.0, 960.0],
            [0.0, 1400.0, 540.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Minimal distortion coefficients
        dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        
        logger.info("Using default camera calibration parameters")
        return camera_matrix, dist_coeffs, False
    
    def save_calibration(self, camera_matrix, dist_coeffs, filepath='calibration_params.npz'):
        """Save calibration parameters to file."""
        np.savez(filepath, mtx=camera_matrix, dist=dist_coeffs)
        logger.info(f"Calibration parameters saved to {filepath}")
    
    def load_calibration(self, filepath='calibration_params.npz'):
        """Load calibration parameters from file."""
        try:
            data = np.load(filepath)
            return data['mtx'], data['dist'], True
        except FileNotFoundError:
            logger.warning(f"Calibration file {filepath} not found")
            return self._get_default_calibration()