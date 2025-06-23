"""Advanced pose analysis with detailed kinematic calculations for cricket technique."""
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Tuple, Optional
from scipy import signal
import math

logger = logging.getLogger(__name__)

class AdvancedPoseAnalyzer:
    """Enhanced pose analyzer with detailed biomechanical analysis for cricket."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Cricket-specific joint mappings
        self.cricket_joints = {
            'head': [self.mp_pose.PoseLandmark.NOSE],
            'shoulders': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'elbows': [self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            'wrists': [self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'hips': [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP],
            'knees': [self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'ankles': [self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        # Initialize history for motion analysis
        self.pose_history = []
        self.max_history = 30  # Keep last 30 frames
        
    def analyze_bowling_technique(self, frames: List[np.ndarray]) -> Dict:
        """
        Comprehensive bowling technique analysis.
        
        Args:
            frames: List of video frames
            
        Returns:
            Detailed bowling analysis
        """
        frame_analyses = []
        
        for frame_idx, frame in enumerate(frames):
            frame_analysis = self._analyze_single_frame(frame, frame_idx, 'bowling')
            if frame_analysis:
                frame_analyses.append(frame_analysis)
        
        if not frame_analyses:
            return {'error': 'No valid pose detections found'}
        
        # Analyze bowling phases
        bowling_phases = self._identify_bowling_phases(frame_analyses)
        
        # Calculate detailed metrics
        kinematic_analysis = self._calculate_bowling_kinematics(frame_analyses, bowling_phases)
        
        # Assess technique quality
        technique_assessment = self._assess_bowling_technique(kinematic_analysis)
        
        return {
            'success': True,
            'detection_rate': f"{len(frame_analyses)}/{len(frames)} ({len(frame_analyses)/len(frames)*100:.1f}%)",
            'bowling_phases': bowling_phases,
            'kinematics': kinematic_analysis,
            'technique_assessment': technique_assessment,
            'frame_by_frame': frame_analyses,
            'recommendations': self._generate_bowling_recommendations(technique_assessment)
        }
    
    def analyze_batting_technique(self, frames: List[np.ndarray]) -> Dict:
        """
        Comprehensive batting technique analysis.
        
        Args:
            frames: List of video frames
            
        Returns:
            Detailed batting analysis
        """
        frame_analyses = []
        
        for frame_idx, frame in enumerate(frames):
            frame_analysis = self._analyze_single_frame(frame, frame_idx, 'batting')
            if frame_analysis:
                frame_analyses.append(frame_analysis)
        
        if not frame_analyses:
            return {'error': 'No valid pose detections found'}
        
        # Analyze batting phases
        batting_phases = self._identify_batting_phases(frame_analyses)
        
        # Calculate detailed metrics
        kinematic_analysis = self._calculate_batting_kinematics(frame_analyses, batting_phases)
        
        # Assess technique quality
        technique_assessment = self._assess_batting_technique(kinematic_analysis)
        
        # Stroke classification
        stroke_analysis = self._classify_batting_stroke(kinematic_analysis, batting_phases)
        
        return {
            'success': True,
            'detection_rate': f"{len(frame_analyses)}/{len(frames)} ({len(frame_analyses)/len(frames)*100:.1f}%)",
            'batting_phases': batting_phases,
            'kinematics': kinematic_analysis,
            'technique_assessment': technique_assessment,
            'stroke_analysis': stroke_analysis,
            'frame_by_frame': frame_analyses,
            'recommendations': self._generate_batting_recommendations(technique_assessment, stroke_analysis)
        }
    
    def _analyze_single_frame(self, frame: np.ndarray, frame_idx: int, sport_type: str) -> Optional[Dict]:
        """Analyze pose in a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
        
        # Calculate joint angles
        joint_angles = self._calculate_joint_angles(landmarks)
        
        # Calculate body alignment metrics
        alignment_metrics = self._calculate_alignment_metrics(landmarks)
        
        # Calculate motion metrics if history available
        motion_metrics = self._calculate_motion_metrics(landmarks, frame_idx)
        
        # Store in history
        self.pose_history.append({
            'frame_idx': frame_idx,
            'landmarks': landmarks,
            'joint_angles': joint_angles,
            'alignment': alignment_metrics
        })
        
        # Keep history size manageable
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return {
            'frame_idx': frame_idx,
            'landmarks': landmarks,
            'joint_angles': joint_angles,
            'alignment': alignment_metrics,
            'motion': motion_metrics,
            'confidence': self._calculate_pose_confidence(results.pose_landmarks)
        }
    
    def _extract_landmarks(self, pose_landmarks, frame_shape: Tuple) -> Dict:
        """Extract and normalize pose landmarks."""
        landmarks = {}
        h, w = frame_shape[:2]
        
        for landmark_name, landmark_indices in self.cricket_joints.items():
            landmark_coords = []
            for idx in landmark_indices:
                if idx.value < len(pose_landmarks.landmark):
                    landmark = pose_landmarks.landmark[idx.value]
                    # Convert to pixel coordinates
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z  # Relative depth
                    visibility = landmark.visibility
                    landmark_coords.append({
                        'x': x, 'y': y, 'z': z, 
                        'visibility': visibility,
                        'normalized': {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                    })
            landmarks[landmark_name] = landmark_coords
        
        return landmarks
    
    def _calculate_joint_angles(self, landmarks: Dict) -> Dict:
        """Calculate important joint angles for cricket analysis."""
        angles = {}
        
        try:
            # Shoulder angle (arm elevation)
            if len(landmarks['shoulders']) == 2 and len(landmarks['elbows']) >= 1:
                left_shoulder = landmarks['shoulders'][0]
                right_shoulder = landmarks['shoulders'][1]
                left_elbow = landmarks['elbows'][0] if len(landmarks['elbows']) > 0 else None
                right_elbow = landmarks['elbows'][1] if len(landmarks['elbows']) > 1 else None
                
                if left_elbow:
                    angles['left_arm_elevation'] = self._calculate_angle_3points(
                        (left_shoulder['x'], left_shoulder['y']),
                        (left_elbow['x'], left_elbow['y']),
                        (left_shoulder['x'], left_shoulder['y'] + 100)  # Reference vertical
                    )
                
                if right_elbow:
                    angles['right_arm_elevation'] = self._calculate_angle_3points(
                        (right_shoulder['x'], right_shoulder['y']),
                        (right_elbow['x'], right_elbow['y']),
                        (right_shoulder['x'], right_shoulder['y'] + 100)  # Reference vertical
                    )
            
            # Elbow flexion angles
            if (len(landmarks['shoulders']) >= 1 and len(landmarks['elbows']) >= 1 
                and len(landmarks['wrists']) >= 1):
                
                # Left elbow flexion
                if (len(landmarks['shoulders']) > 0 and len(landmarks['elbows']) > 0 
                    and len(landmarks['wrists']) > 0):
                    left_shoulder = landmarks['shoulders'][0]
                    left_elbow = landmarks['elbows'][0]
                    left_wrist = landmarks['wrists'][0]
                    
                    angles['left_elbow_flexion'] = self._calculate_angle_3points(
                        (left_shoulder['x'], left_shoulder['y']),
                        (left_elbow['x'], left_elbow['y']),
                        (left_wrist['x'], left_wrist['y'])
                    )
                
                # Right elbow flexion
                if (len(landmarks['shoulders']) > 1 and len(landmarks['elbows']) > 1 
                    and len(landmarks['wrists']) > 1):
                    right_shoulder = landmarks['shoulders'][1]
                    right_elbow = landmarks['elbows'][1]
                    right_wrist = landmarks['wrists'][1]
                    
                    angles['right_elbow_flexion'] = self._calculate_angle_3points(
                        (right_shoulder['x'], right_shoulder['y']),
                        (right_elbow['x'], right_elbow['y']),
                        (right_wrist['x'], right_wrist['y'])
                    )
            
            # Hip-knee-ankle angles (leg alignment)
            if (len(landmarks['hips']) >= 1 and len(landmarks['knees']) >= 1 
                and len(landmarks['ankles']) >= 1):
                
                # Left leg angle
                if (len(landmarks['hips']) > 0 and len(landmarks['knees']) > 0 
                    and len(landmarks['ankles']) > 0):
                    left_hip = landmarks['hips'][0]
                    left_knee = landmarks['knees'][0]
                    left_ankle = landmarks['ankles'][0]
                    
                    angles['left_knee_angle'] = self._calculate_angle_3points(
                        (left_hip['x'], left_hip['y']),
                        (left_knee['x'], left_knee['y']),
                        (left_ankle['x'], left_ankle['y'])
                    )
                
                # Right leg angle
                if (len(landmarks['hips']) > 1 and len(landmarks['knees']) > 1 
                    and len(landmarks['ankles']) > 1):
                    right_hip = landmarks['hips'][1]
                    right_knee = landmarks['knees'][1]
                    right_ankle = landmarks['ankles'][1]
                    
                    angles['right_knee_angle'] = self._calculate_angle_3points(
                        (right_hip['x'], right_hip['y']),
                        (right_knee['x'], right_knee['y']),
                        (right_ankle['x'], right_ankle['y'])
                    )
            
            # Trunk angle (forward lean)
            if len(landmarks['shoulders']) >= 1 and len(landmarks['hips']) >= 1:
                shoulder_center = self._get_center_point(landmarks['shoulders'])
                hip_center = self._get_center_point(landmarks['hips'])
                
                if shoulder_center and hip_center:
                    # Calculate trunk angle from vertical
                    trunk_vector = (shoulder_center['x'] - hip_center['x'], 
                                  shoulder_center['y'] - hip_center['y'])
                    vertical_vector = (0, -100)  # Upward direction
                    
                    angles['trunk_angle'] = self._calculate_vector_angle(trunk_vector, vertical_vector)
        
        except Exception as e:
            logger.warning(f"Error calculating joint angles: {e}")
        
        return angles
    
    def _calculate_alignment_metrics(self, landmarks: Dict) -> Dict:
        """Calculate body alignment metrics."""
        alignment = {}
        
        try:
            # Shoulder level (horizontal alignment)
            if len(landmarks['shoulders']) == 2:
                left_shoulder = landmarks['shoulders'][0]
                right_shoulder = landmarks['shoulders'][1]
                
                shoulder_slope = ((right_shoulder['y'] - left_shoulder['y']) / 
                                 (right_shoulder['x'] - left_shoulder['x'])) if left_shoulder['x'] != right_shoulder['x'] else 0
                alignment['shoulder_tilt'] = math.degrees(math.atan(shoulder_slope))
            
            # Hip level (pelvic alignment)
            if len(landmarks['hips']) == 2:
                left_hip = landmarks['hips'][0]
                right_hip = landmarks['hips'][1]
                
                hip_slope = ((right_hip['y'] - left_hip['y']) / 
                            (right_hip['x'] - left_hip['x'])) if left_hip['x'] != right_hip['x'] else 0
                alignment['hip_tilt'] = math.degrees(math.atan(hip_slope))
            
            # Center line alignment (spine straightness)
            if (len(landmarks['shoulders']) >= 1 and len(landmarks['hips']) >= 1 
                and len(landmarks['head']) >= 1):
                
                head_pos = landmarks['head'][0] if landmarks['head'] else None
                shoulder_center = self._get_center_point(landmarks['shoulders'])
                hip_center = self._get_center_point(landmarks['hips'])
                
                if head_pos and shoulder_center and hip_center:
                    # Calculate deviation from straight line
                    spine_deviation = self._calculate_line_deviation([
                        (head_pos['x'], head_pos['y']),
                        (shoulder_center['x'], shoulder_center['y']),
                        (hip_center['x'], hip_center['y'])
                    ])
                    alignment['spine_straightness'] = max(0, 100 - spine_deviation * 10)
            
            # Weight distribution (based on ankle positions)
            if len(landmarks['ankles']) == 2:
                left_ankle = landmarks['ankles'][0]
                right_ankle = landmarks['ankles'][1]
                
                ankle_distance = math.sqrt((right_ankle['x'] - left_ankle['x'])**2 + 
                                         (right_ankle['y'] - left_ankle['y'])**2)
                alignment['stance_width'] = ankle_distance
        
        except Exception as e:
            logger.warning(f"Error calculating alignment metrics: {e}")
        
        return alignment
    
    def _calculate_motion_metrics(self, landmarks: Dict, frame_idx: int) -> Dict:
        """Calculate motion-based metrics from pose history."""
        motion = {'available': False}
        
        if len(self.pose_history) < 2:
            return motion
        
        try:
            # Compare with previous frame
            prev_landmarks = self.pose_history[-2]['landmarks']
            
            # Calculate joint velocities
            velocities = {}
            for joint_name in self.cricket_joints.keys():
                if joint_name in landmarks and joint_name in prev_landmarks:
                    current_joints = landmarks[joint_name]
                    prev_joints = prev_landmarks[joint_name]
                    
                    if current_joints and prev_joints and len(current_joints) == len(prev_joints):
                        joint_velocities = []
                        for i, (curr, prev) in enumerate(zip(current_joints, prev_joints)):
                            velocity = math.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
                            joint_velocities.append(velocity)
                        velocities[joint_name] = joint_velocities
            
            motion['velocities'] = velocities
            motion['available'] = True
            
            # Calculate overall body motion
            if velocities:
                all_velocities = []
                for joint_vels in velocities.values():
                    all_velocities.extend(joint_vels)
                
                if all_velocities:
                    motion['overall_velocity'] = sum(all_velocities) / len(all_velocities)
                    motion['max_velocity'] = max(all_velocities)
        
        except Exception as e:
            logger.warning(f"Error calculating motion metrics: {e}")
        
        return motion
    
    def _identify_bowling_phases(self, frame_analyses: List[Dict]) -> Dict:
        """Identify key phases in bowling action."""
        phases = {
            'run_up': [],
            'gather': [],
            'delivery_stride': [],
            'release': [],
            'follow_through': []
        }
        
        if len(frame_analyses) < 5:
            return phases
        
        try:
            # Analyze motion patterns to identify phases
            velocities = []
            arm_elevations = []
            
            for analysis in frame_analyses:
                motion = analysis.get('motion', {})
                if motion.get('available'):
                    velocities.append(motion.get('overall_velocity', 0))
                else:
                    velocities.append(0)
                
                angles = analysis.get('joint_angles', {})
                arm_elevation = max(
                    angles.get('left_arm_elevation', 0),
                    angles.get('right_arm_elevation', 0)
                )
                arm_elevations.append(arm_elevation)
            
            # Identify phases based on velocity and arm movement patterns
            if velocities and arm_elevations:
                # Find peaks in velocity for phase transitions
                velocity_peaks = signal.find_peaks(velocities, height=np.mean(velocities))[0]
                
                # Find peak arm elevation (likely release point)
                max_elevation_idx = np.argmax(arm_elevations)
                
                # Assign phases based on identified key points
                total_frames = len(frame_analyses)
                
                # Run-up: First 30% or until velocity peak
                runup_end = min(velocity_peaks[0] if velocity_peaks.size > 0 else int(total_frames * 0.3), 
                               int(total_frames * 0.3))
                phases['run_up'] = list(range(0, runup_end))
                
                # Gather: Next 20%
                gather_end = min(runup_end + int(total_frames * 0.2), total_frames - 3)
                phases['gather'] = list(range(runup_end, gather_end))
                
                # Delivery stride: Approach to release
                delivery_end = min(max_elevation_idx, total_frames - 2)
                phases['delivery_stride'] = list(range(gather_end, delivery_end))
                
                # Release: Peak arm elevation ± 1 frame
                release_start = max(0, max_elevation_idx - 1)
                release_end = min(total_frames - 1, max_elevation_idx + 1)
                phases['release'] = list(range(release_start, release_end + 1))
                
                # Follow-through: Remaining frames
                phases['follow_through'] = list(range(release_end + 1, total_frames))
        
        except Exception as e:
            logger.warning(f"Error identifying bowling phases: {e}")
        
        return phases
    
    def _identify_batting_phases(self, frame_analyses: List[Dict]) -> Dict:
        """Identify key phases in batting stroke."""
        phases = {
            'stance': [],
            'backswing': [],
            'downswing': [],
            'impact': [],
            'follow_through': []
        }
        
        if len(frame_analyses) < 5:
            return phases
        
        try:
            # Analyze bat position (approximated by wrist positions)
            wrist_heights = []
            wrist_positions = []
            
            for analysis in frame_analyses:
                landmarks = analysis.get('landmarks', {})
                wrists = landmarks.get('wrists', [])
                
                if wrists:
                    # Use dominant hand wrist (assume right-handed for now)
                    wrist = wrists[-1] if len(wrists) > 1 else wrists[0]
                    wrist_heights.append(wrist['y'])
                    wrist_positions.append((wrist['x'], wrist['y']))
                else:
                    wrist_heights.append(0)
                    wrist_positions.append((0, 0))
            
            if wrist_heights:
                # Find key points in batting motion
                min_height_idx = np.argmin(wrist_heights)  # Likely backswing peak
                max_velocity_idx = self._find_max_motion_frame(frame_analyses)
                
                total_frames = len(frame_analyses)
                
                # Stance: First 20% or until significant movement
                stance_end = min(int(total_frames * 0.2), min_height_idx)
                phases['stance'] = list(range(0, stance_end))
                
                # Backswing: From stance end to highest bat position
                phases['backswing'] = list(range(stance_end, min_height_idx + 1))
                
                # Downswing: From backswing peak to impact point
                impact_point = max_velocity_idx if max_velocity_idx > min_height_idx else min_height_idx + int(total_frames * 0.1)
                phases['downswing'] = list(range(min_height_idx + 1, impact_point))
                
                # Impact: Around maximum velocity point
                impact_start = max(0, impact_point - 1)
                impact_end = min(total_frames - 1, impact_point + 1)
                phases['impact'] = list(range(impact_start, impact_end + 1))
                
                # Follow-through: Remaining frames
                phases['follow_through'] = list(range(impact_end + 1, total_frames))
        
        except Exception as e:
            logger.warning(f"Error identifying batting phases: {e}")
        
        return phases
    
    def _calculate_angle_3points(self, point1: Tuple, point2: Tuple, point3: Tuple) -> float:
        """Calculate angle between three points (point2 is vertex)."""
        try:
            # Vector from point2 to point1
            vec1 = (point1[0] - point2[0], point1[1] - point2[1])
            # Vector from point2 to point3
            vec2 = (point3[0] - point2[0], point3[1] - point2[1])
            
            # Calculate angle using dot product
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
            magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
            
            angle_rad = math.acos(cos_angle)
            return math.degrees(angle_rad)
        
        except:
            return 0
    
    def _calculate_vector_angle(self, vec1: Tuple, vec2: Tuple) -> float:
        """Calculate angle between two vectors."""
        return self._calculate_angle_3points((0, 0), vec1, vec2)
    
    def _get_center_point(self, points: List[Dict]) -> Optional[Dict]:
        """Calculate center point of multiple landmarks."""
        if not points:
            return None
        
        valid_points = [p for p in points if p.get('visibility', 0) > 0.5]
        if not valid_points:
            return None
        
        center_x = sum(p['x'] for p in valid_points) / len(valid_points)
        center_y = sum(p['y'] for p in valid_points) / len(valid_points)
        center_z = sum(p['z'] for p in valid_points) / len(valid_points)
        
        return {'x': center_x, 'y': center_y, 'z': center_z}
    
    def _calculate_line_deviation(self, points: List[Tuple]) -> float:
        """Calculate deviation from straight line."""
        if len(points) < 3:
            return 0
        
        # Calculate deviation of middle points from line connecting first and last
        start, end = points[0], points[-1]
        max_deviation = 0
        
        for point in points[1:-1]:
            deviation = self._point_to_line_distance_2d(point, start, end)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation
    
    def _point_to_line_distance_2d(self, point: Tuple, line_start: Tuple, line_end: Tuple) -> float:
        """Calculate perpendicular distance from point to line in 2D."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Handle vertical line
        if x2 == x1:
            return abs(x0 - x1)
        
        # Calculate distance using cross product formula
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    def _calculate_pose_confidence(self, pose_landmarks) -> float:
        """Calculate overall pose detection confidence."""
        if not pose_landmarks:
            return 0.0
        
        confidences = []
        for landmark in pose_landmarks.landmark:
            if hasattr(landmark, 'visibility'):
                confidences.append(landmark.visibility)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _find_max_motion_frame(self, frame_analyses: List[Dict]) -> int:
        """Find frame with maximum motion."""
        max_motion = 0
        max_frame_idx = 0
        
        for i, analysis in enumerate(frame_analyses):
            motion = analysis.get('motion', {})
            if motion.get('available'):
                current_motion = motion.get('overall_velocity', 0)
                if current_motion > max_motion:
                    max_motion = current_motion
                    max_frame_idx = i
        
        return max_frame_idx
    
    def _calculate_bowling_kinematics(self, frame_analyses: List[Dict], phases: Dict) -> Dict:
        """Calculate detailed bowling kinematics."""
        kinematics = {}
        
        # Analyze each phase
        for phase_name, frame_indices in phases.items():
            if not frame_indices:
                continue
            
            phase_data = [frame_analyses[i] for i in frame_indices if i < len(frame_analyses)]
            
            if phase_data:
                phase_kinematics = {
                    'duration_frames': len(phase_data),
                    'avg_joint_angles': self._calculate_average_angles(phase_data),
                    'motion_characteristics': self._analyze_phase_motion(phase_data),
                    'alignment_quality': self._assess_phase_alignment(phase_data)
                }
                kinematics[phase_name] = phase_kinematics
        
        return kinematics
    
    def _calculate_batting_kinematics(self, frame_analyses: List[Dict], phases: Dict) -> Dict:
        """Calculate detailed batting kinematics."""
        return self._calculate_bowling_kinematics(frame_analyses, phases)  # Similar analysis
    
    def _calculate_average_angles(self, phase_data: List[Dict]) -> Dict:
        """Calculate average joint angles for a phase."""
        angle_sums = {}
        angle_counts = {}
        
        for frame_data in phase_data:
            angles = frame_data.get('joint_angles', {})
            for angle_name, angle_value in angles.items():
                if angle_name not in angle_sums:
                    angle_sums[angle_name] = 0
                    angle_counts[angle_name] = 0
                angle_sums[angle_name] += angle_value
                angle_counts[angle_name] += 1
        
        return {name: angle_sums[name] / angle_counts[name] 
                for name in angle_sums if angle_counts[name] > 0}
    
    def _analyze_phase_motion(self, phase_data: List[Dict]) -> Dict:
        """Analyze motion characteristics within a phase."""
        velocities = []
        for frame_data in phase_data:
            motion = frame_data.get('motion', {})
            if motion.get('available'):
                velocities.append(motion.get('overall_velocity', 0))
        
        if not velocities:
            return {'avg_velocity': 0, 'max_velocity': 0, 'velocity_consistency': 0}
        
        return {
            'avg_velocity': sum(velocities) / len(velocities),
            'max_velocity': max(velocities),
            'velocity_consistency': 100 - (np.std(velocities) / np.mean(velocities) * 100 if np.mean(velocities) > 0 else 0)
        }
    
    def _assess_phase_alignment(self, phase_data: List[Dict]) -> Dict:
        """Assess alignment quality within a phase."""
        alignment_scores = []
        
        for frame_data in phase_data:
            alignment = frame_data.get('alignment', {})
            spine_straightness = alignment.get('spine_straightness', 50)
            shoulder_tilt = abs(alignment.get('shoulder_tilt', 0))
            hip_tilt = abs(alignment.get('hip_tilt', 0))
            
            # Calculate overall alignment score
            alignment_score = (spine_straightness + 
                             max(0, 90 - shoulder_tilt) + 
                             max(0, 90 - hip_tilt)) / 3
            alignment_scores.append(alignment_score)
        
        if not alignment_scores:
            return {'avg_alignment': 50, 'consistency': 50}
        
        return {
            'avg_alignment': sum(alignment_scores) / len(alignment_scores),
            'consistency': max(0, 100 - np.std(alignment_scores))
        }
    
    def _assess_bowling_technique(self, kinematics: Dict) -> Dict:
        """Assess overall bowling technique quality."""
        assessment = {
            'overall_score': 0,
            'phase_scores': {},
            'strengths': [],
            'weaknesses': []
        }
        
        phase_scores = []
        
        for phase_name, phase_data in kinematics.items():
            motion_score = phase_data.get('motion_characteristics', {}).get('velocity_consistency', 50)
            alignment_score = phase_data.get('alignment_quality', {}).get('avg_alignment', 50)
            
            phase_score = (motion_score + alignment_score) / 2
            assessment['phase_scores'][phase_name] = phase_score
            phase_scores.append(phase_score)
            
            # Identify strengths and weaknesses
            if phase_score > 75:
                assessment['strengths'].append(f"Excellent {phase_name} technique")
            elif phase_score < 50:
                assessment['weaknesses'].append(f"Needs improvement in {phase_name}")
        
        assessment['overall_score'] = sum(phase_scores) / len(phase_scores) if phase_scores else 50
        
        return assessment
    
    def _assess_batting_technique(self, kinematics: Dict) -> Dict:
        """Assess overall batting technique quality."""
        return self._assess_bowling_technique(kinematics)  # Similar assessment logic
    
    def _classify_batting_stroke(self, kinematics: Dict, phases: Dict) -> Dict:
        """Classify the type of batting stroke."""
        stroke_analysis = {
            'stroke_type': 'defensive',
            'confidence': 0.5,
            'characteristics': []
        }
        
        # Analyze motion patterns to classify stroke
        if 'downswing' in kinematics and 'impact' in kinematics:
            downswing_data = kinematics['downswing']
            impact_data = kinematics['impact']
            
            max_velocity = downswing_data.get('motion_characteristics', {}).get('max_velocity', 0)
            
            if max_velocity > 50:  # High velocity indicates aggressive stroke
                stroke_analysis['stroke_type'] = 'attacking'
                stroke_analysis['confidence'] = 0.8
                stroke_analysis['characteristics'].append('High bat speed')
            elif max_velocity > 25:
                stroke_analysis['stroke_type'] = 'controlled'
                stroke_analysis['confidence'] = 0.7
                stroke_analysis['characteristics'].append('Moderate bat speed')
            else:
                stroke_analysis['stroke_type'] = 'defensive'
                stroke_analysis['confidence'] = 0.6
                stroke_analysis['characteristics'].append('Low bat speed')
        
        return stroke_analysis
    
    def _generate_bowling_recommendations(self, technique_assessment: Dict) -> List[str]:
        """Generate specific bowling technique recommendations."""
        recommendations = []
        
        overall_score = technique_assessment.get('overall_score', 50)
        weaknesses = technique_assessment.get('weaknesses', [])
        
        if overall_score < 60:
            recommendations.append("Focus on overall technique consistency across all phases")
        
        for weakness in weaknesses:
            if 'run_up' in weakness.lower():
                recommendations.append("Work on run-up rhythm and approach consistency")
            elif 'gather' in weakness.lower():
                recommendations.append("Improve gathering phase balance and alignment")
            elif 'delivery' in weakness.lower():
                recommendations.append("Focus on delivery stride length and direction")
            elif 'release' in weakness.lower():
                recommendations.append("Practice release point consistency and arm action")
            elif 'follow' in weakness.lower():
                recommendations.append("Complete follow-through for better control and reduced injury risk")
        
        if not recommendations:
            recommendations.append("Maintain current technique with focus on consistency")
        
        return recommendations
    
    def _generate_batting_recommendations(self, technique_assessment: Dict, stroke_analysis: Dict) -> List[str]:
        """Generate specific batting technique recommendations."""
        recommendations = []
        
        overall_score = technique_assessment.get('overall_score', 50)
        stroke_type = stroke_analysis.get('stroke_type', 'defensive')
        
        if overall_score < 60:
            recommendations.append("Focus on fundamental batting stance and head position")
        
        if stroke_type == 'defensive':
            recommendations.append("Practice more attacking shots with controlled aggression")
        elif stroke_type == 'attacking':
            recommendations.append("Work on shot selection and controlled aggression")
        
        recommendations.extend([
            "Maintain balanced stance throughout the shot",
            "Keep eye level consistent during ball approach",
            "Complete follow-through for better shot execution"
        ])
        
        return recommendations