import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import mediapipe as mp

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    """Handles human pose estimation and batting technique analysis."""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Reduced for better performance
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Cricket batting key pose landmarks
        self.key_landmarks = {
            'head': [self.mp_pose.PoseLandmark.NOSE],
            'shoulders': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            ],
            'arms': [
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ],
            'torso': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            'legs': [
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
        }
        
    def analyze_batting_pose(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze batting pose and technique throughout video frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing pose analysis results
        """
        try:
            if not frames:
                return {'error': 'No frames provided'}
            
            pose_data = []
            technique_scores = []
            
            for frame_idx, frame in enumerate(frames):
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose detection
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract landmark positions
                    landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
                    
                    # Analyze batting technique for this frame
                    technique_analysis = self._analyze_batting_technique(landmarks, frame.shape)
                    
                    frame_data = {
                        'frame': frame_idx,
                        'pose_detected': True,
                        'landmarks': landmarks,
                        'technique_analysis': technique_analysis
                    }
                    
                    # Calculate overall technique score for this frame
                    frame_score = self._calculate_technique_score(technique_analysis)
                    technique_scores.append(frame_score)
                    
                else:
                    frame_data = {
                        'frame': frame_idx,
                        'pose_detected': False,
                        'landmarks': None,
                        'technique_analysis': None
                    }
                
                pose_data.append(frame_data)
            
            # Generate overall batting analysis
            overall_analysis = self._generate_overall_analysis(pose_data, technique_scores)
            
            return {
                'frames_analyzed': len(frames),
                'pose_detected_frames': len([f for f in pose_data if f['pose_detected']]),
                'detection_rate': len([f for f in pose_data if f['pose_detected']]) / len(frames),
                'frame_data': pose_data,
                'overall_analysis': overall_analysis,
                'average_technique_score': np.mean(technique_scores) if technique_scores else 0,
                'technique_consistency': np.std(technique_scores) if len(technique_scores) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Pose analysis error: {str(e)}")
            return {'error': f'Pose analysis failed: {str(e)}'}
    
    def _extract_landmarks(self, pose_landmarks, frame_shape: Tuple) -> Dict:
        """
        Extract and normalize pose landmarks.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the video frame
            
        Returns:
            Dictionary of normalized landmark positions
        """
        try:
            height, width = frame_shape[:2]
            landmarks = {}
            
            for category, landmark_list in self.key_landmarks.items():
                category_landmarks = {}
                
                for landmark_type in landmark_list:
                    landmark = pose_landmarks.landmark[landmark_type]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    visibility = landmark.visibility
                    
                    category_landmarks[landmark_type.name] = {
                        'x': x,
                        'y': y,
                        'normalized_x': landmark.x,
                        'normalized_y': landmark.y,
                        'visibility': visibility
                    }
                
                landmarks[category] = category_landmarks
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {str(e)}")
            return {}
    
    def _analyze_batting_technique(self, landmarks: Dict, frame_shape: Tuple) -> Dict:
        """
        Analyze batting technique based on pose landmarks.
        
        Args:
            landmarks: Extracted pose landmarks
            frame_shape: Shape of the video frame
            
        Returns:
            Dictionary with technique analysis
        """
        try:
            analysis = {}
            
            # Stance Analysis
            stance_analysis = self._analyze_stance(landmarks)
            analysis['stance'] = stance_analysis
            
            # Head Position Analysis
            head_analysis = self._analyze_head_position(landmarks)
            analysis['head_position'] = head_analysis
            
            # Shoulder Alignment Analysis
            shoulder_analysis = self._analyze_shoulder_alignment(landmarks)
            analysis['shoulder_alignment'] = shoulder_analysis
            
            # Arm Position Analysis
            arm_analysis = self._analyze_arm_position(landmarks)
            analysis['arm_position'] = arm_analysis
            
            # Balance Analysis
            balance_analysis = self._analyze_balance(landmarks)
            analysis['balance'] = balance_analysis
            
            # Weight Transfer Analysis
            weight_analysis = self._analyze_weight_transfer(landmarks)
            analysis['weight_transfer'] = weight_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technique analysis error: {str(e)}")
            return {}
    
    def _analyze_stance(self, landmarks: Dict) -> Dict:
        """Analyze batting stance."""
        try:
            if not landmarks.get('legs'):
                return {'error': 'Insufficient leg landmarks'}
            
            legs = landmarks['legs']
            
            # Check if key leg landmarks are visible
            left_ankle = legs.get('LEFT_ANKLE')
            right_ankle = legs.get('RIGHT_ANKLE')
            left_knee = legs.get('LEFT_KNEE')
            right_knee = legs.get('RIGHT_KNEE')
            
            if not all([left_ankle, right_ankle, left_knee, right_knee]):
                return {'error': 'Missing leg landmarks'}
            
            # Calculate foot distance (stance width)
            foot_distance = abs(left_ankle['x'] - right_ankle['x'])
            
            # Calculate knee bend angles (approximate)
            left_knee_bend = self._calculate_knee_bend(landmarks, 'left')
            right_knee_bend = self._calculate_knee_bend(landmarks, 'right')
            
            # Determine stance type
            stance_width = "Normal"
            if foot_distance < 50:  # Pixels, adjust based on frame size
                stance_width = "Narrow"
            elif foot_distance > 150:
                stance_width = "Wide"
            
            # Stance balance
            stance_balance = "Balanced"
            weight_distribution = abs(left_knee_bend - right_knee_bend)
            if weight_distribution > 20:
                stance_balance = "Unbalanced"
            
            return {
                'stance_width': stance_width,
                'foot_distance_pixels': foot_distance,
                'left_knee_bend': left_knee_bend,
                'right_knee_bend': right_knee_bend,
                'stance_balance': stance_balance,
                'score': 85 if stance_balance == "Balanced" and stance_width == "Normal" else 65
            }
            
        except Exception as e:
            logger.error(f"Stance analysis error: {str(e)}")
            return {'error': f'Stance analysis failed: {str(e)}'}
    
    def _analyze_head_position(self, landmarks: Dict) -> Dict:
        """Analyze head position and eye line."""
        try:
            if not landmarks.get('head') or not landmarks.get('shoulders'):
                return {'error': 'Missing head or shoulder landmarks'}
            
            head = landmarks['head']['NOSE']
            shoulders = landmarks['shoulders']
            
            # Calculate head position relative to shoulders
            left_shoulder = shoulders['LEFT_SHOULDER']
            right_shoulder = shoulders['RIGHT_SHOULDER']
            
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Head alignment
            head_offset_x = abs(head['x'] - shoulder_center_x)
            head_offset_y = head['y'] - shoulder_center_y
            
            # Determine head position
            head_alignment = "Good"
            if head_offset_x > 30:  # Pixels
                head_alignment = "Off-center"
            
            head_position = "Upright"
            if head_offset_y > 20:
                head_position = "Dropped"
            elif head_offset_y < -10:
                head_position = "Raised"
            
            return {
                'head_alignment': head_alignment,
                'head_position': head_position,
                'horizontal_offset': head_offset_x,
                'vertical_offset': head_offset_y,
                'score': 90 if head_alignment == "Good" and head_position == "Upright" else 70
            }
            
        except Exception as e:
            logger.error(f"Head position analysis error: {str(e)}")
            return {'error': f'Head position analysis failed: {str(e)}'}
    
    def _analyze_shoulder_alignment(self, landmarks: Dict) -> Dict:
        """Analyze shoulder alignment and level."""
        try:
            if not landmarks.get('shoulders'):
                return {'error': 'Missing shoulder landmarks'}
            
            shoulders = landmarks['shoulders']
            left_shoulder = shoulders['LEFT_SHOULDER']
            right_shoulder = shoulders['RIGHT_SHOULDER']
            
            # Calculate shoulder angle
            dy = right_shoulder['y'] - left_shoulder['y']
            dx = right_shoulder['x'] - left_shoulder['x']
            
            if dx != 0:
                shoulder_angle = np.degrees(np.arctan(dy / dx))
            else:
                shoulder_angle = 0
            
            # Determine shoulder level
            shoulder_level = "Level"
            if abs(shoulder_angle) > 10:
                shoulder_level = "Tilted"
                if shoulder_angle > 0:
                    shoulder_level = "Right shoulder high"
                else:
                    shoulder_level = "Left shoulder high"
            
            return {
                'shoulder_level': shoulder_level,
                'shoulder_angle': shoulder_angle,
                'score': 85 if shoulder_level == "Level" else 60
            }
            
        except Exception as e:
            logger.error(f"Shoulder alignment analysis error: {str(e)}")
            return {'error': f'Shoulder alignment analysis failed: {str(e)}'}
    
    def _analyze_arm_position(self, landmarks: Dict) -> Dict:
        """Analyze arm position and bat grip."""
        try:
            if not landmarks.get('arms'):
                return {'error': 'Missing arm landmarks'}
            
            arms = landmarks['arms']
            
            # Get arm landmarks
            left_elbow = arms.get('LEFT_ELBOW')
            right_elbow = arms.get('RIGHT_ELBOW')
            left_wrist = arms.get('LEFT_WRIST')
            right_wrist = arms.get('RIGHT_WRIST')
            
            if not all([left_elbow, right_elbow, left_wrist, right_wrist]):
                return {'error': 'Missing arm landmarks'}
            
            # Calculate elbow positions
            elbow_height_diff = abs(left_elbow['y'] - right_elbow['y'])
            
            # Analyze grip (based on wrist positions)
            wrist_distance = np.sqrt(
                (left_wrist['x'] - right_wrist['x'])**2 + 
                (left_wrist['y'] - right_wrist['y'])**2
            )
            
            # Determine arm position quality
            arm_position = "Good"
            if elbow_height_diff > 40:
                arm_position = "Uneven elbows"
            
            grip_assessment = "Normal"
            if wrist_distance > 100:
                grip_assessment = "Wide grip"
            elif wrist_distance < 30:
                grip_assessment = "Narrow grip"
            
            return {
                'arm_position': arm_position,
                'elbow_height_difference': elbow_height_diff,
                'grip_assessment': grip_assessment,
                'wrist_distance': wrist_distance,
                'score': 80 if arm_position == "Good" and grip_assessment == "Normal" else 65
            }
            
        except Exception as e:
            logger.error(f"Arm position analysis error: {str(e)}")
            return {'error': f'Arm position analysis failed: {str(e)}'}
    
    def _analyze_balance(self, landmarks: Dict) -> Dict:
        """Analyze overall body balance."""
        try:
            if not landmarks.get('torso') or not landmarks.get('legs'):
                return {'error': 'Missing torso or leg landmarks'}
            
            torso = landmarks['torso']
            legs = landmarks['legs']
            
            # Calculate center of gravity approximation
            left_hip = torso['LEFT_HIP']
            right_hip = torso['RIGHT_HIP']
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            
            # Base of support (between feet)
            left_ankle = legs['LEFT_ANKLE']
            right_ankle = legs['RIGHT_ANKLE']
            foot_center_x = (left_ankle['x'] + right_ankle['x']) / 2
            
            # Balance assessment
            balance_offset = abs(hip_center_x - foot_center_x)
            
            balance_quality = "Excellent"
            if balance_offset > 20:
                balance_quality = "Good"
            if balance_offset > 40:
                balance_quality = "Poor"
            
            return {
                'balance_quality': balance_quality,
                'balance_offset': balance_offset,
                'hip_center': hip_center_x,
                'foot_center': foot_center_x,
                'score': 90 if balance_quality == "Excellent" else (70 if balance_quality == "Good" else 50)
            }
            
        except Exception as e:
            logger.error(f"Balance analysis error: {str(e)}")
            return {'error': f'Balance analysis failed: {str(e)}'}
    
    def _analyze_weight_transfer(self, landmarks: Dict) -> Dict:
        """Analyze weight transfer during batting motion."""
        try:
            # This is a simplified analysis - in a real application,
            # you'd track this across multiple frames
            if not landmarks.get('legs'):
                return {'error': 'Missing leg landmarks'}
            
            legs = landmarks['legs']
            left_knee = legs.get('LEFT_KNEE')
            right_knee = legs.get('RIGHT_KNEE')
            
            if not left_knee or not right_knee:
                return {'error': 'Missing knee landmarks'}
            
            # Analyze knee bend to infer weight distribution
            left_knee_bend = self._calculate_knee_bend(landmarks, 'left')
            right_knee_bend = self._calculate_knee_bend(landmarks, 'right')
            
            # Determine weight distribution
            weight_distribution = "Balanced"
            weight_bias = abs(left_knee_bend - right_knee_bend)
            
            if weight_bias > 15:
                if left_knee_bend > right_knee_bend:
                    weight_distribution = "Left-weighted"
                else:
                    weight_distribution = "Right-weighted"
            
            return {
                'weight_distribution': weight_distribution,
                'left_knee_bend': left_knee_bend,
                'right_knee_bend': right_knee_bend,
                'weight_bias': weight_bias,
                'score': 85 if weight_distribution == "Balanced" else 70
            }
            
        except Exception as e:
            logger.error(f"Weight transfer analysis error: {str(e)}")
            return {'error': f'Weight transfer analysis failed: {str(e)}'}
    
    def _calculate_knee_bend(self, landmarks: Dict, side: str) -> float:
        """Calculate knee bend angle."""
        try:
            legs = landmarks.get('legs', {})
            torso = landmarks.get('torso', {})
            
            if side == 'left':
                hip = torso.get('LEFT_HIP')
                knee = legs.get('LEFT_KNEE')
                ankle = legs.get('LEFT_ANKLE')
            else:
                hip = torso.get('RIGHT_HIP')
                knee = legs.get('RIGHT_KNEE')
                ankle = legs.get('RIGHT_ANKLE')
            
            if not all([hip, knee, ankle]):
                return 0
            
            # Calculate vectors
            thigh_vector = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
            shin_vector = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
            
            # Calculate angle between vectors
            cos_angle = np.dot(thigh_vector, shin_vector) / (
                np.linalg.norm(thigh_vector) * np.linalg.norm(shin_vector)
            )
            
            # Clamp to valid range for arccos
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Convert to bend angle (180 - angle for knee bend)
            bend_angle = 180 - angle
            
            return max(0, bend_angle)
            
        except Exception as e:
            logger.error(f"Knee bend calculation error: {str(e)}")
            return 0
    
    def _calculate_technique_score(self, technique_analysis: Dict) -> float:
        """Calculate overall technique score for a frame."""
        try:
            scores = []
            
            for category, analysis in technique_analysis.items():
                if isinstance(analysis, dict) and 'score' in analysis:
                    scores.append(analysis['score'])
            
            return np.mean(scores) if scores else 0
            
        except Exception as e:
            logger.error(f"Technique score calculation error: {str(e)}")
            return 0
    
    def _generate_overall_analysis(self, pose_data: List[Dict], technique_scores: List[float]) -> Dict:
        """Generate overall batting analysis summary."""
        try:
            if not technique_scores:
                return {'error': 'No technique scores available'}
            
            # Calculate statistics
            avg_score = np.mean(technique_scores)
            consistency = 100 - min(100, np.std(technique_scores) * 2)  # Convert std to consistency score
            
            # Determine overall grade
            if avg_score >= 85:
                grade = "A"
                performance = "Excellent"
            elif avg_score >= 75:
                grade = "B"
                performance = "Good"
            elif avg_score >= 65:
                grade = "C"
                performance = "Fair"
            else:
                grade = "D"
                performance = "Needs Improvement"
            
            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            
            # Analyze each technique category
            for frame_data in pose_data:
                if frame_data.get('technique_analysis'):
                    for category, analysis in frame_data['technique_analysis'].items():
                        if isinstance(analysis, dict) and 'score' in analysis:
                            if analysis['score'] >= 80:
                                if category not in [s['category'] for s in strengths]:
                                    strengths.append({
                                        'category': category,
                                        'score': analysis['score']
                                    })
                            elif analysis['score'] < 65:
                                if category not in [w['category'] for w in weaknesses]:
                                    weaknesses.append({
                                        'category': category,
                                        'score': analysis['score']
                                    })
            
            return {
                'overall_score': avg_score,
                'consistency_score': consistency,
                'grade': grade,
                'performance_level': performance,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': self._generate_recommendations(weaknesses)
            }
            
        except Exception as e:
            logger.error(f"Overall analysis generation error: {str(e)}")
            return {'error': f'Overall analysis failed: {str(e)}'}
    
    def _generate_recommendations(self, weaknesses: List[Dict]) -> List[str]:
        """Generate technique improvement recommendations."""
        recommendations = []
        
        for weakness in weaknesses:
            category = weakness['category']
            
            if category == 'stance':
                recommendations.append("Work on maintaining a balanced stance with feet shoulder-width apart")
            elif category == 'head_position':
                recommendations.append("Keep your head still and eyes level, watching the ball closely")
            elif category == 'shoulder_alignment':
                recommendations.append("Ensure shoulders are level and aligned toward the target")
            elif category == 'arm_position':
                recommendations.append("Focus on keeping elbows at the same height and maintaining proper grip")
            elif category == 'balance':
                recommendations.append("Practice balance drills to improve stability during shots")
            elif category == 'weight_transfer':
                recommendations.append("Work on smooth weight transfer from back foot to front foot")
        
        if not recommendations:
            recommendations.append("Continue practicing to maintain your excellent technique!")
        
        return recommendations
