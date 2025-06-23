import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import io
import base64
import logging
from typing import List, Dict, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generates cricket analysis visualizations including pitch plotting and labeled videos."""
    
    def __init__(self):
        # Cricket pitch dimensions (in meters)
        self.pitch_length = 20.12  # 22 yards
        self.pitch_width = 3.05    # 10 feet
        self.crease_length = 2.64  # 8 feet 8 inches
        
        # Visualization settings
        self.fig_width = 12
        self.fig_height = 8
        
    def generate_bowling_pitch_plot(self, tracking_data: Dict, video_info: Dict) -> str:
        """
        Generate Hawk-Eye style pitch plotting for bowling analysis.
        
        Args:
            tracking_data: Ball tracking analysis data
            video_info: Video metadata
            
        Returns:
            Base64 encoded image string of the pitch plot
        """
        try:
            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
            fig.suptitle('Bowling Analysis - Pitch Plot Visualization', fontsize=16, fontweight='bold')
            
            # Plot 1: Top-down pitch view with ball trajectory
            self._draw_cricket_pitch(ax1)
            self._plot_ball_trajectory(ax1, tracking_data)
            ax1.set_title('Ball Trajectory - Top View')
            ax1.set_xlabel('Pitch Width (m)')
            ax1.set_ylabel('Pitch Length (m)')
            
            # Plot 2: Side view with bounce analysis
            self._plot_bounce_analysis(ax2, tracking_data)
            ax2.set_title('Delivery Analysis - Side View')
            ax2.set_xlabel('Distance from Bowler (m)')
            ax2.set_ylabel('Height (m)')
            
            # Add analysis summary
            self._add_bowling_summary(fig, tracking_data)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating bowling pitch plot: {str(e)}")
            return ""
    
    def generate_labeled_batting_video(self, frames: List[np.ndarray], pose_data: Dict) -> str:
        """
        Generate labeled frames with skeleton tracking for batting analysis.
        
        Args:
            frames: List of video frames
            pose_data: Pose analysis data
            
        Returns:
            Base64 encoded composite image showing key frames with skeleton tracking
        """
        try:
            if not frames or not pose_data.get('pose_data'):
                return ""
            
            pose_frames = pose_data['pose_data']
            
            # Select key frames to display (every 5th frame or frames with good pose detection)
            key_frames = []
            for i, frame in enumerate(frames):
                if i < len(pose_frames) and pose_frames[i]['pose_detected'] and i % 5 == 0:
                    labeled_frame = frame.copy()
                    
                    # Draw pose landmarks and overlays
                    labeled_frame = self._draw_pose_landmarks(
                        labeled_frame, 
                        pose_frames[i]['landmarks'],
                        pose_frames[i]['technique_analysis']
                    )
                    
                    labeled_frame = self._add_technique_overlay(
                        labeled_frame,
                        pose_frames[i]['technique_analysis']
                    )
                    
                    # Add frame information
                    cv2.putText(labeled_frame, f'Frame: {i+1}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    key_frames.append(labeled_frame)
                    
                    if len(key_frames) >= 6:  # Limit to 6 key frames
                        break
            
            if not key_frames:
                return ""
            
            # Create composite image with key frames
            frame_height, frame_width = key_frames[0].shape[:2]
            
            # Resize frames for composite (smaller for better layout)
            target_width = 320
            target_height = int(frame_height * target_width / frame_width)
            
            resized_frames = []
            for frame in key_frames:
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)
            
            # Create grid layout (2x3 or 3x2 depending on count)
            rows = 2 if len(resized_frames) <= 4 else 3
            cols = min(3, len(resized_frames))
            
            composite_width = cols * target_width
            composite_height = rows * target_height
            composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
            
            for i, frame in enumerate(resized_frames):
                row = i // cols
                col = i % cols
                y_start = row * target_height
                y_end = y_start + target_height
                x_start = col * target_width
                x_end = x_start + target_width
                
                composite[y_start:y_end, x_start:x_end] = frame
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
            composite_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return composite_b64
            
        except Exception as e:
            logger.error(f"Error generating labeled batting frames: {str(e)}")
            return ""
    
    def _draw_cricket_pitch(self, ax):
        """Draw cricket pitch layout on the given axis."""
        # Pitch rectangle
        pitch_rect = Rectangle((0, 0), self.pitch_width, self.pitch_length, 
                              linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(pitch_rect)
        
        # Creases
        # Bowling crease (bowler's end)
        ax.plot([0, self.pitch_width], [0, 0], 'k-', linewidth=3, label='Bowling Crease')
        
        # Batting crease (batsman's end)
        ax.plot([0, self.pitch_width], [self.pitch_length, self.pitch_length], 'k-', linewidth=3, label='Batting Crease')
        
        # Popping creases
        ax.plot([0, self.pitch_width], [1.22, 1.22], 'k--', linewidth=1, alpha=0.7)
        ax.plot([0, self.pitch_width], [self.pitch_length-1.22, self.pitch_length-1.22], 'k--', linewidth=1, alpha=0.7)
        
        # Stumps
        stump_width = 0.1
        # Bowler's stumps
        ax.add_patch(Rectangle((self.pitch_width/2 - stump_width/2, -0.05), stump_width, 0.1, 
                              facecolor='brown', edgecolor='black'))
        # Batsman's stumps
        ax.add_patch(Rectangle((self.pitch_width/2 - stump_width/2, self.pitch_length-0.05), stump_width, 0.1, 
                              facecolor='brown', edgecolor='black'))
        
        ax.set_xlim(-0.5, self.pitch_width + 0.5)
        ax.set_ylim(-1, self.pitch_length + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_ball_trajectory(self, ax, tracking_data: Dict):
        """Plot ball trajectory on pitch."""
        try:
            frame_data = tracking_data.get('frame_data', [])
            if not frame_data:
                return
            
            # Extract positions from detected frames
            x_positions = []
            y_positions = []
            
            for frame in frame_data:
                if frame['ball_detected'] and frame['position']:
                    # Convert pixel coordinates to pitch coordinates (approximate)
                    # This is a simplified conversion - in reality you'd need camera calibration
                    x_pitch = (frame['position']['x'] / 1000) * self.pitch_width  # Rough conversion
                    y_pitch = (frame['position']['y'] / 1000) * self.pitch_length  # Rough conversion
                    
                    x_positions.append(x_pitch)
                    y_positions.append(y_pitch)
            
            if x_positions and y_positions:
                # Plot trajectory
                ax.plot(x_positions, y_positions, 'ro-', markersize=4, linewidth=2, 
                       color='red', alpha=0.8, label='Ball Path')
                
                # Mark bounce point if available
                trajectory_analysis = tracking_data.get('trajectory_analysis', {})
                if 'bounce_point' in trajectory_analysis:
                    bounce_pos = trajectory_analysis['bounce_point']['position']
                    if len(bounce_pos) >= 2:
                        bounce_x = (bounce_pos[0] / 1000) * self.pitch_width
                        bounce_y = (bounce_pos[1] / 1000) * self.pitch_length
                        ax.plot(bounce_x, bounce_y, 'bs', markersize=10, label='Bounce Point')
                
                # Mark release point
                if x_positions and y_positions:
                    ax.plot(x_positions[0], y_positions[0], 'go', markersize=8, label='Release Point')
                    
                ax.legend()
            
        except Exception as e:
            logger.error(f"Error plotting ball trajectory: {str(e)}")
    
    def _plot_bounce_analysis(self, ax, tracking_data: Dict):
        """Plot delivery analysis in side view."""
        try:
            delivery_metrics = tracking_data.get('delivery_metrics', {})
            
            # Create a simple side view representation
            pitch_distances = np.linspace(0, self.pitch_length, 100)
            
            # Simulate ball height based on trajectory type
            trajectory_analysis = tracking_data.get('trajectory_analysis', {})
            trajectory_type = trajectory_analysis.get('trajectory_type', 'Parabolic')
            
            if trajectory_type == 'Parabolic':
                # Parabolic trajectory
                heights = 2.5 - 0.1 * pitch_distances + 0.002 * pitch_distances**2
            elif trajectory_type == 'Rising':
                heights = 1.5 + 0.1 * pitch_distances - 0.003 * pitch_distances**2
            else:  # Falling
                heights = 3.0 - 0.2 * pitch_distances + 0.001 * pitch_distances**2
            
            # Ensure heights don't go below ground
            heights = np.maximum(heights, 0)
            
            ax.plot(pitch_distances, heights, 'r-', linewidth=3, label='Ball Trajectory')
            ax.fill_between(pitch_distances, 0, heights, alpha=0.2, color='red')
            
            # Add delivery information
            speed_kmh = delivery_metrics.get('average_speed_kmh', 0)
            speed_category = delivery_metrics.get('speed_category', 'Unknown')
            pitch_position = delivery_metrics.get('pitch_position', 'Unknown')
            
            # Add annotations
            ax.text(0.05 * self.pitch_length, max(heights) * 0.8, 
                   f'Speed: {speed_kmh:.1f} km/h\nCategory: {speed_category}\nLength: {pitch_position}',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   fontsize=10)
            
            ax.set_xlim(0, self.pitch_length)
            ax.set_ylim(0, max(heights) * 1.2 if heights.any() else 4)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            logger.error(f"Error plotting bounce analysis: {str(e)}")
    
    def _add_bowling_summary(self, fig, tracking_data: Dict):
        """Add bowling analysis summary to the figure."""
        try:
            summary_text = []
            
            # Detection rate
            detection_rate = tracking_data.get('detection_rate', 0) * 100
            summary_text.append(f"Ball Detection: {detection_rate:.1f}%")
            
            # Speed analysis
            avg_speed = tracking_data.get('average_velocity', 0)
            max_speed = tracking_data.get('max_velocity', 0)
            summary_text.append(f"Avg Speed: {avg_speed:.1f} m/s ({avg_speed*3.6:.1f} km/h)")
            summary_text.append(f"Max Speed: {max_speed:.1f} m/s ({max_speed*3.6:.1f} km/h)")
            
            # Delivery metrics
            delivery_metrics = tracking_data.get('delivery_metrics', {})
            if delivery_metrics:
                summary_text.append(f"Line: {delivery_metrics.get('line_consistency', 'Unknown')}")
                summary_text.append(f"Length: {delivery_metrics.get('pitch_position', 'Unknown')}")
            
            # Add text box
            fig.text(0.02, 0.02, '\n'.join(summary_text), 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                    fontsize=10, verticalalignment='bottom')
            
        except Exception as e:
            logger.error(f"Error adding bowling summary: {str(e)}")
    
    def _draw_pose_landmarks(self, frame: np.ndarray, landmarks: Dict, technique_analysis: Dict) -> np.ndarray:
        """Draw pose landmarks and skeleton on frame."""
        try:
            height, width = frame.shape[:2]
            
            # Define pose connections for skeleton
            connections = [
                # Body outline
                ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
                ('LEFT_SHOULDER', 'LEFT_ELBOW'),
                ('LEFT_ELBOW', 'LEFT_WRIST'),
                ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                ('RIGHT_ELBOW', 'RIGHT_WRIST'),
                ('LEFT_SHOULDER', 'LEFT_HIP'),
                ('RIGHT_SHOULDER', 'RIGHT_HIP'),
                ('LEFT_HIP', 'RIGHT_HIP'),
                ('LEFT_HIP', 'LEFT_KNEE'),
                ('LEFT_KNEE', 'LEFT_ANKLE'),
                ('RIGHT_HIP', 'RIGHT_KNEE'),
                ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            ]
            
            # Draw connections
            for connection in connections:
                start_point = landmarks.get(connection[0])
                end_point = landmarks.get(connection[1])
                
                if start_point and end_point:
                    start_pos = (int(start_point['x']), int(start_point['y']))
                    end_pos = (int(end_point['x']), int(end_point['y']))
                    
                    # Color based on technique quality
                    color = self._get_skeleton_color(technique_analysis)
                    cv2.line(frame, start_pos, end_pos, color, 3)
            
            # Draw landmarks
            for landmark_name, landmark in landmarks.items():
                if landmark:
                    pos = (int(landmark['x']), int(landmark['y']))
                    
                    # Different colors for different body parts
                    if 'HEAD' in landmark_name or 'NOSE' in landmark_name:
                        color = (255, 255, 0)  # Yellow for head
                    elif 'SHOULDER' in landmark_name:
                        color = (0, 255, 255)  # Cyan for shoulders
                    elif 'ELBOW' in landmark_name or 'WRIST' in landmark_name:
                        color = (255, 0, 255)  # Magenta for arms
                    elif 'HIP' in landmark_name:
                        color = (0, 255, 0)    # Green for hips
                    else:
                        color = (255, 0, 0)    # Red for legs
                    
                    cv2.circle(frame, pos, 6, color, -1)
                    cv2.circle(frame, pos, 8, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing pose landmarks: {str(e)}")
            return frame
    
    def _add_technique_overlay(self, frame: np.ndarray, technique_analysis: Dict) -> np.ndarray:
        """Add technique analysis overlay to frame."""
        try:
            height, width = frame.shape[:2]
            
            # Create overlay for technique scores
            overlay = frame.copy()
            
            # Background for text
            cv2.rectangle(overlay, (10, height - 200), (400, height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add technique scores
            y_offset = height - 180
            techniques = ['stance', 'head_position', 'shoulder_alignment', 'arm_position', 'balance']
            
            for technique in techniques:
                if technique in technique_analysis:
                    score = technique_analysis[technique].get('score', 0)
                    
                    # Color based on score
                    if score >= 80:
                        color = (0, 255, 0)      # Green for good
                    elif score >= 60:
                        color = (0, 255, 255)    # Yellow for average
                    else:
                        color = (0, 0, 255)      # Red for poor
                    
                    text = f"{technique.replace('_', ' ').title()}: {score:.0f}"
                    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)
                    y_offset += 30
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding technique overlay: {str(e)}")
            return frame
    
    def _get_skeleton_color(self, technique_analysis: Dict) -> Tuple[int, int, int]:
        """Get skeleton color based on overall technique quality."""
        try:
            scores = []
            for technique_data in technique_analysis.values():
                if isinstance(technique_data, dict) and 'score' in technique_data:
                    scores.append(technique_data['score'])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= 80:
                    return (0, 255, 0)      # Green for good
                elif avg_score >= 60:
                    return (0, 255, 255)    # Yellow for average
                else:
                    return (0, 0, 255)      # Red for poor
            
            return (255, 255, 255)  # White default
            
        except Exception as e:
            logger.error(f"Error getting skeleton color: {str(e)}")
            return (255, 255, 255)
    
    def generate_batting_summary_plot(self, pose_data: Dict) -> str:
        """Generate batting technique summary visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height))
            fig.suptitle('Batting Technique Analysis Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Technique scores radar chart
            self._plot_technique_radar(ax1, pose_data)
            
            # Plot 2: Consistency over time
            self._plot_consistency_timeline(ax2, pose_data)
            
            # Plot 3: Stance analysis
            self._plot_stance_analysis(ax3, pose_data)
            
            # Plot 4: Balance analysis
            self._plot_balance_analysis(ax4, pose_data)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating batting summary plot: {str(e)}")
            return ""
    
    def _plot_technique_radar(self, ax, pose_data: Dict):
        """Plot radar chart of technique scores."""
        try:
            overall_analysis = pose_data.get('overall_analysis', {})
            strengths = overall_analysis.get('strengths', [])
            
            if not strengths:
                ax.text(0.5, 0.5, 'No technique data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            categories = [strength['category'].replace('_', '\n') for strength in strengths]
            scores = [strength['score'] for strength in strengths]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]  # Close the circle
            angles += angles[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
            ax.fill(angles, scores, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_title('Technique Scores')
            ax.grid(True)
            
        except Exception as e:
            logger.error(f"Error plotting technique radar: {str(e)}")
    
    def _plot_consistency_timeline(self, ax, pose_data: Dict):
        """Plot technique consistency over time."""
        try:
            pose_frames = pose_data.get('pose_data', [])
            if not pose_frames:
                ax.text(0.5, 0.5, 'No pose data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            frame_numbers = []
            overall_scores = []
            
            for i, frame_data in enumerate(pose_frames):
                if frame_data.get('pose_detected'):
                    frame_numbers.append(i)
                    
                    # Calculate average technique score for frame
                    technique_analysis = frame_data.get('technique_analysis', {})
                    scores = []
                    for technique_data in technique_analysis.values():
                        if isinstance(technique_data, dict) and 'score' in technique_data:
                            scores.append(technique_data['score'])
                    
                    avg_score = sum(scores) / len(scores) if scores else 0
                    overall_scores.append(avg_score)
            
            if frame_numbers and overall_scores:
                ax.plot(frame_numbers, overall_scores, 'g-', linewidth=2, marker='o')
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Overall Technique Score')
                ax.set_title('Consistency Over Time')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
            
        except Exception as e:
            logger.error(f"Error plotting consistency timeline: {str(e)}")
    
    def _plot_stance_analysis(self, ax, pose_data: Dict):
        """Plot stance width and balance analysis."""
        try:
            pose_frames = pose_data.get('pose_data', [])
            if not pose_frames:
                ax.text(0.5, 0.5, 'No stance data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            stance_widths = []
            balance_scores = []
            
            for frame_data in pose_frames:
                if frame_data.get('pose_detected'):
                    technique = frame_data.get('technique_analysis', {})
                    stance = technique.get('stance', {})
                    balance = technique.get('balance', {})
                    
                    if 'foot_distance_pixels' in stance:
                        stance_widths.append(stance['foot_distance_pixels'])
                    if 'score' in balance:
                        balance_scores.append(balance['score'])
            
            if stance_widths and balance_scores:
                ax.scatter(stance_widths, balance_scores, alpha=0.7, c='red', s=50)
                ax.set_xlabel('Stance Width (pixels)')
                ax.set_ylabel('Balance Score')
                ax.set_title('Stance vs Balance Analysis')
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting stance analysis: {str(e)}")
    
    def _plot_balance_analysis(self, ax, pose_data: Dict):
        """Plot balance distribution analysis."""
        try:
            pose_frames = pose_data.get('pose_data', [])
            if not pose_frames:
                ax.text(0.5, 0.5, 'No balance data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            balance_qualities = []
            
            for frame_data in pose_frames:
                if frame_data.get('pose_detected'):
                    technique = frame_data.get('technique_analysis', {})
                    balance = technique.get('balance', {})
                    quality = balance.get('balance_quality', 'Unknown')
                    balance_qualities.append(quality)
            
            if balance_qualities:
                # Count balance quality occurrences
                from collections import Counter
                quality_counts = Counter(balance_qualities)
                
                qualities = list(quality_counts.keys())
                counts = list(quality_counts.values())
                
                colors = ['green' if q == 'Excellent' else 'yellow' if q == 'Good' else 'red' 
                         for q in qualities]
                
                ax.pie(counts, labels=qualities, colors=colors, autopct='%1.1f%%')
                ax.set_title('Balance Quality Distribution')
            
        except Exception as e:
            logger.error(f"Error plotting balance analysis: {str(e)}")