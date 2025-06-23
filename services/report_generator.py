import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive analysis reports for cricket performance."""
    
    def __init__(self):
        self.bowling_metrics = [
            'speed', 'accuracy', 'line', 'length', 'trajectory'
        ]
        self.batting_metrics = [
            'stance', 'balance', 'timing', 'technique', 'consistency'
        ]
    
    def generate_bowling_report(self, tracking_data: Dict, video_info: Dict) -> Dict:
        """
        Generate comprehensive bowling analysis report.
        
        Args:
            tracking_data: Ball tracking analysis data
            video_info: Video metadata
            
        Returns:
            Dictionary containing detailed bowling report
        """
        try:
            if 'error' in tracking_data:
                return {
                    'error': 'Cannot generate report due to tracking errors',
                    'details': tracking_data['error']
                }
            
            # Extract key metrics
            avg_velocity = tracking_data.get('average_velocity', 0)
            max_velocity = tracking_data.get('max_velocity', 0)
            detection_rate = tracking_data.get('detection_rate', 0)
            delivery_metrics = tracking_data.get('delivery_metrics', {})
            trajectory_analysis = tracking_data.get('trajectory_analysis', {})
            
            # Generate performance summary
            performance_summary = self._generate_bowling_performance_summary(
                avg_velocity, max_velocity, detection_rate, delivery_metrics
            )
            
            # Generate detailed analysis
            detailed_analysis = self._generate_bowling_detailed_analysis(
                tracking_data, trajectory_analysis, delivery_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_bowling_recommendations(
                performance_summary, delivery_metrics
            )
            
            # Calculate overall scores
            overall_scores = self._calculate_bowling_scores(
                delivery_metrics, trajectory_analysis, detection_rate
            )
            
            report = {
                'report_type': 'bowling_analysis',
                'generated_at': datetime.now().isoformat(),
                'video_info': video_info,
                'analysis_summary': {
                    'total_frames': tracking_data.get('frames_analyzed', 0),
                    'ball_detected_frames': tracking_data.get('ball_detected_frames', 0),
                    'detection_rate': f"{detection_rate * 100:.1f}%",
                    'analysis_quality': self._assess_analysis_quality(detection_rate)
                },
                'performance_metrics': {
                    'speed_analysis': {
                        'average_speed_kmh': round(avg_velocity * 3.6, 1),
                        'max_speed_kmh': round(max_velocity * 3.6, 1),
                        'speed_category': delivery_metrics.get('speed_category', 'Unknown'),
                        'speed_consistency': self._calculate_speed_consistency(tracking_data.get('velocity_data', []))
                    },
                    'accuracy_analysis': {
                        'line_consistency': delivery_metrics.get('line_consistency', 'Unknown'),
                        'length_assessment': delivery_metrics.get('pitch_position', 'Unknown'),
                        'accuracy_score': delivery_metrics.get('delivery_accuracy', {}).get('line_score', 0)
                    },
                    'trajectory_analysis': {
                        'trajectory_type': trajectory_analysis.get('trajectory_type', 'Unknown'),
                        'trajectory_smoothness': trajectory_analysis.get('trajectory_smoothness', 0),
                        'bounce_point': trajectory_analysis.get('bounce_point', {})
                    }
                },
                'performance_summary': performance_summary,
                'detailed_analysis': detailed_analysis,
                'overall_scores': overall_scores,
                'recommendations': recommendations,
                'technical_data': {
                    'ball_tracking_data': tracking_data,
                    'processing_notes': self._generate_processing_notes(tracking_data)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Bowling report generation error: {str(e)}")
            return {
                'error': 'Failed to generate bowling report',
                'details': str(e)
            }
    
    def generate_batting_report(self, pose_data: Dict, video_info: Dict) -> Dict:
        """
        Generate comprehensive batting analysis report.
        
        Args:
            pose_data: Pose analysis data
            video_info: Video metadata
            
        Returns:
            Dictionary containing detailed batting report
        """
        try:
            if 'error' in pose_data:
                return {
                    'error': 'Cannot generate report due to pose analysis errors',
                    'details': pose_data['error']
                }
            
            # Extract key metrics
            avg_technique_score = pose_data.get('average_technique_score', 0)
            detection_rate = pose_data.get('detection_rate', 0)
            overall_analysis = pose_data.get('overall_analysis', {})
            technique_consistency = pose_data.get('technique_consistency', 0)
            
            # Generate performance summary
            performance_summary = self._generate_batting_performance_summary(
                avg_technique_score, detection_rate, overall_analysis
            )
            
            # Generate detailed technique analysis
            detailed_analysis = self._generate_batting_detailed_analysis(
                pose_data, overall_analysis
            )
            
            # Generate recommendations
            recommendations = overall_analysis.get('recommendations', [])
            
            # Calculate scores for different aspects
            aspect_scores = self._calculate_batting_aspect_scores(pose_data)
            
            report = {
                'report_type': 'batting_analysis',
                'generated_at': datetime.now().isoformat(),
                'video_info': video_info,
                'analysis_summary': {
                    'total_frames': pose_data.get('frames_analyzed', 0),
                    'pose_detected_frames': pose_data.get('pose_detected_frames', 0),
                    'detection_rate': f"{detection_rate * 100:.1f}%",
                    'analysis_quality': self._assess_analysis_quality(detection_rate)
                },
                'performance_metrics': {
                    'overall_technique_score': round(avg_technique_score, 1),
                    'technique_grade': overall_analysis.get('grade', 'N/A'),
                    'performance_level': overall_analysis.get('performance_level', 'Unknown'),
                    'consistency_score': round(100 - min(100, technique_consistency * 2), 1),
                    'aspect_scores': aspect_scores
                },
                'technique_analysis': {
                    'strengths': overall_analysis.get('strengths', []),
                    'weaknesses': overall_analysis.get('weaknesses', []),
                    'key_observations': self._generate_key_observations(pose_data)
                },
                'performance_summary': performance_summary,
                'detailed_analysis': detailed_analysis,
                'improvement_plan': {
                    'immediate_focus': self._get_immediate_focus_areas(overall_analysis.get('weaknesses', [])),
                    'recommendations': recommendations,
                    'practice_drills': self._suggest_practice_drills(overall_analysis.get('weaknesses', []))
                },
                'technical_data': {
                    'pose_analysis_data': pose_data,
                    'processing_notes': self._generate_pose_processing_notes(pose_data)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Batting report generation error: {str(e)}")
            return {
                'error': 'Failed to generate batting report',
                'details': str(e)
            }
    
    def _generate_bowling_performance_summary(self, avg_velocity: float, max_velocity: float, 
                                            detection_rate: float, delivery_metrics: Dict) -> Dict:
        """Generate bowling performance summary."""
        try:
            speed_kmh = avg_velocity * 3.6
            
            summary = {
                'overall_rating': 'Good',
                'key_highlights': [],
                'areas_for_improvement': []
            }
            
            # Speed analysis
            if speed_kmh > 140:
                summary['key_highlights'].append(f"Excellent pace - averaging {speed_kmh:.1f} km/h")
                summary['overall_rating'] = 'Excellent'
            elif speed_kmh > 120:
                summary['key_highlights'].append(f"Good pace - averaging {speed_kmh:.1f} km/h")
            else:
                summary['areas_for_improvement'].append(f"Consider working on pace - currently {speed_kmh:.1f} km/h")
            
            # Line and length
            line_consistency = delivery_metrics.get('line_consistency', '')
            if line_consistency == 'Good':
                summary['key_highlights'].append("Consistent line bowling")
            else:
                summary['areas_for_improvement'].append("Work on line consistency")
            
            # Length analysis
            pitch_position = delivery_metrics.get('pitch_position', '')
            if pitch_position == 'Good Length':
                summary['key_highlights'].append("Excellent length control")
            else:
                summary['areas_for_improvement'].append(f"Focus on length - currently bowling {pitch_position.lower()}")
            
            # Overall rating adjustment
            if len(summary['areas_for_improvement']) > len(summary['key_highlights']):
                summary['overall_rating'] = 'Needs Improvement'
            elif len(summary['key_highlights']) > 2:
                summary['overall_rating'] = 'Excellent'
            
            return summary
            
        except Exception as e:
            logger.error(f"Bowling performance summary error: {str(e)}")
            return {'error': 'Failed to generate performance summary'}
    
    def _generate_batting_performance_summary(self, avg_score: float, detection_rate: float, 
                                            overall_analysis: Dict) -> Dict:
        """Generate batting performance summary."""
        try:
            summary = {
                'overall_rating': overall_analysis.get('performance_level', 'Unknown'),
                'technique_grade': overall_analysis.get('grade', 'N/A'),
                'key_strengths': [],
                'primary_focus_areas': []
            }
            
            # Extract strengths and weaknesses
            strengths = overall_analysis.get('strengths', [])
            weaknesses = overall_analysis.get('weaknesses', [])
            
            # Format strengths
            for strength in strengths[:3]:  # Top 3 strengths
                category = strength.get('category', '').replace('_', ' ').title()
                score = strength.get('score', 0)
                summary['key_strengths'].append(f"{category} - {score:.0f}/100")
            
            # Format weaknesses
            for weakness in weaknesses[:3]:  # Top 3 areas for improvement
                category = weakness.get('category', '').replace('_', ' ').title()
                score = weakness.get('score', 0)
                summary['primary_focus_areas'].append(f"{category} - {score:.0f}/100")
            
            return summary
            
        except Exception as e:
            logger.error(f"Batting performance summary error: {str(e)}")
            return {'error': 'Failed to generate performance summary'}
    
    def _generate_bowling_detailed_analysis(self, tracking_data: Dict, trajectory_analysis: Dict, 
                                          delivery_metrics: Dict) -> Dict:
        """Generate detailed bowling analysis."""
        try:
            analysis = {
                'ball_tracking_quality': {
                    'frames_with_ball_detected': tracking_data.get('ball_detected_frames', 0),
                    'total_frames_analyzed': tracking_data.get('frames_analyzed', 0),
                    'tracking_reliability': f"{tracking_data.get('detection_rate', 0) * 100:.1f}%"
                },
                'speed_breakdown': {
                    'average_speed': f"{tracking_data.get('average_velocity', 0) * 3.6:.1f} km/h",
                    'peak_speed': f"{tracking_data.get('max_velocity', 0) * 3.6:.1f} km/h",
                    'speed_category': delivery_metrics.get('speed_category', 'Unknown'),
                    'speed_variations': self._analyze_speed_variations(tracking_data.get('velocity_data', []))
                },
                'accuracy_assessment': {
                    'line_analysis': {
                        'consistency': delivery_metrics.get('line_consistency', 'Unknown'),
                        'variation': f"{delivery_metrics.get('line_variation_pixels', 0):.1f} pixels"
                    },
                    'length_analysis': {
                        'pitch_position': delivery_metrics.get('pitch_position', 'Unknown'),
                        'length_score': delivery_metrics.get('delivery_accuracy', {}).get('length_score', 0)
                    }
                },
                'trajectory_details': {
                    'ball_path': trajectory_analysis.get('trajectory_type', 'Unknown'),
                    'smoothness_score': f"{trajectory_analysis.get('trajectory_smoothness', 0):.1f}/100",
                    'bounce_analysis': trajectory_analysis.get('bounce_point', {})
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Detailed bowling analysis error: {str(e)}")
            return {'error': 'Failed to generate detailed analysis'}
    
    def _generate_batting_detailed_analysis(self, pose_data: Dict, overall_analysis: Dict) -> Dict:
        """Generate detailed batting analysis."""
        try:
            frame_data = pose_data.get('frame_data', [])
            
            # Analyze technique categories across all frames
            technique_breakdown = {}
            
            for frame in frame_data:
                if frame.get('technique_analysis'):
                    for category, analysis in frame['technique_analysis'].items():
                        if isinstance(analysis, dict) and 'score' in analysis:
                            if category not in technique_breakdown:
                                technique_breakdown[category] = []
                            technique_breakdown[category].append(analysis['score'])
            
            # Calculate averages for each technique aspect
            technique_averages = {}
            for category, scores in technique_breakdown.items():
                technique_averages[category] = {
                    'average_score': np.mean(scores),
                    'consistency': 100 - min(100, np.std(scores) * 2),
                    'category_name': category.replace('_', ' ').title()
                }
            
            analysis = {
                'pose_detection_quality': {
                    'frames_with_pose_detected': pose_data.get('pose_detected_frames', 0),
                    'total_frames_analyzed': pose_data.get('frames_analyzed', 0),
                    'detection_reliability': f"{pose_data.get('detection_rate', 0) * 100:.1f}%"
                },
                'technique_breakdown': technique_averages,
                'consistency_analysis': {
                    'overall_consistency': f"{100 - min(100, pose_data.get('technique_consistency', 0) * 2):.1f}%",
                    'most_consistent_aspect': self._find_most_consistent_aspect(technique_breakdown),
                    'least_consistent_aspect': self._find_least_consistent_aspect(technique_breakdown)
                },
                'performance_trends': self._analyze_performance_trends(frame_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Detailed batting analysis error: {str(e)}")
            return {'error': 'Failed to generate detailed analysis'}
    
    def _calculate_bowling_scores(self, delivery_metrics: Dict, trajectory_analysis: Dict, detection_rate: float) -> Dict:
        """Calculate overall bowling scores."""
        try:
            scores = {}
            
            # Speed score (based on category)
            speed_category = delivery_metrics.get('speed_category', 'Slow')
            speed_scores = {'Slow': 60, 'Medium': 75, 'Fast': 90, 'Express': 95}
            scores['speed'] = speed_scores.get(speed_category, 60)
            
            # Accuracy score
            line_score = delivery_metrics.get('delivery_accuracy', {}).get('line_score', 0)
            length_score = delivery_metrics.get('delivery_accuracy', {}).get('length_score', 0)
            scores['accuracy'] = (line_score + length_score) / 2
            
            # Consistency score (based on trajectory smoothness)
            scores['consistency'] = trajectory_analysis.get('trajectory_smoothness', 0)
            
            # Overall score
            scores['overall'] = np.mean(list(scores.values()))
            
            return scores
            
        except Exception as e:
            logger.error(f"Bowling scores calculation error: {str(e)}")
            return {'error': 'Failed to calculate scores'}
    
    def _calculate_batting_aspect_scores(self, pose_data: Dict) -> Dict:
        """Calculate scores for different batting aspects."""
        try:
            frame_data = pose_data.get('frame_data', [])
            aspect_scores = {}
            
            # Collect scores for each aspect
            for frame in frame_data:
                if frame.get('technique_analysis'):
                    for aspect, analysis in frame['technique_analysis'].items():
                        if isinstance(analysis, dict) and 'score' in analysis:
                            if aspect not in aspect_scores:
                                aspect_scores[aspect] = []
                            aspect_scores[aspect].append(analysis['score'])
            
            # Calculate average scores
            averaged_scores = {}
            for aspect, scores in aspect_scores.items():
                averaged_scores[aspect.replace('_', ' ').title()] = round(np.mean(scores), 1)
            
            return averaged_scores
            
        except Exception as e:
            logger.error(f"Batting aspect scores calculation error: {str(e)}")
            return {}
    
    def _generate_bowling_recommendations(self, performance_summary: Dict, delivery_metrics: Dict) -> List[str]:
        """Generate bowling improvement recommendations."""
        recommendations = []
        
        # Speed recommendations
        speed_category = delivery_metrics.get('speed_category', 'Slow')
        if speed_category in ['Slow', 'Medium']:
            recommendations.append("Focus on building pace through strength training and proper technique")
        
        # Line and length recommendations
        line_consistency = delivery_metrics.get('line_consistency', '')
        if line_consistency != 'Good':
            recommendations.append("Practice line drills using target areas on the pitch")
        
        pitch_position = delivery_metrics.get('pitch_position', '')
        if pitch_position not in ['Good Length', 'Full']:
            recommendations.append("Work on length control - practice hitting specific areas consistently")
        
        # General recommendations
        recommendations.extend([
            "Regular video analysis to track improvement",
            "Focus on consistent release point",
            "Maintain smooth bowling action"
        ])
        
        return recommendations
    
    def _generate_key_observations(self, pose_data: Dict) -> List[str]:
        """Generate key observations from pose analysis."""
        observations = []
        
        frame_data = pose_data.get('frame_data', [])
        if not frame_data:
            return observations
        
        # Find common technique patterns
        technique_issues = {}
        for frame in frame_data:
            if frame.get('technique_analysis'):
                for category, analysis in frame['technique_analysis'].items():
                    if isinstance(analysis, dict):
                        # Look for specific technique issues
                        if category == 'stance' and analysis.get('stance_balance') == 'Unbalanced':
                            technique_issues['stance_balance'] = technique_issues.get('stance_balance', 0) + 1
                        elif category == 'head_position' and analysis.get('head_alignment') != 'Good':
                            technique_issues['head_alignment'] = technique_issues.get('head_alignment', 0) + 1
        
        # Generate observations based on frequency of issues
        total_frames = len([f for f in frame_data if f.get('pose_detected')])
        for issue, count in technique_issues.items():
            if count > total_frames * 0.3:  # If issue appears in >30% of frames
                issue_name = issue.replace('_', ' ').title()
                observations.append(f"{issue_name} needs attention - observed in {count}/{total_frames} frames")
        
        if not observations:
            observations.append("Overall technique shows good consistency")
        
        return observations
    
    def _get_immediate_focus_areas(self, weaknesses: List[Dict]) -> List[str]:
        """Get immediate focus areas from weaknesses."""
        focus_areas = []
        
        # Prioritize critical areas
        priority_order = ['balance', 'stance', 'head_position', 'shoulder_alignment', 'arm_position']
        
        for priority in priority_order:
            for weakness in weaknesses:
                if weakness.get('category') == priority:
                    focus_areas.append(priority.replace('_', ' ').title())
                    break
        
        return focus_areas[:3]  # Top 3 priorities
    
    def _suggest_practice_drills(self, weaknesses: List[Dict]) -> List[str]:
        """Suggest specific practice drills based on weaknesses."""
        drills = []
        
        weakness_categories = [w.get('category') for w in weaknesses]
        
        if 'stance' in weakness_categories:
            drills.append("Mirror work for stance positioning")
            drills.append("Balance exercises on one foot")
        
        if 'head_position' in weakness_categories:
            drills.append("Head still drills with tennis ball on head")
            drills.append("Eye tracking exercises")
        
        if 'balance' in weakness_categories:
            drills.append("Single-leg balance exercises")
            drills.append("Wobble board training")
        
        if 'arm_position' in weakness_categories:
            drills.append("Shadow batting with focus on arm position")
            drills.append("Grip strengthening exercises")
        
        if not drills:
            drills.extend([
                "Regular net practice",
                "Video analysis sessions",
                "Technique refinement with coach"
            ])
        
        return drills
    
    def _assess_analysis_quality(self, detection_rate: float) -> str:
        """Assess the quality of analysis based on detection rate."""
        if detection_rate >= 0.8:
            return "High"
        elif detection_rate >= 0.6:
            return "Good"
        elif detection_rate >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_speed_consistency(self, velocity_data: List[float]) -> str:
        """Calculate speed consistency rating."""
        if not velocity_data or len(velocity_data) < 2:
            return "Unknown"
        
        std_dev = np.std(velocity_data)
        mean_vel = np.mean(velocity_data)
        
        if mean_vel == 0:
            return "Unknown"
        
        coefficient_of_variation = std_dev / mean_vel
        
        if coefficient_of_variation < 0.1:
            return "Excellent"
        elif coefficient_of_variation < 0.2:
            return "Good"
        elif coefficient_of_variation < 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def _analyze_speed_variations(self, velocity_data: List[float]) -> Dict:
        """Analyze speed variations in delivery."""
        if not velocity_data:
            return {'error': 'No velocity data available'}
        
        return {
            'min_speed_kmh': round(min(velocity_data) * 3.6, 1),
            'max_speed_kmh': round(max(velocity_data) * 3.6, 1),
            'speed_range_kmh': round((max(velocity_data) - min(velocity_data)) * 3.6, 1),
            'standard_deviation': round(np.std(velocity_data) * 3.6, 1)
        }
    
    def _find_most_consistent_aspect(self, technique_breakdown: Dict) -> str:
        """Find the most consistent technique aspect."""
        if not technique_breakdown:
            return "Unknown"
        
        min_std = float('inf')
        most_consistent = "Unknown"
        
        for category, scores in technique_breakdown.items():
            std_dev = np.std(scores)
            if std_dev < min_std:
                min_std = std_dev
                most_consistent = category.replace('_', ' ').title()
        
        return most_consistent
    
    def _find_least_consistent_aspect(self, technique_breakdown: Dict) -> str:
        """Find the least consistent technique aspect."""
        if not technique_breakdown:
            return "Unknown"
        
        max_std = 0
        least_consistent = "Unknown"
        
        for category, scores in technique_breakdown.items():
            std_dev = np.std(scores)
            if std_dev > max_std:
                max_std = std_dev
                least_consistent = category.replace('_', ' ').title()
        
        return least_consistent
    
    def _analyze_performance_trends(self, frame_data: List[Dict]) -> Dict:
        """Analyze performance trends across frames."""
        try:
            scores = []
            
            for frame in frame_data:
                if frame.get('technique_analysis'):
                    frame_scores = []
                    for analysis in frame['technique_analysis'].values():
                        if isinstance(analysis, dict) and 'score' in analysis:
                            frame_scores.append(analysis['score'])
                    
                    if frame_scores:
                        scores.append(np.mean(frame_scores))
            
            if len(scores) < 3:
                return {'trend': 'Insufficient data'}
            
            # Simple trend analysis
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            
            if second_avg > first_avg + 5:
                trend = "Improving"
            elif second_avg < first_avg - 5:
                trend = "Declining"
            else:
                trend = "Stable"
            
            return {
                'trend': trend,
                'first_half_average': round(first_avg, 1),
                'second_half_average': round(second_avg, 1),
                'improvement': round(second_avg - first_avg, 1)
            }
            
        except Exception as e:
            logger.error(f"Performance trends analysis error: {str(e)}")
            return {'trend': 'Analysis failed'}
    
    def _generate_processing_notes(self, tracking_data: Dict) -> List[str]:
        """Generate processing notes for bowling analysis."""
        notes = []
        
        detection_rate = tracking_data.get('detection_rate', 0)
        if detection_rate < 0.5:
            notes.append("Low ball detection rate - consider better lighting or camera angle")
        
        if tracking_data.get('ball_detected_frames', 0) < 10:
            notes.append("Limited tracking data - longer video sequences recommended")
        
        if not notes:
            notes.append("Analysis completed successfully with good data quality")
        
        return notes
    
    def _generate_pose_processing_notes(self, pose_data: Dict) -> List[str]:
        """Generate processing notes for pose analysis."""
        notes = []
        
        detection_rate = pose_data.get('detection_rate', 0)
        if detection_rate < 0.6:
            notes.append("Low pose detection rate - ensure clear view of player")
        
        if pose_data.get('pose_detected_frames', 0) < 10:
            notes.append("Limited pose data - longer video sequences with clear player visibility recommended")
        
        if not notes:
            notes.append("Pose analysis completed successfully with good detection quality")
        
        return notes
