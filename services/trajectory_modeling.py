"""Advanced trajectory modeling for cricket ball physics simulation."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class BallTrajectoryModel:
    """Physics-based cricket ball trajectory modeling with air resistance and Magnus effect."""
    
    def __init__(self):
        # Physical constants
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.rho = 1.225  # Air density at sea level (kg/m³)
        self.c_d = 0.47  # Drag coefficient for sphere
        self.r = 0.0365  # Cricket ball radius (m)
        self.mass = 0.1633  # Cricket ball mass (kg)
        self.area = np.pi * self.r**2  # Cross-sectional area
        
        # Cricket-specific parameters
        self.pitch_length = 20.12  # Length of cricket pitch (m)
        self.pitch_width = 3.05  # Width of cricket pitch (m)
        self.stump_height = 0.71  # Height of stumps (m)
        
        # Calibration parameters
        self.pixels_per_meter = 100  # Default, should be calibrated
        
    def predict_trajectory(self, initial_pos: np.ndarray, initial_vel: np.ndarray, 
                          spin_vector: np.ndarray = None, time_span: float = 2.0) -> Dict:
        """
        Predict complete ball trajectory using physics simulation.
        
        Args:
            initial_pos: Initial position [x, y, z] in meters
            initial_vel: Initial velocity [vx, vy, vz] in m/s
            spin_vector: Angular velocity [wx, wy, wz] in rad/s
            time_span: Maximum simulation time in seconds
            
        Returns:
            Dictionary with trajectory data and analysis
        """
        if spin_vector is None:
            spin_vector = np.array([0.0, 0.0, 0.0])
        
        # Initial state vector [x, y, z, vx, vy, vz]
        initial_state = np.concatenate([initial_pos, initial_vel])
        
        # Solve trajectory differential equation
        try:
            solution = solve_ivp(
                self._trajectory_equations,
                [0, time_span],
                initial_state,
                args=(spin_vector,),
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
                max_step=0.01
            )
            
            if not solution.success:
                logger.warning("Trajectory integration failed, using simplified model")
                return self._simplified_trajectory(initial_pos, initial_vel, time_span)
            
            # Extract trajectory points
            time_points = np.linspace(0, solution.t[-1], 100)
            trajectory = solution.sol(time_points)
            
            # Analyze trajectory
            analysis = self._analyze_trajectory(trajectory, time_points, initial_vel, spin_vector)
            
            return {
                'trajectory': trajectory,
                'time_points': time_points,
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Trajectory prediction failed: {e}")
            return self._simplified_trajectory(initial_pos, initial_vel, time_span)
    
    def _trajectory_equations(self, t: float, state: np.ndarray, spin_vector: np.ndarray) -> np.ndarray:
        """
        Differential equations for cricket ball trajectory with aerodynamics.
        
        Args:
            t: Time
            state: Current state [x, y, z, vx, vy, vz]
            spin_vector: Ball spin [wx, wy, wz]
            
        Returns:
            Derivative of state vector
        """
        # Extract position and velocity
        pos = state[:3]
        vel = state[3:]
        speed = np.linalg.norm(vel)
        
        if speed < 0.1:  # Prevent division by zero
            return np.array([*vel, 0, 0, -self.g])
        
        # Drag force (opposing velocity)
        drag_force = -0.5 * self.rho * self.c_d * self.area * speed * vel
        
        # Magnus force (perpendicular to velocity and spin)
        magnus_force = self._calculate_magnus_force(vel, spin_vector)
        
        # Total acceleration
        acceleration = (drag_force + magnus_force) / self.mass + np.array([0, 0, -self.g])
        
        return np.concatenate([vel, acceleration])
    
    def _calculate_magnus_force(self, velocity: np.ndarray, spin_vector: np.ndarray) -> np.ndarray:
        """
        Calculate Magnus force due to ball spin.
        
        Args:
            velocity: Velocity vector [vx, vy, vz]
            spin_vector: Angular velocity vector [wx, wy, wz]
            
        Returns:
            Magnus force vector
        """
        # Magnus coefficient for cricket ball
        c_m = 0.5  # Typical value for cricket ball
        
        # Magnus force = (1/2) * rho * area * c_m * |v| * (omega × v)
        cross_product = np.cross(spin_vector, velocity)
        speed = np.linalg.norm(velocity)
        
        if speed < 0.1:
            return np.zeros(3)
        
        magnus_force = 0.5 * self.rho * self.area * c_m * speed * cross_product
        
        return magnus_force
    
    def _simplified_trajectory(self, initial_pos: np.ndarray, initial_vel: np.ndarray, 
                              time_span: float) -> Dict:
        """
        Simplified trajectory calculation for fallback.
        
        Args:
            initial_pos: Initial position
            initial_vel: Initial velocity
            time_span: Time span
            
        Returns:
            Simplified trajectory data
        """
        time_points = np.linspace(0, time_span, 50)
        trajectory = np.zeros((6, len(time_points)))
        
        for i, t in enumerate(time_points):
            # Simple projectile motion with basic drag
            drag_factor = np.exp(-0.1 * t)  # Simplified drag decay
            
            # Position
            trajectory[0, i] = initial_pos[0] + initial_vel[0] * t * drag_factor
            trajectory[1, i] = initial_pos[1] + initial_vel[1] * t * drag_factor
            trajectory[2, i] = initial_pos[2] + initial_vel[2] * t - 0.5 * self.g * t**2
            
            # Velocity
            trajectory[3, i] = initial_vel[0] * drag_factor
            trajectory[4, i] = initial_vel[1] * drag_factor
            trajectory[5, i] = initial_vel[2] - self.g * t
        
        analysis = {
            'max_height': np.max(trajectory[2, :]),
            'range': trajectory[0, -1] - trajectory[0, 0],
            'flight_time': time_span,
            'impact_velocity': np.linalg.norm(trajectory[3:, -1]),
            'simplified': True
        }
        
        return {
            'trajectory': trajectory,
            'time_points': time_points,
            'analysis': analysis,
            'success': False
        }
    
    def _analyze_trajectory(self, trajectory: np.ndarray, time_points: np.ndarray,
                           initial_vel: np.ndarray, spin_vector: np.ndarray) -> Dict:
        """
        Analyze trajectory for cricket-specific metrics.
        
        Args:
            trajectory: Trajectory data [6 x n] (position and velocity)
            time_points: Time points
            initial_vel: Initial velocity
            spin_vector: Ball spin
            
        Returns:
            Trajectory analysis dictionary
        """
        positions = trajectory[:3, :]
        velocities = trajectory[3:, :]
        
        # Basic trajectory metrics
        max_height = np.max(positions[2, :])
        range_distance = positions[0, -1] - positions[0, 0]
        flight_time = time_points[-1]
        
        # Speed analysis
        speeds = np.linalg.norm(velocities, axis=0)
        initial_speed = np.linalg.norm(initial_vel)
        final_speed = speeds[-1]
        avg_speed = np.mean(speeds)
        
        # Find bounce point (when ball hits ground level)
        bounce_idx = self._find_bounce_point(positions[2, :])
        bounce_point = None
        bounce_angle = None
        
        if bounce_idx is not None and bounce_idx < len(positions[0, :]):
            bounce_point = positions[:, bounce_idx]
            if bounce_idx > 0:
                bounce_velocity = velocities[:, bounce_idx]
                bounce_angle = np.degrees(np.arctan2(bounce_velocity[2], 
                                                   np.linalg.norm(bounce_velocity[:2])))
        
        # Swing analysis
        lateral_deviation = self._calculate_lateral_deviation(positions)
        
        # Release point analysis
        release_height = positions[2, 0]
        release_angle = np.degrees(np.arctan2(initial_vel[2], 
                                            np.linalg.norm(initial_vel[:2])))
        
        return {
            'max_height': max_height,
            'range': range_distance,
            'flight_time': flight_time,
            'initial_speed': initial_speed,
            'final_speed': final_speed,
            'avg_speed': avg_speed,
            'speed_loss': initial_speed - final_speed,
            'bounce_point': bounce_point,
            'bounce_angle': bounce_angle,
            'lateral_deviation': lateral_deviation,
            'release_height': release_height,
            'release_angle': release_angle,
            'spin_rate': np.linalg.norm(spin_vector),
            'swing_type': self._classify_swing(lateral_deviation, spin_vector),
            'simplified': False
        }
    
    def _find_bounce_point(self, heights: np.ndarray) -> Optional[int]:
        """Find the index where ball hits ground (z=0)."""
        for i in range(len(heights) - 1):
            if heights[i] > 0 and heights[i + 1] <= 0:
                return i
        return None
    
    def _calculate_lateral_deviation(self, positions: np.ndarray) -> float:
        """Calculate maximum lateral deviation from straight path."""
        if positions.shape[1] < 2:
            return 0.0
        
        # Calculate straight line from start to end
        start_point = positions[:2, 0]
        end_point = positions[:2, -1]
        
        max_deviation = 0.0
        for i in range(1, positions.shape[1] - 1):
            current_point = positions[:2, i]
            
            # Calculate perpendicular distance to straight line
            deviation = self._point_to_line_distance(current_point, start_point, end_point)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, 
                               line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line."""
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # Project point onto line
        projection = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        projection = np.clip(projection, 0, 1)
        
        closest_point = line_start + projection * line_vec
        return np.linalg.norm(point - closest_point)
    
    def _classify_swing(self, lateral_deviation: float, spin_vector: np.ndarray) -> str:
        """Classify the type of swing based on deviation and spin."""
        if lateral_deviation < 0.05:  # Less than 5cm deviation
            return "straight"
        
        spin_magnitude = np.linalg.norm(spin_vector)
        
        if spin_magnitude < 10:  # Low spin
            return "conventional_swing"
        elif lateral_deviation > 0.2:  # Significant deviation
            return "reverse_swing"
        else:
            return "swing"
    
    def estimate_ball_parameters(self, detected_positions: List[Tuple[float, float]], 
                               timestamps: List[float]) -> Dict:
        """
        Estimate initial velocity and spin from detected ball positions.
        
        Args:
            detected_positions: List of (x, y) pixel positions
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with estimated parameters
        """
        if len(detected_positions) < 3:
            return {'success': False, 'error': 'Insufficient data points'}
        
        # Convert to numpy arrays
        positions = np.array(detected_positions)
        times = np.array(timestamps)
        
        # Convert pixel coordinates to world coordinates (simplified)
        world_positions = self._pixels_to_world(positions)
        
        # Estimate initial velocity using finite differences
        if len(world_positions) >= 2:
            dt = times[1] - times[0]
            initial_velocity = (world_positions[1] - world_positions[0]) / dt
        else:
            initial_velocity = np.array([20.0, 0.0, 0.0])  # Default bowling speed
        
        # Estimate spin from trajectory curvature
        spin_estimate = self._estimate_spin_from_trajectory(world_positions, times)
        
        return {
            'success': True,
            'initial_position': world_positions[0],
            'initial_velocity': initial_velocity,
            'spin_vector': spin_estimate,
            'confidence': min(1.0, len(detected_positions) / 10.0)
        }
    
    def _pixels_to_world(self, pixel_positions: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to world coordinates."""
        # Simplified conversion - in practice, use camera calibration
        world_positions = np.zeros((len(pixel_positions), 3))
        
        for i, (px, py) in enumerate(pixel_positions):
            # Assume camera is positioned at standard cricket filming position
            # This is a simplified conversion - real implementation needs calibration
            world_positions[i, 0] = px / self.pixels_per_meter  # x-direction
            world_positions[i, 1] = (480 - py) / self.pixels_per_meter  # y-direction (flip Y)
            world_positions[i, 2] = 2.0  # Assume constant height for simplicity
        
        return world_positions
    
    def _estimate_spin_from_trajectory(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Estimate ball spin from trajectory curvature."""
        if len(positions) < 4:
            return np.array([0.0, 0.0, 0.0])
        
        # Calculate curvature from trajectory
        # This is a simplified estimation - real implementation would use curve fitting
        spin_magnitude = 0.0
        
        try:
            # Fit parabola to trajectory points
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            if len(x_coords) >= 3:
                # Fit quadratic curve
                coeffs = np.polyfit(x_coords, y_coords, 2)
                curvature = abs(coeffs[0])  # Second derivative gives curvature
                
                # Convert curvature to approximate spin
                spin_magnitude = curvature * 100  # Empirical scaling
        
        except Exception as e:
            logger.warning(f"Spin estimation failed: {e}")
            spin_magnitude = 0.0
        
        # Assume spin is primarily about vertical axis for side spin
        return np.array([0.0, 0.0, spin_magnitude])
    
    def predict_hawkeye_path(self, ball_detections: List[Dict], frame_rate: float = 30.0) -> Dict:
        """
        Generate Hawk-Eye style ball path prediction.
        
        Args:
            ball_detections: List of ball detection dictionaries
            frame_rate: Video frame rate
            
        Returns:
            Hawk-Eye prediction data
        """
        if len(ball_detections) < 3:
            return {'success': False, 'error': 'Insufficient detections for Hawk-Eye prediction'}
        
        # Extract positions and timestamps
        positions = []
        timestamps = []
        
        for i, detection in enumerate(ball_detections):
            if detection and 'bbox' in detection:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append((center_x, center_y))
                timestamps.append(i / frame_rate)
        
        if len(positions) < 3:
            return {'success': False, 'error': 'Insufficient valid positions'}
        
        # Estimate ball parameters
        params = self.estimate_ball_parameters(positions, timestamps)
        
        if not params['success']:
            return params
        
        # Predict full trajectory
        trajectory_data = self.predict_trajectory(
            params['initial_position'],
            params['initial_velocity'],
            params['spin_vector'],
            time_span=3.0
        )
        
        # Generate Hawk-Eye visualization data
        hawkeye_data = {
            'success': True,
            'ball_path': trajectory_data['trajectory'][:3, :],  # Only position data
            'time_points': trajectory_data['time_points'],
            'predicted_bounce': trajectory_data['analysis'].get('bounce_point'),
            'predicted_stumps_impact': self._predict_stumps_impact(trajectory_data),
            'swing_analysis': {
                'swing_type': trajectory_data['analysis'].get('swing_type', 'unknown'),
                'lateral_deviation': trajectory_data['analysis'].get('lateral_deviation', 0),
                'max_height': trajectory_data['analysis'].get('max_height', 0)
            },
            'speed_analysis': {
                'release_speed': trajectory_data['analysis'].get('initial_speed', 0) * 3.6,  # km/h
                'predicted_speed_at_batsman': trajectory_data['analysis'].get('final_speed', 0) * 3.6
            },
            'confidence': params['confidence']
        }
        
        return hawkeye_data
    
    def _predict_stumps_impact(self, trajectory_data: Dict) -> Optional[Dict]:
        """Predict if ball would hit stumps."""
        trajectory = trajectory_data['trajectory']
        
        # Find where ball crosses batsman's crease (simplified)
        crease_position = 17.68  # Distance from bowling crease to batting crease
        
        for i in range(len(trajectory[0, :]) - 1):
            if trajectory[0, i] <= crease_position <= trajectory[0, i + 1]:
                # Interpolate position at crease
                t = (crease_position - trajectory[0, i]) / (trajectory[0, i + 1] - trajectory[0, i])
                height_at_crease = trajectory[2, i] + t * (trajectory[2, i + 1] - trajectory[2, i])
                width_at_crease = trajectory[1, i] + t * (trajectory[1, i + 1] - trajectory[1, i])
                
                # Check if ball would hit stumps
                stump_impact = (0 <= height_at_crease <= self.stump_height and 
                               abs(width_at_crease) <= 0.11)  # Stump width
                
                return {
                    'impact': stump_impact,
                    'height_at_crease': height_at_crease,
                    'width_at_crease': width_at_crease,
                    'distance_from_center': abs(width_at_crease)
                }
        
        return None