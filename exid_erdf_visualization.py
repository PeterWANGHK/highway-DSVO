"""
exiD Dataset: Enhanced Driving Risk Field (EDRF) Visualization
================================================================
Implements the EDRF model from:
"EDRF: Enhanced Driving Risk Field Based on Multimodal Trajectory 
Prediction and Its Applications" (IEEE ITSC 2024)

Features:
1. Driving Risk Probability (DRP) along predicted trajectories
2. Gaussian cross-section model with uncertainty propagation
3. Virtual mass model for consequence assessment
4. Interaction Risk (IR) computation between vehicle pairs
5. Applications: Traffic risk monitoring, ego-vehicle risk analysis

Reference: Jiang et al. (2024) "EDRF: Enhanced Driving Risk Field Based on 
Multimodal Trajectory Prediction and Its Applications" IEEE ITSC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import defaultdict
from scipy.interpolate import splprep, splev
from scipy.integrate import cumulative_trapezoid
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EDRFConfig:
    """Configuration for EDRF model based on the paper."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range (meters, relative to ego)
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 30.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # EDRF grid resolution
    EDRF_GRID_X: int = 80
    EDRF_GRID_Y: int = 40
    
    # DRP Parameters (Table II in paper)
    Q: float = 0.0001          # Parabola steepness for height a(s)
    B: float = 0.04            # Base slope for width σ(s)
    K: float = 1.0             # Curvature influence on width
    C: float = 0.5             # Initial Gaussian width
    
    # Virtual Mass Parameters (Equation 5 in paper)
    ALPHA: float = 1.566e-14   # Velocity coefficient
    BETA_VM: float = 6.687     # Velocity exponent
    GAMMA: float = 0.3345      # Base term
    
    # Ego-vehicle DRP Parameters (Table III in paper) - uses Laplace distribution
    Q_EGO: float = 0.004       # Linear slope for ego height
    B_EGO: float = 0.05        # Base slope for ego width
    K_EGO: float = 1.0         # Steering angle influence
    C_EGO: float = 0.5         # Initial Laplace width
    
    # Trajectory prediction parameters
    PREDICTION_HORIZON: float = 6.0   # seconds (t_la in paper)
    NUM_PREDICTION_MODES: int = 6     # Number of multimodal predictions
    DT: float = 0.1                   # Time step for trajectory sampling
    
    # Vehicle parameters
    MASS_HV: float = 15000.0   # Heavy vehicle mass (kg)
    MASS_PC: float = 3000.0    # Passenger car mass (kg)
    WHEELBASE: float = 2.7     # Default wheelbase (m)
    
    # Vehicle type factors
    TYPE_FACTOR_HV: float = 1.5
    TYPE_FACTOR_PC: float = 1.0
    
    # Vehicle dimensions (for visualization)
    TRUCK_LENGTH: float = 12.0
    TRUCK_WIDTH: float = 2.5
    CAR_LENGTH: float = 4.5
    CAR_WIDTH: float = 1.8
    
    # Risk threshold
    RISK_THRESHOLD: float = 0.1
    
    # Visualization
    FPS: int = 25
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
        'van': '#9B59B6',
    })


# =============================================================================
# Trajectory Prediction (Lane-Constrained Multimodal)
# =============================================================================

# Standard lane width (meters)
LANE_WIDTH = 3.5


def generate_lane_keeping_trajectory(veh: Dict, horizon: float, dt: float,
                                      speed_factor: float = 1.0) -> np.ndarray:
    """
    Generate trajectory for lane keeping behavior.
    Vehicle continues straight along its current heading.
    
    Args:
        veh: Vehicle state
        horizon: Prediction horizon in seconds
        dt: Time step
        speed_factor: Speed multiplier (for acceleration/deceleration modes)
    
    Returns:
        trajectory: Array of (x, y) points
    """
    x, y = veh['x'], veh['y']
    heading = veh['heading']
    v = veh['speed'] * speed_factor
    
    trajectory = [(x, y)]
    
    n_steps = int(horizon / dt)
    for _ in range(n_steps):
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        trajectory.append((x, y))
    
    return np.array(trajectory)


def generate_lane_change_trajectory(veh: Dict, horizon: float, dt: float,
                                     direction: str = 'left',
                                     lane_width: float = LANE_WIDTH,
                                     lc_duration: float = 4.0) -> np.ndarray:
    """
    Generate trajectory for lane change maneuver using sinusoidal lateral profile.
    
    The lane change follows a smooth sinusoidal profile:
    d(t) = (D/2) * (1 - cos(π * t / T))
    
    where D is the lane width and T is the lane change duration.
    
    Args:
        veh: Vehicle state
        horizon: Prediction horizon in seconds
        dt: Time step
        direction: 'left' or 'right'
        lane_width: Width of lane to change into
        lc_duration: Duration of lane change maneuver
    
    Returns:
        trajectory: Array of (x, y) points
    """
    x0, y0 = veh['x'], veh['y']
    heading = veh['heading']
    v = veh['speed']
    
    # Direction multiplier
    d_sign = 1.0 if direction == 'left' else -1.0
    
    # Unit vectors in vehicle frame
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    trajectory = [(x0, y0)]
    
    n_steps = int(horizon / dt)
    lc_steps = int(lc_duration / dt)
    
    for i in range(1, n_steps + 1):
        t = i * dt
        
        # Longitudinal position (along heading)
        s = v * t
        
        # Lateral offset using sinusoidal profile
        if i <= lc_steps:
            # During lane change: smooth sinusoidal transition
            d = d_sign * (lane_width / 2) * (1 - np.cos(np.pi * t / lc_duration))
        else:
            # After lane change: maintain new lane position
            d = d_sign * lane_width
        
        # Convert to global coordinates
        # x_local = s (forward), y_local = d (lateral)
        x = x0 + s * cos_h - d * sin_h
        y = y0 + s * sin_h + d * cos_h
        
        trajectory.append((x, y))
    
    return np.array(trajectory)


def generate_multimodal_predictions(veh: Dict, config: EDRFConfig) -> List[Dict]:
    """
    Generate multimodal trajectory predictions with lane-constrained maneuvers.
    
    Generates trajectories for:
    1. Lane keeping (maintain speed)
    2. Lane keeping (deceleration)
    3. Lane keeping (acceleration)
    4. Lane change left
    5. Lane change right
    6. Lane change left (slower)
    
    Probabilities are adjusted based on current lateral velocity (lane change intention).
    
    Args:
        veh: Vehicle state
        config: EDRF configuration
    
    Returns:
        List of prediction dicts with 'trajectory', 'probability', 'curvature'
    """
    predictions = []
    
    # Detect lane change intention from lateral velocity
    heading = veh['heading']
    # Lateral velocity in vehicle frame (positive = left)
    lat_v = -veh.get('vx', 0) * np.sin(heading) + veh.get('vy', 0) * np.cos(heading)
    
    # Base probabilities for different maneuvers
    # These will be adjusted based on observed lateral motion
    base_probs = {
        'lane_keep': 0.40,
        'lane_keep_decel': 0.12,
        'lane_keep_accel': 0.08,
        'lane_change_left': 0.15,
        'lane_change_right': 0.15,
        'lane_change_left_slow': 0.10,
    }
    
    # Adjust probabilities based on lateral velocity
    if lat_v > 0.3:  # Moving left
        base_probs['lane_change_left'] *= 2.0
        base_probs['lane_change_left_slow'] *= 1.5
        base_probs['lane_change_right'] *= 0.3
    elif lat_v < -0.3:  # Moving right
        base_probs['lane_change_right'] *= 2.0
        base_probs['lane_change_left'] *= 0.3
        base_probs['lane_change_left_slow'] *= 0.3
    
    # Lane change duration varies with speed
    base_lc_duration = 4.0  # seconds
    speed = max(veh['speed'], 5.0)
    lc_duration = max(2.5, min(5.0, base_lc_duration * 15.0 / speed))
    
    # Generate trajectories for each mode
    modes = [
        {
            'name': 'lane_keep',
            'trajectory': generate_lane_keeping_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT, speed_factor=1.0
            ),
            'prob': base_probs['lane_keep']
        },
        {
            'name': 'lane_keep_decel',
            'trajectory': generate_lane_keeping_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT, speed_factor=0.7
            ),
            'prob': base_probs['lane_keep_decel']
        },
        {
            'name': 'lane_keep_accel',
            'trajectory': generate_lane_keeping_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT, speed_factor=1.2
            ),
            'prob': base_probs['lane_keep_accel']
        },
        {
            'name': 'lane_change_left',
            'trajectory': generate_lane_change_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT,
                direction='left', lc_duration=lc_duration
            ),
            'prob': base_probs['lane_change_left']
        },
        {
            'name': 'lane_change_right',
            'trajectory': generate_lane_change_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT,
                direction='right', lc_duration=lc_duration
            ),
            'prob': base_probs['lane_change_right']
        },
        {
            'name': 'lane_change_left_slow',
            'trajectory': generate_lane_change_trajectory(
                veh, config.PREDICTION_HORIZON, config.DT,
                direction='left', lc_duration=lc_duration * 1.3
            ),
            'prob': base_probs['lane_change_left_slow']
        },
    ]
    
    # Normalize probabilities
    total_prob = sum(m['prob'] for m in modes)
    
    for mode in modes:
        traj = mode['trajectory']
        prob = mode['prob'] / total_prob
        
        # Compute average curvature
        curvature = compute_trajectory_curvature(traj)
        
        predictions.append({
            'trajectory': traj,
            'probability': prob,
            'curvature': curvature,
            'name': mode['name']
        })
    
    return predictions[:config.NUM_PREDICTION_MODES]


def compute_trajectory_curvature(trajectory: np.ndarray) -> float:
    """Compute average curvature of a trajectory."""
    if len(trajectory) < 3:
        return 0.0
    
    # Compute curvature using finite differences
    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])
    
    # Avoid division by zero for stationary points
    ds = np.sqrt(dx**2 + dy**2)
    ds = np.where(ds < 1e-6, 1e-6, ds)
    
    # Tangent angles
    theta = np.arctan2(dy, dx)
    
    # Curvature = dθ/ds
    dtheta = np.diff(theta)
    # Handle angle wrapping
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    
    curvatures = np.abs(dtheta) / ds[:-1]
    
    return float(np.mean(curvatures))


# =============================================================================
# Frenet Coordinate System
# =============================================================================

def trajectory_to_frenet(trajectory: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Build Frenet coordinate system along a trajectory.
    
    Returns:
        s_values: Arc length along trajectory for each point
        total_length: Total trajectory length
    """
    if len(trajectory) < 2:
        return np.array([0.0]), 0.0
    
    # Compute arc length
    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    s_values = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    return s_values, s_values[-1]


def cartesian_to_frenet(point: np.ndarray, trajectory: np.ndarray, 
                        s_values: np.ndarray) -> Tuple[float, float]:
    """
    Convert Cartesian point to Frenet coordinates (s, d) along trajectory.
    
    Args:
        point: (x, y) point in Cartesian coordinates
        trajectory: Reference trajectory points
        s_values: Arc length values for trajectory
    
    Returns:
        (s, d): Frenet coordinates - s is arc length, d is lateral offset
    """
    if len(trajectory) < 2:
        return 0.0, np.linalg.norm(point - trajectory[0])
    
    # Find closest point on trajectory
    distances = np.sqrt(np.sum((trajectory - point)**2, axis=1))
    closest_idx = np.argmin(distances)
    
    # Get s coordinate
    s = s_values[closest_idx]
    
    # Compute signed lateral distance d
    if closest_idx < len(trajectory) - 1:
        tangent = trajectory[closest_idx + 1] - trajectory[closest_idx]
    else:
        tangent = trajectory[closest_idx] - trajectory[closest_idx - 1]
    
    tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
    normal = np.array([-tangent[1], tangent[0]])
    
    vec_to_point = point - trajectory[closest_idx]
    d = np.dot(vec_to_point, normal)
    
    return s, d


# =============================================================================
# DRP Model (Driving Risk Probability)
# =============================================================================

def compute_drp_single_trajectory(X: np.ndarray, Y: np.ndarray, 
                                   trajectory: np.ndarray,
                                   probability: float,
                                   curvature: float,
                                   config: EDRFConfig) -> np.ndarray:
    """
    Compute DRP (Driving Risk Probability) for a single predicted trajectory.
    
    Based on Equation (1)-(3) in the paper:
    DRP(s,d) = a(s) * exp(-d² / (2σ(s)²))
    a(s) = q(s - s_pt)²
    σ(s) = (b + k * κ_pt) * s + c
    
    Args:
        X, Y: Grid coordinates
        trajectory: Predicted trajectory points
        probability: Trajectory probability p_i
        curvature: Average curvature κ_pt
        config: EDRF configuration
    
    Returns:
        DRP field weighted by probability
    """
    s_values, s_pt = trajectory_to_frenet(trajectory)
    
    if s_pt < 0.1:
        return np.zeros_like(X)
    
    drp = np.zeros_like(X)
    
    # For each grid point, compute DRP
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            
            # Convert to Frenet coordinates
            s, d = cartesian_to_frenet(point, trajectory, s_values)
            
            # Skip points too far from trajectory
            if s < 0 or s > s_pt * 1.1 or abs(d) > 20:
                continue
            
            # Height function a(s) - Equation (2)
            # Parabolic decrease: maximum at origin, zero at trajectory end
            a_s = config.Q * (s - s_pt)**2
            
            # Width function σ(s) - Equation (3)
            # Linear increase with curvature influence
            sigma_s = (config.B + config.K * curvature) * s + config.C
            sigma_s = max(sigma_s, 0.1)  # Prevent division by zero
            
            # Gaussian cross-section - Equation (1)
            drp[i, j] += probability * a_s * np.exp(-d**2 / (2 * sigma_s**2))
    
    return drp


def compute_drp_ego(X: np.ndarray, Y: np.ndarray,
                    trajectory: np.ndarray,
                    steering_angle: float,
                    config: EDRFConfig) -> np.ndarray:
    """
    Compute DRP for ego vehicle using Laplace-like distribution.
    
    Based on Equations (10)-(12) in the paper:
    DRP_ego(s,d) = a_ego(s) * exp(-|d| / λ(s))
    a_ego(s) = q_ego * |s - v*t_la|
    λ(s) = (b_ego + k_ego * |δ|) * s + c_ego
    
    Ego vehicle has lower uncertainty, hence Laplace instead of Gaussian.
    """
    s_values, s_pt = trajectory_to_frenet(trajectory)
    
    if s_pt < 0.1:
        return np.zeros_like(X)
    
    drp = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            s, d = cartesian_to_frenet(point, trajectory, s_values)
            
            if s < 0 or s > s_pt * 1.1 or abs(d) > 15:
                continue
            
            # Height function - Equation (11) - linear decrease
            a_ego_s = config.Q_EGO * abs(s - s_pt)
            
            # Width function - Equation (12)
            lambda_s = (config.B_EGO + config.K_EGO * abs(steering_angle)) * s + config.C_EGO
            lambda_s = max(lambda_s, 0.1)
            
            # Laplace distribution - Equation (10)
            drp[i, j] = a_ego_s * np.exp(-abs(d) / lambda_s)
    
    return drp


def compute_complete_drp(X: np.ndarray, Y: np.ndarray,
                          predictions: List[Dict],
                          config: EDRFConfig) -> np.ndarray:
    """
    Compute complete DRP from multimodal predictions.
    
    Based on Equation (4) in the paper:
    DRP^C(x,y) = Σ p_i * a(s_i) * exp(-d_i² / (2σ(s_i)²))
    """
    drp_complete = np.zeros_like(X)
    
    for pred in predictions:
        drp_single = compute_drp_single_trajectory(
            X, Y,
            pred['trajectory'],
            pred['probability'],
            pred['curvature'],
            config
        )
        drp_complete += drp_single
    
    return drp_complete


# =============================================================================
# Virtual Mass Model
# =============================================================================

def compute_virtual_mass(veh: Dict, config: EDRFConfig) -> float:
    """
    Compute virtual mass M representing consequence severity.
    
    Based on Equation (5) in the paper:
    M = m * T * (α * v^β + γ)
    
    Args:
        veh: Vehicle dict with mass, speed, class
        config: EDRF configuration
    
    Returns:
        Virtual mass M
    """
    m = veh['mass']
    v = max(veh['speed'], 0.1)  # Avoid zero velocity
    
    # Type factor T
    if veh['class'] in config.HEAVY_VEHICLE_CLASSES:
        T = config.TYPE_FACTOR_HV
    else:
        T = config.TYPE_FACTOR_PC
    
    # Virtual mass formula
    M = m * T * (config.ALPHA * v**config.BETA_VM + config.GAMMA)
    
    return M


# =============================================================================
# EDRF Model (Enhanced Driving Risk Field)
# =============================================================================

def compute_edrf(X: np.ndarray, Y: np.ndarray, 
                  veh: Dict, 
                  predictions: List[Dict],
                  config: EDRFConfig,
                  is_ego: bool = False) -> np.ndarray:
    """
    Compute complete EDRF for a vehicle.
    
    Based on Equation (6) in the paper:
    EDRF(x,y) = DRP^C(x,y) * M
    
    Args:
        X, Y: Grid coordinates
        veh: Vehicle state
        predictions: Multimodal trajectory predictions
        config: EDRF configuration
        is_ego: Whether this is the ego vehicle
    
    Returns:
        EDRF field
    """
    # Compute DRP
    if is_ego and len(predictions) > 0:
        # Ego uses Laplace distribution
        steering = veh.get('steering_angle', 0.0) or 0.0
        drp = compute_drp_ego(X, Y, predictions[0]['trajectory'], steering, config)
    else:
        # Other vehicles use Gaussian with multimodal predictions
        drp = compute_complete_drp(X, Y, predictions, config)
    
    # Compute virtual mass
    M = compute_virtual_mass(veh, config)
    
    # EDRF = DRP * M
    edrf = drp * M
    
    return edrf


# =============================================================================
# Interaction Risk (IR)
# =============================================================================

def compute_interaction_risk(edrf_i: np.ndarray, edrf_j: np.ndarray) -> np.ndarray:
    """
    Compute Interaction Risk between two vehicles.
    
    Based on Equation (7) in the paper:
    IR_ij(x,y) = EDRF_i(x,y) * EDRF_j(x,y)
    """
    return edrf_i * edrf_j


def compute_risk_level(ir: np.ndarray) -> float:
    """
    Compute risk level F between two entities.
    
    Based on Equation (8) in the paper:
    F_ij = max_{x,y} IR_ij(x,y)
    """
    return float(np.max(ir))


# =============================================================================
# Data Loader (Same as original)
# =============================================================================

class ExiDLoader:
    """Load exiD data for EDRF visualization."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = EDRFConfig()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.merge_bounds: Dict[int, Tuple[float, float]] = {}
        self.recording_meta = None
        self.background_image = None
        self.ortho_px_to_meter = 0.1
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_meta_df = pd.read_csv(rec_meta_path)
                if not rec_meta_df.empty:
                    self.recording_meta = rec_meta_df.iloc[0]
                    self.ortho_px_to_meter = float(self.recording_meta.get('orthoPxToMeter', self.ortho_px_to_meter))
            
            bg_path = self.data_dir / f"{prefix}background.png"
            if bg_path.exists():
                self.background_image = plt.imread(str(bg_path))
                logger.info("Loaded lane layout background image.")
            
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            logger.info(f"Loaded recording {recording_id}")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int, heading_tol_deg: float = 60.0) -> Optional[Dict]:
        """Get a snapshot of ego and surrounding vehicles at a frame."""
        
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        
        if frame_data.empty:
            return None
        
        ego_row = frame_data[frame_data['trackId'] == ego_id]
        if ego_row.empty:
            return None
        
        ego_row = ego_row.iloc[0]
        vclass = str(ego_row.get('class', 'car')).lower()
        
        ego = {
            'id': ego_id,
            'x': float(ego_row['xCenter']),
            'y': float(ego_row['yCenter']),
            'heading': np.radians(float(ego_row.get('heading', 0))),
            'vx': float(ego_row.get('xVelocity', 0)),
            'vy': float(ego_row.get('yVelocity', 0)),
            'ax': float(ego_row.get('xAcceleration', 0)),
            'ay': float(ego_row.get('yAcceleration', 0)),
            'speed': np.sqrt(ego_row.get('xVelocity', 0)**2 + ego_row.get('yVelocity', 0)**2),
            'width': float(ego_row.get('width', 2.0)),
            'length': float(ego_row.get('length', 5.0)),
            'class': vclass,
            'mass': self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            'steering_angle': None,
        }
        
        surrounding = []
        
        for _, row in frame_data.iterrows():
            if row['trackId'] == ego_id:
                continue
            
            other_class = str(row.get('class', 'car')).lower()
            other = {
                'id': int(row['trackId']),
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'heading': np.radians(float(row.get('heading', 0))),
                'vx': float(row.get('xVelocity', 0)),
                'vy': float(row.get('yVelocity', 0)),
                'ax': float(row.get('xAcceleration', 0)),
                'ay': float(row.get('yAcceleration', 0)),
                'speed': np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                'width': float(row.get('width', 2.0)),
                'length': float(row.get('length', 5.0)),
                'class': other_class,
                'mass': self.config.MASS_HV if other_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
                'steering_angle': None,
            }
            
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            if (-self.config.OBS_RANGE_BEHIND <= dx <= self.config.OBS_RANGE_AHEAD and
                -self.config.OBS_RANGE_RIGHT <= dy <= self.config.OBS_RANGE_LEFT and
                self._is_same_direction(ego, other, heading_tol_deg)):
                surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def _is_same_direction(self, ego: Dict, other: Dict, heading_tol_deg: float = 60.0) -> bool:
        """Check if another vehicle travels roughly the same direction as ego."""
        
        d_heading = np.abs(np.arctan2(
            np.sin(other['heading'] - ego['heading']),
            np.cos(other['heading'] - ego['heading'])
        ))
        if d_heading > np.radians(heading_tol_deg):
            return False
        
        ego_v = np.array([ego['vx'], ego['vy']])
        other_v = np.array([other['vx'], other['vy']])
        if np.linalg.norm(ego_v) > 0.5 and np.linalg.norm(other_v) > 0.5:
            cos_sim = np.dot(ego_v, other_v) / (np.linalg.norm(ego_v) * np.linalg.norm(other_v))
            return cos_sim > np.cos(np.radians(heading_tol_deg))
        
        return True
    
    def find_best_interaction_frame(self, ego_id: int) -> Optional[int]:
        """Find frame with most surrounding vehicles for ego."""
        
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        
        merge_bounds = self._get_merge_bounds(ego_id)
        if merge_bounds is not None:
            s_series = None
            if 'lonLaneletPos' in ego_data.columns:
                s_series = ego_data['lonLaneletPos']
            if (s_series is None or s_series.isna().all()) and 'traveledDistance' in ego_data.columns:
                s_series = ego_data['traveledDistance']
            if s_series is not None and not s_series.isna().all():
                s_vals = np.array(s_series, dtype=float)
                mask_merge = (s_vals >= merge_bounds[0] - 5.0) & (s_vals <= merge_bounds[1] + 5.0)
                if mask_merge.any():
                    frames = ego_data['frame'].values[mask_merge]
        
        best_frame = None
        best_count = -1
        
        for tol in (60.0, 120.0, 179.0):
            for frame in frames[::10]:
                snapshot = self.get_snapshot(ego_id, frame, heading_tol_deg=tol)
                if snapshot is None:
                    continue
                count = len(snapshot['surrounding'])
                if count > best_count:
                    best_count = count
                    best_frame = frame
            if best_count > 0:
                break
        
        if best_frame is None and len(frames) > 0:
            best_frame = int(np.median(frames))
        
        return best_frame
    
    def _get_merge_bounds(self, ego_id: int) -> Optional[Tuple[float, float]]:
        """Estimate merge start/end along s."""
        if ego_id in self.merge_bounds:
            return self.merge_bounds[ego_id]
        
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        s_series = None
        if 'lonLaneletPos' in ego_data.columns:
            s_series = ego_data['lonLaneletPos']
        if (s_series is None or s_series.isna().all()) and 'traveledDistance' in ego_data.columns:
            s_series = ego_data['traveledDistance']
        if s_series is None or s_series.isna().all():
            return None
        
        s_clean = s_series.replace([np.inf, -np.inf], np.nan).dropna()
        if s_clean.empty:
            return None
        
        start = float(np.nanpercentile(s_clean, 5))
        end = float(np.nanpercentile(s_clean, 95))
        if end <= start:
            end = start + 1.0
        
        self.merge_bounds[ego_id] = (start, end)
        return self.merge_bounds[ego_id]
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def get_merging_heavy_vehicles(self, min_frames: int = 5) -> List[int]:
        """Return heavy vehicles that exhibit merge-area interactions."""
        heavy_ids = self.get_heavy_vehicles()
        candidates = []
        
        for hv_id in heavy_ids:
            merge_bounds = self._get_merge_bounds(hv_id)
            if merge_bounds is not None:
                candidates.append(hv_id)
        
        return candidates
    
    def get_background_extent(self) -> List[float]:
        """Extent for plotting background image in meters."""
        if self.background_image is None:
            return [0, 0, 0, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]


# =============================================================================
# EDRF Visualization
# =============================================================================

class EDRFVisualizer:
    """Creates EDRF visualizations based on the paper."""
    
    def __init__(self, config: EDRFConfig = None, loader: ExiDLoader = None, light_theme: bool = False):
        self.config = config or EDRFConfig()
        self.loader = loader
        self.light_theme = light_theme
        if light_theme:
            self.bg_color = 'white'
            self.panel_color = 'white'
            self.fg_color = 'black'
            self.spine_color = '#4A4A4A'
        else:
            self.bg_color = '#0D1117'
            self.panel_color = '#1A1A2E'
            self.fg_color = 'white'
            self.spine_color = '#4A4A6A'
    
    def create_combined_figure(self, snapshot: Dict, output_path: str = None):
        """
        Create a combined figure with:
        - Individual EDRF for ego vehicle
        - Individual EDRF for a representative surrounding vehicle
        - Combined Interaction Risk field
        - DRP along predicted trajectories
        - Risk analysis summary
        - Traffic snapshot
        """
        
        ego = snapshot['ego']
        surrounding = snapshot['surrounding']
        
        if not surrounding:
            logger.warning("No surrounding vehicles")
            return
        
        # Setup figure
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor(self.bg_color)
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # EDRF for ego
        ax2 = fig.add_subplot(gs[0, 1])  # EDRF for representative other
        ax3 = fig.add_subplot(gs[0, 2])  # Combined IR field
        ax4 = fig.add_subplot(gs[1, 0])  # DRP along trajectories
        ax5 = fig.add_subplot(gs[1, 1])  # Traffic snapshot with predictions
        ax6 = fig.add_subplot(gs[1, 2])  # Risk summary
        
        # Generate predictions for all vehicles
        ego_predictions = generate_multimodal_predictions(ego, self.config)
        other_predictions_dict = {}
        for other in surrounding:
            other_predictions_dict[other['id']] = generate_multimodal_predictions(other, self.config)
        
        # Compute grid bounds
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        rel_positions = []
        for other in surrounding:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            rel_positions.append([dx_rel, dy_rel])
        rel_positions = np.array(rel_positions) if rel_positions else np.array([[0.0, 0.0]])
        
        margin_x = 10.0
        margin_y = 5.0
        x_min = -max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin_x)
        x_max = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin_x)
        y_span = max(self.config.OBS_RANGE_LEFT, self.config.OBS_RANGE_RIGHT, 
                     np.ptp(rel_positions[:, 1])/2 + margin_y, (x_max - x_min)/4, 30.0)
        # Force lateral range to +-30m for visualization consistency
        y_span = 30.0
        
        # Create grid in ego's reference frame
        X = np.linspace(x_min, x_max, self.config.EDRF_GRID_X)
        Y = np.linspace(-y_span, y_span, self.config.EDRF_GRID_Y)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        # Transform grid to global coordinates for computations
        X_global = ego['x'] + X_mesh * cos_h + Y_mesh * sin_h
        Y_global = ego['y'] - X_mesh * sin_h + Y_mesh * cos_h
        
        # 1. EDRF for ego vehicle
        edrf_ego = compute_edrf(X_global, Y_global, ego, ego_predictions, self.config, is_ego=True)
        self._plot_edrf(ax1, X_mesh, Y_mesh, edrf_ego, ego, surrounding, 
                        f"EDRF: {ego['class'].title()} (Ego)", ego_predictions)
        
        # 2. EDRF for closest surrounding vehicle
        closest_other = min(surrounding, key=lambda c: np.sqrt((c['x']-ego['x'])**2 + (c['y']-ego['y'])**2))
        closest_predictions = other_predictions_dict[closest_other['id']]
        edrf_other = compute_edrf(X_global, Y_global, closest_other, closest_predictions, self.config, is_ego=False)
        self._plot_edrf(ax2, X_mesh, Y_mesh, edrf_other, ego, [closest_other],
                        f"EDRF: Vehicle {closest_other['id']}", closest_predictions, show_ego=True)
        
        # 3. Combined Interaction Risk field
        ir_total = np.zeros_like(X_mesh)
        risk_results = []
        
        for other in surrounding:
            edrf_o = compute_edrf(X_global, Y_global, other, 
                                   other_predictions_dict[other['id']], 
                                   self.config, is_ego=False)
            ir = compute_interaction_risk(edrf_ego, edrf_o)
            ir_total += ir
            
            F = compute_risk_level(ir)
            risk_results.append({
                'other_id': other['id'],
                'other_class': other['class'],
                'risk_level': F,
                'distance': np.sqrt((other['x']-ego['x'])**2 + (other['y']-ego['y'])**2)
            })
        
        self._plot_ir_field(ax3, X_mesh, Y_mesh, ir_total, ego, surrounding, risk_results)
        
        # 4. DRP along trajectories visualization
        self._plot_drp_along_trajectories(ax4, ego, ego_predictions, closest_other, closest_predictions)
        
        # 5. Traffic snapshot with predicted trajectories
        self._plot_traffic_with_predictions(ax5, ego, surrounding, ego_predictions, other_predictions_dict)
        
        # 6. Risk summary
        self._plot_risk_summary(ax6, ego, risk_results)
        
        # Title
        fig.suptitle(
            f"EDRF Analysis: {ego['class'].title()} (ID: {ego['id']}) | Frame: {snapshot['frame']} | "
            f"Surrounding: {len(surrounding)} vehicles",
            fontsize=14, fontweight='bold', color=self.fg_color, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_edrf(self, ax, X: np.ndarray, Y: np.ndarray, edrf: np.ndarray,
                   ego: Dict, others: List[Dict], title: str, predictions: List[Dict],
                   show_ego: bool = False):
        """Plot EDRF field for a vehicle."""
        
        ax.set_facecolor(self.panel_color)
        
        # Plot EDRF field
        cmap = LinearSegmentedColormap.from_list('edrf', ['#0D1117', '#1E3A5F', '#3498DB', '#F39C12', '#E74C3C'])
        
        # Normalize for better visualization
        edrf_normalized = edrf / (np.max(edrf) + 1e-10)
        
        pcm = ax.pcolormesh(X, Y, edrf_normalized, cmap=cmap, shading='gouraud', alpha=0.85)
        
        # Transform trajectories to ego frame
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Draw predicted trajectories
        mode_colors = {
            'lane_keep': '#2ECC71',
            'lane_keep_decel': '#27AE60',
            'lane_keep_accel': '#58D68D',
            'lane_change_left': '#3498DB',
            'lane_change_right': '#E74C3C',
            'lane_change_left_slow': '#5DADE2',
        }
        
        for pred in predictions:
            traj = pred['trajectory']
            prob = pred['probability']
            mode_name = pred.get('name', 'unknown')
            
            # Transform to ego frame
            traj_rel = np.zeros_like(traj)
            for i, (tx, ty) in enumerate(traj):
                dx = tx - ego['x']
                dy = ty - ego['y']
                traj_rel[i, 0] = dx * cos_h - dy * sin_h
                traj_rel[i, 1] = dx * sin_h + dy * cos_h
            
            alpha = 0.4 + 0.6 * prob
            color = mode_colors.get(mode_name, '#2ECC71')
            ax.plot(traj_rel[:, 0], traj_rel[:, 1], '-', color=color, alpha=alpha, linewidth=2)
            
            # Short label at trajectory end
            short_labels = {
                'lane_keep': 'LK',
                'lane_keep_decel': 'LK-',
                'lane_keep_accel': 'LK+',
                'lane_change_left': 'LCL',
                'lane_change_right': 'LCR',
                'lane_change_left_slow': 'LCL-',
            }
            label = short_labels.get(mode_name, '?')
            ax.text(traj_rel[-1, 0], traj_rel[-1, 1], f'{label}\n{prob:.0%}', 
                    fontsize=6, color=color, alpha=alpha, ha='center')
        
        # Draw ego vehicle at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS.get(ego['class'], '#E74C3C'),
            edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # Draw other vehicles
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other['length']/2, dy_rel - other['width']/2),
                other['length'], other['width'],
                boxstyle="round,pad=0.02",
                facecolor=self.config.COLORS.get(other['class'], '#3498DB'),
                edgecolor=self.fg_color, linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + other['width']/2 + 1.5, str(other['id']),
                   ha='center', va='bottom', fontsize=8, color='yellow')
        
        # Lane markings
        ax.axhline(3.5, color=self.fg_color, linestyle='--', alpha=0.5)
        ax.axhline(-3.5, color=self.fg_color, linestyle='--', alpha=0.5)
        ax.axhline(7, color=self.fg_color, linestyle='-', alpha=0.5)
        ax.axhline(-7, color=self.fg_color, linestyle='-', alpha=0.5)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color)
        ax.set_ylabel('Lateral (m)', color=self.fg_color)
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('EDRF (normalized)', color=self.fg_color)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_ir_field(self, ax, X: np.ndarray, Y: np.ndarray, ir: np.ndarray,
                       ego: Dict, others: List[Dict], risk_results: List[Dict]):
        """Plot combined Interaction Risk field."""
        
        ax.set_facecolor(self.panel_color)
        
        # IR colormap (emphasize high-risk areas)
        cmap = LinearSegmentedColormap.from_list('ir', ['#1A1A2E', '#2C3E50', '#F39C12', '#E74C3C', '#C0392B'])
        
        ir_normalized = ir / (np.max(ir) + 1e-10)
        pcm = ax.pcolormesh(X, Y, ir_normalized, cmap=cmap, shading='gouraud', alpha=0.85)
        
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Draw ego
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2),
            ego['length'], ego['width'],
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS.get(ego['class'], '#E74C3C'),
            edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # Draw others with risk level coloring
        for other, result in zip(others, risk_results):
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Color based on risk level
            F = result['risk_level']
            max_F = max(r['risk_level'] for r in risk_results) if risk_results else 1
            risk_ratio = F / (max_F + 1e-10)
            
            if risk_ratio > 0.7:
                color = '#E74C3C'
            elif risk_ratio > 0.4:
                color = '#F39C12'
            elif risk_ratio > 0.1:
                color = '#3498DB'
            else:
                color = '#27AE60'
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other['length']/2, dy_rel - other['width']/2),
                other['length'], other['width'],
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=self.fg_color, linewidth=1.5, alpha=0.9
            )
            ax.add_patch(rect)
            
            # Risk annotation
            ax.text(dx_rel, dy_rel + other['width']/2 + 1.5,
                   f"{other['id']}\nF={F:.2e}",
                   ha='center', va='bottom', fontsize=7, color=self.fg_color,
                   bbox=dict(boxstyle='round', facecolor='white' if self.light_theme else 'black', alpha=0.5))
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color)
        ax.set_ylabel('Lateral (m)', color=self.fg_color)
        ax.set_title('Combined Interaction Risk (IR)', fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('IR (normalized)', color=self.fg_color)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_drp_along_trajectories(self, ax, ego: Dict, ego_preds: List[Dict],
                                      other: Dict, other_preds: List[Dict]):
        """Plot DRP along predicted trajectories (similar to Fig. 6 in paper)."""
        
        ax.set_facecolor(self.panel_color)
        
        # Mode colors for ego
        mode_colors = {
            'lane_keep': '#2ECC71',
            'lane_keep_decel': '#27AE60',
            'lane_keep_accel': '#58D68D',
            'lane_change_left': '#3498DB',
            'lane_change_right': '#E74C3C',
            'lane_change_left_slow': '#5DADE2',
        }
        
        # Short labels
        short_labels = {
            'lane_keep': 'LK',
            'lane_keep_decel': 'LK-',
            'lane_keep_accel': 'LK+',
            'lane_change_left': 'LCL',
            'lane_change_right': 'LCR',
            'lane_change_left_slow': 'LCL-',
        }
        
        # Ego trajectories
        for i, pred in enumerate(ego_preds):
            traj = pred['trajectory']
            prob = pred['probability']
            mode_name = pred.get('name', f'mode_{i}')
            
            s_values, s_pt = trajectory_to_frenet(traj)
            
            # Compute DRP profile along centerline (d=0)
            s_profile = np.linspace(0, s_pt, 50)
            drp_profile = []
            
            for s in s_profile:
                # Height function (linear for ego - Laplace)
                a_s = self.config.Q_EGO * abs(s - s_pt)
                drp_profile.append(a_s * prob)
            
            color = mode_colors.get(mode_name, '#2ECC71')
            label = f'Ego {short_labels.get(mode_name, mode_name)} (p={prob:.0%})'
            ax.plot(s_profile, drp_profile, color=color, 
                   label=label, alpha=0.8, linewidth=1.5)
        
        # Other vehicle trajectories (use dashed lines)
        colors_other = plt.cm.Oranges(np.linspace(0.4, 0.9, len(other_preds)))
        for i, pred in enumerate(other_preds[:3]):  # Show top 3
            traj = pred['trajectory']
            prob = pred['probability']
            mode_name = pred.get('name', f'mode_{i}')
            curvature = pred['curvature']
            
            s_values, s_pt = trajectory_to_frenet(traj)
            
            s_profile = np.linspace(0, s_pt, 50)
            drp_profile = []
            
            for s in s_profile:
                # Height function (parabolic for others - Gaussian)
                a_s = self.config.Q * (s - s_pt)**2
                drp_profile.append(a_s * prob)
            
            label = f'V{other["id"]} {short_labels.get(mode_name, mode_name)} (p={prob:.0%})'
            ax.plot(s_profile, drp_profile, color=colors_other[i],
                   label=label, alpha=0.8, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Arc length s (m)', color=self.fg_color)
        ax.set_ylabel('DRP (weighted)', color=self.fg_color)
        ax.set_title('DRP along Predicted Trajectories', fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color)
        ax.legend(loc='upper right', fontsize=6, ncol=2)
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_traffic_with_predictions(self, ax, ego: Dict, surrounding: List[Dict],
                                        ego_preds: List[Dict], other_preds: Dict):
        """Plot traffic snapshot with predicted trajectories in global coordinates."""
        
        ax.set_facecolor(self.panel_color)
        
        if self.loader and self.loader.background_image is not None:
            bg_extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=bg_extent, alpha=0.5, aspect='equal', zorder=0)
        
        # Compute bounds
        all_x = [ego['x']] + [s['x'] for s in surrounding]
        all_y = [ego['y']] + [s['y'] for s in surrounding]
        
        x_center = ego['x']
        y_center = ego['y']
        half_x = max(40, (max(all_x) - min(all_x))/2 + 15)
        half_y = max(30, (max(all_y) - min(all_y))/2 + 10)
        
        # Mode colors
        mode_colors = {
            'lane_keep': '#2ECC71',
            'lane_keep_decel': '#27AE60',
            'lane_keep_accel': '#58D68D',
            'lane_change_left': '#3498DB',
            'lane_change_right': '#E74C3C',
            'lane_change_left_slow': '#5DADE2',
        }
        
        # Draw ego predicted trajectories
        for pred in ego_preds:
            traj = pred['trajectory']
            prob = pred['probability']
            mode_name = pred.get('name', 'unknown')
            alpha = 0.3 + 0.5 * prob
            color = mode_colors.get(mode_name, '#2ECC71')
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=alpha, linewidth=2.5, zorder=5)
        
        # Draw other vehicles' predicted trajectories
        for other in surrounding:
            preds = other_preds.get(other['id'], [])
            for pred in preds[:3]:  # Top 3 modes
                traj = pred['trajectory']
                prob = pred['probability']
                mode_name = pred.get('name', 'unknown')
                alpha = 0.2 + 0.4 * prob
                # Use orange tones for other vehicles
                ax.plot(traj[:, 0], traj[:, 1], '-', color='#F39C12', alpha=alpha, linewidth=1.5, zorder=4)
        
        # Draw ego vehicle
        self._draw_vehicle(ax, ego, is_ego=True)
        
        # Draw surrounding vehicles
        for other in surrounding:
            self._draw_vehicle(ax, other, is_ego=False)
        
        ax.set_xlim(x_center - half_x, x_center + half_x)
        ax.set_ylim(y_center - half_y, y_center + half_y)
        ax.set_xlabel('X (m)', color=self.fg_color)
        ax.set_ylabel('Y (m)', color=self.fg_color)
        ax.set_title('Traffic Scene with Lane-Based Predictions', fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Add legend for trajectory types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2ECC71', linewidth=2, label='Lane Keep'),
            Line2D([0], [0], color='#3498DB', linewidth=2, label='LC Left'),
            Line2D([0], [0], color='#E74C3C', linewidth=2, label='LC Right'),
            Line2D([0], [0], color='#F39C12', linewidth=2, linestyle='-', label='Other Vehicles'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _draw_vehicle(self, ax, veh: Dict, is_ego: bool = False):
        """Draw a vehicle."""
        
        color = self.config.COLORS.get(veh['class'], '#3498DB')
        if is_ego:
            alpha = 1.0
            lw = 2
        else:
            alpha = 0.8
            lw = 1
        
        corners = self._get_rotated_rect(
            veh['x'], veh['y'], veh['length'], veh['width'], veh['heading']
        )
        
        rect = plt.Polygon(corners, closed=True, facecolor=color,
                          edgecolor=self.fg_color, linewidth=lw, alpha=alpha, zorder=10)
        ax.add_patch(rect)
        
        # Velocity arrow
        arrow_scale = 0.5
        ax.arrow(veh['x'], veh['y'], veh['vx']*arrow_scale, veh['vy']*arrow_scale,
                head_width=0.8, head_length=0.4, fc='yellow' if is_ego else 'cyan',
                ec='yellow' if is_ego else 'cyan', alpha=0.8, zorder=11)
        
        # Label
        label = f"EGO" if is_ego else str(veh['id'])
        ax.text(veh['x'], veh['y'] + veh['width']/2 + 1, label,
               ha='center', va='bottom', fontsize=8,
               color=self.fg_color, fontweight='bold' if is_ego else 'normal', zorder=12)
    
    def _get_rotated_rect(self, cx, cy, length, width, heading):
        """Get corners of rotated rectangle."""
        half_l, half_w = length/2, width/2
        
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        return corners @ R.T + np.array([cx, cy])
    
    def _plot_risk_summary(self, ax, ego: Dict, risk_results: List[Dict]):
        """Plot risk summary panel."""
        
        ax.set_facecolor(self.panel_color)
        ax.axis('off')
        
        if not risk_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color=self.fg_color)
            return
        
        # Sort by risk level
        risk_results = sorted(risk_results, key=lambda x: x['risk_level'], reverse=True)
        
        total_risk = sum(r['risk_level'] for r in risk_results)
        max_risk = max(r['risk_level'] for r in risk_results)
        mean_distance = np.mean([r['distance'] for r in risk_results])
        
        def interpret_risk(F, max_F):
            ratio = F / (max_F + 1e-10)
            if ratio > 0.7: return 'HIGH', '#E74C3C'
            elif ratio > 0.4: return 'MEDIUM', '#F39C12'
            elif ratio > 0.1: return 'LOW', '#3498DB'
            else: return 'MINIMAL', '#27AE60'
        
        summary_lines = [
            "═" * 40,
            "EDRF RISK ANALYSIS SUMMARY",
            "═" * 40,
            f"",
            f"Ego Vehicle: {ego['class'].title()} (ID: {ego['id']})",
            f"Speed: {ego['speed']*3.6:.1f} km/h",
            f"Mass: {ego['mass']/1000:.1f} tons",
            f"Virtual Mass: {compute_virtual_mass(ego, self.config):.2e}",
            f"",
            "─" * 40,
            "INTERACTION RISK ANALYSIS",
            "─" * 40,
            f"",
            f"Total vehicles: {len(risk_results)}",
            f"Mean distance: {mean_distance:.1f} m",
            f"Total IR: {total_risk:.2e}",
            f"Max IR (F): {max_risk:.2e}",
            f"",
            "─" * 40,
            "RISK BY VEHICLE (sorted)",
            "─" * 40,
        ]
        
        for r in risk_results[:5]:  # Top 5
            level, color = interpret_risk(r['risk_level'], max_risk)
            summary_lines.append(f"  Veh {r['other_id']:3d}: F={r['risk_level']:.2e} [{level}]")
            summary_lines.append(f"           dist={r['distance']:.1f}m")
        
        if len(risk_results) > 5:
            summary_lines.append(f"  ... and {len(risk_results)-5} more")
        
        summary_lines.extend([
            "",
            "─" * 40,
            "EDRF MODEL PARAMETERS",
            "─" * 40,
            f"  Prediction horizon: {self.config.PREDICTION_HORIZON}s",
            f"  DRP: q={self.config.Q}, b={self.config.B}",
            f"       k={self.config.K}, c={self.config.C}",
            f"  Virtual Mass: α={self.config.ALPHA:.2e}",
            f"               β={self.config.BETA_VM}, γ={self.config.GAMMA}",
        ])
        
        summary = "\n".join(summary_lines)
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=9, color=self.fg_color, family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)


# =============================================================================
# Main
# =============================================================================

def _select_from_occlusion_log(csv_path: Path, row_idx: int, ego_role: str) -> Tuple[int, int, int]:
    """Select recording, frame, ego_id from occlusion log row."""
    df = pd.read_csv(csv_path)
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"Row {row_idx} out of range for occlusion log with {len(df)} rows")
    occ = df.iloc[row_idx]
    recording_id = int(occ.get('recording_id', occ.get('recordingId', 0)))
    frame = int(occ['frame'])
    if ego_role == 'occluder':
        ego_id = int(occ['occluder_id'])
    elif ego_role == 'occluded':
        ego_id = int(occ['occluded_id'])
    else:
        ego_id = int(occ['blocked_id'])
    return recording_id, frame, ego_id


def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None,
         frame: Optional[int] = None, output_dir: str = './output_edrf',
         occlusion_csv: Optional[str] = None, occlusion_row: Optional[int] = None,
         occlusion_ego_role: str = 'blocked', light_theme: bool = False):
    """Main entry point for EDRF visualization."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if occlusion_csv is not None:
        csv_path = Path(occlusion_csv)
        if not csv_path.exists():
            logger.error(f"Occlusion CSV not found: {csv_path}")
            return
        if occlusion_row is None:
            logger.error("Please provide --occlusion-row when using --occlusion-csv")
            return
        try:
            recording_id, frame, ego_id = _select_from_occlusion_log(csv_path, occlusion_row, occlusion_ego_role)
            logger.info(f"Using occlusion log row {occlusion_row}: rec {recording_id}, frame {frame}, ego {ego_id} ({occlusion_ego_role})")
        except Exception as e:
            logger.error(f"Failed to parse occlusion log: {e}")
            return
    
    logger.info("=" * 60)
    logger.info("EDRF (Enhanced Driving Risk Field) Visualization")
    logger.info("Based on: Jiang et al. IEEE ITSC 2024")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    # Find ego vehicle
    if ego_id is None:
        merging_ids = loader.get_merging_heavy_vehicles()
        if merging_ids:
            ego_id = merging_ids[0]
            logger.info(f"Auto-selected ego (merge-capable): {ego_id}")
        else:
            heavy_ids = loader.get_heavy_vehicles()
            if not heavy_ids:
                logger.error("No heavy vehicles found")
                return
            ego_id = heavy_ids[0]
            logger.info(f"Auto-selected ego: {ego_id}")
    
    # Find best frame
    if frame is None:
        frame = loader.find_best_interaction_frame(ego_id)
        if frame is None:
            logger.error("Could not find suitable frame")
            return
        logger.info(f"Auto-selected frame: {frame}")
    
    # Get snapshot
    snapshot = loader.get_snapshot(ego_id, frame)
    if snapshot is None:
        ego_frames = loader.tracks_df[loader.tracks_df['trackId'] == ego_id]['frame'].values
        if len(ego_frames) > 0:
            nearest = int(ego_frames[np.abs(ego_frames - frame).argmin()])
            if nearest != frame:
                logger.warning(f"Requested frame {frame} not found; falling back to nearest ego frame {nearest}")
                snapshot = loader.get_snapshot(ego_id, nearest)
        if snapshot is None:
            logger.error("Could not get snapshot")
            return
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    # Create visualization
    config = EDRFConfig()
    viz = EDRFVisualizer(config=config, loader=loader, light_theme=light_theme)
    
    output_file = output_path / f'edrf_recording{recording_id}_ego{ego_id}_frame{frame}.png'
    viz.create_combined_figure(snapshot, str(output_file))
    
    logger.info(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDRF (Enhanced Driving Risk Field) Visualization')
    parser.add_argument('--data_dir', type=str, default='C:\\exiD-tools\\data',
                        help='Path to exiD dataset directory')
    parser.add_argument('--recording', type=int, default=25,
                        help='Recording ID')
    parser.add_argument('--ego_id', type=int, default=None,
                        help='Ego vehicle ID (auto-selected if not provided)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame number (auto-selected if not provided)')
    parser.add_argument('--output_dir', type=str, default='./output_edrf',
                        help='Output directory for visualizations')
    parser.add_argument('--occlusion-csv', type=str, default=None,
                        help='Path to occlusion_log.csv from role/occlusion analysis')
    parser.add_argument('--occlusion-row', type=int, default=None,
                        help='Row index in occlusion log to visualize')
    parser.add_argument('--occlusion-ego-role', type=str, default='blocked',
                        choices=['blocked', 'occluder', 'occluded'],
                        help='Which vehicle from occlusion row to treat as ego')
    parser.add_argument('--light-theme', action='store_true',
                        help='Use light (white) background instead of dark theme')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id, args.frame, args.output_dir,
         occlusion_csv=args.occlusion_csv, occlusion_row=args.occlusion_row,
         occlusion_ego_role=args.occlusion_ego_role, light_theme=args.light_theme)
