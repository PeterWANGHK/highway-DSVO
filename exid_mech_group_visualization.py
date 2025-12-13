"""
exiD Dataset: Mechanical Wave-Based Vehicle Aggressiveness with Group Analysis
===============================================================================
Implements the vehicle aggressiveness model from:
Hu et al. (2023) "Formulating Vehicle Aggressiveness Towards Social 
Cognitive Autonomous Driving" IEEE Transactions on Intelligent Vehicles

Enhanced Features:
1. Group wave visualization - surrounding vehicles treated as a collective unit
2. Expanded observation range to capture more vehicles
3. Comparison between ego truck wave field and surrounding group wave field
4. Superposition of individual waves (Equation 24)

Mathematical Model (Simplified Formulation - Equation 23-24):
    Ω_{i→j} = (m_i |v_i|) / (2δ m_j) * exp(ξ₁ + ξ₂)
    
    Total from group: Ω_j = Σ_{i=1}^{K} Ω_{i→j}  (Equation 24)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - Extended observation range for more vehicles
# =============================================================================

@dataclass
class Config:
    """Configuration for Mechanical Wave Aggressiveness model."""
    
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # EXPANDED observation range (meters, relative to ego) - to capture more vehicles
    OBS_RANGE_AHEAD: float = 100.0   # Extended from 60
    OBS_RANGE_BEHIND: float = 60.0   # Extended from 30
    OBS_RANGE_LEFT: float = 25.0     # Extended from 15
    OBS_RANGE_RIGHT: float = 25.0    # Extended from 15
    
    # Grid resolution for field visualization
    FIELD_GRID_X: int = 100
    FIELD_GRID_Y: int = 60
    
    # ========================================================================
    # Mechanical Wave Model Parameters (Table I from paper)
    # ========================================================================
    GAMMA_1: float = 1200.0
    GAMMA_2: float = 100.0
    EPSILON_1: float = 0.65
    EPSILON_2: float = 0.5
    
    TAU: float = 0.2
    BETA: float = 0.05
    SIGMA: float = 600.0
    DELTA: float = 5e-4
    
    MU_1: float = 0.15
    MU_2: float = 0.16
    T_0: float = 2.0
    
    # ========================================================================
    # Vehicle mass parameters
    # ========================================================================
    MASS_HV: float = 15000.0   # Heavy vehicle mass (kg) - 15 tons
    MASS_PC: float = 3000.0    # Passenger car mass (kg) - 3 tons
    
    V_REF: float = 25.0
    DIST_REF: float = 25.0
    
    # SVO weights
    WEIGHT_AGGR: float = 0.45
    WEIGHT_DECEL: float = 0.30
    WEIGHT_YIELD: float = 0.25
    
    # Vehicle dimensions
    TRUCK_LENGTH: float = 12.0
    TRUCK_WIDTH: float = 2.5
    CAR_LENGTH: float = 4.5
    CAR_WIDTH: float = 1.8
    
    # Visualization
    FPS: int = 25
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
        'van': '#9B59B6',
        'group': '#2ECC71',  # Color for group representation
    })


# =============================================================================
# Mechanical Wave Aggressiveness Model
# =============================================================================

def compute_pseudo_distance(aggressor: Dict, sufferer: Dict, config: Config) -> float:
    """Compute elliptical pseudo-distance R_ij (Equation 17)."""
    
    dx = sufferer['x'] - aggressor['x']
    dy = sufferer['y'] - aggressor['y']
    
    phi_i = aggressor['heading']
    cos_phi = np.cos(phi_i)
    sin_phi = np.sin(phi_i)
    
    x_local = dx * cos_phi + dy * sin_phi
    y_local = -dx * sin_phi + dy * cos_phi
    
    v_i = aggressor['speed']
    if v_i > 0.1:
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.1:
            cos_theta_i = (aggressor['vx'] * dx + aggressor['vy'] * dy) / (v_i * dist)
            cos_theta_i = np.clip(cos_theta_i, -1, 1)
        else:
            cos_theta_i = 1.0
    else:
        cos_theta_i = 0.0
    
    a_i = aggressor['ax'] * cos_phi + aggressor['ay'] * sin_phi
    accel_term = max(0, cos_theta_i * a_i * config.T_0)
    rho = config.TAU * np.exp(config.BETA * (v_i + accel_term))
    eta = config.TAU
    
    rho = max(rho, 0.1)
    eta = max(eta, 0.1)
    
    R_ij = np.sqrt((x_local**2) / (rho**2) + (y_local**2) / (eta**2))
    R_ij = R_ij * min(rho, eta)
    
    return max(R_ij, 0.1)


def compute_doppler_factor(aggressor: Dict, sufferer: Dict, config: Config) -> Tuple[float, float]:
    """Compute Doppler-like frequency modification factors (Equation 19-20)."""
    
    dx = sufferer['x'] - aggressor['x']
    dy = sufferer['y'] - aggressor['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 0.1:
        return 0.0, 0.0
    
    ux_ij = dx / dist
    uy_ij = dy / dist
    
    v_i = aggressor['speed']
    if v_i > 0.1:
        cos_theta_i = (aggressor['vx'] * ux_ij + aggressor['vy'] * uy_ij) / v_i
    else:
        cos_theta_i = 0.0
    
    v_j = sufferer['speed']
    if v_j > 0.1:
        cos_theta_j = -(sufferer['vx'] * ux_ij + sufferer['vy'] * uy_ij) / v_j
    else:
        cos_theta_j = 0.0
    
    cos_theta_i = np.clip(cos_theta_i, -1, 1)
    cos_theta_j = np.clip(cos_theta_j, -1, 1)
    
    return cos_theta_i, cos_theta_j


def compute_aggressiveness(aggressor: Dict, sufferer: Dict, config: Config) -> float:
    """Compute directional aggressiveness Ω_{i→j} (Equation 23)."""
    
    m_i = aggressor['mass']
    m_j = sufferer['mass']
    v_i = aggressor['speed']
    v_j = sufferer['speed']
    
    R_ij = compute_pseudo_distance(aggressor, sufferer, config)
    cos_theta_i, cos_theta_j = compute_doppler_factor(aggressor, sufferer, config)
    
    xi_1 = config.MU_1 * v_i * cos_theta_i + config.MU_2 * v_j * cos_theta_j
    xi_2 = -config.SIGMA * (m_i ** -1) * R_ij
    
    if m_j > 0:
        mass_term = (m_i * v_i) / (2 * config.DELTA * m_j)
    else:
        mass_term = 0.0
    
    omega = mass_term * np.exp(xi_1 + xi_2)
    
    return np.clip(omega, 0, 5000)


def compute_total_aggressiveness(sufferer: Dict, aggressors: List[Dict], config: Config) -> float:
    """Compute total aggressiveness from multiple AGVs (Equation 24)."""
    
    total_aggr = 0.0
    for aggressor in aggressors:
        total_aggr += compute_aggressiveness(aggressor, sufferer, config)
    
    return total_aggr


def create_group_centroid(vehicles: List[Dict], config: Config) -> Dict:
    """
    Create a virtual 'group centroid' vehicle representing the collective.
    Uses mass-weighted averaging for position/velocity.
    """
    if not vehicles:
        return None
    
    total_mass = sum(v['mass'] for v in vehicles)
    
    # Mass-weighted centroid
    x_c = sum(v['x'] * v['mass'] for v in vehicles) / total_mass
    y_c = sum(v['y'] * v['mass'] for v in vehicles) / total_mass
    
    # Mass-weighted velocity
    vx_c = sum(v['vx'] * v['mass'] for v in vehicles) / total_mass
    vy_c = sum(v['vy'] * v['mass'] for v in vehicles) / total_mass
    
    # Mass-weighted acceleration
    ax_c = sum(v['ax'] * v['mass'] for v in vehicles) / total_mass
    ay_c = sum(v['ay'] * v['mass'] for v in vehicles) / total_mass
    
    # Average heading (circular mean)
    sin_sum = sum(np.sin(v['heading']) for v in vehicles)
    cos_sum = sum(np.cos(v['heading']) for v in vehicles)
    heading_c = np.arctan2(sin_sum, cos_sum)
    
    return {
        'id': 'GROUP',
        'x': x_c,
        'y': y_c,
        'heading': heading_c,
        'vx': vx_c,
        'vy': vy_c,
        'ax': ax_c,
        'ay': ay_c,
        'speed': np.sqrt(vx_c**2 + vy_c**2),
        'mass': total_mass,  # Combined mass for group
        'width': 3.0,  # Visual representation
        'length': 6.0,
        'class': 'group',
        'n_vehicles': len(vehicles)
    }


def compute_group_wave_field(ego: Dict, group_vehicles: List[Dict],
                             x_range: Tuple[float, float],
                             y_range: Tuple[float, float],
                             grid_size: Tuple[int, int],
                             config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the superposed wave field from a group of vehicles (Equation 24).
    Each vehicle contributes its individual wave, which are then summed.
    """
    
    nx, ny = grid_size
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    field = np.zeros_like(X_mesh)
    
    cos_h = np.cos(ego['heading'])
    sin_h = np.sin(ego['heading'])
    
    for i in range(ny):
        for j in range(nx):
            x_local = X_mesh[i, j]
            y_local = Y_mesh[i, j]
            
            # Transform to global coordinates
            x_global = ego['x'] + x_local * cos_h - y_local * sin_h
            y_global = ego['y'] + x_local * sin_h + y_local * cos_h
            
            # Virtual SFV at this grid point (with car mass for standard comparison)
            virtual_sfv = {
                'x': x_global,
                'y': y_global,
                'heading': ego['heading'],
                'vx': ego['vx'],
                'vy': ego['vy'],
                'speed': ego['speed'],
                'ax': 0.0,
                'ay': 0.0,
                'mass': config.MASS_PC  # Use car mass as reference
            }
            
            # Sum contributions from all group vehicles (Equation 24)
            field[i, j] = compute_total_aggressiveness(virtual_sfv, group_vehicles, config)
    
    return X_mesh, Y_mesh, field


def compute_ego_wave_field(ego: Dict, 
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           grid_size: Tuple[int, int],
                           config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the wave field emitted by ego vehicle alone.
    """
    
    nx, ny = grid_size
    X = np.linspace(x_range[0], x_range[1], nx)
    Y = np.linspace(y_range[0], y_range[1], ny)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    
    field = np.zeros_like(X_mesh)
    
    cos_h = np.cos(ego['heading'])
    sin_h = np.sin(ego['heading'])
    
    for i in range(ny):
        for j in range(nx):
            x_local = X_mesh[i, j]
            y_local = Y_mesh[i, j]
            
            x_global = ego['x'] + x_local * cos_h - y_local * sin_h
            y_global = ego['y'] + x_local * sin_h + y_local * cos_h
            
            # Virtual SFV with car mass
            virtual_sfv = {
                'x': x_global,
                'y': y_global,
                'heading': ego['heading'],
                'vx': 0.0,  # Stationary reference
                'vy': 0.0,
                'speed': 0.0,
                'ax': 0.0,
                'ay': 0.0,
                'mass': config.MASS_PC
            }
            
            field[i, j] = compute_aggressiveness(ego, virtual_sfv, config)
    
    return X_mesh, Y_mesh, field


# =============================================================================
# SVO Computation
# =============================================================================

def compute_svo_from_aggressiveness(ego: Dict, other: Dict, config: Config) -> Dict:
    """Compute bidirectional SVO based on mechanical wave aggressiveness."""
    
    dx = other['x'] - ego['x']
    dy = other['y'] - ego['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1.0:
        return {'ego_svo': 45.0, 'other_svo': 45.0, 'distance': dist,
                'ego_aggr': 0.0, 'other_aggr': 0.0}
    
    aggr_ego_to_other = compute_aggressiveness(ego, other, config)
    aggr_other_to_ego = compute_aggressiveness(other, ego, config)
    
    ego_svo = _aggr_to_svo(ego, other, aggr_ego_to_other, dist, config)
    other_svo = _aggr_to_svo(other, ego, aggr_other_to_ego, dist, config)
    
    return {
        'ego_svo': ego_svo,
        'other_svo': other_svo,
        'ego_aggr': aggr_ego_to_other,
        'other_aggr': aggr_other_to_ego,
        'distance': dist,
        'dx': dx,
        'dy': dy,
    }


def _aggr_to_svo(veh: Dict, other: Dict, aggr: float, dist: float, config: Config) -> float:
    """Convert aggressiveness to SVO angle."""
    
    v_ego = max(veh['speed'], 1.0)
    v_other = max(other['speed'], 1.0)
    speed_factor = (v_ego + v_other) / (2 * config.V_REF)
    dist_factor = config.DIST_REF / max(dist, 5.0)
    mass_factor = veh['mass'] / config.MASS_PC
    context = max(100.0 * speed_factor * dist_factor * mass_factor, 10.0)
    
    norm_aggr = np.clip(aggr / context, 0, 1)
    
    accel = (veh['vx'] * veh['ax'] + veh['vy'] * veh['ay']) / max(veh['speed'], 0.1)
    decel = max(0, -accel)
    norm_decel = np.tanh(decel / 2.0)
    
    speed_ratio = veh['speed'] / max(config.V_REF, 1)
    norm_yield = np.clip(1 - speed_ratio, 0, 1)
    
    svo = (config.WEIGHT_AGGR * (90 - 135 * norm_aggr) +
           config.WEIGHT_DECEL * (45 * norm_decel) +
           config.WEIGHT_YIELD * (-22.5 + 67.5 * norm_yield))
    
    return np.clip(svo, -45, 90)


# =============================================================================
# Data Loader - Extended range
# =============================================================================

class ExiDLoader:
    """Load exiD data with extended observation range."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
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
    
    def get_snapshot(self, ego_id: int, frame: int, heading_tol_deg: float = 90.0,
                    extended_range: bool = True) -> Optional[Dict]:
        """Get snapshot with extended range option."""
        
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
        }
        
        # Use extended range if requested
        if extended_range:
            range_ahead = self.config.OBS_RANGE_AHEAD
            range_behind = self.config.OBS_RANGE_BEHIND
            range_left = self.config.OBS_RANGE_LEFT
            range_right = self.config.OBS_RANGE_RIGHT
        else:
            range_ahead = 60.0
            range_behind = 30.0
            range_left = 15.0
            range_right = 15.0
        
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
            }
            
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            
            if (-range_behind <= dx <= range_ahead and
                -range_right <= dy <= range_left and
                self._is_same_direction(ego, other, heading_tol_deg)):
                surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def _is_same_direction(self, ego: Dict, other: Dict, heading_tol_deg: float = 90.0) -> bool:
        """Check if vehicles travel roughly the same direction."""
        
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
    
    def find_best_interaction_frame(self, ego_id: int, min_vehicles: int = 5) -> Optional[int]:
        """Find frame with most surrounding vehicles (prefer frames with many vehicles)."""
        
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        
        best_frame = None
        best_count = -1
        
        # Sample frames and find one with many neighbors
        for tol in (90.0, 120.0, 179.0):
            for frame in frames[::5]:  # Sample every 5 frames for more coverage
                snapshot = self.get_snapshot(ego_id, frame, heading_tol_deg=tol, extended_range=True)
                if snapshot is None:
                    continue
                count = len(snapshot['surrounding'])
                if count > best_count:
                    best_count = count
                    best_frame = frame
            if best_count >= min_vehicles:
                break
        
        if best_frame is None and len(frames) > 0:
            best_frame = int(np.median(frames))
        
        return best_frame
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def get_background_extent(self) -> List[float]:
        """Extent for plotting background image in meters."""
        if self.background_image is None:
            return [0, 0, 0, 0]
        h, w = self.background_image.shape[:2]
        return [0, w * self.ortho_px_to_meter, -h * self.ortho_px_to_meter, 0]


# =============================================================================
# Visualization with Group Wave Analysis
# =============================================================================

class GroupWaveVisualizer:
    """Visualizer for group-based mechanical wave analysis."""
    
    def __init__(self, config: Config = None, loader: ExiDLoader = None):
        self.config = config or Config()
        self.loader = loader
    
    def create_group_analysis_figure(self, snapshot: Dict, output_path: str = None):
        """
        Create figure showing:
        - Ego truck's wave field
        - Surrounding group's combined wave field (superposition)
        - Wave interference/interaction zones
        - Individual vehicle contributions
        """
        
        ego = snapshot['ego']
        surrounding = snapshot['surrounding']
        
        if not surrounding:
            logger.warning("No surrounding vehicles")
            return
        
        # Filter to get only cars for group analysis
        cars = [v for v in surrounding if v['class'] in self.config.CAR_CLASSES]
        other_heavy = [v for v in surrounding if v['class'] in self.config.HEAVY_VEHICLE_CLASSES]
        
        logger.info(f"Found {len(cars)} cars and {len(other_heavy)} other heavy vehicles")
        
        # Setup figure - larger for group analysis
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#0D1117')
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Ego truck wave field
        ax2 = fig.add_subplot(gs[0, 1])  # Car group wave field
        ax3 = fig.add_subplot(gs[0, 2])  # Combined/interference field
        ax4 = fig.add_subplot(gs[1, :2])  # Wide traffic view with waves
        ax5 = fig.add_subplot(gs[1, 2])  # Individual contributions
        ax6 = fig.add_subplot(gs[2, 0])  # Aggressiveness comparison
        ax7 = fig.add_subplot(gs[2, 1])  # Wave profile cross-section
        ax8 = fig.add_subplot(gs[2, 2])  # Summary statistics
        
        # Compute SVOs
        svo_results = []
        for other in surrounding:
            svo = compute_svo_from_aggressiveness(ego, other, self.config)
            svo['other_id'] = other['id']
            svo['other_class'] = other['class']
            svo_results.append(svo)
        
        # Create group centroid
        if cars:
            group_centroid = create_group_centroid(cars, self.config)
        else:
            group_centroid = None
        
        # Compute bounds
        rel_positions = self._compute_rel_positions(ego, surrounding)
        margin = 15.0
        ahead = max(self.config.OBS_RANGE_AHEAD, rel_positions[:, 0].max() + margin)
        behind = max(self.config.OBS_RANGE_BEHIND, -rel_positions[:, 0].min() + margin)
        lat_span = max(self.config.OBS_RANGE_LEFT, np.abs(rel_positions[:, 1]).max() + margin)
        x_range = (-behind, ahead)
        y_range = (-lat_span, lat_span)
        
        # 1. Ego truck wave field
        self._plot_ego_wave_field(ax1, ego, surrounding, x_range, y_range)
        
        # 2. Car group combined wave field
        self._plot_group_wave_field(ax2, ego, cars, x_range, y_range, group_centroid)
        
        # 3. Combined interference field
        self._plot_interference_field(ax3, ego, cars, x_range, y_range)
        
        # 4. Wide traffic view
        self._plot_wide_traffic_view(ax4, ego, surrounding, svo_results, group_centroid)
        
        # 5. Individual wave contributions
        self._plot_individual_contributions(ax5, ego, cars)
        
        # 6. Aggressiveness comparison
        self._plot_group_aggressiveness(ax6, ego, cars, svo_results)
        
        # 7. Wave profile cross-section
        self._plot_wave_cross_section(ax7, ego, cars, x_range)
        
        # 8. Summary panel
        self._plot_group_summary(ax8, ego, surrounding, cars, svo_results, group_centroid)
        
        # Title
        fig.suptitle(
            f"Mechanical Wave Group Analysis: {ego['class'].title()} (ID: {ego['id']}) vs "
            f"{len(cars)} Cars | Frame: {snapshot['frame']}\n"
            f"Superposition of Individual Waves (Equation 24: Ω_j = Σ Ω_{{i→j}})",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _compute_rel_positions(self, ego: Dict, vehicles: List[Dict]) -> np.ndarray:
        """Compute relative positions in ego frame."""
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        positions = []
        for v in vehicles:
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            positions.append([dx_rel, dy_rel])
        return np.array(positions) if positions else np.array([[0.0, 0.0]])
    
    def _plot_ego_wave_field(self, ax, ego: Dict, surrounding: List[Dict],
                            x_range: Tuple, y_range: Tuple):
        """Plot wave field emitted by ego truck."""
        
        ax.set_facecolor('#1A1A2E')
        
        X, Y, field = compute_ego_wave_field(
            ego, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config
        )
        
        field_norm = np.log1p(field)
        
        cmap = LinearSegmentedColormap.from_list('ego_wave',
            ['#1A1A2E', '#2D132C', '#801336', '#C72C41', '#EE4540', '#FF8C42', '#FFD93D'])
        pcm = ax.pcolormesh(X, Y, field_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Contours - ensure levels are unique and increasing
        if np.any(field_norm > 0):
            levels = np.percentile(field_norm[field_norm > 0], [30, 50, 70, 90])
            levels = np.unique(levels)
            if len(levels) >= 2 and levels[-1] > levels[0]:
                ax.contour(X, Y, field_norm, levels=levels, colors='white', alpha=0.4, linewidths=0.5)
        
        # Draw vehicles
        self._draw_vehicles_local(ax, ego, surrounding)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title(f"Truck Wave Field (Ω_{{truck→*}})\nMass: {ego['mass']/1000:.0f}t, Speed: {ego['speed']:.1f}m/s",
                    fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('log(1+Ω)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_group_wave_field(self, ax, ego: Dict, cars: List[Dict],
                               x_range: Tuple, y_range: Tuple, group_centroid: Dict):
        """Plot combined wave field from car group (superposition)."""
        
        ax.set_facecolor('#1A1A2E')
        
        if not cars:
            ax.text(0.5, 0.5, 'No cars in range', ha='center', va='center', 
                   color='white', transform=ax.transAxes)
            return
        
        X, Y, field = compute_group_wave_field(
            ego, cars, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config
        )
        
        field_norm = np.log1p(field)
        
        cmap = LinearSegmentedColormap.from_list('group_wave',
            ['#1A1A2E', '#1B4332', '#2D6A4F', '#40916C', '#52B788', '#74C69D', '#95D5B2'])
        pcm = ax.pcolormesh(X, Y, field_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Contours - ensure levels are unique and increasing
        if np.any(field_norm > 0):
            levels = np.percentile(field_norm[field_norm > 0], [30, 50, 70, 90])
            levels = np.unique(levels)
            if len(levels) >= 2 and levels[-1] > levels[0]:
                ax.contour(X, Y, field_norm, levels=levels, colors='white', alpha=0.4, linewidths=0.5)
        
        # Draw ego and cars
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Ego at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2), ego['length'], ego['width'],
            boxstyle="round,pad=0.02", facecolor=self.config.COLORS['truck'],
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 2, 'EGO', ha='center', color='white', fontsize=8, fontweight='bold')
        
        # Cars
        for car in cars:
            dx = car['x'] - ego['x']
            dy = car['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - car['length']/2, dy_rel - car['width']/2),
                car['length'], car['width'],
                boxstyle="round,pad=0.02", facecolor=self.config.COLORS['car'],
                edgecolor='white', linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
        
        # Group centroid marker
        if group_centroid:
            dx = group_centroid['x'] - ego['x']
            dy = group_centroid['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            ax.scatter(dx_rel, dy_rel, s=200, c='#2ECC71', marker='*', 
                      edgecolors='white', linewidths=2, zorder=10, label='Group Centroid')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        total_mass = sum(c['mass'] for c in cars) / 1000
        ax.set_title(f"Car Group Wave Field (Σ Ω_{{car_i→*}})\n{len(cars)} cars, Total: {total_mass:.0f}t",
                    fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('log(1+Σ Ω)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_interference_field(self, ax, ego: Dict, cars: List[Dict],
                                  x_range: Tuple, y_range: Tuple):
        """Plot interference between ego wave and group wave."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Compute both fields
        X, Y, ego_field = compute_ego_wave_field(
            ego, x_range, y_range,
            (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
            self.config
        )
        
        if cars:
            _, _, group_field = compute_group_wave_field(
                ego, cars, x_range, y_range,
                (self.config.FIELD_GRID_X, self.config.FIELD_GRID_Y),
                self.config
            )
        else:
            group_field = np.zeros_like(ego_field)
        
        # Compute difference (positive = ego dominates, negative = group dominates)
        diff_field = np.log1p(ego_field) - np.log1p(group_field)
        
        # Diverging colormap
        cmap = LinearSegmentedColormap.from_list('interference',
            ['#2ECC71', '#27AE60', '#1A1A2E', '#E74C3C', '#C0392B'])
        
        vmax = max(abs(diff_field.min()), abs(diff_field.max()), 0.1)
        pcm = ax.pcolormesh(X, Y, diff_field, cmap=cmap, shading='gouraud',
                           vmin=-vmax, vmax=vmax, alpha=0.9)
        
        # Draw vehicles
        self._draw_vehicles_local(ax, ego, cars)
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Longitudinal (m)', color='white')
        ax.set_ylabel('Lateral (m)', color='white')
        ax.set_title("Wave Dominance (Truck vs Cars)\nRed=Truck dominant, Green=Cars dominant",
                    fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('log(Ω_truck) - log(Σ Ω_cars)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_wide_traffic_view(self, ax, ego: Dict, surrounding: List[Dict],
                                svo_results: List[Dict], group_centroid: Dict):
        """Plot wide traffic view with wave indicators."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Background if available
        if self.loader and self.loader.background_image is not None:
            bg_extent = self.loader.get_background_extent()
            ax.imshow(self.loader.background_image, extent=bg_extent, alpha=0.4, aspect='equal', zorder=0)
        
        # Compute bounds
        all_x = [ego['x']] + [s['x'] for s in surrounding]
        all_y = [ego['y']] + [s['y'] for s in surrounding]
        
        margin = 30
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin/2, max(all_y) + margin/2
        
        # Draw wave circles around ego
        for r in [20, 40, 60]:
            circle = plt.Circle((ego['x'], ego['y']), r, fill=False, 
                               color='#E74C3C', alpha=0.3, linestyle='--', linewidth=1)
            ax.add_patch(circle)
        
        # Draw wave circles around group centroid
        if group_centroid:
            for r in [15, 30, 45]:
                circle = plt.Circle((group_centroid['x'], group_centroid['y']), r, 
                                   fill=False, color='#2ECC71', alpha=0.3, 
                                   linestyle='--', linewidth=1)
                ax.add_patch(circle)
        
        # Draw vehicles
        self._draw_vehicle_global(ax, ego, is_ego=True)
        ax.arrow(ego['x'], ego['y'], ego['vx']*0.8, ego['vy']*0.8,
                head_width=1.5, head_length=0.8, fc='yellow', ec='yellow', zorder=5)
        
        for other, svo in zip(surrounding, svo_results):
            self._draw_vehicle_global(ax, other, is_ego=False, svo=svo['other_svo'])
            ax.arrow(other['x'], other['y'], other['vx']*0.5, other['vy']*0.5,
                    head_width=1, head_length=0.5, fc='cyan', ec='cyan', alpha=0.6, zorder=5)
        
        # Group centroid marker
        if group_centroid:
            ax.scatter(group_centroid['x'], group_centroid['y'], s=300, c='#2ECC71',
                      marker='*', edgecolors='white', linewidths=2, zorder=10)
            ax.annotate(f"Group\n({group_centroid['n_vehicles']} cars)",
                       (group_centroid['x'], group_centroid['y']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='#2ECC71', fontweight='bold')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_title('Traffic Overview with Wave Propagation Indicators', 
                    fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#E74C3C', linestyle='--', label='Truck waves'),
            Line2D([0], [0], color='#2ECC71', linestyle='--', label='Group waves'),
            Line2D([0], [0], marker='*', color='#2ECC71', markersize=10, 
                  linestyle='None', label='Group centroid')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_individual_contributions(self, ax, ego: Dict, cars: List[Dict]):
        """Plot individual wave contributions from each car."""
        
        ax.set_facecolor('#1A1A2E')
        
        if not cars:
            ax.text(0.5, 0.5, 'No cars', ha='center', va='center', color='white')
            return
        
        # Compute aggressiveness from each car to ego
        contributions = []
        for car in cars:
            aggr = compute_aggressiveness(car, ego, self.config)
            dist = np.sqrt((car['x']-ego['x'])**2 + (car['y']-ego['y'])**2)
            contributions.append({
                'id': car['id'],
                'aggr': aggr,
                'distance': dist,
                'speed': car['speed']
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['aggr'], reverse=True)
        
        # Bar chart
        n = len(contributions)
        x = np.arange(n)
        aggrs = [c['aggr'] for c in contributions]
        labels = [f"Car {c['id']}" for c in contributions]
        
        # Color by contribution level
        colors = ['#E74C3C' if a > np.median(aggrs) else '#3498DB' for a in aggrs]
        
        bars = ax.barh(x, aggrs, color=colors, edgecolor='white', alpha=0.8)
        
        # Annotations
        for i, (c, bar) in enumerate(zip(contributions, bars)):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                   f'd={c["distance"]:.0f}m, v={c["speed"]:.1f}m/s',
                   va='center', fontsize=7, color='white')
        
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8, color='white')
        ax.set_xlabel('Aggressiveness (Ω)', color='white')
        ax.set_title('Individual Car Contributions to Ego', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        
        # Total line
        total = sum(aggrs)
        ax.axvline(total/n, color='yellow', linestyle='--', alpha=0.7, label=f'Mean: {total/n:.0f}')
        ax.legend(loc='lower right', fontsize=8)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_group_aggressiveness(self, ax, ego: Dict, cars: List[Dict], svo_results: List[Dict]):
        """Plot bidirectional aggressiveness: ego vs group."""
        
        ax.set_facecolor('#1A1A2E')
        
        if not cars:
            ax.text(0.5, 0.5, 'No cars', ha='center', va='center', color='white')
            return
        
        # Compute totals
        ego_to_cars = sum(compute_aggressiveness(ego, car, self.config) for car in cars)
        cars_to_ego = sum(compute_aggressiveness(car, ego, self.config) for car in cars)
        
        # Bar chart
        categories = [f'Truck → Cars\n(N={len(cars)})', f'Cars → Truck\n(N={len(cars)})']
        values = [ego_to_cars, cars_to_ego]
        colors = ['#E74C3C', '#3498DB']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='white', alpha=0.8)
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{val:.0f}', ha='center', fontsize=10, color='white', fontweight='bold')
        
        # Asymmetry annotation
        asymm = cars_to_ego - ego_to_cars
        ratio = cars_to_ego / ego_to_cars if ego_to_cars > 0 else 0
        
        ax.text(0.5, 0.95, f'Asymmetry: ΔΩ = {asymm:.0f} | Ratio: {ratio:.2f}x',
               transform=ax.transAxes, ha='center', fontsize=10, color='yellow',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax.set_ylabel('Total Aggressiveness (Σ Ω)', color='white')
        ax.set_title('Bidirectional Group Aggressiveness', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_wave_cross_section(self, ax, ego: Dict, cars: List[Dict], x_range: Tuple):
        """Plot wave intensity cross-section along longitudinal axis."""
        
        ax.set_facecolor('#1A1A2E')
        
        # Sample points along x-axis (y=0)
        x_points = np.linspace(x_range[0], x_range[1], 200)
        
        ego_profile = []
        group_profile = []
        
        cos_h = np.cos(ego['heading'])
        sin_h = np.sin(ego['heading'])
        
        for x_local in x_points:
            # Global position
            x_global = ego['x'] + x_local * cos_h
            y_global = ego['y'] + x_local * sin_h
            
            virtual_sfv = {
                'x': x_global, 'y': y_global,
                'heading': ego['heading'],
                'vx': 0, 'vy': 0, 'speed': 0,
                'ax': 0, 'ay': 0,
                'mass': self.config.MASS_PC
            }
            
            # Ego wave
            ego_aggr = compute_aggressiveness(ego, virtual_sfv, self.config)
            ego_profile.append(ego_aggr)
            
            # Group wave
            if cars:
                group_aggr = compute_total_aggressiveness(virtual_sfv, cars, self.config)
            else:
                group_aggr = 0
            group_profile.append(group_aggr)
        
        # Plot profiles
        ax.fill_between(x_points, 0, np.log1p(ego_profile), alpha=0.4, color='#E74C3C', label='Truck wave')
        ax.plot(x_points, np.log1p(ego_profile), color='#E74C3C', linewidth=2)
        
        ax.fill_between(x_points, 0, np.log1p(group_profile), alpha=0.4, color='#2ECC71', label='Car group wave')
        ax.plot(x_points, np.log1p(group_profile), color='#2ECC71', linewidth=2)
        
        # Mark ego position
        ax.axvline(0, color='white', linestyle='--', alpha=0.5, label='Ego position')
        
        # Mark car positions
        for car in cars:
            dx = car['x'] - ego['x']
            dy = car['y'] - ego['y']
            dx_rel = dx * np.cos(-ego['heading']) - dy * np.sin(-ego['heading'])
            if x_range[0] <= dx_rel <= x_range[1]:
                ax.axvline(dx_rel, color='#3498DB', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Longitudinal Distance from Ego (m)', color='white')
        ax.set_ylabel('log(1 + Ω)', color='white')
        ax.set_title('Wave Intensity Profile (y=0 cross-section)', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _plot_group_summary(self, ax, ego: Dict, surrounding: List[Dict], 
                           cars: List[Dict], svo_results: List[Dict], group_centroid: Dict):
        """Plot summary statistics panel."""
        
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
        
        # Compute statistics
        ego_to_cars = [compute_aggressiveness(ego, car, self.config) for car in cars] if cars else [0]
        cars_to_ego = [compute_aggressiveness(car, ego, self.config) for car in cars] if cars else [0]
        
        total_ego_to_cars = sum(ego_to_cars)
        total_cars_to_ego = sum(cars_to_ego)
        
        car_svo_results = [s for s, v in zip(svo_results, surrounding) if v['class'] in self.config.CAR_CLASSES]
        
        lines = [
            "═" * 42,
            "     GROUP WAVE ANALYSIS SUMMARY",
            "═" * 42,
            "",
            f"EGO VEHICLE: {ego['class'].upper()}",
            f"  ID: {ego['id']}",
            f"  Mass: {ego['mass']/1000:.1f} tons",
            f"  Speed: {ego['speed']:.1f} m/s ({ego['speed']*3.6:.1f} km/h)",
            "",
            "─" * 42,
            f"SURROUNDING GROUP: {len(cars)} CARS",
            "─" * 42,
        ]
        
        if group_centroid:
            lines.extend([
                f"  Combined mass: {group_centroid['mass']/1000:.1f} tons",
                f"  Centroid speed: {group_centroid['speed']:.1f} m/s",
                f"  Mass ratio (group/ego): {group_centroid['mass']/ego['mass']:.2f}x",
            ])
        
        lines.extend([
            "",
            "─" * 42,
            "WAVE SUPERPOSITION (Eq. 24)",
            "─" * 42,
            f"  Truck → Cars (Σ Ω): {total_ego_to_cars:.0f}",
            f"    Mean per car: {np.mean(ego_to_cars):.0f}",
            f"    Max: {max(ego_to_cars):.0f}, Min: {min(ego_to_cars):.0f}",
            "",
            f"  Cars → Truck (Σ Ω): {total_cars_to_ego:.0f}",
            f"    Mean per car: {np.mean(cars_to_ego):.0f}",
            f"    Max: {max(cars_to_ego):.0f}, Min: {min(cars_to_ego):.0f}",
            "",
            "─" * 42,
            "ASYMMETRY ANALYSIS",
            "─" * 42,
            f"  ΔΩ (total): {total_cars_to_ego - total_ego_to_cars:.0f}",
            f"  Ratio: {total_cars_to_ego/total_ego_to_cars:.2f}x" if total_ego_to_cars > 0 else "  Ratio: N/A",
        ])
        
        if total_cars_to_ego > total_ego_to_cars:
            lines.append("  → Cars collectively threaten truck more")
        else:
            lines.append("  → Truck threatens cars more (individually)")
        
        if car_svo_results:
            lines.extend([
                "",
                "─" * 42,
                "SVO STATISTICS",
                "─" * 42,
                f"  Truck SVO (mean): {np.mean([s['ego_svo'] for s in car_svo_results]):.1f}°",
                f"  Cars SVO (mean): {np.mean([s['other_svo'] for s in car_svo_results]):.1f}°",
            ])
        
        lines.extend([
            "",
            "═" * 42,
            "Model: Hu et al. (2023) IEEE T-IV",
            "Ω = (m_i|v_i|)/(2δm_j) exp(ξ₁+ξ₂)",
            "═" * 42,
        ])
        
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
               fontsize=8, color='white', family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    def _draw_vehicles_local(self, ax, ego: Dict, vehicles: List[Dict]):
        """Draw vehicles in ego-relative coordinates."""
        
        cos_h = np.cos(-ego['heading'])
        sin_h = np.sin(-ego['heading'])
        
        # Ego at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-ego['length']/2, -ego['width']/2), ego['length'], ego['width'],
            boxstyle="round,pad=0.02", facecolor=self.config.COLORS.get(ego['class'], '#E74C3C'),
            edgecolor='white', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego['width']/2 - 2, 'EGO', ha='center', color='white', fontsize=8, fontweight='bold')
        
        # Others
        for v in vehicles:
            dx = v['x'] - ego['x']
            dy = v['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            color = self.config.COLORS.get(v['class'], '#3498DB')
            rect = mpatches.FancyBboxPatch(
                (dx_rel - v['length']/2, dy_rel - v['width']/2),
                v['length'], v['width'],
                boxstyle="round,pad=0.02", facecolor=color,
                edgecolor='white', linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + v['width']/2 + 1, str(v['id']),
                   ha='center', fontsize=7, color='yellow')
    
    def _draw_vehicle_global(self, ax, veh: Dict, is_ego: bool = False, svo: float = None):
        """Draw vehicle in global coordinates."""
        
        if is_ego:
            color = self.config.COLORS.get(veh['class'], '#E74C3C')
            lw = 2
            alpha = 1.0
        else:
            if svo is not None:
                if svo > 60: color = '#27AE60'
                elif svo > 30: color = '#3498DB'
                elif svo > 0: color = '#F39C12'
                else: color = '#E74C3C'
            else:
                color = self.config.COLORS.get(veh['class'], '#3498DB')
            lw = 1
            alpha = 0.8
        
        corners = self._get_rotated_rect(veh['x'], veh['y'], veh['length'], veh['width'], veh['heading'])
        rect = plt.Polygon(corners, closed=True, facecolor=color,
                          edgecolor='white', linewidth=lw, alpha=alpha)
        ax.add_patch(rect)
        
        label = 'EGO' if is_ego else str(veh['id'])
        ax.text(veh['x'], veh['y'] + veh['width']/2 + 2, label,
               ha='center', fontsize=8, color='white' if is_ego else 'yellow',
               fontweight='bold' if is_ego else 'normal')
    
    def _get_rotated_rect(self, cx, cy, length, width, heading):
        """Get corners of rotated rectangle."""
        half_l, half_w = length/2, width/2
        corners = np.array([
            [-half_l, -half_w], [half_l, -half_w],
            [half_l, half_w], [-half_l, half_w]
        ])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return corners @ R.T + np.array([cx, cy])


# =============================================================================
# Main
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None,
         frame: Optional[int] = None, output_dir: str = './output_group'):
    """Main entry point."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Mechanical Wave Group Analysis")
    logger.info("Extended observation range for more vehicles")
    logger.info("=" * 60)
    
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return
    
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if not heavy_ids:
            logger.error("No heavy vehicles found")
            return
        ego_id = heavy_ids[0]
        logger.info(f"Auto-selected ego: {ego_id}")
    
    if frame is None:
        frame = loader.find_best_interaction_frame(ego_id, min_vehicles=5)
        if frame is None:
            logger.error("Could not find suitable frame")
            return
        logger.info(f"Auto-selected frame: {frame}")
    
    snapshot = loader.get_snapshot(ego_id, frame, heading_tol_deg=90.0, extended_range=True)
    if snapshot is None:
        logger.error("Could not get snapshot")
        return
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding vehicles: {len(snapshot['surrounding'])}")
    
    cars = [v for v in snapshot['surrounding'] if v['class'] in loader.config.CAR_CLASSES]
    logger.info(f"Cars in range: {len(cars)}")
    
    viz = GroupWaveVisualizer(loader=loader)
    
    output_file = output_path / f'group_wave_rec{recording_id}_ego{ego_id}_frame{snapshot["frame"]}.png'
    viz.create_group_analysis_figure(snapshot, str(output_file))
    
    logger.info(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mechanical Wave Group Analysis (Extended Range)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--recording', type=int, default=25)
    parser.add_argument('--ego_id', type=int, default=None)
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_group')
    
    args = parser.parse_args()
    main(args.data_dir, args.recording, args.ego_id, args.frame, args.output_dir)