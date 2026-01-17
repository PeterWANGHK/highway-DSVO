"""
Unified Risk Field Metrics Integration
========================================
Comprehensive evaluation framework integrating:
1. Field-Structural Metrics (Smoothness, Boundary, Occlusion)
2. Behavioral Alignment Metrics (Spatial, Temporal, Validity, Sensitivity)

Works across multiple field theories:
- GVF (Gaussian Velocity Field)
- EDRF (Elliptic Driving Risk Field)
- ADA (Asymmetric Driving Aggressiveness)
- APF-Wang (Artificial Potential Field from Wang et al. 2024)

Command-line usage:
    # All metrics on exiD data
    python unified_metrics_integration.py --data_dir ./exiD --recording 25 --ego_id 123
    
    # Only field metrics
    python unified_metrics_integration.py --data_dir ./exiD --recording 25 --metrics field
    
    # Only behavioral metrics
    python unified_metrics_integration.py --data_dir ./exiD --recording 25 --metrics behavioral
    
    # Specific field methods
    python unified_metrics_integration.py --data_dir ./exiD --recording 25 --methods gvf edrf ada apf
    
    # Demo mode (no data required)
    python unified_metrics_integration.py --demo --methods gvf edrf ada

Author: Research Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import argparse
import logging
import sys
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class MetricGroup(Enum):
    """Available metric groups."""
    FIELD = 'field'           # Field-structural metrics
    BEHAVIORAL = 'behavioral' # Behavioral alignment metrics
    ALL = 'all'               # Both groups


class FieldMethod(Enum):
    """Available field construction methods."""
    GVF = 'gvf'       # Gaussian Velocity Field
    EDRF = 'edrf'     # Elliptic Driving Risk Field
    ADA = 'ada'       # Asymmetric Driving Aggressiveness
    APF = 'apf'       # Artificial Potential Field (Wang et al.)


@dataclass
class UnifiedConfig:
    """Unified configuration for all metrics."""
    # Vehicle classes
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    
    # Mass estimates
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Grid parameters
    OBS_RANGE_AHEAD: float = 80.0
    OBS_RANGE_BEHIND: float = 40.0
    OBS_RANGE_LEFT: float = 20.0
    OBS_RANGE_RIGHT: float = 20.0
    GRID_NX: int = 80
    GRID_NY: int = 40
    
    # Behavioral analysis
    TEMPORAL_WINDOW_BEFORE: float = 3.0
    TEMPORAL_WINDOW_AFTER: float = 2.0
    SAMPLING_RATE: float = 10.0
    DECEL_THRESHOLD: float = -1.5
    ACCEL_THRESHOLD: float = 1.0
    LATERAL_THRESHOLD: float = 0.3
    
    # Risk thresholds (percentiles)
    HIGH_RISK_PERCENTILE: float = 75.0
    MODERATE_RISK_PERCENTILE: float = 50.0


# =============================================================================
# Import Metrics Modules (with fallback)
# =============================================================================

# Try to import existing modules
try:
    from risk_field_metrics import (
        SmoothnessAnalyzer, BoundaryAnalyzer, OcclusionAnalyzer,
        RiskFieldConstructor, OcclusionGeometry,
        AllMetrics as FieldMetrics, SmoothnessMetrics, BoundaryMetrics, OcclusionMetrics
    )
    FIELD_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("risk_field_metrics not found, using built-in implementation")
    FIELD_METRICS_AVAILABLE = False

try:
    from behavioral_alignment_metrics import (
        BehavioralAlignmentEvaluator, BehavioralAlignmentConfig,
        AllBehavioralMetrics, DriverBehaviorSequence, ManeuverEvent, ManeuverType,
        SpatialAlignmentMetrics, TemporalCoherenceMetrics,
        BehavioralValidityMetrics, DecisionSensitivityMetrics
    )
    BEHAVIORAL_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("behavioral_alignment_metrics not found, using built-in implementation")
    BEHAVIORAL_METRICS_AVAILABLE = False


# =============================================================================
# Built-in Field Metrics (if external module not available)
# =============================================================================

if not FIELD_METRICS_AVAILABLE:
    from numpy.fft import fft2, fftshift
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
    from scipy.stats import entropy
    
    @dataclass
    class SmoothnessMetrics:
        SNHS: float = 0.0
        ASD_s: float = 0.0
        ASD_d: float = 0.0
        AR: float = 0.0
        TVR: float = 0.0
        SSI: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class BoundaryMetrics:
        ISI: float = 0.0
        PPI: float = 0.0
        BCI: float = 0.0
        GCI: float = 0.0
        DBS: float = 0.0
        threshold: float = 0.5
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class OcclusionMetrics:
        MU: float = 0.0
        UV: float = 0.0
        CZU: float = 0.0
        ITOI_JS: float = 0.0
        OFSR: float = 0.0
        VWRD: float = 0.0
        SZRI_norm: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class FieldMetrics:
        smoothness: SmoothnessMetrics = field(default_factory=SmoothnessMetrics)
        boundary: BoundaryMetrics = field(default_factory=BoundaryMetrics)
        occlusion: OcclusionMetrics = field(default_factory=OcclusionMetrics)
        def to_dict(self):
            return {
                'smoothness': self.smoothness.to_dict(),
                'boundary': self.boundary.to_dict(),
                'occlusion': self.occlusion.to_dict()
            }
    
    class SmoothnessAnalyzer:
        def __init__(self, epsilon=1e-10): self.epsilon = epsilon
        def compute_all(self, R, dx=1.0, dy=1.0, k_cutoff_ratio=0.3):
            m = SmoothnessMetrics()
            grad_y, grad_x = np.gradient(R, dy, dx)
            grad_xx = np.gradient(grad_x, dx, axis=1)
            grad_yy = np.gradient(grad_y, dy, axis=0)
            grad_xy = np.gradient(grad_x, dy, axis=0)
            hess_frob_sq = grad_xx**2 + grad_yy**2 + 2*grad_xy**2
            m.SNHS = np.sum(hess_frob_sq) / (np.sum(R**2) + self.epsilon)
            m.ASD_s = np.sum(grad_xx**2) / R.size
            m.ASD_d = np.sum(grad_yy**2) / R.size
            m.AR = m.ASD_s / (m.ASD_d + self.epsilon)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            m.TVR = np.sum(grad_mag)*dx*dy / (np.max(np.abs(R))*np.sqrt(R.size*dx*dy) + self.epsilon)
            R_fft = fftshift(fft2(R))
            ps = np.abs(R_fft)**2
            ny, nx = R.shape
            kx, ky = np.fft.fftshift(np.fft.fftfreq(nx)), np.fft.fftshift(np.fft.fftfreq(ny))
            KX, KY = np.meshgrid(kx, ky)
            K_mag = np.sqrt(KX**2 + KY**2)
            k_cut = k_cutoff_ratio * np.max(K_mag)
            m.SSI = np.sum(ps[K_mag > k_cut]) / (np.sum(ps) + self.epsilon)
            return m
    
    class BoundaryAnalyzer:
        def __init__(self, epsilon=1e-10): self.epsilon = epsilon
        def compute_all(self, R, threshold=None, dx=1.0, dy=1.0, delta_band=0.1, **kwargs):
            m = BoundaryMetrics()
            if threshold is None: threshold = np.median(R)
            m.threshold = threshold
            R_min, R_max = np.min(R), np.max(R)
            delta_R = R_max - R_min
            if delta_R < self.epsilon: return m
            R_norm = 2*(R - threshold)/delta_R
            grad_y, grad_x = np.gradient(R, dy, dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            boundary_mask = np.abs(R - threshold) < delta_band*delta_R
            if np.any(boundary_mask):
                m.ISI = np.mean(grad_mag[boundary_mask]) / (delta_R + self.epsilon)
            m.PPI = np.mean(R_norm**2)
            m.GCI = np.sum(grad_mag[boundary_mask]**2) / (np.sum(grad_mag**2) + self.epsilon) if np.any(boundary_mask) else 0
            m.BCI = 1.0
            m.DBS = 1.0
            return m
    
    class OcclusionAnalyzer:
        def __init__(self, epsilon=1e-10): self.epsilon = epsilon
        def compute_all(self, R_full, R_occluded, visibility_map=None, shadow_mask=None,
                       critical_zone_mask=None, threshold=None, dx=1.0, dy=1.0):
            m = OcclusionMetrics()
            if threshold is None: threshold = np.median(R_full)
            delta_R = np.maximum(0, R_full - R_occluded)
            m.MU = np.max(delta_R)
            m.UV = np.sum(delta_R)*dx*dy
            if critical_zone_mask is not None and np.any(critical_zone_mask):
                m.CZU = np.sum(delta_R[critical_zone_mask]) / (np.sum(R_full[critical_zone_mask]) + self.epsilon)
            truly_unsafe = R_full > threshold
            appears_safe = R_occluded <= threshold
            false_safe = truly_unsafe & appears_safe
            if np.any(truly_unsafe):
                m.OFSR = np.sum(false_safe) / np.sum(truly_unsafe)
            if visibility_map is not None:
                m.VWRD = np.sum((1-visibility_map)*R_full) / (np.sum(R_full) + self.epsilon)
            return m


# =============================================================================
# Built-in Behavioral Metrics (if external module not available)
# =============================================================================

if not BEHAVIORAL_METRICS_AVAILABLE:
    from scipy import signal, stats
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    @dataclass
    class SpatialAlignmentMetrics:
        CTAI: float = 0.0
        DZOC: float = 0.0
        LPDA: float = 0.0
        TBD: float = 0.0
        GMDC: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class TemporalCoherenceMetrics:
        DTC: float = 0.0
        RATC: float = 0.0
        RATL: float = 0.0
        PRPT: float = 0.0
        TGSC: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class BehavioralValidityMetrics:
        RZBV: float = 0.0
        ZCPS: float = 0.0
        BPSR: float = 0.0
        MZAC: float = 0.0
        ARI: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class DecisionSensitivityMetrics:
        MODR_precision: float = 0.0
        MODR_recall: float = 0.0
        MODR_f1: float = 0.0
        BRC: float = 0.0
        SCAC: float = 0.0
        DGS: float = 0.0
        def to_dict(self): return self.__dict__.copy()
    
    @dataclass
    class AllBehavioralMetrics:
        spatial: SpatialAlignmentMetrics = field(default_factory=SpatialAlignmentMetrics)
        temporal: TemporalCoherenceMetrics = field(default_factory=TemporalCoherenceMetrics)
        validity: BehavioralValidityMetrics = field(default_factory=BehavioralValidityMetrics)
        sensitivity: DecisionSensitivityMetrics = field(default_factory=DecisionSensitivityMetrics)
        overall_alignment_score: float = 0.0
        def to_dict(self):
            return {
                'spatial': self.spatial.to_dict(),
                'temporal': self.temporal.to_dict(),
                'validity': self.validity.to_dict(),
                'sensitivity': self.sensitivity.to_dict(),
                'overall_alignment_score': self.overall_alignment_score
            }
        def compute_overall_score(self, weights=None):
            if weights is None:
                weights = {'CTAI': 0.15, 'DZOC': 0.10, 'LPDA': 0.10, 'DTC': 0.10, 'RATC': 0.10,
                          'RZBV': 0.15, 'ARI': 0.10, 'MODR_f1': 0.10, 'SCAC': 0.10}
            all_vals = {**self.spatial.to_dict(), **self.temporal.to_dict(),
                       **self.validity.to_dict(), **self.sensitivity.to_dict()}
            score, total = 0.0, 0.0
            for m, w in weights.items():
                if m in all_vals and not np.isnan(all_vals[m]):
                    score += w * all_vals[m]
                    total += w
            self.overall_alignment_score = score/total if total > 0 else 0
            return self.overall_alignment_score
    
    @dataclass
    class DriverBehaviorSequence:
        timestamps: np.ndarray
        positions_x: np.ndarray
        positions_y: np.ndarray
        velocities_x: np.ndarray
        velocities_y: np.ndarray
        accelerations_x: np.ndarray
        accelerations_y: np.ndarray
        headings: np.ndarray
        lane_ids: np.ndarray
        maneuver_labels: np.ndarray = None
        risk_values: np.ndarray = None
        @property
        def speed(self): return np.sqrt(self.velocities_x**2 + self.velocities_y**2)
    
    class ManeuverType(Enum):
        LANE_CHANGE_LEFT = 'lc_left'
        LANE_CHANGE_RIGHT = 'lc_right'
        DECELERATION = 'decel'
        NONE = 'none'
    
    @dataclass
    class ManeuverEvent:
        maneuver_type: ManeuverType
        start_time: float
        end_time: float
        start_position: Tuple[float, float]
        end_position: Tuple[float, float]
        peak_intensity: float
    
    @dataclass
    class BehavioralAlignmentConfig:
        high_risk_percentile: float = 75.0
        moderate_risk_percentile: float = 50.0
        temporal_window_before: float = 3.0
        temporal_window_after: float = 2.0
        sampling_rate: float = 10.0
        longitudinal_resolution: float = 5.0
        lateral_resolution: float = 0.5
        decel_threshold: float = -1.5
        accel_threshold: float = 1.0
        lateral_threshold: float = 0.3
        min_samples_per_zone: int = 10
        significance_level: float = 0.05


# =============================================================================
# Risk Field Constructors
# =============================================================================

class UnifiedRiskFieldConstructor:
    """Construct risk fields using multiple methodologies."""
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self.epsilon = 1e-10
    
    def construct(self, method: str, ego: Dict, others: List[Dict],
                  x_range: Tuple[float, float], y_range: Tuple[float, float],
                  grid_size: Tuple[int, int] = (80, 40)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct risk field using specified method.
        
        Args:
            method: 'gvf', 'edrf', 'ada', or 'apf'
            ego: Ego vehicle dict
            others: List of other vehicle dicts
            x_range, y_range: Grid extent
            grid_size: (nx, ny)
        
        Returns:
            (X_mesh, Y_mesh, risk_field)
        """
        method = method.lower()
        
        if method == 'gvf':
            return self._construct_gvf(ego, others, x_range, y_range, grid_size)
        elif method == 'edrf':
            return self._construct_edrf(ego, others, x_range, y_range, grid_size)
        elif method == 'ada':
            return self._construct_ada(ego, others, x_range, y_range, grid_size)
        elif method == 'apf':
            return self._construct_apf_wang(ego, others, x_range, y_range, grid_size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _construct_gvf(self, ego, others, x_range, y_range, grid_size):
        """Gaussian Velocity Field risk."""
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        if not others:
            return X_mesh, Y_mesh, np.zeros_like(X_mesh)
        
        cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
        
        positions, rel_velocities = [], []
        for other in others:
            dx, dy = other['x'] - ego['x'], other['y'] - ego['y']
            dx_rel = dx*cos_h - dy*sin_h
            dy_rel = dx*sin_h + dy*cos_h
            dvx = other.get('vx', 0) - ego.get('vx', 0)
            dvy = other.get('vy', 0) - ego.get('vy', 0)
            dvx_rel = dvx*cos_h - dvy*sin_h
            dvy_rel = dvx*sin_h + dvy*cos_h
            positions.append([dx_rel, dy_rel])
            rel_velocities.append([dvx_rel, dvy_rel])
        
        positions = np.array(positions)
        rel_velocities = np.array(rel_velocities)
        
        P_test = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        sigma_x, sigma_y = 15, 2.5
        
        K = self._anisotropic_rbf(positions, positions, sigma_x, sigma_y)
        K_s = self._anisotropic_rbf(P_test, positions, sigma_x, sigma_y)
        K_inv = np.linalg.inv(K + 1e-6*np.eye(len(K)))
        
        VX = (K_s @ K_inv @ rel_velocities[:, 0]).reshape(X_mesh.shape)
        VY = (K_s @ K_inv @ rel_velocities[:, 1]).reshape(X_mesh.shape)
        
        risk = np.sqrt(VX**2 + VY**2)
        risk[VX < 0] *= 1.5  # Approaching vehicles
        
        return X_mesh, Y_mesh, risk
    
    def _construct_edrf(self, ego, others, x_range, y_range, grid_size):
        """Elliptic Driving Risk Field."""
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        risk = np.zeros_like(X_mesh)
        
        if not others:
            return X_mesh, Y_mesh, risk
        
        cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
        
        for other in others:
            dx, dy = other['x'] - ego['x'], other['y'] - ego['y']
            dx_rel = dx*cos_h - dy*sin_h
            dy_rel = dx*sin_h + dy*cos_h
            speed = other.get('speed', np.sqrt(other.get('vx', 0)**2 + other.get('vy', 0)**2))
            
            sigma_x = 5.0 + 0.5*speed
            sigma_y = 2.0
            
            risk_contrib = np.exp(-((X_mesh - dx_rel)**2/(2*sigma_x**2) + 
                                    (Y_mesh - dy_rel)**2/(2*sigma_y**2)))
            dist = np.sqrt(dx_rel**2 + dy_rel**2)
            scale = (speed + 5) / (dist + 5)
            risk += scale * risk_contrib
        
        return X_mesh, Y_mesh, risk
    
    def _construct_ada(self, ego, others, x_range, y_range, grid_size):
        """Asymmetric Driving Aggressiveness field."""
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        risk = np.zeros_like(X_mesh)
        
        if not others:
            return X_mesh, Y_mesh, risk
        
        cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
        ego_mass = ego.get('mass', 3000)
        ego_v = ego.get('speed', np.sqrt(ego.get('vx', 0)**2 + ego.get('vy', 0)**2))
        
        mu1, mu2, delta = 0.25, 0.25, 0.001
        
        for other in others:
            dx, dy = other['x'] - ego['x'], other['y'] - ego['y']
            dx_rel = dx*cos_h - dy*sin_h
            dy_rel = dx*sin_h + dy*cos_h
            dist = max(1.0, np.sqrt(dx**2 + dy**2))
            ux, uy = dx/dist, dy/dist
            
            other_mass = other.get('mass', 3000)
            other_v = other.get('speed', np.sqrt(other.get('vx', 0)**2 + other.get('vy', 0)**2))
            
            cos_i = np.clip((other.get('vx', 0)*ux + other.get('vy', 0)*uy)/other_v, -1, 1) if other_v > 0.1 else 0
            cos_j = np.clip(-(ego.get('vx', 0)*ux + ego.get('vy', 0)*uy)/ego_v, -1, 1) if ego_v > 0.1 else 0
            
            xi1 = mu1*other_v*cos_i + mu2*ego_v*cos_j
            omega = np.clip((other_mass*other_v)/(2*delta*ego_mass)*np.exp(xi1), 0, 2000)
            
            spatial = np.exp(-((X_mesh - dx_rel)**2/288 + (Y_mesh - dy_rel)**2/32))
            risk += omega * spatial
        
        return X_mesh, Y_mesh, risk
    
    def _construct_apf_wang(self, ego, others, x_range, y_range, grid_size):
        """
        Artificial Potential Field from Wang et al. (2024).
        
        Implements:
        - LGPF: Longitudinal Gravitational Potential Field
        - TGPF: Transverse Gravitational Potential Field
        - LRPF: Longitudinal Repulsion Potential Field
        - TRPF: Transverse Repulsion Potential Field
        """
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        # Parameters from Wang et al.
        L = 3.5  # Lane width
        rho_action = 50.0  # LRPF action threshold
        S_LF = 11.0  # Safe gap
        k11 = 0.85  # LGPF coefficient
        k12 = 0.26  # TGPF coefficient
        k21 = 0.95  # LRPF coefficient
        k24, k25 = 0.30, 0.25  # TRPF coefficients
        
        # Weights from paper (Table 8)
        w1, w2, w3, w4, w5 = 0.277, 0.063, 0.312, 0.246, 0.102
        
        # Target point (end of taper section)
        taper_end_x = x_range[1] * 0.8  # Assume 80% of range
        
        # 1. LGPF - Longitudinal Gravitational (attraction to target)
        rho_11 = np.sqrt((X_mesh - taper_end_x)**2) / (x_range[1] - x_range[0])
        U_lgpf = 0.5 * k11 * np.exp(-2 * rho_11)
        U_lgpf[np.abs(Y_mesh) > L/2] = 0
        
        # 2. TGPF - Transverse Gravitational (attraction to adjacent lane)
        target_lane_y = L  # Adjacent lane center
        rho_tgpf = np.sqrt((Y_mesh - target_lane_y)**2)
        U_tgpf = 0.5 * k12 * rho_tgpf**2
        U_tgpf[np.abs(Y_mesh) > L/2] = 0
        
        # 3. LRPF - Longitudinal Repulsion from other vehicles
        U_lrpf_lv = np.zeros_like(X_mesh)  # Leading vehicle
        U_lrpf_fv = np.zeros_like(X_mesh)  # Following vehicle
        
        if others:
            cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
            
            for other in others:
                dx, dy = other['x'] - ego['x'], other['y'] - ego['y']
                dx_rel = dx*cos_h - dy*sin_h
                dy_rel = dx*sin_h + dy*cos_h
                
                # Relative velocity for bias effect
                dvx = other.get('vx', 0) - ego.get('vx', 0)
                dvy = other.get('vy', 0) - ego.get('vy', 0)
                dvx_rel = dvx*cos_h - dvy*sin_h
                
                # Bias effect (Eq. 6-7)
                A_hat = 1.0
                bias = np.sign(dvx_rel) * A_hat * np.log(1 + abs(dvx_rel)) if dvx_rel != 0 else 0
                
                # Biased position
                px_biased = dx_rel + bias
                
                dist = np.sqrt((X_mesh - px_biased)**2 + (Y_mesh - dy_rel)**2)
                
                # k22 depends on ICV environment (using guidance group value)
                k22 = 0.54
                
                # Gap factor k23 (Eq. 10)
                k23 = 0.5  # Simplified
                
                # LRPF (Eq. 8-9)
                lrpf_contrib = 0.5 * k21 / (1 + np.exp(k22*(dist - rho_action))) * (1 + k23)
                
                if dx_rel > 0:  # Leading vehicle
                    U_lrpf_lv += lrpf_contrib
                else:  # Following vehicle
                    U_lrpf_fv += lrpf_contrib
        
        # 4. TRPF - Transverse Repulsion from lane boundaries
        # Eq. 12: Different functions for different regions
        delta_D = Y_mesh  # Offset from lane center
        U_trpf = np.where(
            np.abs(delta_D) > L/4,
            k24 * (np.exp(L/2 - np.abs(delta_D)) - 1),
            k25 * np.sin(2*(L/2 - np.abs(delta_D))/L)
        )
        U_trpf = np.clip(U_trpf, 0, 1)
        
        # Total Potential Field (Eq. 13)
        TPF = w1*U_lgpf + w2*U_tgpf + w3*U_lrpf_lv + w4*U_lrpf_fv + w5*U_trpf
        
        return X_mesh, Y_mesh, TPF
    
    def _anisotropic_rbf(self, XA, XB, sigma_x, sigma_y):
        diff_x = XA[:, 0:1] - XB[:, 0:1].T
        diff_y = XA[:, 1:2] - XB[:, 1:2].T
        return np.exp(-0.5*(diff_x**2/sigma_x**2 + diff_y**2/sigma_y**2))


# =============================================================================
# Occlusion Geometry
# =============================================================================

class UnifiedOcclusionGeometry:
    """Compute occlusion-related geometry."""
    
    def compute_shadow_mask(self, ego: Dict, occluders: List[Dict],
                           X_mesh: np.ndarray, Y_mesh: np.ndarray) -> np.ndarray:
        shadow_mask = np.zeros(X_mesh.shape, dtype=bool)
        if not occluders:
            return shadow_mask
        
        cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
        
        for occ in occluders:
            dx, dy = occ['x'] - ego['x'], occ['y'] - ego['y']
            dx_rel = dx*cos_h - dy*sin_h
            dy_rel = dx*sin_h + dy*cos_h
            d_occ = np.sqrt(dx_rel**2 + dy_rel**2)
            if d_occ < 1.0:
                continue
            
            phi_occ = np.arctan2(dy_rel, dx_rel)
            occ_l, occ_w = occ.get('length', 5.0), occ.get('width', 2.0)
            occ_h_rel = occ.get('heading', 0) - ego.get('heading', 0)
            w_eff = abs(occ_l*np.sin(phi_occ - occ_h_rel)) + abs(occ_w*np.cos(phi_occ - occ_h_rel))
            alpha_occ = 2*np.arctan(w_eff/(2*d_occ))
            
            phi_grid = np.arctan2(Y_mesh, X_mesh)
            d_grid = np.sqrt(X_mesh**2 + Y_mesh**2)
            angle_diff = np.abs(np.arctan2(np.sin(phi_grid - phi_occ), np.cos(phi_grid - phi_occ)))
            
            shadow_mask |= (d_grid > d_occ) & (angle_diff < alpha_occ/2)
        
        return shadow_mask
    
    def compute_visibility_map(self, ego: Dict, occluders: List[Dict],
                              X_mesh: np.ndarray, Y_mesh: np.ndarray) -> np.ndarray:
        visibility = np.ones(X_mesh.shape)
        if not occluders:
            return visibility
        
        cos_h, sin_h = np.cos(-ego.get('heading', 0)), np.sin(-ego.get('heading', 0))
        
        for occ in occluders:
            dx, dy = occ['x'] - ego['x'], occ['y'] - ego['y']
            dx_rel = dx*cos_h - dy*sin_h
            dy_rel = dx*sin_h + dy*cos_h
            d_occ = np.sqrt(dx_rel**2 + dy_rel**2)
            if d_occ < 1.0:
                continue
            
            phi_occ = np.arctan2(dy_rel, dx_rel)
            occ_l, occ_w = occ.get('length', 5.0), occ.get('width', 2.0)
            occ_h_rel = occ.get('heading', 0) - ego.get('heading', 0)
            w_eff = abs(occ_l*np.sin(phi_occ - occ_h_rel)) + abs(occ_w*np.cos(phi_occ - occ_h_rel))
            alpha_occ = 2*np.arctan(w_eff/(2*d_occ))
            
            phi_grid = np.arctan2(Y_mesh, X_mesh)
            d_grid = np.sqrt(X_mesh**2 + Y_mesh**2)
            angle_diff = np.abs(np.arctan2(np.sin(phi_grid - phi_occ), np.cos(phi_grid - phi_occ)))
            
            behind = np.clip((d_grid - d_occ)/10.0, 0, 1)
            angle_factor = np.clip(1 - angle_diff/(alpha_occ/2 + 0.01), 0, 1)
            visibility *= (1 - 0.9*behind*angle_factor)
        
        return visibility


# =============================================================================
# Unified Evaluator
# =============================================================================

class UnifiedRiskFieldEvaluator:
    """
    Unified evaluator for both field-structural and behavioral alignment metrics.
    """
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        
        # Field metrics analyzers
        self.smoothness_analyzer = SmoothnessAnalyzer()
        self.boundary_analyzer = BoundaryAnalyzer()
        self.occlusion_analyzer = OcclusionAnalyzer()
        
        # Constructors
        self.constructor = UnifiedRiskFieldConstructor(self.config)
        self.occ_geometry = UnifiedOcclusionGeometry()
        
        # Behavioral analyzer (if available)
        if BEHAVIORAL_METRICS_AVAILABLE:
            self.behavioral_evaluator = BehavioralAlignmentEvaluator()
        else:
            self.behavioral_evaluator = None
    
    def evaluate_scenario(self,
                          snapshot: Dict,
                          methods: List[str] = None,
                          metric_groups: List[str] = None,
                          behavior: DriverBehaviorSequence = None,
                          occluder_ids: List[int] = None) -> Dict[str, Dict]:
        """
        Evaluate a scenario with multiple methods and metric groups.
        
        Args:
            snapshot: Data with 'ego' and 'surrounding'
            methods: List of field methods ['gvf', 'edrf', 'ada', 'apf']
            metric_groups: List of groups ['field', 'behavioral', 'all']
            behavior: Driver behavior sequence (for behavioral metrics)
            occluder_ids: IDs of occluding vehicles
        
        Returns:
            Nested dict: {method: {'field': FieldMetrics, 'behavioral': AllBehavioralMetrics}}
        """
        if methods is None:
            methods = ['gvf', 'edrf', 'ada']
        
        if metric_groups is None:
            metric_groups = ['all']
        
        compute_field = 'field' in metric_groups or 'all' in metric_groups
        compute_behavioral = 'behavioral' in metric_groups or 'all' in metric_groups
        
        ego = snapshot['ego']
        others = snapshot['surrounding']
        
        # Identify occluders
        if occluder_ids is None:
            occluders = [v for v in others if v.get('class', '') in self.config.HEAVY_VEHICLE_CLASSES]
        else:
            occluders = [v for v in others if v.get('id') in occluder_ids]
        
        # Grid setup
        x_range = (-self.config.OBS_RANGE_BEHIND, self.config.OBS_RANGE_AHEAD)
        y_range = (-self.config.OBS_RANGE_LEFT, self.config.OBS_RANGE_RIGHT)
        grid_size = (self.config.GRID_NX, self.config.GRID_NY)
        
        dx = (x_range[1] - x_range[0]) / grid_size[0]
        dy = (y_range[1] - y_range[0]) / grid_size[1]
        
        results = {}
        
        for method in methods:
            try:
                X, Y, R = self.constructor.construct(method, ego, others, x_range, y_range, grid_size)
            except Exception as e:
                logger.warning(f"Failed to construct {method} field: {e}")
                continue
            
            method_results = {'X': X, 'Y': Y, 'R': R}
            
            # Field-structural metrics
            if compute_field:
                shadow_mask = self.occ_geometry.compute_shadow_mask(ego, occluders, X, Y)
                visibility_map = self.occ_geometry.compute_visibility_map(ego, occluders, X, Y)
                critical_zone = (X > 0) & (X < 60) & (np.abs(Y) < 6)
                
                smooth = self.smoothness_analyzer.compute_all(R, dx, dy)
                threshold = np.percentile(R, 70)
                boundary = self.boundary_analyzer.compute_all(R, threshold, dx, dy)
                
                R_occluded = R * visibility_map
                occlusion = self.occlusion_analyzer.compute_all(
                    R, R_occluded, visibility_map, shadow_mask, critical_zone, threshold, dx, dy
                )
                
                method_results['field'] = FieldMetrics(smooth, boundary, occlusion)
            
            # Behavioral alignment metrics
            if compute_behavioral and behavior is not None:
                # Generate risk sequence along trajectory
                risk_sequence = self._sample_risk_along_trajectory(R, X, Y, behavior)
                
                # Extract decision positions
                decision_positions = self._extract_decision_positions(behavior)
                
                if BEHAVIORAL_METRICS_AVAILABLE and self.behavioral_evaluator is not None:
                    behavioral_metrics = self.behavioral_evaluator.evaluate_full(
                        R, X, Y, risk_sequence, behavior.timestamps, behavior,
                        decision_positions
                    )
                else:
                    behavioral_metrics = self._compute_behavioral_metrics_builtin(
                        R, X, Y, risk_sequence, behavior, decision_positions
                    )
                
                method_results['behavioral'] = behavioral_metrics
            
            results[method] = method_results
        
        return results
    
    def _sample_risk_along_trajectory(self, R: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                      behavior: DriverBehaviorSequence) -> np.ndarray:
        """Sample risk field values along the driver's trajectory."""
        from scipy.interpolate import RegularGridInterpolator
        
        x_1d = X[0, :] if len(X.shape) > 1 else X
        y_1d = Y[:, 0] if len(Y.shape) > 1 else Y
        
        interp = RegularGridInterpolator((y_1d, x_1d), R, bounds_error=False, fill_value=0)
        
        # Transform positions to ego frame (assuming ego at origin)
        points = np.column_stack([behavior.positions_y, behavior.positions_x])
        risk_sequence = interp(points)
        
        return risk_sequence
    
    def _extract_decision_positions(self, behavior: DriverBehaviorSequence) -> List[Tuple[float, float]]:
        """Extract decision positions from behavior."""
        positions = []
        
        action_intensity = np.abs(behavior.accelerations_x) + 0.5*np.abs(behavior.velocities_y)
        threshold = np.percentile(action_intensity, 75)
        high_action = action_intensity > threshold
        
        from scipy import ndimage
        labels, n_features = ndimage.label(high_action)
        
        for i in range(1, n_features + 1):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                mid_idx = indices[len(indices) // 2]
                positions.append((behavior.positions_x[mid_idx], behavior.positions_y[mid_idx]))
        
        return positions
    
    def _compute_behavioral_metrics_builtin(self, R, X, Y, risk_sequence, behavior, decision_positions):
        """Built-in behavioral metrics computation."""
        metrics = AllBehavioralMetrics()
        
        # Simplified spatial alignment
        if len(decision_positions) > 0:
            metrics.spatial.LPDA = 0.5  # Placeholder
            metrics.spatial.GMDC = 0.5
        
        # Simplified temporal
        if len(risk_sequence) > 10:
            dt = np.mean(np.diff(behavior.timestamps))
            risk_grad = np.gradient(risk_sequence, dt)
            accel_corr = np.corrcoef(risk_grad[:-1], behavior.accelerations_x[:len(risk_grad)-1])[0, 1]
            metrics.temporal.TGSC = abs(accel_corr) if not np.isnan(accel_corr) else 0
        
        # Simplified validity
        if len(risk_sequence) >= 30:
            try:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                risk_labels = kmeans.fit_predict(risk_sequence.reshape(-1, 1))
                
                # Create behavior labels
                behavior_labels = np.zeros(len(risk_sequence))
                active = (behavior.accelerations_x[:len(risk_sequence)] < self.config.DECEL_THRESHOLD) | \
                         (np.abs(behavior.velocities_y[:len(risk_sequence)]) > self.config.LATERAL_THRESHOLD)
                behavior_labels[active] = 1
                
                metrics.validity.ARI = adjusted_rand_score(behavior_labels, risk_labels)
                metrics.validity.RZBV = normalized_mutual_info_score(risk_labels, behavior_labels.astype(int))
            except:
                pass
        
        metrics.compute_overall_score()
        return metrics


# =============================================================================
# ExiD Data Loader
# =============================================================================

class ExiDDataLoader:
    """Load exiD dataset for evaluation."""
    
    def __init__(self, data_dir: str, config: UnifiedConfig = None):
        self.data_dir = Path(data_dir)
        self.config = config or UnifiedConfig()
        self.tracks_df = None
        self.tracks_meta_df = None
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['trackId', 'class', 'width', 'length']],
                on='trackId', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['width'] = self.tracks_df['width'].fillna(self.tracks_df['width_meta'])
                self.tracks_df['length'] = self.tracks_df['length'].fillna(self.tracks_df['length_meta'])
            
            logger.info(f"Loaded recording {recording_id}: {len(self.tracks_df)} rows")
            return True
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int) -> Optional[Dict]:
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
        
        surrounding = []
        for _, row in frame_data.iterrows():
            if row['trackId'] == ego_id:
                continue
            
            dx = row['xCenter'] - ego['x']
            dy = row['yCenter'] - ego['y']
            if abs(dx) > 100 or abs(dy) > 30:
                continue
            
            other_class = str(row.get('class', 'car')).lower()
            other = {
                'id': int(row['trackId']),
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'heading': np.radians(float(row.get('heading', 0))),
                'vx': float(row.get('xVelocity', 0)),
                'vy': float(row.get('yVelocity', 0)),
                'speed': np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                'width': float(row.get('width', 1.8)),
                'length': float(row.get('length', 4.5)),
                'class': other_class,
                'mass': self.config.MASS_HV if other_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            }
            surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def get_behavior_sequence(self, vehicle_id: int, 
                              start_frame: int = None, 
                              end_frame: int = None) -> Optional[DriverBehaviorSequence]:
        """Extract behavior sequence for a vehicle."""
        veh_data = self.tracks_df[self.tracks_df['trackId'] == vehicle_id].sort_values('frame')
        
        if veh_data.empty:
            return None
        
        if start_frame is not None:
            veh_data = veh_data[veh_data['frame'] >= start_frame]
        if end_frame is not None:
            veh_data = veh_data[veh_data['frame'] <= end_frame]
        
        if len(veh_data) < 10:
            return None
        
        # Assume 25 Hz for exiD
        dt = 1.0 / 25.0
        timestamps = (veh_data['frame'].values - veh_data['frame'].values[0]) * dt
        
        return DriverBehaviorSequence(
            timestamps=timestamps,
            positions_x=veh_data['xCenter'].values,
            positions_y=veh_data['yCenter'].values,
            velocities_x=veh_data['xVelocity'].values,
            velocities_y=veh_data['yVelocity'].values,
            accelerations_x=veh_data['xAcceleration'].values if 'xAcceleration' in veh_data else np.gradient(veh_data['xVelocity'].values, dt),
            accelerations_y=veh_data['yAcceleration'].values if 'yAcceleration' in veh_data else np.gradient(veh_data['yVelocity'].values, dt),
            headings=np.radians(veh_data['heading'].values) if 'heading' in veh_data else np.zeros(len(veh_data)),
            lane_ids=veh_data['laneId'].values if 'laneId' in veh_data else np.zeros(len(veh_data))
        )


# =============================================================================
# Demo Data Generation
# =============================================================================

def create_demo_scenario():
    """Create demo scenario for testing."""
    ego = {
        'id': 0, 'x': 0.0, 'y': 0.0, 'vx': 15.0, 'vy': 0.0, 'speed': 15.0,
        'heading': 0.0, 'length': 12.0, 'width': 2.5, 'mass': 15000, 'class': 'truck'
    }
    
    others = [
        {'id': 1, 'x': 35.0, 'y': 0.0, 'vx': 12.0, 'vy': 0.0, 'speed': 12.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        {'id': 2, 'x': 25.0, 'y': 8.0, 'vx': 18.0, 'vy': -1.5, 'speed': 18.1,
         'heading': -0.08, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        {'id': 3, 'x': 20.0, 'y': 3.5, 'vx': 14.0, 'vy': 0.0, 'speed': 14.0,
         'heading': 0.0, 'length': 10.0, 'width': 2.5, 'mass': 12000, 'class': 'truck'},
        {'id': 4, 'x': 45.0, 'y': 5.0, 'vx': 16.0, 'vy': -0.5, 'speed': 16.0,
         'heading': -0.03, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        {'id': 5, 'x': -25.0, 'y': 0.0, 'vx': 17.0, 'vy': 0.0, 'speed': 17.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
    ]
    
    return {'ego': ego, 'surrounding': others, 'frame': 100}


def create_demo_behavior(n_samples: int = 200):
    """Create demo behavior sequence."""
    dt = 0.1
    timestamps = np.arange(n_samples) * dt
    
    positions_x = np.cumsum(np.ones(n_samples) * 20 * dt)
    positions_y = np.zeros(n_samples)
    
    # Lane change at t=8-12s
    lc_start, lc_end = 80, 120
    positions_y[lc_start:lc_end] = np.linspace(0, 3.5, lc_end - lc_start)
    positions_y[lc_end:] = 3.5
    
    velocities_x = np.gradient(positions_x, dt)
    velocities_y = np.gradient(positions_y, dt)
    velocities_x[50:80] -= np.linspace(0, 3, 30)
    velocities_x[80:] = velocities_x[79]
    
    accelerations_x = np.gradient(velocities_x, dt)
    accelerations_y = np.gradient(velocities_y, dt)
    headings = np.arctan2(velocities_y, velocities_x)
    lane_ids = np.zeros(n_samples)
    lane_ids[lc_end:] = 1
    
    return DriverBehaviorSequence(
        timestamps=timestamps,
        positions_x=positions_x,
        positions_y=positions_y,
        velocities_x=velocities_x,
        velocities_y=velocities_y,
        accelerations_x=accelerations_x,
        accelerations_y=accelerations_y,
        headings=headings,
        lane_ids=lane_ids
    )


# =============================================================================
# Results Formatting and Output
# =============================================================================

def format_results(results: Dict[str, Dict], methods: List[str]) -> str:
    """Format results as a string table."""
    lines = []
    lines.append("=" * 80)
    lines.append("UNIFIED RISK FIELD METRICS EVALUATION")
    lines.append("=" * 80)
    
    for method in methods:
        if method not in results:
            continue
        
        r = results[method]
        lines.append(f"\n{'─' * 80}")
        lines.append(f"METHOD: {method.upper()}")
        lines.append(f"{'─' * 80}")
        
        if 'field' in r:
            fm = r['field']
            lines.append("\n[FIELD-STRUCTURAL METRICS]")
            lines.append("  Smoothness:")
            lines.append(f"    SNHS: {fm.smoothness.SNHS:.4f}  (Scale-Normalized Hessian)")
            lines.append(f"    TVR:  {fm.smoothness.TVR:.4f}  (Total Variation Regularity)")
            lines.append(f"    SSI:  {fm.smoothness.SSI:.4f}  (Spectral Smoothness Index)")
            lines.append(f"    AR:   {fm.smoothness.AR:.4f}  (Anisotropy Ratio)")
            lines.append("  Boundary:")
            lines.append(f"    ISI:  {fm.boundary.ISI:.4f}  (Interface Sharpness)")
            lines.append(f"    PPI:  {fm.boundary.PPI:.4f}  (Phase Purity)")
            lines.append(f"    GCI:  {fm.boundary.GCI:.4f}  (Gradient Concentration)")
            lines.append(f"    DBS:  {fm.boundary.DBS:.4f}  (Decision Boundary Stability)")
            lines.append("  Occlusion:")
            lines.append(f"    OFSR: {fm.occlusion.OFSR:.4f}  (False Safety Rate)")
            lines.append(f"    CZU:  {fm.occlusion.CZU:.4f}  (Critical Zone Underestimation)")
            lines.append(f"    VWRD: {fm.occlusion.VWRD:.4f}  (Visibility-Weighted Discrepancy)")
        
        if 'behavioral' in r:
            bm = r['behavioral']
            lines.append("\n[BEHAVIORAL ALIGNMENT METRICS]")
            lines.append("  Spatial:")
            lines.append(f"    CTAI: {bm.spatial.CTAI:.4f}  (Critical Transition Alignment)")
            lines.append(f"    DZOC: {bm.spatial.DZOC:.4f}  (Decision Zone Overlap)")
            lines.append(f"    LPDA: {bm.spatial.LPDA:.4f}  (Longitudinal Position Accuracy)")
            lines.append(f"    GMDC: {bm.spatial.GMDC:.4f}  (Gradient-Decision Correspondence)")
            lines.append("  Temporal:")
            lines.append(f"    DTC:  {bm.temporal.DTC:.4f}  (Decision Timing Concordance)")
            lines.append(f"    RATC: {bm.temporal.RATC:.4f}  (Risk-Action Coherence)")
            lines.append(f"    TGSC: {bm.temporal.TGSC:.4f}  (Gradient-Signal Correlation)")
            lines.append("  Validity:")
            lines.append(f"    RZBV: {bm.validity.RZBV:.4f}  (Risk Zone Behavioral Validity)")
            lines.append(f"    ARI:  {bm.validity.ARI:.4f}  (Adjusted Rand Index)")
            lines.append("  Sensitivity:")
            lines.append(f"    MODR: {bm.sensitivity.MODR_f1:.4f}  (Maneuver Detection F1)")
            lines.append(f"    SCAC: {bm.sensitivity.SCAC:.4f}  (Safe-Critical Correspondence)")
            lines.append(f"  Overall Score: {bm.overall_alignment_score:.4f}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def save_results_json(results: Dict, output_path: str):
    """Save results to JSON file."""
    serializable = {}
    for method, r in results.items():
        serializable[method] = {}
        if 'field' in r:
            serializable[method]['field'] = r['field'].to_dict()
        if 'behavioral' in r:
            serializable[method]['behavioral'] = r['behavioral'].to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Risk Field Metrics Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no data required)
  python unified_metrics_integration.py --demo
  
  # All metrics on exiD data
  python unified_metrics_integration.py --data_dir ./exiD --recording 25 --ego_id 123
  
  # Only field-structural metrics
  python unified_metrics_integration.py --data_dir ./exiD --recording 25 --metrics field
  
  # Only behavioral alignment metrics
  python unified_metrics_integration.py --data_dir ./exiD --recording 25 --metrics behavioral
  
  # Specific field methods
  python unified_metrics_integration.py --demo --methods gvf edrf ada apf
  
  # Save results to JSON
  python unified_metrics_integration.py --demo --output results.json
        """
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--demo', action='store_true', 
                            help='Use demo data (no external data required)')
    data_group.add_argument('--data_dir', type=str, 
                            help='Path to exiD data directory')
    
    # Recording/vehicle selection
    parser.add_argument('--recording', type=int, default=25,
                        help='Recording ID (default: 25)')
    parser.add_argument('--ego_id', type=int, default=None,
                        help='Ego vehicle ID (auto-select if not specified)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame number (auto-select if not specified)')
    
    # Metric groups
    parser.add_argument('--metrics', type=str, nargs='+',
                        choices=['field', 'behavioral', 'all'],
                        default=['all'],
                        help='Metric groups to compute (default: all)')
    
    # Field methods
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['gvf', 'edrf', 'ada', 'apf'],
                        default=['gvf', 'edrf', 'ada'],
                        help='Risk field methods to evaluate (default: gvf edrf ada)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--output_dir', type=str, default='./output_unified',
                        help='Output directory for visualizations')
    
    # Visualization
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--light-theme', action='store_true',
                        help='Use light theme for plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.demo and args.data_dir is None:
        parser.print_help()
        print("\nError: Either --demo or --data_dir must be specified")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Unified Risk Field Metrics Evaluation")
    logger.info("=" * 60)
    logger.info(f"Methods: {', '.join(args.methods)}")
    logger.info(f"Metric groups: {', '.join(args.metrics)}")
    
    # Initialize evaluator
    evaluator = UnifiedRiskFieldEvaluator()
    
    if args.demo:
        # Demo mode
        logger.info("\nRunning in DEMO mode")
        snapshot = create_demo_scenario()
        behavior = create_demo_behavior()
        
        logger.info(f"Demo scenario: {len(snapshot['surrounding'])} vehicles")
        logger.info(f"Demo behavior: {len(behavior.timestamps)} samples")
    else:
        # Load exiD data
        loader = ExiDDataLoader(args.data_dir)
        if not loader.load_recording(args.recording):
            sys.exit(1)
        
        # Find ego vehicle
        ego_id = args.ego_id
        if ego_id is None:
            heavy_ids = [row['trackId'] for _, row in loader.tracks_meta_df.iterrows()
                        if str(row.get('class', '')).lower() in evaluator.config.HEAVY_VEHICLE_CLASSES]
            if heavy_ids:
                ego_id = heavy_ids[0]
                logger.info(f"Auto-selected ego: {ego_id}")
            else:
                logger.error("No heavy vehicles found, please specify --ego_id")
                sys.exit(1)
        
        # Find frame
        frame = args.frame
        if frame is None:
            veh_data = loader.tracks_df[loader.tracks_df['trackId'] == ego_id]
            frame = int(np.median(veh_data['frame'].values))
            logger.info(f"Auto-selected frame: {frame}")
        
        snapshot = loader.get_snapshot(ego_id, frame)
        if snapshot is None:
            logger.error("Could not get snapshot")
            sys.exit(1)
        
        # Get behavior sequence for behavioral metrics
        behavior = None
        if 'behavioral' in args.metrics or 'all' in args.metrics:
            behavior = loader.get_behavior_sequence(ego_id)
            if behavior is not None:
                logger.info(f"Loaded behavior: {len(behavior.timestamps)} samples")
            else:
                logger.warning("Could not load behavior sequence, behavioral metrics will be skipped")
    
    # Evaluate
    logger.info("\nEvaluating...")
    results = evaluator.evaluate_scenario(
        snapshot=snapshot,
        methods=args.methods,
        metric_groups=args.metrics,
        behavior=behavior
    )
    
    # Format and print results
    output_text = format_results(results, args.methods)
    print(output_text)
    
    # Save results
    if args.output:
        save_results_json(results, args.output)
    
    # Save to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"metrics_{timestamp}.json"
    save_results_json(results, str(json_path))
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Done!")
    
    return results


if __name__ == '__main__':
    main()
