"""
Risk Field Metrics Implementation
=================================
Comprehensive evaluation metrics for field-based risk modeling:
1. Gradient Smoothness Metrics (SNHS, ASD, TVR, SSI)
2. Safe/Unsafe Boundary Metrics (ISI, PPI, BCI, GCI, DBS)
3. Occlusion/Observation Uncertainty Metrics (SRUM, ITOI, OFSR, VWRD, SZRI)

References:
- Wang et al. "Field-Based Risk Modeling for Interactive Driving" (2025)
- Tian et al. "Mean Field Game-Based Interactive Trajectory Planning" (2026)

Author: Research Implementation
"""

import numpy as np
from numpy.fft import fft2, fftshift
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import entropy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes for Metrics Results
# =============================================================================

@dataclass
class SmoothnessMetrics:
    """Container for gradient smoothness metrics."""
    SNHS: float = 0.0          # Scale-Normalized Hessian Smoothness
    ASD_s: float = 0.0         # Anisotropic Smoothness - longitudinal
    ASD_d: float = 0.0         # Anisotropic Smoothness - lateral
    AR: float = 0.0            # Anisotropy Ratio
    TVR: float = 0.0           # Total Variation Regularity
    SSI: float = 0.0           # Spectral Smoothness Index
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'SNHS': self.SNHS,
            'ASD_s': self.ASD_s,
            'ASD_d': self.ASD_d,
            'AR': self.AR,
            'TVR': self.TVR,
            'SSI': self.SSI
        }


@dataclass
class BoundaryMetrics:
    """Container for safe/unsafe boundary metrics."""
    ISI: float = 0.0           # Interface Sharpness Index
    PPI: float = 0.0           # Phase Purity Index
    BCI: float = 0.0           # Boundary Complexity Index
    GCI: float = 0.0           # Gradient Concentration Index
    DBS: float = 0.0           # Decision Boundary Stability
    threshold: float = 0.5     # Risk threshold used
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'ISI': self.ISI,
            'PPI': self.PPI,
            'BCI': self.BCI,
            'GCI': self.GCI,
            'DBS': self.DBS,
            'threshold': self.threshold
        }


@dataclass  
class OcclusionMetrics:
    """Container for occlusion/uncertainty metrics."""
    MU: float = 0.0            # Maximum Underestimation
    UV: float = 0.0            # Underestimation Volume
    CZU: float = 0.0           # Critical Zone Underestimation
    ITOI_JS: float = 0.0       # Information-Theoretic Occlusion Impact (JS divergence)
    OFSR: float = 0.0          # Occlusion-induced False Safety Rate
    VWRD: float = 0.0          # Visibility-Weighted Risk Discrepancy
    SZRI_norm: float = 0.0     # Shadow Zone Risk Intensity (normalized)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MU': self.MU,
            'UV': self.UV,
            'CZU': self.CZU,
            'ITOI_JS': self.ITOI_JS,
            'OFSR': self.OFSR,
            'VWRD': self.VWRD,
            'SZRI_norm': self.SZRI_norm
        }


@dataclass
class AllMetrics:
    """Combined metrics container."""
    smoothness: SmoothnessMetrics = field(default_factory=SmoothnessMetrics)
    boundary: BoundaryMetrics = field(default_factory=BoundaryMetrics)
    occlusion: OcclusionMetrics = field(default_factory=OcclusionMetrics)
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'smoothness': self.smoothness.to_dict(),
            'boundary': self.boundary.to_dict(),
            'occlusion': self.occlusion.to_dict()
        }


# =============================================================================
# 1. Gradient Smoothness Metrics
# =============================================================================

class SmoothnessAnalyzer:
    """Compute gradient smoothness metrics for risk fields."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def compute_all(self, R: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                    k_cutoff_ratio: float = 0.3) -> SmoothnessMetrics:
        """
        Compute all smoothness metrics.
        
        Args:
            R: Risk field array (ny, nx)
            dx: Grid spacing in x (longitudinal)
            dy: Grid spacing in y (lateral)
            k_cutoff_ratio: Cutoff frequency ratio for SSI
        
        Returns:
            SmoothnessMetrics object
        """
        metrics = SmoothnessMetrics()
        
        # Compute gradients
        grad_y, grad_x = np.gradient(R, dy, dx)
        
        # Compute Hessian components
        grad_xx = np.gradient(grad_x, dx, axis=1)
        grad_yy = np.gradient(grad_y, dy, axis=0)
        grad_xy = np.gradient(grad_x, dy, axis=0)
        
        # 1. Scale-Normalized Hessian Smoothness (SNHS)
        hessian_frobenius_sq = grad_xx**2 + grad_yy**2 + 2 * grad_xy**2
        field_energy = np.sum(R**2)
        metrics.SNHS = np.sum(hessian_frobenius_sq) / (field_energy + self.epsilon)
        
        # 2. Anisotropic Smoothness Decomposition (ASD)
        domain_area = R.size
        metrics.ASD_s = np.sum(grad_xx**2) / domain_area  # Longitudinal
        metrics.ASD_d = np.sum(grad_yy**2) / domain_area  # Lateral
        metrics.AR = metrics.ASD_s / (metrics.ASD_d + self.epsilon)
        
        # 3. Total Variation Regularity (TVR)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        total_variation = np.sum(grad_magnitude) * dx * dy
        R_max = np.max(np.abs(R))
        domain_size = np.sqrt(domain_area * dx * dy)
        metrics.TVR = total_variation / (R_max * domain_size + self.epsilon)
        
        # 4. Spectral Smoothness Index (SSI)
        metrics.SSI = self._compute_ssi(R, k_cutoff_ratio)
        
        return metrics
    
    def _compute_ssi(self, R: np.ndarray, k_cutoff_ratio: float) -> float:
        """Compute Spectral Smoothness Index using FFT."""
        # 2D FFT
        R_fft = fft2(R)
        R_fft_shifted = fftshift(R_fft)
        power_spectrum = np.abs(R_fft_shifted)**2
        
        # Create frequency grid
        ny, nx = R.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        KX, KY = np.meshgrid(kx, ky)
        K_mag = np.sqrt(KX**2 + KY**2)
        
        # Cutoff frequency
        k_max = np.max(K_mag)
        k_cutoff = k_cutoff_ratio * k_max
        
        # High-frequency energy ratio
        high_freq_mask = K_mag > k_cutoff
        high_freq_energy = np.sum(power_spectrum[high_freq_mask])
        total_energy = np.sum(power_spectrum)
        
        return high_freq_energy / (total_energy + self.epsilon)


# =============================================================================
# 2. Safe/Unsafe Boundary Metrics
# =============================================================================

class BoundaryAnalyzer:
    """Compute boundary metrics for risk fields."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def compute_all(self, R: np.ndarray, threshold: float = None,
                    dx: float = 1.0, dy: float = 1.0,
                    delta_band: float = 0.1,
                    n_perturbations: int = 10,
                    perturbation_sigma: float = 0.05) -> BoundaryMetrics:
        """
        Compute all boundary metrics.
        
        Args:
            R: Risk field array (ny, nx)
            threshold: Risk threshold (default: median)
            dx, dy: Grid spacing
            delta_band: Width of transition band for GCI
            n_perturbations: Number of perturbations for DBS
            perturbation_sigma: Std of perturbation noise
        
        Returns:
            BoundaryMetrics object
        """
        metrics = BoundaryMetrics()
        
        # Default threshold
        if threshold is None:
            threshold = np.median(R)
        metrics.threshold = threshold
        
        # Normalize field to [-1, 1]
        R_min, R_max = np.min(R), np.max(R)
        delta_R = R_max - R_min
        if delta_R < self.epsilon:
            return metrics
        
        R_norm = 2 * (R - threshold) / delta_R  # Centered at threshold
        
        # Compute gradient magnitude
        grad_y, grad_x = np.gradient(R, dy, dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find boundary points (iso-contour)
        boundary_mask = np.abs(R - threshold) < delta_band * delta_R
        
        # 1. Interface Sharpness Index (ISI)
        if np.any(boundary_mask):
            boundary_gradients = grad_mag[boundary_mask]
            metrics.ISI = np.mean(boundary_gradients) / (delta_R + self.epsilon)
        
        # 2. Phase Purity Index (PPI)
        metrics.PPI = np.mean(R_norm**2)
        
        # 3. Boundary Complexity Index (BCI)
        metrics.BCI = self._compute_bci(R, threshold, dx, dy)
        
        # 4. Gradient Concentration Index (GCI)
        metrics.GCI = self._compute_gci(R, threshold, grad_mag, delta_band)
        
        # 5. Decision Boundary Stability (DBS)
        metrics.DBS = self._compute_dbs(R, threshold, n_perturbations, 
                                        perturbation_sigma, dx, dy)
        
        return metrics
    
    def _compute_bci(self, R: np.ndarray, threshold: float, 
                     dx: float, dy: float) -> float:
        """Compute Boundary Complexity Index (isoperimetric ratio)."""
        # Binary mask of unsafe regions
        unsafe_mask = R > threshold
        
        if not np.any(unsafe_mask) or np.all(unsafe_mask):
            return 1.0  # No meaningful boundary
        
        # Compute boundary length using contour
        from skimage import measure
        try:
            contours = measure.find_contours(R, threshold)
            total_length = 0
            for contour in contours:
                # Approximate length
                diffs = np.diff(contour, axis=0)
                lengths = np.sqrt((diffs[:, 0] * dy)**2 + (diffs[:, 1] * dx)**2)
                total_length += np.sum(lengths)
            
            # Area of unsafe region
            area_unsafe = np.sum(unsafe_mask) * dx * dy
            
            if area_unsafe > self.epsilon:
                bci = total_length**2 / (4 * np.pi * area_unsafe)
            else:
                bci = 1.0
        except:
            bci = 1.0
        
        return bci
    
    def _compute_gci(self, R: np.ndarray, threshold: float,
                     grad_mag: np.ndarray, delta_band: float) -> float:
        """Compute Gradient Concentration Index."""
        R_min, R_max = np.min(R), np.max(R)
        delta_R = R_max - R_min
        
        # Boundary region
        boundary_mask = np.abs(R - threshold) < delta_band * delta_R
        
        # Gradient energy in boundary vs total
        boundary_grad_energy = np.sum(grad_mag[boundary_mask]**2)
        total_grad_energy = np.sum(grad_mag**2)
        
        return boundary_grad_energy / (total_grad_energy + self.epsilon)
    
    def _compute_dbs(self, R: np.ndarray, threshold: float,
                     n_perturbations: int, sigma: float,
                     dx: float, dy: float) -> float:
        """Compute Decision Boundary Stability."""
        from skimage import measure
        
        # Get original boundary
        try:
            original_contours = measure.find_contours(R, threshold)
            if not original_contours:
                return 1.0
            original_boundary = np.vstack(original_contours)
        except:
            return 1.0
        
        # Reference length
        L_ref = np.sqrt(R.shape[0] * R.shape[1]) * max(dx, dy)
        
        # Compute Hausdorff distances for perturbations
        hausdorff_distances = []
        
        for _ in range(n_perturbations):
            # Add Gaussian noise
            noise = np.random.normal(0, sigma * (np.max(R) - np.min(R)), R.shape)
            R_perturbed = R + noise
            
            try:
                perturbed_contours = measure.find_contours(R_perturbed, threshold)
                if perturbed_contours:
                    perturbed_boundary = np.vstack(perturbed_contours)
                    
                    # Hausdorff distance
                    h1 = directed_hausdorff(original_boundary, perturbed_boundary)[0]
                    h2 = directed_hausdorff(perturbed_boundary, original_boundary)[0]
                    h_dist = max(h1, h2) * max(dx, dy)  # Scale by grid spacing
                    hausdorff_distances.append(h_dist)
            except:
                continue
        
        if hausdorff_distances:
            mean_hausdorff = np.mean(hausdorff_distances)
            dbs = 1 - mean_hausdorff / L_ref
            return max(0, min(1, dbs))
        
        return 1.0


# =============================================================================
# 3. Occlusion/Observation Uncertainty Metrics
# =============================================================================

class OcclusionAnalyzer:
    """Compute occlusion-related metrics for risk fields."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def compute_all(self, R_full: np.ndarray, R_occluded: np.ndarray,
                    visibility_map: np.ndarray = None,
                    shadow_mask: np.ndarray = None,
                    critical_zone_mask: np.ndarray = None,
                    threshold: float = None,
                    dx: float = 1.0, dy: float = 1.0) -> OcclusionMetrics:
        """
        Compute all occlusion metrics.
        
        Args:
            R_full: Full-observability risk field
            R_occluded: Occluded (partial) risk field
            visibility_map: Visibility values [0,1] per cell
            shadow_mask: Binary mask of shadow region
            critical_zone_mask: Binary mask of ego's trajectory corridor
            threshold: Risk threshold for safety classification
            dx, dy: Grid spacing
        
        Returns:
            OcclusionMetrics object
        """
        metrics = OcclusionMetrics()
        
        if threshold is None:
            threshold = np.median(R_full)
        
        # 1. Spatial Risk Underestimation Map (SRUM) and derived metrics
        delta_R = np.maximum(0, R_full - R_occluded)
        
        # Maximum Underestimation (MU)
        metrics.MU = np.max(delta_R)
        
        # Underestimation Volume (UV)
        metrics.UV = np.sum(delta_R) * dx * dy
        
        # 2. Critical Zone Underestimation (CZU)
        if critical_zone_mask is not None and np.any(critical_zone_mask):
            czu_num = np.sum(delta_R[critical_zone_mask])
            czu_den = np.sum(R_full[critical_zone_mask])
            metrics.CZU = czu_num / (czu_den + self.epsilon)
        
        # 3. Information-Theoretic Occlusion Impact (ITOI) - Jensen-Shannon
        metrics.ITOI_JS = self._compute_js_divergence(R_full, R_occluded)
        
        # 4. Occlusion-induced False Safety Rate (OFSR)
        truly_unsafe = R_full > threshold
        appears_safe = R_occluded <= threshold
        false_safe = truly_unsafe & appears_safe
        
        if np.any(truly_unsafe):
            metrics.OFSR = np.sum(false_safe) / np.sum(truly_unsafe)
        
        # 5. Visibility-Weighted Risk Discrepancy (VWRD)
        if visibility_map is not None:
            hidden_risk = np.sum((1 - visibility_map) * R_full)
            total_risk = np.sum(R_full)
            metrics.VWRD = hidden_risk / (total_risk + self.epsilon)
        
        # 6. Shadow Zone Risk Intensity (SZRI)
        if shadow_mask is not None and np.any(shadow_mask):
            shadow_mean_risk = np.mean(R_full[shadow_mask])
            global_mean_risk = np.mean(R_full)
            metrics.SZRI_norm = shadow_mean_risk / (global_mean_risk + self.epsilon)
        
        return metrics
    
    def _compute_js_divergence(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two risk distributions."""
        # Normalize to probability distributions
        R1_flat = R1.flatten()
        R2_flat = R2.flatten()
        
        # Ensure non-negative and normalize
        R1_pos = np.maximum(R1_flat, 0)
        R2_pos = np.maximum(R2_flat, 0)
        
        p1 = R1_pos / (np.sum(R1_pos) + self.epsilon)
        p2 = R2_pos / (np.sum(R2_pos) + self.epsilon)
        
        # Mixture distribution
        m = 0.5 * (p1 + p2)
        
        # JS divergence
        kl1 = entropy(p1, m)
        kl2 = entropy(p2, m)
        
        js = 0.5 * (kl1 + kl2)
        
        # Normalize to [0, 1]
        return min(1.0, js / np.log(2))


# =============================================================================
# Risk Field Constructors (Multiple Methods)
# =============================================================================

class RiskFieldConstructor:
    """Construct risk fields using different methodologies."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.epsilon = 1e-10
    
    def construct_gvf_risk(self, ego: Dict, others: List[Dict],
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float],
                          grid_size: Tuple[int, int] = (60, 30),
                          length_scale: Tuple[float, float] = (15, 2.5)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct GVF-based risk field.
        
        Returns:
            (X_mesh, Y_mesh, risk_field)
        """
        from sklearn.metrics.pairwise import rbf_kernel
        
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        if not others:
            return X_mesh, Y_mesh, np.zeros_like(X_mesh)
        
        # Collect positions and velocities relative to ego
        positions = []
        rel_velocities = []
        
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            dvx = other.get('vx', 0) - ego.get('vx', 0)
            dvy = other.get('vy', 0) - ego.get('vy', 0)
            dvx_rel = dvx * cos_h - dvy * sin_h
            dvy_rel = dvx * sin_h + dvy * cos_h
            
            positions.append([dx_rel, dy_rel])
            rel_velocities.append([dvx_rel, dvy_rel])
        
        positions = np.array(positions)
        rel_velocities = np.array(rel_velocities)
        
        # Construct velocity field using GP
        P_test = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        
        sigma_x, sigma_y = length_scale
        K = self._anisotropic_rbf(positions, positions, sigma_x, sigma_y)
        K_s = self._anisotropic_rbf(P_test, positions, sigma_x, sigma_y)
        
        K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        
        VX = (K_s @ K_inv @ rel_velocities[:, 0]).reshape(X_mesh.shape)
        VY = (K_s @ K_inv @ rel_velocities[:, 1]).reshape(X_mesh.shape)
        
        # Risk as velocity magnitude (approaching = higher risk)
        risk_field = np.sqrt(VX**2 + VY**2)
        
        # Enhance risk in front of ego (approaching vehicles)
        approach_mask = VX < 0  # Negative = approaching
        risk_field[approach_mask] *= 1.5
        
        return X_mesh, Y_mesh, risk_field
    
    def construct_edrf_risk(self, ego: Dict, others: List[Dict],
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           grid_size: Tuple[int, int] = (60, 30),
                           sigma_base: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct EDRF-style risk field (Gaussian along predicted trajectories).
        
        Returns:
            (X_mesh, Y_mesh, risk_field)
        """
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        risk_field = np.zeros_like(X_mesh)
        
        if not others:
            return X_mesh, Y_mesh, risk_field
        
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for other in others:
            # Position relative to ego
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            # Velocity for trajectory prediction
            speed = other.get('speed', np.sqrt(other.get('vx', 0)**2 + other.get('vy', 0)**2))
            
            # Adaptive sigma based on speed
            sigma_x = sigma_base + 0.5 * speed
            sigma_y = sigma_base * 0.4
            
            # Gaussian risk
            risk_contribution = np.exp(
                -((X_mesh - dx_rel)**2 / (2 * sigma_x**2) +
                  (Y_mesh - dy_rel)**2 / (2 * sigma_y**2))
            )
            
            # Scale by inverse distance and speed
            dist = np.sqrt(dx_rel**2 + dy_rel**2)
            scale = (speed + 5) / (dist + 5)
            
            risk_field += scale * risk_contribution
        
        return X_mesh, Y_mesh, risk_field
    
    def construct_ada_risk(self, ego: Dict, others: List[Dict],
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float],
                          grid_size: Tuple[int, int] = (60, 30),
                          mu1: float = 0.25, mu2: float = 0.25,
                          delta: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct ADA (Asymmetric Driving Aggressiveness) risk field.
        
        Returns:
            (X_mesh, Y_mesh, risk_field)
        """
        nx, ny = grid_size
        X = np.linspace(x_range[0], x_range[1], nx)
        Y = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        risk_field = np.zeros_like(X_mesh)
        
        if not others:
            return X_mesh, Y_mesh, risk_field
        
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        ego_mass = ego.get('mass', 3000)
        ego_v = ego.get('speed', np.sqrt(ego.get('vx', 0)**2 + ego.get('vy', 0)**2))
        
        for other in others:
            # Position relative to ego
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1.0:
                dist = 1.0
            
            # Direction unit vector
            ux, uy = dx / dist, dy / dist
            
            other_mass = other.get('mass', 3000)
            other_v = other.get('speed', np.sqrt(other.get('vx', 0)**2 + other.get('vy', 0)**2))
            
            # Cosine factors
            if other_v > 0.1:
                cos_i = (other.get('vx', 0) * ux + other.get('vy', 0) * uy) / other_v
            else:
                cos_i = 0
            
            if ego_v > 0.1:
                cos_j = -(ego.get('vx', 0) * ux + ego.get('vy', 0) * uy) / ego_v
            else:
                cos_j = 0
            
            cos_i = np.clip(cos_i, -1, 1)
            cos_j = np.clip(cos_j, -1, 1)
            
            # ADA aggressiveness
            xi1 = mu1 * other_v * cos_i + mu2 * ego_v * cos_j
            omega = (other_mass * other_v) / (2 * delta * ego_mass) * np.exp(xi1)
            omega = np.clip(omega, 0, 2000)
            
            # Spatial decay
            sigma_x = 12.0
            sigma_y = 4.0
            spatial_decay = np.exp(
                -((X_mesh - dx_rel)**2 / (2 * sigma_x**2) +
                  (Y_mesh - dy_rel)**2 / (2 * sigma_y**2))
            )
            
            risk_field += omega * spatial_decay
        
        return X_mesh, Y_mesh, risk_field
    
    def _anisotropic_rbf(self, XA: np.ndarray, XB: np.ndarray,
                         sigma_x: float, sigma_y: float) -> np.ndarray:
        """Compute anisotropic RBF kernel."""
        diff_x = XA[:, 0:1] - XB[:, 0:1].T
        diff_y = XA[:, 1:2] - XB[:, 1:2].T
        
        return np.exp(-0.5 * (diff_x**2 / sigma_x**2 + diff_y**2 / sigma_y**2))


# =============================================================================
# Occlusion Geometry Computation
# =============================================================================

class OcclusionGeometry:
    """Compute occlusion-related geometric quantities."""
    
    def __init__(self):
        pass
    
    def compute_shadow_mask(self, ego: Dict, occluders: List[Dict],
                           X_mesh: np.ndarray, Y_mesh: np.ndarray) -> np.ndarray:
        """
        Compute shadow mask from occluders.
        
        Args:
            ego: Ego vehicle dict with x, y, heading
            occluders: List of occluder vehicles
            X_mesh, Y_mesh: Grid coordinates (in ego frame)
        
        Returns:
            Binary shadow mask
        """
        shadow_mask = np.zeros(X_mesh.shape, dtype=bool)
        
        if not occluders:
            return shadow_mask
        
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for occ in occluders:
            # Occluder position relative to ego
            dx = occ['x'] - ego['x']
            dy = occ['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            d_occ = np.sqrt(dx_rel**2 + dy_rel**2)
            if d_occ < 1.0:
                continue
            
            # Bearing angle to occluder
            phi_occ = np.arctan2(dy_rel, dx_rel)
            
            # Angular width of occluder
            occ_length = occ.get('length', 5.0)
            occ_width = occ.get('width', 2.0)
            occ_heading_rel = occ.get('heading', 0) - ego.get('heading', 0)
            
            w_eff = (np.abs(occ_length * np.sin(phi_occ - occ_heading_rel)) +
                    np.abs(occ_width * np.cos(phi_occ - occ_heading_rel)))
            
            alpha_occ = 2 * np.arctan(w_eff / (2 * d_occ))
            
            # Shadow cone
            phi_grid = np.arctan2(Y_mesh, X_mesh)
            d_grid = np.sqrt(X_mesh**2 + Y_mesh**2)
            
            # Points in shadow: behind occluder and within angular cone
            behind_occluder = d_grid > d_occ
            angle_diff = np.abs(np.arctan2(np.sin(phi_grid - phi_occ),
                                          np.cos(phi_grid - phi_occ)))
            in_cone = angle_diff < alpha_occ / 2
            
            shadow_mask |= (behind_occluder & in_cone)
        
        return shadow_mask
    
    def compute_visibility_map(self, ego: Dict, occluders: List[Dict],
                              X_mesh: np.ndarray, Y_mesh: np.ndarray) -> np.ndarray:
        """
        Compute continuous visibility map [0, 1].
        
        Returns:
            Visibility map (1 = fully visible, 0 = fully occluded)
        """
        visibility = np.ones(X_mesh.shape)
        
        if not occluders:
            return visibility
        
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for occ in occluders:
            dx = occ['x'] - ego['x']
            dy = occ['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            d_occ = np.sqrt(dx_rel**2 + dy_rel**2)
            if d_occ < 1.0:
                continue
            
            phi_occ = np.arctan2(dy_rel, dx_rel)
            
            occ_length = occ.get('length', 5.0)
            occ_width = occ.get('width', 2.0)
            occ_heading_rel = occ.get('heading', 0) - ego.get('heading', 0)
            
            w_eff = (np.abs(occ_length * np.sin(phi_occ - occ_heading_rel)) +
                    np.abs(occ_width * np.cos(phi_occ - occ_heading_rel)))
            
            alpha_occ = 2 * np.arctan(w_eff / (2 * d_occ))
            
            phi_grid = np.arctan2(Y_mesh, X_mesh)
            d_grid = np.sqrt(X_mesh**2 + Y_mesh**2)
            
            angle_diff = np.abs(np.arctan2(np.sin(phi_grid - phi_occ),
                                          np.cos(phi_grid - phi_occ)))
            
            # Smooth occlusion factor
            behind_factor = np.clip((d_grid - d_occ) / 10.0, 0, 1)
            angle_factor = np.clip(1 - angle_diff / (alpha_occ / 2 + 0.01), 0, 1)
            
            occlusion_factor = behind_factor * angle_factor
            visibility *= (1 - 0.9 * occlusion_factor)
        
        return visibility


# =============================================================================
# Comprehensive Visualizer
# =============================================================================

class RiskFieldMetricsVisualizer:
    """Visualize risk fields and their evaluation metrics."""
    
    def __init__(self, light_theme: bool = False):
        self.light_theme = light_theme
        if light_theme:
            self.bg_color = 'white'
            self.panel_color = '#F5F5F5'
            self.fg_color = 'black'
            self.spine_color = '#4A4A4A'
        else:
            self.bg_color = '#0D1117'
            self.panel_color = '#1A1A2E'
            self.fg_color = 'white'
            self.spine_color = '#4A4A6A'
    
    def create_comprehensive_figure(self, 
                                    ego: Dict,
                                    others: List[Dict],
                                    occluders: List[Dict] = None,
                                    output_path: str = None,
                                    methods: List[str] = None):
        """
        Create comprehensive visualization comparing risk field methods.
        
        Args:
            ego: Ego vehicle data
            others: List of other vehicles
            occluders: List of occluding vehicles (subset of others)
            output_path: Output file path
            methods: List of methods to compare ('gvf', 'edrf', 'ada')
        """
        if methods is None:
            methods = ['gvf', 'edrf', 'ada']
        
        if occluders is None:
            # Default: use heavy vehicles as occluders
            occluders = [v for v in others if v.get('class', '') in ['truck', 'bus']]
        
        # Setup grid ranges
        x_range = (-40, 80)
        y_range = (-20, 20)
        grid_size = (70, 40)
        
        # Construct fields
        constructor = RiskFieldConstructor()
        fields = {}
        
        for method in methods:
            if method == 'gvf':
                X, Y, R = constructor.construct_gvf_risk(ego, others, x_range, y_range, grid_size)
            elif method == 'edrf':
                X, Y, R = constructor.construct_edrf_risk(ego, others, x_range, y_range, grid_size)
            elif method == 'ada':
                X, Y, R = constructor.construct_ada_risk(ego, others, x_range, y_range, grid_size)
            else:
                continue
            fields[method] = {'X': X, 'Y': Y, 'R': R}
        
        if not fields:
            print("No valid methods specified")
            return
        
        # Compute occlusion geometry
        occ_geo = OcclusionGeometry()
        X_mesh = fields[methods[0]]['X']
        Y_mesh = fields[methods[0]]['Y']
        
        shadow_mask = occ_geo.compute_shadow_mask(ego, occluders, X_mesh, Y_mesh)
        visibility_map = occ_geo.compute_visibility_map(ego, occluders, X_mesh, Y_mesh)
        
        # Critical zone: corridor in front of ego
        critical_zone = (X_mesh > 0) & (X_mesh < 50) & (np.abs(Y_mesh) < 5)
        
        # Compute metrics for each method
        smoothness_analyzer = SmoothnessAnalyzer()
        boundary_analyzer = BoundaryAnalyzer()
        occlusion_analyzer = OcclusionAnalyzer()
        
        dx = (x_range[1] - x_range[0]) / grid_size[0]
        dy = (y_range[1] - y_range[0]) / grid_size[1]
        
        all_metrics = {}
        for method, field_data in fields.items():
            R = field_data['R']
            
            # Smoothness metrics
            smooth = smoothness_analyzer.compute_all(R, dx, dy)
            
            # Boundary metrics
            threshold = np.percentile(R, 70)  # Upper 30% as unsafe
            boundary = boundary_analyzer.compute_all(R, threshold, dx, dy)
            
            # Occluded field (risk reduced in shadow zones)
            R_occluded = R * visibility_map
            
            # Occlusion metrics
            occlusion = occlusion_analyzer.compute_all(
                R, R_occluded, visibility_map, shadow_mask, critical_zone, threshold, dx, dy
            )
            
            all_metrics[method] = AllMetrics(smooth, boundary, occlusion)
        
        # Create figure
        n_methods = len(methods)
        fig = plt.figure(figsize=(7 * n_methods, 16))
        fig.patch.set_facecolor(self.bg_color)
        
        gs = GridSpec(4, n_methods, figure=fig, height_ratios=[1, 1, 0.8, 0.8],
                     hspace=0.35, wspace=0.25)
        
        # Row 1: Risk fields
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            self._plot_risk_field(ax, fields[method], ego, others, 
                                 f"{method.upper()} Risk Field")
        
        # Row 2: Risk fields with occlusion overlay
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[1, i])
            R_occluded = fields[method]['R'] * visibility_map
            self._plot_risk_field_with_occlusion(
                ax, fields[method]['X'], fields[method]['Y'], 
                fields[method]['R'], R_occluded, shadow_mask,
                ego, others, occluders, f"{method.upper()} with Occlusion"
            )
        
        # Row 3: Smoothness & Boundary metrics comparison
        ax_smooth = fig.add_subplot(gs[2, :n_methods//2+1])
        ax_boundary = fig.add_subplot(gs[2, n_methods//2+1:]) if n_methods > 1 else None
        
        self._plot_smoothness_comparison(ax_smooth, all_metrics, methods)
        if ax_boundary:
            self._plot_boundary_comparison(ax_boundary, all_metrics, methods)
        
        # Row 4: Occlusion metrics and summary
        ax_occ = fig.add_subplot(gs[3, :n_methods//2+1])
        ax_summary = fig.add_subplot(gs[3, n_methods//2+1:]) if n_methods > 1 else None
        
        self._plot_occlusion_comparison(ax_occ, all_metrics, methods)
        if ax_summary:
            self._plot_metrics_summary(ax_summary, all_metrics, methods)
        
        # Title
        fig.suptitle(
            f"Risk Field Evaluation: Ego {ego.get('class', 'vehicle').title()} | "
            f"{len(others)} surrounding vehicles | {len(occluders)} occluders",
            fontsize=14, fontweight='bold', color=self.fg_color, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            print(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
        
        return all_metrics
    
    def _plot_risk_field(self, ax, field_data: Dict, ego: Dict, 
                        others: List[Dict], title: str):
        """Plot risk field with vehicles."""
        ax.set_facecolor(self.panel_color)
        
        X, Y, R = field_data['X'], field_data['Y'], field_data['R']
        
        # Normalize for visualization
        R_norm = R / (np.max(R) + 1e-10)
        
        cmap = LinearSegmentedColormap.from_list('risk', 
            ['#1A1A2E', '#2E4057', '#F39C12', '#E74C3C', '#8B0000'])
        pcm = ax.pcolormesh(X, Y, R_norm, cmap=cmap, shading='gouraud', alpha=0.85)
        
        # Contour lines
        levels = np.linspace(0.2, 0.9, 4)
        ax.contour(X, Y, R_norm, levels=levels, colors='white', alpha=0.4, linewidths=0.5)
        
        # Draw ego
        ego_rect = mpatches.FancyBboxPatch(
            (-ego.get('length', 5)/2, -ego.get('width', 2)/2),
            ego.get('length', 5), ego.get('width', 2),
            boxstyle="round,pad=0.02",
            facecolor='#E74C3C', edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(0, -ego.get('width', 2)/2 - 2, 'EGO', ha='center', 
               color=self.fg_color, fontsize=9, fontweight='bold')
        
        # Draw others
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            color = '#9B59B6' if other.get('class', '') in ['truck', 'bus'] else '#3498DB'
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other.get('length', 4.5)/2, dy_rel - other.get('width', 1.8)/2),
                other.get('length', 4.5), other.get('width', 1.8),
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=self.fg_color, linewidth=1, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(dx_rel, dy_rel + other.get('width', 1.8)/2 + 1.5, 
                   str(other.get('id', '')), ha='center', color='yellow', fontsize=7)
        
        # Lane markings
        for y_lane in [-7, -3.5, 0, 3.5, 7]:
            ax.axhline(y_lane, color=self.fg_color, linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color, fontsize=9)
        ax.set_ylabel('Lateral (m)', color=self.fg_color, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Risk (normalized)', color=self.fg_color, fontsize=8)
        cbar.ax.tick_params(colors=self.fg_color, labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_risk_field_with_occlusion(self, ax, X, Y, R_full, R_occluded, 
                                        shadow_mask, ego, others, occluders, title):
        """Plot risk field with occlusion visualization."""
        ax.set_facecolor(self.panel_color)
        
        # Use occluded field for display
        R_norm = R_occluded / (np.max(R_full) + 1e-10)
        
        cmap = LinearSegmentedColormap.from_list('risk', 
            ['#1A1A2E', '#2E4057', '#F39C12', '#E74C3C', '#8B0000'])
        pcm = ax.pcolormesh(X, Y, R_norm, cmap=cmap, shading='gouraud', alpha=0.85)
        
        # Overlay shadow regions
        shadow_overlay = np.ma.masked_where(~shadow_mask, shadow_mask.astype(float))
        ax.pcolormesh(X, Y, shadow_overlay, cmap='gray', alpha=0.5, shading='auto')
        
        # Draw ego
        ego_rect = mpatches.FancyBboxPatch(
            (-ego.get('length', 5)/2, -ego.get('width', 2)/2),
            ego.get('length', 5), ego.get('width', 2),
            boxstyle="round,pad=0.02",
            facecolor='#E74C3C', edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # Draw others with occlusion highlighting
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        occluder_ids = {o.get('id') for o in occluders}
        
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            is_occluder = other.get('id') in occluder_ids
            color = '#FFD700' if is_occluder else '#3498DB'  # Gold for occluders
            lw = 2.5 if is_occluder else 1
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other.get('length', 4.5)/2, dy_rel - other.get('width', 1.8)/2),
                other.get('length', 4.5), other.get('width', 1.8),
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=self.fg_color, linewidth=lw, alpha=0.9
            )
            ax.add_patch(rect)
            
            label = f"{other.get('id', '')}"
            if is_occluder:
                label += "\n(OCC)"
            ax.text(dx_rel, dy_rel + other.get('width', 1.8)/2 + 1.5,
                   label, ha='center', color='white' if is_occluder else 'yellow', 
                   fontsize=7, fontweight='bold' if is_occluder else 'normal')
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color, fontsize=9)
        ax.set_ylabel('Lateral (m)', color=self.fg_color, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Visible Risk', color=self.fg_color, fontsize=8)
        cbar.ax.tick_params(colors=self.fg_color, labelsize=7)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_smoothness_comparison(self, ax, all_metrics: Dict, methods: List[str]):
        """Plot smoothness metrics comparison."""
        ax.set_facecolor(self.panel_color)
        
        metric_names = ['SNHS', 'TVR', 'SSI', 'AR']
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            values = [
                np.log10(all_metrics[method].smoothness.SNHS + 1e-10),
                all_metrics[method].smoothness.TVR,
                all_metrics[method].smoothness.SSI,
                np.log10(all_metrics[method].smoothness.AR + 1e-10)
            ]
            ax.bar(x + i * width - width * len(methods) / 2 + width / 2, 
                  values, width, label=method.upper(), color=colors[i], edgecolor=self.fg_color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['log(SNHS)', 'TVR', 'SSI', 'log(AR)'], 
                          color=self.fg_color, fontsize=9)
        ax.set_ylabel('Metric Value', color=self.fg_color, fontsize=9)
        ax.set_title('Smoothness Metrics Comparison', fontsize=11, 
                    fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_boundary_comparison(self, ax, all_metrics: Dict, methods: List[str]):
        """Plot boundary metrics comparison."""
        ax.set_facecolor(self.panel_color)
        
        metric_names = ['ISI', 'PPI', 'GCI', 'DBS']
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            values = [
                min(all_metrics[method].boundary.ISI, 2),  # Cap for visualization
                all_metrics[method].boundary.PPI,
                all_metrics[method].boundary.GCI,
                all_metrics[method].boundary.DBS
            ]
            ax.bar(x + i * width - width * len(methods) / 2 + width / 2,
                  values, width, label=method.upper(), color=colors[i], edgecolor=self.fg_color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, color=self.fg_color, fontsize=9)
        ax.set_ylabel('Metric Value', color=self.fg_color, fontsize=9)
        ax.set_title('Boundary Metrics Comparison', fontsize=11,
                    fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(0, 1.2)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_occlusion_comparison(self, ax, all_metrics: Dict, methods: List[str]):
        """Plot occlusion metrics comparison."""
        ax.set_facecolor(self.panel_color)
        
        metric_names = ['CZU', 'OFSR', 'VWRD', 'ITOI_JS']
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            values = [
                all_metrics[method].occlusion.CZU,
                all_metrics[method].occlusion.OFSR,
                all_metrics[method].occlusion.VWRD,
                all_metrics[method].occlusion.ITOI_JS
            ]
            ax.bar(x + i * width - width * len(methods) / 2 + width / 2,
                  values, width, label=method.upper(), color=colors[i], edgecolor=self.fg_color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, color=self.fg_color, fontsize=9)
        ax.set_ylabel('Metric Value', color=self.fg_color, fontsize=9)
        ax.set_title('Occlusion Metrics Comparison', fontsize=11,
                    fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(0, 1.0)
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _plot_metrics_summary(self, ax, all_metrics: Dict, methods: List[str]):
        """Plot summary table of all metrics."""
        ax.set_facecolor(self.panel_color)
        ax.axis('off')
        
        # Build summary text
        lines = ["=" * 45, "METRICS SUMMARY", "=" * 45, ""]
        
        for method in methods:
            m = all_metrics[method]
            lines.append(f">>> {method.upper()} <<<")
            lines.append("")
            lines.append("Smoothness:")
            lines.append(f"  SNHS: {m.smoothness.SNHS:.4f}")
            lines.append(f"  TVR:  {m.smoothness.TVR:.4f}")
            lines.append(f"  SSI:  {m.smoothness.SSI:.4f}")
            lines.append(f"  AR:   {m.smoothness.AR:.4f}")
            lines.append("")
            lines.append("Boundary:")
            lines.append(f"  ISI: {m.boundary.ISI:.4f}")
            lines.append(f"  PPI: {m.boundary.PPI:.4f}")
            lines.append(f"  GCI: {m.boundary.GCI:.4f}")
            lines.append(f"  DBS: {m.boundary.DBS:.4f}")
            lines.append("")
            lines.append("Occlusion:")
            lines.append(f"  OFSR:    {m.occlusion.OFSR:.4f}")
            lines.append(f"  CZU:     {m.occlusion.CZU:.4f}")
            lines.append(f"  VWRD:    {m.occlusion.VWRD:.4f}")
            lines.append(f"  ITOI_JS: {m.occlusion.ITOI_JS:.4f}")
            lines.append("-" * 45)
            lines.append("")
        
        summary = "\n".join(lines)
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
               fontsize=8, color=self.fg_color, family='monospace',
               verticalalignment='top')
        
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)


# =============================================================================
# Integration with ExiD Data Loader
# =============================================================================

def evaluate_from_snapshot(snapshot: Dict, occluder_ids: List[int] = None,
                          methods: List[str] = None,
                          output_path: str = None,
                          light_theme: bool = False) -> Dict[str, AllMetrics]:
    """
    Evaluate risk field metrics from an exiD snapshot.
    
    Args:
        snapshot: Dictionary with 'ego' and 'surrounding' keys
        occluder_ids: List of vehicle IDs to treat as occluders
        methods: Risk field methods to compare
        output_path: Path to save visualization
        light_theme: Use light color theme
    
    Returns:
        Dictionary of AllMetrics for each method
    """
    ego = snapshot['ego']
    others = snapshot['surrounding']
    
    if occluder_ids is None:
        # Default: use heavy vehicles as occluders
        occluders = [v for v in others if v.get('class', '') in ['truck', 'bus', 'trailer']]
    else:
        occluders = [v for v in others if v.get('id') in occluder_ids]
    
    if methods is None:
        methods = ['gvf', 'edrf', 'ada']
    
    visualizer = RiskFieldMetricsVisualizer(light_theme=light_theme)
    return visualizer.create_comprehensive_figure(ego, others, occluders, output_path, methods)


# =============================================================================
# Standalone Demo
# =============================================================================

def create_demo_scenario() -> Dict:
    """Create a demo scenario for testing."""
    ego = {
        'id': 0,
        'x': 0.0,
        'y': 0.0,
        'vx': 15.0,
        'vy': 0.0,
        'speed': 15.0,
        'heading': 0.0,
        'length': 12.0,
        'width': 2.5,
        'mass': 15000,
        'class': 'truck'
    }
    
    others = [
        # Leading car (ahead in same lane)
        {'id': 1, 'x': 35.0, 'y': 0.0, 'vx': 12.0, 'vy': 0.0, 'speed': 12.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Car merging from right
        {'id': 2, 'x': 25.0, 'y': 8.0, 'vx': 18.0, 'vy': -1.5, 'speed': 18.1,
         'heading': -0.08, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Truck in adjacent lane (occluder)
        {'id': 3, 'x': 20.0, 'y': 3.5, 'vx': 14.0, 'vy': 0.0, 'speed': 14.0,
         'heading': 0.0, 'length': 10.0, 'width': 2.5, 'mass': 12000, 'class': 'truck'},
        # Car behind truck (potentially occluded)
        {'id': 4, 'x': 45.0, 'y': 5.0, 'vx': 16.0, 'vy': -0.5, 'speed': 16.0,
         'heading': -0.03, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Car behind ego
        {'id': 5, 'x': -25.0, 'y': 0.0, 'vx': 17.0, 'vy': 0.0, 'speed': 17.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Car in left lane
        {'id': 6, 'x': 10.0, 'y': -3.5, 'vx': 20.0, 'vy': 0.0, 'speed': 20.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
    ]
    
    return {'ego': ego, 'surrounding': others, 'frame': 100}


def main():
    """Run demo evaluation."""
    print("=" * 60)
    print("Risk Field Metrics Evaluation Demo")
    print("=" * 60)
    
    # Create demo scenario
    snapshot = create_demo_scenario()
    
    print(f"\nScenario:")
    print(f"  Ego: {snapshot['ego']['class']} (ID: {snapshot['ego']['id']})")
    print(f"  Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    # Evaluate with all methods
    occluder_ids = [3]  # Truck ID 3 as occluder
    methods = ['gvf', 'edrf', 'ada']
    
    metrics = evaluate_from_snapshot(
        snapshot, 
        occluder_ids=occluder_ids,
        methods=methods,
        output_path='./risk_field_metrics_demo.png',
        light_theme=False
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    if metrics:
        for method, m in metrics.items():
            print(f"\n{method.upper()} Summary:")
            print(f"  Smoothness - SNHS: {m.smoothness.SNHS:.4f}, TVR: {m.smoothness.TVR:.4f}")
            print(f"  Boundary   - ISI: {m.boundary.ISI:.4f}, PPI: {m.boundary.PPI:.4f}")
            print(f"  Occlusion  - OFSR: {m.occlusion.OFSR:.4f}, CZU: {m.occlusion.CZU:.4f}")


if __name__ == '__main__':
    main()
