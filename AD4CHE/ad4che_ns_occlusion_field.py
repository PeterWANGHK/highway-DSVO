"""
AD4CHE Dataset: Navier-Stokes Inspired Interaction Field with Stochastic Occlusion
===================================================================================
Physics-informed traffic interaction modeling using:
1. Navier-Stokes analogues: pressure, viscosity, vorticity fields
2. Stochastic occlusion modeling: Yukawa potentials, GP uncertainty fields
3. Fokker-Planck inspired density evolution for risk assessment

Reference Physics:
- Continuity: ∂ρ/∂t + ∇·(ρv) = 0
- Momentum: ∂v/∂t + (v·∇)v = -(1/ρ)∇P + ν∇²v + (Vₑ-v)/τ + F_ext
- Fokker-Planck: ∂p/∂t = -∇·(μp) + (1/2)∇²(σ²p)

For PINN-based interaction field learning with occlusion uncertainty.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
from enum import Enum
import warnings
import argparse
import logging
import json
import re

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from numpy.linalg import inv

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OcclusionType(Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class VehicleState:
    """Complete vehicle state for field computation."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    heading: float
    speed: float
    length: float
    width: float
    vehicle_class: str
    mass: float
    is_occluded: bool = False
    occlusion_ratio: float = 0.0
    visibility: float = 1.0  # 1.0 = fully visible, 0.0 = fully occluded


@dataclass 
class OcclusionEvent:
    """Occlusion relationship between vehicles."""
    occluder_id: int
    occluded_id: int
    blocked_id: int
    occlusion_ratio: float
    shadow_polygon: Optional[np.ndarray] = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NSConfig:
    """Configuration for Navier-Stokes inspired field computation."""
    
    # Vehicle classes
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    # Observation range (meters)
    OBS_RANGE_AHEAD: float = 60.0
    OBS_RANGE_BEHIND: float = 30.0
    OBS_RANGE_LEFT: float = 15.0
    OBS_RANGE_RIGHT: float = 15.0
    
    # Grid resolution
    GRID_NX: int = 80
    GRID_NY: int = 40
    
    # === Navier-Stokes Parameters ===
    # Traffic pressure (repulsion strength)
    PRESSURE_AMPLITUDE: float = 100.0
    PRESSURE_DECAY_LONG: float = 15.0  # Longitudinal decay length
    PRESSURE_DECAY_LAT: float = 4.0    # Lateral decay length
    
    # Traffic viscosity (driver heterogeneity)
    VISCOSITY_BASE: float = 0.5
    VISCOSITY_SPEED_FACTOR: float = 0.02
    
    # Relaxation time to equilibrium velocity
    TAU_RELAX: float = 1.5  # seconds
    
    # Reference/desired velocity
    V_EQUILIBRIUM: float = 30.0  # m/s (~108 km/h)
    
    # Vorticity parameters (for lane change dynamics)
    VORTICITY_AMPLITUDE: float = 0.5
    VORTICITY_DECAY: float = 8.0
    
    # === Stochastic/Occlusion Parameters ===
    # Yukawa screening for occluded interactions
    YUKAWA_SCREENING_LENGTH: float = 20.0  # meters
    YUKAWA_AMPLITUDE: float = 50.0
    
    # Uncertainty field parameters
    UNCERTAINTY_BASE: float = 0.1
    UNCERTAINTY_OCCLUSION_MULT: float = 3.0
    UNCERTAINTY_DISTANCE_DECAY: float = 0.02
    
    # Gaussian Process kernel parameters
    GP_LENGTH_SCALE_X: float = 12.0
    GP_LENGTH_SCALE_Y: float = 3.0
    GP_NOISE: float = 1e-4
    
    # Fokker-Planck diffusion coefficient
    FP_DIFFUSION: float = 2.0
    
    # === Physical Constants ===
    MASS_HV: float = 15000.0  # kg
    MASS_PC: float = 1500.0   # kg
    RHO_JAM: float = 0.15     # vehicles/m (jam density)
    
    # === Visualization ===
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C',
        'car': '#3498DB',
        'bus': '#F39C12',
    })
    
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#161B22'
    SPINE_COLOR: str = '#30363D'


# =============================================================================
# Navier-Stokes Inspired Field Computation
# =============================================================================

class NavierStokesTrafficField:
    """
    Computes traffic interaction fields using Navier-Stokes analogues.
    
    Key fields computed:
    1. Pressure field P(x,y) - repulsive potential from vehicles
    2. Velocity field (u,v)(x,y) - local flow velocity
    3. Viscosity field ν(x,y) - local driver heterogeneity
    4. Vorticity field ω(x,y) - rotational tendency (lane changes)
    5. Momentum flux tensor τ_ij - stress distribution
    """
    
    def __init__(self, config: NSConfig = None):
        self.config = config or NSConfig()
        
    def compute_pressure_field(self, ego: VehicleState, 
                               others: List[VehicleState],
                               X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute traffic pressure field (analogous to fluid pressure).
        
        P(x,y) = Σᵢ Aᵢ · exp(-[(x-xᵢ)²/2σₓ² + (y-yᵢ)²/2σᵧ²]) · f(vᵢ) · g(mᵢ)
        
        where f(v) accounts for velocity-dependent pressure and g(m) for mass.
        """
        P = np.zeros_like(X)
        
        if not others:
            return P  # Return zero field if no other vehicles
        
        # Ego contribution (self-pressure for visualization)
        sigma_x = self.config.PRESSURE_DECAY_LONG * (1 + 0.5 * ego.speed / max(self.config.V_EQUILIBRIUM, 1))
        sigma_y = self.config.PRESSURE_DECAY_LAT
        
        # Rotate grid to ego frame
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        for other in others:
            # Relative position
            dx = other.x - ego.x
            dy = other.y - ego.y
            
            # Transform to ego frame
            dx_ego = dx * cos_h - dy * sin_h
            dy_ego = dx * sin_h + dy * cos_h
            
            # Velocity-dependent pressure scaling
            rel_speed = np.sqrt((other.vx - ego.vx)**2 + (other.vy - ego.vy)**2)
            speed_factor = 1 + 0.5 * rel_speed / max(self.config.V_EQUILIBRIUM, 1)
            
            # Mass-dependent pressure
            mass_factor = other.mass / self.config.MASS_PC
            
            # Visibility/occlusion scaling (reduced pressure if occluded)
            visibility_factor = other.visibility
            
            # Anisotropic Gaussian pressure contribution
            # Elongated in longitudinal direction based on speed
            sigma_x_i = sigma_x * (1 + 0.3 * other.speed / max(self.config.V_EQUILIBRIUM, 1))
            sigma_y_i = sigma_y
            
            # Grid points in ego frame
            X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
            Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
            
            # Gaussian contribution
            gauss = np.exp(-0.5 * ((X_ego - dx_ego)**2 / sigma_x_i**2 + 
                                   (Y_ego - dy_ego)**2 / sigma_y_i**2))
            
            P += self.config.PRESSURE_AMPLITUDE * speed_factor * mass_factor * visibility_factor * gauss
        
        return P
    
    def compute_velocity_field(self, ego: VehicleState,
                               others: List[VehicleState],
                               X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity field using Gaussian Process interpolation.
        
        Uses anisotropic RBF kernel to interpolate velocity from vehicle positions.
        """
        # Collect vehicle positions and velocities in ego frame
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        positions = []
        velocities_x = []
        velocities_y = []
        weights = []
        
        # Add ego at origin
        positions.append([0, 0])
        velocities_x.append(0)  # Ego is reference frame
        velocities_y.append(0)
        weights.append(1.0)
        
        for other in others:
            dx = other.x - ego.x
            dy = other.y - ego.y
            dx_ego = dx * cos_h - dy * sin_h
            dy_ego = dx * sin_h + dy * cos_h
            
            # Relative velocity in ego frame
            dvx = other.vx - ego.vx
            dvy = other.vy - ego.vy
            dvx_ego = dvx * cos_h - dvy * sin_h
            dvy_ego = dvx * sin_h + dvy * cos_h
            
            positions.append([dx_ego, dy_ego])
            velocities_x.append(dvx_ego)
            velocities_y.append(dvy_ego)
            weights.append(other.visibility)  # Weight by visibility
        
        positions = np.array(positions)
        velocities_x = np.array(velocities_x)
        velocities_y = np.array(velocities_y)
        
        # Grid in ego frame
        X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
        Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
        grid_points = np.column_stack([X_ego.ravel(), Y_ego.ravel()])
        
        # RBF interpolation with anisotropic kernel (approximated by scaling)
        scale_x = self.config.GP_LENGTH_SCALE_X
        scale_y = self.config.GP_LENGTH_SCALE_Y
        
        # Scale positions for isotropic RBF
        positions_scaled = positions.copy()
        positions_scaled[:, 0] /= scale_x
        positions_scaled[:, 1] /= scale_y
        
        grid_scaled = grid_points.copy()
        grid_scaled[:, 0] /= scale_x
        grid_scaled[:, 1] /= scale_y
        
        # Need at least 2 points for interpolation
        if len(positions) < 2:
            return np.zeros_like(X), np.zeros_like(Y)
        
        try:
            rbf_vx = RBFInterpolator(positions_scaled, velocities_x, kernel='gaussian', 
                                     epsilon=1.0, smoothing=self.config.GP_NOISE)
            rbf_vy = RBFInterpolator(positions_scaled, velocities_y, kernel='gaussian',
                                     epsilon=1.0, smoothing=self.config.GP_NOISE)
            
            U = rbf_vx(grid_scaled).reshape(X.shape)
            V = rbf_vy(grid_scaled).reshape(X.shape)
            
            # Clip extreme values
            U = np.clip(U, -50, 50)
            V = np.clip(V, -50, 50)
        except Exception as e:
            logger.warning(f"RBF interpolation failed: {e}, using simple averaging")
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
        
        return U, V
    
    def compute_viscosity_field(self, ego: VehicleState,
                                others: List[VehicleState],
                                X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute traffic viscosity field (driver reaction heterogeneity).
        
        ν(x,y) = ν₀ + Σᵢ νᵢ · K(x-xᵢ, y-yᵢ)
        
        Higher viscosity near vehicles = more "friction" in flow.
        """
        nu = np.ones_like(X) * self.config.VISCOSITY_BASE
        
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
        Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
        
        for other in others:
            dx = other.x - ego.x
            dy = other.y - ego.y
            dx_ego = dx * cos_h - dy * sin_h
            dy_ego = dx * sin_h + dy * cos_h
            
            # Viscosity contribution based on speed difference
            speed_diff = abs(other.speed - ego.speed)
            nu_contrib = self.config.VISCOSITY_SPEED_FACTOR * speed_diff
            
            # Gaussian kernel
            r2 = (X_ego - dx_ego)**2 + (Y_ego - dy_ego)**2
            kernel = np.exp(-r2 / (2 * 10**2))
            
            nu += nu_contrib * kernel * other.visibility
        
        return nu
    
    def compute_vorticity_field(self, U: np.ndarray, V: np.ndarray,
                                dx: float, dy: float) -> np.ndarray:
        """
        Compute vorticity (curl of velocity field).
        
        ω = ∂v/∂x - ∂u/∂y
        
        Positive vorticity indicates counterclockwise rotation (left LC tendency).
        Negative vorticity indicates clockwise rotation (right LC tendency).
        """
        # Compute partial derivatives using central differences
        dv_dx = np.gradient(V, dx, axis=1)
        du_dy = np.gradient(U, dy, axis=0)
        
        omega = dv_dx - du_dy
        
        return omega
    
    def compute_momentum_flux_tensor(self, U: np.ndarray, V: np.ndarray,
                                     P: np.ndarray, nu: np.ndarray,
                                     dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute momentum flux (stress) tensor components.
        
        τ_xx = -P + 2ν(∂u/∂x)
        τ_yy = -P + 2ν(∂v/∂y)  
        τ_xy = ν(∂u/∂y + ∂v/∂x)
        
        These represent the "forces" acting on fluid elements.
        """
        du_dx = np.gradient(U, dx, axis=1)
        du_dy = np.gradient(U, dy, axis=0)
        dv_dx = np.gradient(V, dx, axis=1)
        dv_dy = np.gradient(V, dy, axis=0)
        
        tau_xx = -P + 2 * nu * du_dx
        tau_yy = -P + 2 * nu * dv_dy
        tau_xy = nu * (du_dy + dv_dx)
        
        return tau_xx, tau_yy, tau_xy
    
    def compute_all_fields(self, ego: VehicleState,
                          others: List[VehicleState],
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """
        Compute all Navier-Stokes inspired fields.
        
        Returns dict with: X, Y, P, U, V, nu, omega, tau_xx, tau_yy, tau_xy, speed_mag
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], self.config.GRID_NX)
        y = np.linspace(y_range[0], y_range[1], self.config.GRID_NY)
        X, Y = np.meshgrid(x, y)
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Compute fields
        P = self.compute_pressure_field(ego, others, X, Y)
        U, V = self.compute_velocity_field(ego, others, X, Y)
        nu = self.compute_viscosity_field(ego, others, X, Y)
        omega = self.compute_vorticity_field(U, V, dx, dy)
        tau_xx, tau_yy, tau_xy = self.compute_momentum_flux_tensor(U, V, P, nu, dx, dy)
        
        # Speed magnitude
        speed_mag = np.sqrt(U**2 + V**2)
        
        return {
            'X': X, 'Y': Y,
            'P': P,
            'U': U, 'V': V,
            'nu': nu,
            'omega': omega,
            'tau_xx': tau_xx,
            'tau_yy': tau_yy,
            'tau_xy': tau_xy,
            'speed_mag': speed_mag,
            'dx': dx, 'dy': dy
        }


# =============================================================================
# Stochastic Occlusion Field
# =============================================================================

class StochasticOcclusionField:
    """
    Models occlusion uncertainty using stochastic field approaches.
    
    Key components:
    1. Yukawa (screened Coulomb) potential for occluded interactions
    2. Uncertainty field based on visibility
    3. Gaussian Process with occlusion-dependent variance
    4. Fokker-Planck diffusion for belief propagation
    """
    
    def __init__(self, config: NSConfig = None):
        self.config = config or NSConfig()
        self.occlusion_detector = OcclusionDetector(config)
    
    def compute_visibility_field(self, ego: VehicleState,
                                 others: List[VehicleState],
                                 X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute visibility field from ego's perspective.
        
        V(x,y) ∈ [0, 1] where:
        - 1.0 = fully visible region
        - 0.0 = fully occluded (shadow) region
        """
        visibility = np.ones_like(X)
        
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        # Transform grid to ego frame
        X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
        Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
        
        # Find heavy vehicles that can cause occlusion
        occluders = [o for o in others if o.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES]
        
        for occluder in occluders:
            dx = occluder.x - ego.x
            dy = occluder.y - ego.y
            dx_ego = dx * cos_h - dy * sin_h
            dy_ego = dx * sin_h + dy * cos_h
            
            dist = np.sqrt(dx_ego**2 + dy_ego**2)
            if dist < 1.0:
                continue
            
            # Compute shadow cone
            angle_to_occluder = np.arctan2(dy_ego, dx_ego)
            angular_width = np.arctan2(occluder.width / 2, dist)
            
            # Grid point angles from ego
            grid_angles = np.arctan2(Y_ego, X_ego)
            grid_dists = np.sqrt(X_ego**2 + Y_ego**2)
            
            # Points in shadow cone (beyond occluder)
            in_cone = np.abs(grid_angles - angle_to_occluder) < angular_width
            beyond_occluder = grid_dists > dist
            
            # Soft shadow (gradual visibility reduction)
            shadow_mask = in_cone & beyond_occluder
            shadow_depth = (grid_dists - dist) / self.config.YUKAWA_SCREENING_LENGTH
            shadow_attenuation = np.exp(-shadow_depth)
            
            visibility = np.where(shadow_mask, 
                                  visibility * shadow_attenuation, 
                                  visibility)
        
        return np.clip(visibility, 0, 1)
    
    def compute_uncertainty_field(self, visibility: np.ndarray,
                                  X: np.ndarray, Y: np.ndarray,
                                  ego: VehicleState) -> np.ndarray:
        """
        Compute uncertainty field based on visibility and distance.
        
        σ²(x,y) = σ₀² + σ_occ² · (1 - V(x,y)) + σ_dist² · d(x,y)
        
        Higher uncertainty in:
        - Occluded regions
        - Far distances
        - Behind ego vehicle
        """
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
        Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
        
        # Distance from ego
        dist = np.sqrt(X_ego**2 + Y_ego**2)
        
        # Base uncertainty
        sigma2 = np.ones_like(X) * self.config.UNCERTAINTY_BASE**2
        
        # Occlusion contribution (inverse visibility)
        sigma2 += (self.config.UNCERTAINTY_OCCLUSION_MULT * self.config.UNCERTAINTY_BASE)**2 * (1 - visibility)
        
        # Distance contribution
        sigma2 += (self.config.UNCERTAINTY_DISTANCE_DECAY * dist)**2
        
        # Directional bias (more uncertainty behind)
        behind_mask = X_ego < -ego.length
        sigma2 = np.where(behind_mask, sigma2 * 1.5, sigma2)
        
        return np.sqrt(sigma2)
    
    def compute_yukawa_potential(self, ego: VehicleState,
                                others: List[VehicleState],
                                X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Yukawa (screened Coulomb) potential for occluded interactions.
        
        U(r) = A · exp(-r/λ) / r
        
        For occluded vehicles, the screening length λ is reduced,
        creating faster decay of their influence.
        """
        U = np.zeros_like(X)
        
        if not others:
            return U  # Return zero field if no other vehicles
        
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        X_ego = (X - ego.x) * cos_h - (Y - ego.y) * sin_h
        Y_ego = (X - ego.x) * sin_h + (Y - ego.y) * cos_h
        
        for other in others:
            dx = other.x - ego.x
            dy = other.y - ego.y
            dx_ego = dx * cos_h - dy * sin_h
            dy_ego = dx * sin_h + dy * cos_h
            
            # Distance from vehicle center
            r = np.sqrt((X_ego - dx_ego)**2 + (Y_ego - dy_ego)**2)
            r = np.maximum(r, 0.1)  # Avoid singularity
            
            # Screening length depends on visibility
            lambda_screen = self.config.YUKAWA_SCREENING_LENGTH * other.visibility
            lambda_screen = max(lambda_screen, 2.0)  # Minimum screening
            
            # Amplitude depends on mass and speed
            A = self.config.YUKAWA_AMPLITUDE * (other.mass / self.config.MASS_PC) * \
                (1 + other.speed / max(self.config.V_EQUILIBRIUM, 1))
            
            # Yukawa potential
            U += A * np.exp(-r / lambda_screen) / r
        
        return U
    
    def compute_fokker_planck_diffusion(self, P_initial: np.ndarray,
                                       uncertainty: np.ndarray,
                                       U: np.ndarray, V: np.ndarray,
                                       dx: float, dy: float,
                                       dt: float = 0.1,
                                       n_steps: int = 5) -> np.ndarray:
        """
        Evolve probability density using Fokker-Planck equation.
        
        ∂p/∂t = -∇·(vp) + D∇²p
        
        where D is spatially varying diffusion (from uncertainty field).
        """
        P = P_initial.copy()
        P_sum = P.sum()
        if P_sum < 1e-10:
            # If initial distribution is too small, use uniform
            P = np.ones_like(P_initial) / P_initial.size
        else:
            P = P / P_sum  # Normalize
        
        D = self.config.FP_DIFFUSION * (uncertainty**2 + 0.01)  # Add small constant for stability
        
        for _ in range(n_steps):
            # Advection term: -∇·(vp)
            flux_x = U * P
            flux_y = V * P
            
            div_flux = np.gradient(flux_x, dx, axis=1) + np.gradient(flux_y, dy, axis=0)
            
            # Diffusion term: D∇²p
            laplacian_P = (np.gradient(np.gradient(P, dx, axis=1), dx, axis=1) + 
                          np.gradient(np.gradient(P, dy, axis=0), dy, axis=0))
            
            # Also include gradient of D for variable diffusion
            dD_dx = np.gradient(D, dx, axis=1)
            dD_dy = np.gradient(D, dy, axis=0)
            dP_dx = np.gradient(P, dx, axis=1)
            dP_dy = np.gradient(P, dy, axis=0)
            
            diffusion = D * laplacian_P + dD_dx * dP_dx + dD_dy * dP_dy
            
            # Update with stability clipping
            P = P - dt * div_flux + dt * diffusion
            P = np.maximum(P, 0)  # Ensure non-negative
            P_sum = P.sum()
            if P_sum > 1e-10:
                P = P / P_sum  # Normalize
            else:
                P = np.ones_like(P_initial) / P_initial.size
        
        return P
    
    def compute_all_stochastic_fields(self, ego: VehicleState,
                                      others: List[VehicleState],
                                      ns_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute all stochastic occlusion fields.
        
        Returns dict with: visibility, uncertainty, yukawa_potential, 
                          risk_density, combined_risk
        """
        X = ns_fields['X']
        Y = ns_fields['Y']
        
        # Visibility field
        visibility = self.compute_visibility_field(ego, others, X, Y)
        
        # Uncertainty field
        uncertainty = self.compute_uncertainty_field(visibility, X, Y, ego)
        
        # Yukawa potential
        yukawa = self.compute_yukawa_potential(ego, others, X, Y)
        
        # Risk density evolution
        # Initialize with pressure field as proxy for vehicle density
        P_max = ns_fields['P'].max()
        if P_max > 1e-10:
            P_initial = ns_fields['P'] / P_max
        else:
            P_initial = np.ones_like(X) * 0.01  # Uniform low density
        
        risk_density = self.compute_fokker_planck_diffusion(
            P_initial, uncertainty,
            ns_fields['U'], ns_fields['V'],
            ns_fields['dx'], ns_fields['dy']
        )
        
        # Combined risk field
        # Weighted combination of deterministic and stochastic components
        P_norm = ns_fields['P'] / (ns_fields['P'].max() + 1e-10)
        yukawa_norm = yukawa / (yukawa.max() + 1e-10)
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-10)
        
        combined_risk = (0.4 * P_norm + 0.3 * yukawa_norm + 0.3 * uncertainty_norm)
        
        return {
            'visibility': visibility,
            'uncertainty': uncertainty,
            'yukawa_potential': yukawa,
            'risk_density': risk_density,
            'combined_risk': combined_risk
        }


# =============================================================================
# Occlusion Detector (Simplified from AD4CHE analysis)
# =============================================================================

class OcclusionDetector:
    """Simplified occlusion detection for field computation."""
    
    def __init__(self, config: NSConfig = None):
        self.config = config or NSConfig()
    
    def compute_vehicle_visibility(self, ego: VehicleState,
                                   target: VehicleState,
                                   potential_occluders: List[VehicleState]) -> float:
        """Compute visibility of target from ego's perspective."""
        dx = target.x - ego.x
        dy = target.y - ego.y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        if dist_to_target < 1.0:
            return 1.0
        
        target_angle = np.arctan2(dy, dx)
        target_angular_width = np.arctan2(target.width / 2, dist_to_target)
        
        max_occlusion = 0.0
        
        for occluder in potential_occluders:
            if occluder.id == ego.id or occluder.id == target.id:
                continue
            
            occ_dx = occluder.x - ego.x
            occ_dy = occluder.y - ego.y
            dist_to_occ = np.sqrt(occ_dx**2 + occ_dy**2)
            
            # Occluder must be between ego and target
            if dist_to_occ >= dist_to_target:
                continue
            
            occ_angle = np.arctan2(occ_dy, occ_dx)
            occ_angular_width = np.arctan2(occluder.width / 2, dist_to_occ)
            
            # Check angular overlap
            angle_diff = abs(target_angle - occ_angle)
            total_width = target_angular_width + occ_angular_width
            
            if angle_diff < total_width:
                overlap = (total_width - angle_diff) / (2 * target_angular_width)
                max_occlusion = max(max_occlusion, min(overlap, 1.0))
        
        return 1.0 - max_occlusion
    
    def update_vehicle_visibility(self, ego: VehicleState,
                                  others: List[VehicleState]) -> List[VehicleState]:
        """Update visibility attribute for all vehicles."""
        heavy_vehicles = [v for v in others 
                        if v.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES]
        
        for vehicle in others:
            visibility = self.compute_vehicle_visibility(ego, vehicle, heavy_vehicles)
            vehicle.visibility = visibility
            vehicle.is_occluded = visibility < 0.5
            vehicle.occlusion_ratio = 1.0 - visibility
        
        return others


# =============================================================================
# Data Loader (AD4CHE Format)
# =============================================================================

class AD4CHELoader:
    """Load AD4CHE data for NS field computation."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = NSConfig()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_meta = None
        self.background_image = None
        self.scale = 0.0375
    
    def _parse_scale(self, value) -> float:
        try:
            if isinstance(value, str):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                if nums:
                    return float(nums[-1])
            return float(value)
        except:
            return self.scale
    
    def load_recording(self, recording_id: int) -> bool:
        prefix = f"{recording_id:02d}_"
        
        try:
            self.tracks_df = pd.read_csv(self.data_dir / f"{prefix}tracks.csv")
            self.tracks_meta_df = pd.read_csv(self.data_dir / f"{prefix}tracksMeta.csv")
            
            rec_meta_path = self.data_dir / f"{prefix}recordingMeta.csv"
            if rec_meta_path.exists():
                rec_df = pd.read_csv(rec_meta_path)
                if not rec_df.empty:
                    self.recording_meta = rec_df.iloc[0]
                    self.scale = self._parse_scale(self.recording_meta.get('scale', self.scale))
            
            # Load background
            for suffix in ['highway.png', 'lanePicture.png', 'background.png']:
                bg_path = self.data_dir / f"{prefix}{suffix}"
                if bg_path.exists():
                    self.background_image = plt.imread(str(bg_path))
                    break
            
            # Merge meta
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['id', 'class', 'width', 'height', 'drivingDirection']],
                on='id', how='left', suffixes=('', '_meta')
            )
            
            # Fix dimensions (AD4CHE: width=length, height=width)
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['length'] = self.tracks_df['width_meta']
                self.tracks_df['veh_width'] = self.tracks_df['height']
            
            logger.info(f"Loaded recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int) -> Optional[Dict]:
        """Get snapshot of ego and surrounding vehicles."""
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        
        if frame_data.empty:
            return None
        
        ego_row = frame_data[frame_data['id'] == ego_id]
        if ego_row.empty:
            return None
        
        ego_row = ego_row.iloc[0]
        vclass = str(ego_row.get('class', 'car')).lower()
        
        ego = VehicleState(
            id=ego_id,
            x=float(ego_row['x']),
            y=float(ego_row['y']),
            vx=float(ego_row.get('xVelocity', 0)),
            vy=float(ego_row.get('yVelocity', 0)),
            ax=float(ego_row.get('xAcceleration', 0)),
            ay=float(ego_row.get('yAcceleration', 0)),
            heading=float(ego_row.get('orientation', 0)),
            speed=np.sqrt(ego_row.get('xVelocity', 0)**2 + ego_row.get('yVelocity', 0)**2),
            length=float(ego_row.get('length', ego_row.get('width', 12.0))),
            width=float(ego_row.get('veh_width', ego_row.get('height', 2.5))),
            vehicle_class=vclass,
            mass=self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
        )
        
        # Get surrounding vehicles
        surrounding = []
        for _, row in frame_data.iterrows():
            if row['id'] == ego_id:
                continue
            
            other_class = str(row.get('class', 'car')).lower()
            dx = row['x'] - ego.x
            dy = row['y'] - ego.y
            
            # Check observation range
            if not (-self.config.OBS_RANGE_BEHIND <= dx <= self.config.OBS_RANGE_AHEAD and
                    -self.config.OBS_RANGE_RIGHT <= dy <= self.config.OBS_RANGE_LEFT):
                continue
            
            other = VehicleState(
                id=int(row['id']),
                x=float(row['x']),
                y=float(row['y']),
                vx=float(row.get('xVelocity', 0)),
                vy=float(row.get('yVelocity', 0)),
                ax=float(row.get('xAcceleration', 0)),
                ay=float(row.get('yAcceleration', 0)),
                heading=float(row.get('orientation', 0)),
                speed=np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                length=float(row.get('length', row.get('width', 4.5))),
                width=float(row.get('veh_width', row.get('height', 1.8))),
                vehicle_class=other_class,
                mass=self.config.MASS_HV if other_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
            )
            surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def get_heavy_vehicles(self) -> List[int]:
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['id'].tolist()
    
    def find_best_frame(self, ego_id: int) -> Optional[int]:
        ego_data = self.tracks_df[self.tracks_df['id'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        best_frame = None
        best_count = -1
        
        for frame in frames[::10]:
            snapshot = self.get_snapshot(ego_id, frame)
            if snapshot and len(snapshot['surrounding']) > best_count:
                best_count = len(snapshot['surrounding'])
                best_frame = frame
        
        return best_frame if best_frame else int(np.median(frames))
    
    def get_background_extent(self) -> Optional[List[float]]:
        if self.background_image is None:
            return None
        h, w = self.background_image.shape[:2]
        return [0, w * self.scale, h * self.scale, 0]


# =============================================================================
# Visualization
# =============================================================================

class NSFieldVisualizer:
    """Visualize Navier-Stokes inspired fields with occlusion."""
    
    def __init__(self, config: NSConfig = None, loader: AD4CHELoader = None):
        self.config = config or NSConfig()
        self.loader = loader
    
    def create_combined_figure(self, snapshot: Dict, 
                               ns_fields: Dict, 
                               stoch_fields: Dict,
                               output_path: str = None):
        """Create comprehensive visualization figure."""
        
        ego = snapshot['ego']
        others = snapshot['surrounding']
        
        if not others:
            logger.warning("No surrounding vehicles - creating limited visualization")
        
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor(self.config.BG_DARK)
        
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: NS Fields
        ax1 = fig.add_subplot(gs[0, 0])  # Pressure
        ax2 = fig.add_subplot(gs[0, 1])  # Velocity
        ax3 = fig.add_subplot(gs[0, 2])  # Vorticity
        ax4 = fig.add_subplot(gs[0, 3])  # Viscosity
        ax5 = fig.add_subplot(gs[1, 0])  # Visibility
        ax6 = fig.add_subplot(gs[1, 1])  # Uncertainty
        ax7 = fig.add_subplot(gs[1, 2])  # Yukawa Potential
        ax8 = fig.add_subplot(gs[1, 3])  # Risk Density
        
        # Row 3: Combined and Summary
        ax9 = fig.add_subplot(gs[2, :2])   # Combined Risk Field
        ax10 = fig.add_subplot(gs[2, 2])   # Traffic Snapshot
        ax11 = fig.add_subplot(gs[2, 3])   # Summary Statistics
        
        # Extract grid
        X, Y = ns_fields['X'], ns_fields['Y']
        
        # Plot NS fields
        self._plot_pressure_field(ax1, X, Y, ns_fields['P'], ego, others)
        self._plot_velocity_field(ax2, X, Y, ns_fields['U'], ns_fields['V'], ego, others)
        self._plot_vorticity_field(ax3, X, Y, ns_fields['omega'], ego, others)
        self._plot_viscosity_field(ax4, X, Y, ns_fields['nu'], ego, others)
        
        # Plot stochastic fields
        self._plot_visibility_field(ax5, X, Y, stoch_fields['visibility'], ego, others)
        self._plot_uncertainty_field(ax6, X, Y, stoch_fields['uncertainty'], ego, others)
        self._plot_yukawa_field(ax7, X, Y, stoch_fields['yukawa_potential'], ego, others)
        self._plot_risk_density(ax8, X, Y, stoch_fields['risk_density'], ego, others)
        
        # Combined and summary
        self._plot_combined_risk(ax9, X, Y, stoch_fields['combined_risk'], 
                                ns_fields, stoch_fields, ego, others)
        self._plot_traffic_snapshot(ax10, ego, others)
        self._plot_summary_stats(ax11, ego, others, ns_fields, stoch_fields)
        
        # Title
        fig.suptitle(
            f"Navier-Stokes Interaction Field with Stochastic Occlusion | "
            f"{ego.vehicle_class.title()} (ID: {ego.id}) | Frame: {snapshot['frame']} | "
            f"Surrounding: {len(others)} vehicles",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _setup_axis(self, ax, title: str):
        """Common axis setup."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.set_title(title, fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('X (m)', color='white', fontsize=8)
        ax.set_ylabel('Y (m)', color='white', fontsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)
    
    def _draw_vehicles(self, ax, ego: VehicleState, others: List[VehicleState],
                       in_ego_frame: bool = True):
        """Draw vehicles on axis."""
        cos_h = np.cos(-ego.heading) if in_ego_frame else 1
        sin_h = np.sin(-ego.heading) if in_ego_frame else 0
        
        # Draw ego at origin (if ego frame) or actual position
        ego_x = 0 if in_ego_frame else ego.x
        ego_y = 0 if in_ego_frame else ego.y
        
        ego_rect = mpatches.FancyBboxPatch(
            (ego_x - ego.length/2, ego_y - ego.width/2),
            ego.length, ego.width,
            boxstyle="round,pad=0.02",
            facecolor=self.config.COLORS.get(ego.vehicle_class, '#E74C3C'),
            edgecolor='yellow', linewidth=2
        )
        ax.add_patch(ego_rect)
        ax.text(ego_x, ego_y - ego.width/2 - 1.5, 'EGO',
               ha='center', va='top', fontsize=7, color='yellow', fontweight='bold')
        
        for other in others:
            if in_ego_frame:
                dx = other.x - ego.x
                dy = other.y - ego.y
                ox = dx * cos_h - dy * sin_h
                oy = dx * sin_h + dy * cos_h
            else:
                ox, oy = other.x, other.y
            
            # Color by visibility
            if other.visibility < 0.3:
                color = '#7F8C8D'  # Gray for occluded
                alpha = 0.5
            elif other.visibility < 0.7:
                color = '#F39C12'  # Orange for partial
                alpha = 0.7
            else:
                color = self.config.COLORS.get(other.vehicle_class, '#3498DB')
                alpha = 0.9
            
            rect = mpatches.FancyBboxPatch(
                (ox - other.length/2, oy - other.width/2),
                other.length, other.width,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', 
                linewidth=1, alpha=alpha
            )
            ax.add_patch(rect)
            ax.text(ox, oy + other.width/2 + 0.8, f'{other.id}',
                   ha='center', va='bottom', fontsize=6, color='white')
    
    def _plot_pressure_field(self, ax, X, Y, P, ego, others):
        """Plot traffic pressure field."""
        self._setup_axis(ax, 'Traffic Pressure Field P(x,y)')
        
        cmap = LinearSegmentedColormap.from_list('pressure', 
            ['#0D1117', '#1E3A5F', '#3498DB', '#F39C12', '#E74C3C'])
        pcm = ax.pcolormesh(X, Y, P, cmap=cmap, shading='gouraud')
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Pressure (a.u.)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_velocity_field(self, ax, X, Y, U, V, ego, others):
        """Plot velocity field with streamlines."""
        self._setup_axis(ax, 'Relative Velocity Field (u,v)')
        
        speed = np.sqrt(U**2 + V**2)
        pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='gouraud', alpha=0.7)
        
        # Streamlines (check for valid velocity field first)
        if speed.max() > 1e-6:  # Only plot if there's meaningful velocity
            try:
                ax.streamplot(X, Y, U, V, color='white', density=1.2, 
                             linewidth=0.5, arrowsize=0.8)
            except ValueError:
                # Streamplot can fail with certain field configurations
                skip = 4
                ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                         U[::skip, ::skip], V[::skip, ::skip],
                         color='white', scale=50, alpha=0.7)
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Speed (m/s)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_vorticity_field(self, ax, X, Y, omega, ego, others):
        """Plot vorticity field."""
        self._setup_axis(ax, 'Vorticity Field ω = ∂v/∂x - ∂u/∂y')
        
        # Symmetric colormap for vorticity
        vmax = np.percentile(np.abs(omega), 95)
        pcm = ax.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='gouraud',
                           vmin=-vmax, vmax=vmax)
        
        # Contour lines
        ax.contour(X, Y, omega, levels=[-0.1, 0, 0.1], colors='white', 
                  linewidths=0.5, alpha=0.5)
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Vorticity (1/s)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_viscosity_field(self, ax, X, Y, nu, ego, others):
        """Plot viscosity field."""
        self._setup_axis(ax, 'Traffic Viscosity ν(x,y)')
        
        pcm = ax.pcolormesh(X, Y, nu, cmap='magma', shading='gouraud')
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Viscosity (a.u.)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_visibility_field(self, ax, X, Y, V, ego, others):
        """Plot visibility field."""
        self._setup_axis(ax, 'Visibility Field V(x,y)')
        
        pcm = ax.pcolormesh(X, Y, V, cmap='gray', shading='gouraud', 
                           vmin=0, vmax=1)
        
        # Highlight shadow regions
        ax.contour(X, Y, V, levels=[0.3, 0.5, 0.7], colors=['red', 'orange', 'yellow'],
                  linewidths=1, linestyles='--')
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Visibility [0-1]', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_uncertainty_field(self, ax, X, Y, sigma, ego, others):
        """Plot uncertainty field."""
        self._setup_axis(ax, 'Uncertainty Field σ(x,y)')
        
        cmap = LinearSegmentedColormap.from_list('uncertainty',
            ['#0D1117', '#2C3E50', '#8E44AD', '#E74C3C'])
        pcm = ax.pcolormesh(X, Y, sigma, cmap=cmap, shading='gouraud')
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Uncertainty σ', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_yukawa_field(self, ax, X, Y, U_yukawa, ego, others):
        """Plot Yukawa (screened) potential field."""
        self._setup_axis(ax, 'Yukawa Potential U(r) = A·exp(-r/λ)/r')
        
        # Log scale for better visualization
        U_plot = np.log1p(U_yukawa)
        pcm = ax.pcolormesh(X, Y, U_plot, cmap='hot', shading='gouraud')
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('log(1 + U)', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_risk_density(self, ax, X, Y, rho, ego, others):
        """Plot Fokker-Planck evolved risk density."""
        self._setup_axis(ax, 'Risk Density (Fokker-Planck Evolution)')
        
        cmap = LinearSegmentedColormap.from_list('risk',
            ['#0D1117', '#1A5276', '#2980B9', '#F39C12', '#E74C3C', '#FFFFFF'])
        pcm = ax.pcolormesh(X, Y, rho, cmap=cmap, shading='gouraud')
        
        # Contours for high risk regions
        ax.contour(X, Y, rho, levels=[0.01, 0.05, 0.1], colors='white',
                  linewidths=0.8, alpha=0.7)
        
        self._draw_vehicles(ax, ego, others)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Probability Density', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
    
    def _plot_combined_risk(self, ax, X, Y, combined_risk, ns_fields, stoch_fields,
                           ego, others):
        """Plot combined risk field with gradient arrows."""
        self._setup_axis(ax, 'Combined Risk Field (NS + Stochastic Occlusion)')
        
        cmap = LinearSegmentedColormap.from_list('combined',
            ['#0D1117', '#1E3A5F', '#2ECC71', '#F1C40F', '#E74C3C', '#FFFFFF'])
        pcm = ax.pcolormesh(X, Y, combined_risk, cmap=cmap, shading='gouraud')
        
        # Risk gradient (negative = escape direction)
        dx = ns_fields['dx']
        dy = ns_fields['dy']
        grad_x = -np.gradient(combined_risk, dx, axis=1)
        grad_y = -np.gradient(combined_risk, dy, axis=0)
        
        # Quiver for gradient
        skip = 4
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                 color='white', alpha=0.6, scale=20)
        
        # Velocity streamlines overlay (use RGBA for transparency)
        try:
            speed_check = np.sqrt(ns_fields['U']**2 + ns_fields['V']**2)
            if speed_check.max() > 1e-6:
                stream = ax.streamplot(X, Y, ns_fields['U'], ns_fields['V'], 
                                      color=(0, 1, 1, 0.5), density=0.8, linewidth=0.3, 
                                      arrowsize=0.5)
        except (ValueError, IndexError):
            pass  # Skip streamlines if they fail
        
        self._draw_vehicles(ax, ego, others)
        
        # Highlight occluded vehicles
        for other in others:
            if other.visibility < 0.5:
                cos_h = np.cos(-ego.heading)
                sin_h = np.sin(-ego.heading)
                dx_o = other.x - ego.x
                dy_o = other.y - ego.y
                ox = dx_o * cos_h - dy_o * sin_h
                oy = dx_o * sin_h + dy_o * cos_h
                
                circle = plt.Circle((ox, oy), 3, fill=False, 
                                   edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(circle)
                ax.text(ox, oy + 4, 'OCCLUDED', ha='center', fontsize=7,
                       color='red', fontweight='bold')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Combined Risk', color='white', fontsize=8)
        cbar.ax.tick_params(colors='white', labelsize=7)
        
        ax.set_aspect('equal')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#E74C3C', label='High Risk'),
            mpatches.Patch(facecolor='#2ECC71', label='Low Risk'),
            plt.Line2D([0], [0], color='cyan', label='Flow Streamlines'),
            plt.Line2D([0], [0], color='white', label='Escape Gradient'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
                 facecolor=self.config.BG_PANEL, edgecolor='white', labelcolor='white')
    
    def _plot_traffic_snapshot(self, ax, ego, others):
        """Plot traffic snapshot in ego frame."""
        self._setup_axis(ax, 'Traffic Configuration (Ego Frame)')
        
        self._draw_vehicles(ax, ego, others, in_ego_frame=True)
        
        # Draw observation range
        rect = mpatches.Rectangle(
            (-self.config.OBS_RANGE_BEHIND, -self.config.OBS_RANGE_RIGHT),
            self.config.OBS_RANGE_AHEAD + self.config.OBS_RANGE_BEHIND,
            self.config.OBS_RANGE_LEFT + self.config.OBS_RANGE_RIGHT,
            fill=False, edgecolor='white', linestyle='--', linewidth=1, alpha=0.5
        )
        ax.add_patch(rect)
        
        # Lane markings
        for y in [-7, -3.5, 0, 3.5, 7]:
            ax.axhline(y, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Velocity arrows
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        for other in others:
            dx = other.x - ego.x
            dy = other.y - ego.y
            ox = dx * cos_h - dy * sin_h
            oy = dx * sin_h + dy * cos_h
            
            dvx = other.vx - ego.vx
            dvy = other.vy - ego.vy
            dvx_r = dvx * cos_h - dvy * sin_h
            dvy_r = dvx * sin_h + dvy * cos_h
            
            ax.arrow(ox, oy, dvx_r * 0.3, dvy_r * 0.3,
                    head_width=0.8, head_length=0.3, fc='cyan', ec='cyan', alpha=0.7)
        
        ax.set_xlim(-self.config.OBS_RANGE_BEHIND - 5, self.config.OBS_RANGE_AHEAD + 5)
        ax.set_ylim(-self.config.OBS_RANGE_RIGHT - 5, self.config.OBS_RANGE_LEFT + 5)
        ax.set_aspect('equal')
    
    def _plot_summary_stats(self, ax, ego, others, ns_fields, stoch_fields):
        """Plot summary statistics panel."""
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        # Compute statistics
        n_occluded = sum(1 for o in others if o.visibility < 0.5)
        n_partial = sum(1 for o in others if 0.5 <= o.visibility < 0.8)
        n_visible = len(others) - n_occluded - n_partial
        
        avg_visibility = np.mean([o.visibility for o in others]) if others else 1.0
        max_pressure = ns_fields['P'].max()
        max_vorticity = np.abs(ns_fields['omega']).max()
        max_uncertainty = stoch_fields['uncertainty'].max()
        max_risk = stoch_fields['combined_risk'].max()
        
        # Risk assessment
        if max_risk > 0.7:
            risk_level = "HIGH"
            risk_color = "#E74C3C"
        elif max_risk > 0.4:
            risk_level = "MODERATE"
            risk_color = "#F39C12"
        else:
            risk_level = "LOW"
            risk_color = "#2ECC71"
        
        lines = [
            "=== NS-OCCLUSION FIELD SUMMARY ===",
            "",
            f"Ego Vehicle: {ego.vehicle_class.upper()} (ID: {ego.id})",
            f"Speed: {ego.speed * 3.6:.1f} km/h",
            f"Mass: {ego.mass/1000:.1f} tons",
            "",
            "=== Surrounding Vehicles ===",
            f"Total: {len(others)}",
            f"  Fully Visible: {n_visible}",
            f"  Partially Occluded: {n_partial}",
            f"  Fully Occluded: {n_occluded}",
            f"Avg Visibility: {avg_visibility:.2f}",
            "",
            "=== Navier-Stokes Fields ===",
            f"Max Pressure: {max_pressure:.1f}",
            f"Max |Vorticity|: {max_vorticity:.3f} /s",
            f"Mean Viscosity: {ns_fields['nu'].mean():.3f}",
        ]
        
        lines.append("")
        lines.append("=== Stochastic Fields ===")
        lines.append(f"Max Uncertainty: {max_uncertainty:.3f}")
        lines.append(f"Max Yukawa Pot.: {stoch_fields['yukawa_potential'].max():.1f}")
        
        # Compute entropy safely
        rho = stoch_fields['risk_density']
        rho_sum = rho.sum()
        if rho_sum > 1e-10:
            rho_norm = rho / rho_sum
            rho_safe = np.where(rho_norm > 1e-10, rho_norm, 1e-10)
            risk_entropy = -np.sum(rho_norm * np.log(rho_safe))
        else:
            risk_entropy = 0.0
        
        lines.append(f"Risk Entropy: {risk_entropy:.2f}")
        
        lines.extend([
            "=== Risk Assessment ===",
            f"Combined Risk: {max_risk:.3f}",
            f"Risk Level: {risk_level}",
        ])
        
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, color='white', family='monospace',
               verticalalignment='top')
        
        # Risk indicator box
        ax.add_patch(mpatches.Rectangle(
            (0.7, 0.1), 0.25, 0.15, transform=ax.transAxes,
            facecolor=risk_color, edgecolor='white', linewidth=2
        ))
        ax.text(0.825, 0.175, risk_level, transform=ax.transAxes,
               ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        for spine in ax.spines.values():
            spine.set_color(self.config.SPINE_COLOR)


# =============================================================================
# Metrics Computation
# =============================================================================

class NSFieldMetrics:
    """Compute evaluation metrics for NS-based interaction fields."""
    
    def __init__(self, config: NSConfig = None):
        self.config = config or NSConfig()
    
    def compute_field_metrics(self, ns_fields: Dict, stoch_fields: Dict,
                             ego: VehicleState, others: List[VehicleState]) -> Dict:
        """Compute comprehensive field quality metrics."""
        
        metrics = {}
        
        # === PDE Residual Metrics ===
        # Continuity residual: ∂ρ/∂t + ∇·(ρv) ≈ 0
        # Using pressure as density proxy
        P = ns_fields['P']
        U, V = ns_fields['U'], ns_fields['V']
        dx, dy = ns_fields['dx'], ns_fields['dy']
        
        div_flux = (np.gradient(P * U, dx, axis=1) + np.gradient(P * V, dy, axis=0))
        metrics['continuity_residual_l2'] = float(np.sqrt(np.mean(div_flux**2)))
        
        # === Spatial Field Quality ===
        # Divergence of velocity field
        div_v = np.gradient(U, dx, axis=1) + np.gradient(V, dy, axis=0)
        metrics['velocity_divergence_mean'] = float(np.mean(np.abs(div_v)))
        
        # Curl magnitude (vorticity)
        omega = ns_fields['omega']
        metrics['vorticity_max'] = float(np.max(np.abs(omega)))
        metrics['vorticity_l2'] = float(np.sqrt(np.mean(omega**2)))
        
        # Enstrophy (vorticity squared integral)
        metrics['enstrophy'] = float(np.mean(omega**2))
        
        # === Energy Metrics ===
        # Kinetic energy density
        KE = 0.5 * (U**2 + V**2)
        metrics['kinetic_energy_mean'] = float(np.mean(KE))
        metrics['kinetic_energy_max'] = float(np.max(KE))
        
        # Potential energy (from pressure/Yukawa)
        PE = P + stoch_fields['yukawa_potential']
        metrics['potential_energy_mean'] = float(np.mean(PE))
        
        # === Information-Theoretic Metrics ===
        # Risk field entropy
        rho = stoch_fields['risk_density']
        rho_sum = rho.sum()
        if rho_sum > 1e-10:
            rho_norm = rho / rho_sum
            # Avoid log(0) by adding small epsilon and only computing where rho_norm > 0
            rho_safe = np.where(rho_norm > 1e-10, rho_norm, 1e-10)
            entropy = -np.sum(rho_norm * np.log(rho_safe))
        else:
            entropy = 0.0
        metrics['risk_entropy'] = float(entropy)
        
        # Uncertainty field statistics
        sigma = stoch_fields['uncertainty']
        metrics['uncertainty_mean'] = float(np.mean(sigma))
        metrics['uncertainty_max'] = float(np.max(sigma))
        
        # === Occlusion Metrics ===
        visibilities = [o.visibility for o in others]
        if visibilities:
            metrics['mean_visibility'] = float(np.mean(visibilities))
            metrics['min_visibility'] = float(np.min(visibilities))
            metrics['n_occluded'] = int(sum(1 for v in visibilities if v < 0.5))
        else:
            metrics['mean_visibility'] = 1.0
            metrics['min_visibility'] = 1.0
            metrics['n_occluded'] = 0
        
        # === Stability Metrics ===
        # Lyapunov-like measure (gradient magnitude of risk field)
        grad_risk_x = np.gradient(stoch_fields['combined_risk'], dx, axis=1)
        grad_risk_y = np.gradient(stoch_fields['combined_risk'], dy, axis=0)
        grad_risk_mag = np.sqrt(grad_risk_x**2 + grad_risk_y**2)
        metrics['risk_gradient_max'] = float(np.max(grad_risk_mag))
        metrics['risk_gradient_mean'] = float(np.mean(grad_risk_mag))
        
        return metrics


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_recording(data_dir: str, recording_id: int,
                     ego_id: Optional[int] = None,
                     frame: Optional[int] = None,
                     output_dir: str = './output_ns_field') -> Dict:
    """Main analysis pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Navier-Stokes Interaction Field with Stochastic Occlusion Analysis")
    logger.info("=" * 70)
    
    # Initialize components
    config = NSConfig()
    loader = AD4CHELoader(data_dir)
    
    if not loader.load_recording(recording_id):
        return {}
    
    # Find ego vehicle
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if heavy_ids:
            ego_id = heavy_ids[0]
            logger.info(f"Auto-selected ego (truck): {ego_id}")
        else:
            logger.error("No heavy vehicles found")
            return {}
    
    # Find best frame
    if frame is None:
        frame = loader.find_best_frame(ego_id)
        logger.info(f"Auto-selected frame: {frame}")
    
    # Get snapshot
    snapshot = loader.get_snapshot(ego_id, frame)
    if snapshot is None:
        logger.error("Could not get snapshot")
        return {}
    
    ego = snapshot['ego']
    others = snapshot['surrounding']
    
    logger.info(f"Ego: {ego.vehicle_class} (ID: {ego.id}), Speed: {ego.speed*3.6:.1f} km/h")
    logger.info(f"Surrounding: {len(others)} vehicles")
    
    # Update visibility based on occlusion
    occlusion_detector = OcclusionDetector(config)
    others = occlusion_detector.update_vehicle_visibility(ego, others)
    snapshot['surrounding'] = others
    
    n_occluded = sum(1 for o in others if o.is_occluded)
    logger.info(f"Occluded vehicles: {n_occluded}")
    
    # Compute NS fields
    ns_computer = NavierStokesTrafficField(config)
    x_range = (-config.OBS_RANGE_BEHIND, config.OBS_RANGE_AHEAD)
    y_range = (-config.OBS_RANGE_RIGHT, config.OBS_RANGE_LEFT)
    
    logger.info("Computing Navier-Stokes fields...")
    ns_fields = ns_computer.compute_all_fields(ego, others, x_range, y_range)
    
    # Compute stochastic fields
    stoch_computer = StochasticOcclusionField(config)
    logger.info("Computing stochastic occlusion fields...")
    stoch_fields = stoch_computer.compute_all_stochastic_fields(ego, others, ns_fields)
    
    # Compute metrics
    metrics_computer = NSFieldMetrics(config)
    metrics = metrics_computer.compute_field_metrics(ns_fields, stoch_fields, ego, others)
    
    logger.info("Field Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Create visualization
    visualizer = NSFieldVisualizer(config, loader)
    fig_path = output_path / f'ns_field_rec{recording_id}_ego{ego_id}_frame{frame}.png'
    visualizer.create_combined_figure(snapshot, ns_fields, stoch_fields, str(fig_path))
    
    # Save metrics
    metrics_path = output_path / f'metrics_rec{recording_id}_ego{ego_id}_frame{frame}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'recording_id': recording_id,
            'ego_id': ego_id,
            'frame': frame,
            'n_surrounding': len(others),
            'n_occluded': n_occluded,
            'metrics': metrics
        }, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")
    
    logger.info("=" * 70)
    logger.info("Analysis Complete!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)
    
    return {
        'recording_id': recording_id,
        'ego_id': ego_id,
        'frame': frame,
        'n_surrounding': len(others),
        'n_occluded': n_occluded,
        'metrics': metrics,
        'output_path': str(output_path)
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Navier-Stokes Interaction Field with Stochastic Occlusion Analysis'
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to AD4CHE data directory')
    parser.add_argument('--recording', type=int, default=1,
                       help='Recording ID to analyze')
    parser.add_argument('--ego_id', type=int, default=None,
                       help='Ego vehicle ID (truck/bus)')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame to analyze')
    parser.add_argument('--output_dir', type=str, default='./output_ns_field',
                       help='Output directory')
    
    args = parser.parse_args()
    
    result = analyze_recording(
        args.data_dir,
        args.recording,
        args.ego_id,
        args.frame,
        args.output_dir
    )
    
    if result:
        print(f"\nAnalysis complete. Output saved to: {result['output_path']}")
