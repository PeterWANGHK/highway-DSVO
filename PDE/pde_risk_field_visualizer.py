"""
PDE-Based Risk Field Visualization for Heavy Vehicle Interactions
===================================================================
Implements the Unified Hyperbolic-Parabolic PDE Framework for traffic risk assessment.

Based on the Telegrapher's Equation formulation:
    τ ∂²R/∂t² + ∂R/∂t + ∇·(v_eff R) = ∇·(D∇R) + Q(x,t) - λR

Features:
1. Risk field profile visualization with distance-based dynamics
2. Risk field decomposition (source, occlusion, topology, predator-prey)
3. Finite propagation speed enforcement (hyperbolic PDE)
4. Diffractive barrier modeling for occlusion
5. Singular merging pressure potentials
6. Predator-Prey dynamics for HV-MV interactions

Author: Generated for exiD dataset analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Physical Constants and Configuration
# =============================================================================

@dataclass
class PDEConfig:
    """Configuration for PDE-based risk field computation."""
    
    # Grid parameters
    grid_size_x: int = 200  # Longitudinal grid points
    grid_size_y: int = 80   # Lateral grid points
    dx: float = 1.0         # Spatial resolution (m)
    dt: float = 0.04        # Time step (s) - 25 FPS
    
    # Telegrapher's equation parameters
    tau: float = 0.6        # Relaxation time (driver reaction time, s)
    c: float = 18.0         # Risk propagation speed (m/s)
    lambda_decay: float = 0.15  # Dissipation rate (1/s)
    
    # Diffusion tensor (anisotropic)
    D_longitudinal: float = 8.0   # m²/s
    D_lateral: float = 4.0        # m²/s
    D_occlusion_boost: float = 3.0  # Boost factor in occlusion zones
    
    # Merging pressure parameters
    k_topo: float = 500.0     # Topology pressure coefficient
    gamma: float = 2.0        # Singularity exponent
    merge_end_offset: float = 50.0  # Distance from scene end to merge point
    
    # Occlusion parameters
    occlusion_source_strength: float = 0.8
    diffraction_decay: float = 0.15  # Angular decay rate
    shadow_extension: float = 100.0  # Shadow zone extension (m)
    
    # Predator-prey parameters (Lotka-Volterra)
    alpha: float = 0.3   # Prey growth rate
    beta: float = 0.5    # Predation rate
    gamma_pred: float = 0.2  # Predator death rate
    delta: float = 0.4   # Predator reproduction rate
    
    # Visualization
    cmap_risk: str = 'hot'
    cmap_urgency: str = 'YlOrRd'
    cmap_diffraction: str = 'Blues'
    bg_color: str = '#0D1117'
    road_color: str = '#1A1A2E'
    lane_color: str = '#3A3A5A'


class VehicleClass(Enum):
    """Vehicle classification."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MERGING = "merging"


@dataclass
class Vehicle:
    """Vehicle state representation."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    heading: float
    vehicle_class: VehicleClass
    is_ego: bool = False
    
    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)
    
    @property
    def corners(self) -> np.ndarray:
        """Compute rotated bounding box corners."""
        half_l, half_w = self.length / 2, self.width / 2
        corners_local = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return corners_local @ R.T + np.array([self.x, self.y])


@dataclass 
class Scenario:
    """Traffic scenario container."""
    vehicles: List[Vehicle]
    ego_id: int
    road_bounds: Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    lane_centers: List[float]  # y-coordinates of lane centers
    merge_lane_y: Optional[float] = None
    merge_end_x: Optional[float] = None
    frame_id: int = 0
    recording_id: str = "sample"


# =============================================================================
# PDE Solver: Telegrapher's Equation with Singular Potentials
# =============================================================================

class TelegrapherPDESolver:
    """
    Solves the damped wave equation (Telegrapher's equation) for risk field:
    
    τ ∂²R/∂t² + ∂R/∂t + ∇·(v_eff R) = ∇·(D∇R) + Q(x,t) - λR
    
    Discretized using FDTD (Finite-Difference Time-Domain) with staggered grid.
    """
    
    def __init__(self, config: PDEConfig):
        self.config = config
        self.nx = config.grid_size_x
        self.ny = config.grid_size_y
        self.dx = config.dx
        self.dt = config.dt
        
        # Initialize field arrays
        self.R = np.zeros((self.nx, self.ny))        # Risk field (current)
        self.R_prev = np.zeros((self.nx, self.ny))   # Risk field (previous)
        self.R_velocity = np.zeros((self.nx, self.ny))  # Time derivative
        
        # Source and potential fields
        self.Q_source = np.zeros((self.nx, self.ny))      # Vehicle sources
        self.Q_occlusion = np.zeros((self.nx, self.ny))   # Occlusion sources
        self.Phi_merge = np.zeros((self.nx, self.ny))     # Merging potential
        self.D_tensor = np.ones((self.nx, self.ny, 2))    # Anisotropic diffusion
        
        # Velocity field for advection
        self.v_flow = np.zeros((self.nx, self.ny, 2))
        self.v_topo = np.zeros((self.nx, self.ny, 2))
        
        # Predator-prey fields
        self.U_urgency = np.zeros((self.nx, self.ny))  # Merging urgency (prey)
        self.R_hv = np.zeros((self.nx, self.ny))       # HV risk (predator)
        
        # Grid coordinates
        self.x_coords = np.arange(self.nx) * self.dx
        self.y_coords = np.arange(self.ny) * self.dx
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing='ij')
        
    def reset(self):
        """Reset all fields to zero."""
        self.R.fill(0)
        self.R_prev.fill(0)
        self.R_velocity.fill(0)
        self.Q_source.fill(0)
        self.Q_occlusion.fill(0)
        self.Phi_merge.fill(0)
        self.U_urgency.fill(0)
        self.R_hv.fill(0)
        self.D_tensor[:, :, 0] = self.config.D_longitudinal
        self.D_tensor[:, :, 1] = self.config.D_lateral
        
    def setup_merging_potential(self, merge_end_x: float, merge_lane_y: float, lane_width: float = 3.5):
        """
        Set up singular merging potential: Φ(x) = k/(x_end - x)^γ
        
        Creates lateral drift towards mainline as distance to merge point decreases.
        """
        cfg = self.config
        
        for i in range(self.nx):
            x = self.x_coords[i]
            dist_to_end = max(merge_end_x - x, 1.0)  # Prevent singularity
            
            # Singular potential (capped to prevent numerical issues)
            phi_val = min(cfg.k_topo / (dist_to_end ** cfg.gamma), 1000.0)
            
            for j in range(self.ny):
                y = self.y_coords[j]
                
                # Only apply in merge lane region
                if abs(y - merge_lane_y) < lane_width:
                    self.Phi_merge[i, j] = phi_val
                    
                    # Topological drift: rotate longitudinal pressure to lateral
                    # F = -∇Φ, then apply π/2 rotation for steering response
                    if dist_to_end > 2.0:
                        grad_phi = cfg.gamma * cfg.k_topo / (dist_to_end ** (cfg.gamma + 1))
                        self.v_topo[i, j, 1] = -0.01 * grad_phi  # Lateral drift
                        
    def compute_vehicle_source(self, vehicle: Vehicle, source_strength: float = 1.0):
        """
        Compute risk source term for a vehicle using Gaussian distribution.
        """
        sigma_x = vehicle.length / 2
        sigma_y = vehicle.width / 2
        
        # Higher risk for heavy vehicles
        if vehicle.vehicle_class in [VehicleClass.TRUCK, VehicleClass.BUS]:
            source_strength *= 2.0
            sigma_x *= 1.5
            sigma_y *= 1.2
            
        # Compute Gaussian source
        dx = self.X - vehicle.x
        dy = self.Y - vehicle.y
        
        # Rotate to vehicle frame
        cos_h, sin_h = np.cos(vehicle.heading), np.sin(vehicle.heading)
        dx_rot = dx * cos_h + dy * sin_h
        dy_rot = -dx * sin_h + dy * cos_h
        
        source = source_strength * np.exp(
            -0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2)
        )
        
        return source
    
    def compute_occlusion_source(self, occluder: Vehicle, observer: Vehicle) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute occlusion-aware source term with diffraction effects.
        
        Models the shadow zone as a region of latent risk with diffraction
        at the occluder edges (Sommerfeld diffraction analogy).
        """
        cfg = self.config
        
        # Observer position
        obs_x, obs_y = observer.x, observer.y
        
        # Compute tangent angles to occluder corners
        corners = occluder.corners
        angles = []
        for corner in corners:
            dx, dy = corner[0] - obs_x, corner[1] - obs_y
            angles.append(np.arctan2(dy, dx))
        
        # Shadow cone defined by extreme angles
        angle_min, angle_max = min(angles), max(angles)
        
        # Create shadow zone mask and diffraction field
        shadow_mask = np.zeros((self.nx, self.ny))
        diffraction_field = np.zeros((self.nx, self.ny))
        
        dist_to_occluder = np.sqrt((occluder.x - obs_x)**2 + (occluder.y - obs_y)**2)
        
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.x_coords[i], self.y_coords[j]
                dx, dy = x - obs_x, y - obs_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # Only consider points beyond occluder
                if dist < dist_to_occluder + occluder.length/2:
                    continue
                    
                if dist > cfg.shadow_extension:
                    continue
                    
                angle = np.arctan2(dy, dx)
                
                # Check if in shadow cone
                if angle_min <= angle <= angle_max:
                    # Distance-based decay
                    decay = np.exp(-0.01 * (dist - dist_to_occluder))
                    shadow_mask[i, j] = decay
                    
                else:
                    # Diffraction at edges (Fresnel-like decay)
                    angle_diff = min(abs(angle - angle_min), abs(angle - angle_max))
                    if angle_diff < np.pi/6:  # Within diffraction zone
                        fresnel_decay = np.exp(-cfg.diffraction_decay * angle_diff * dist)
                        diffraction_field[i, j] = fresnel_decay
        
        # Combine shadow and diffraction
        occlusion_source = cfg.occlusion_source_strength * (shadow_mask + 0.5 * diffraction_field)
        
        return occlusion_source, shadow_mask
    
    def compute_diffusion_tensor(self, occlusion_mask: np.ndarray):
        """
        Set up anisotropic diffusion tensor with occlusion boost.
        In occluded regions, uncertainty spreads faster.
        """
        cfg = self.config
        
        self.D_tensor[:, :, 0] = cfg.D_longitudinal * (1 + cfg.D_occlusion_boost * occlusion_mask)
        self.D_tensor[:, :, 1] = cfg.D_lateral * (1 + cfg.D_occlusion_boost * occlusion_mask)
        
    def step_telegrapher(self):
        """
        Advance the Telegrapher's equation by one time step.
        
        Uses explicit FDTD scheme:
        R^{n+1} = (2τR^n - (τ - dt/2)R^{n-1} + dt²[∇·(D∇R) - ∇·(vR) + Q - λR]) / (τ + dt/2)
        """
        cfg = self.config
        dt, dx = cfg.dt, cfg.dx
        tau = cfg.tau
        
        # Compute Laplacian with anisotropic diffusion (finite differences)
        laplacian = np.zeros_like(self.R)
        
        # ∂/∂x(D_x ∂R/∂x)
        Dx = self.D_tensor[:, :, 0]
        laplacian[1:-1, :] += (
            Dx[2:, :] * (self.R[2:, :] - self.R[1:-1, :]) -
            Dx[1:-1, :] * (self.R[1:-1, :] - self.R[:-2, :])
        ) / dx**2
        
        # ∂/∂y(D_y ∂R/∂y)  
        Dy = self.D_tensor[:, :, 1]
        laplacian[:, 1:-1] += (
            Dy[:, 2:] * (self.R[:, 2:] - self.R[:, 1:-1]) -
            Dy[:, 1:-1] * (self.R[:, 1:-1] - self.R[:, :-2])
        ) / dx**2
        
        # Advection term: -∇·(v_eff R) using upwind scheme
        v_eff = self.v_flow + self.v_topo
        advection = np.zeros_like(self.R)
        
        # Upwind for x-component
        vx = v_eff[:, :, 0]
        advection[1:-1, :] -= np.where(
            vx[1:-1, :] > 0,
            vx[1:-1, :] * (self.R[1:-1, :] - self.R[:-2, :]) / dx,
            vx[1:-1, :] * (self.R[2:, :] - self.R[1:-1, :]) / dx
        )
        
        # Upwind for y-component
        vy = v_eff[:, :, 1]
        advection[:, 1:-1] -= np.where(
            vy[:, 1:-1] > 0,
            vy[:, 1:-1] * (self.R[:, 1:-1] - self.R[:, :-2]) / dx,
            vy[:, 1:-1] * (self.R[:, 2:] - self.R[:, 1:-1]) / dx
        )
        
        # Total source
        Q_total = self.Q_source + self.Q_occlusion
        
        # Telegrapher's equation time stepping
        coeff1 = 2 * tau
        coeff2 = tau - dt / 2
        coeff3 = tau + dt / 2
        
        R_new = (
            coeff1 * self.R - coeff2 * self.R_prev +
            dt**2 * (laplacian + advection + Q_total - cfg.lambda_decay * self.R)
        ) / coeff3
        
        # Enforce non-negativity
        R_new = np.maximum(R_new, 0)
        
        # Update fields
        self.R_prev = self.R.copy()
        self.R = R_new
        
    def step_predator_prey(self):
        """
        Advance the Predator-Prey (Lotka-Volterra) coupled system.
        
        Prey (U_urgency): Merging vehicle's desire to enter
        Predator (R_hv): Heavy vehicle's space occupation
        
        ∂U/∂t = D₁ΔU + αU - βRU
        ∂R/∂t = D₂ΔR - γR + δUR
        """
        cfg = self.config
        dt = cfg.dt
        
        # Diffusion for prey (urgency)
        laplacian_U = np.zeros_like(self.U_urgency)
        laplacian_U[1:-1, 1:-1] = (
            self.U_urgency[2:, 1:-1] + self.U_urgency[:-2, 1:-1] +
            self.U_urgency[1:-1, 2:] + self.U_urgency[1:-1, :-2] -
            4 * self.U_urgency[1:-1, 1:-1]
        ) / self.dx**2
        
        # Diffusion for predator (HV risk)
        laplacian_R = np.zeros_like(self.R_hv)
        laplacian_R[1:-1, 1:-1] = (
            self.R_hv[2:, 1:-1] + self.R_hv[:-2, 1:-1] +
            self.R_hv[1:-1, 2:] + self.R_hv[1:-1, :-2] -
            4 * self.R_hv[1:-1, 1:-1]
        ) / self.dx**2
        
        # Lotka-Volterra dynamics
        interaction = self.U_urgency * self.R_hv
        
        dU = cfg.D_lateral * laplacian_U + cfg.alpha * self.U_urgency - cfg.beta * interaction
        dR = cfg.D_longitudinal * laplacian_R - cfg.gamma_pred * self.R_hv + cfg.delta * interaction
        
        self.U_urgency = np.maximum(self.U_urgency + dt * dU, 0)
        self.R_hv = np.maximum(self.R_hv + dt * dR, 0)


# =============================================================================
# Scenario Generation
# =============================================================================

def create_sample_scenario() -> Scenario:
    """
    Create a sample highway merging scenario with heavy vehicle occlusion.
    
    Layout:
    - 3-lane highway (mainline)
    - 1 acceleration lane (merge lane)
    - Heavy truck in rightmost mainline lane (ego/occluder)
    - Cars ahead, behind, and in merge lane
    """
    
    # Road geometry
    lane_width = 3.5
    lane_centers = [lane_width * 0.5, lane_width * 1.5, lane_width * 2.5]  # 3 mainline lanes
    merge_lane_y = lane_width * 3.5  # Acceleration lane
    
    vehicles = []
    
    # Heavy Truck (Ego vehicle - occluder)
    truck = Vehicle(
        id=1,
        x=80.0,
        y=lane_centers[2],  # Rightmost mainline lane
        vx=22.0,  # ~80 km/h
        vy=0.0,
        length=16.5,  # Typical truck+trailer
        width=2.5,
        heading=0.0,
        vehicle_class=VehicleClass.TRUCK,
        is_ego=True
    )
    vehicles.append(truck)
    
    # Car behind truck (observer - gets occluded view)
    car_behind = Vehicle(
        id=2,
        x=45.0,
        y=lane_centers[2] - 0.5,  # Slightly offset
        vx=25.0,
        vy=0.0,
        length=4.5,
        width=1.8,
        heading=0.0,
        vehicle_class=VehicleClass.CAR
    )
    vehicles.append(car_behind)
    
    # Car ahead of truck (potentially occluded from car_behind)
    car_ahead = Vehicle(
        id=3,
        x=120.0,
        y=lane_centers[2] + 1.0,
        vx=20.0,
        vy=0.0,
        length=4.5,
        width=1.8,
        heading=0.0,
        vehicle_class=VehicleClass.CAR
    )
    vehicles.append(car_ahead)
    
    # Merging vehicle in acceleration lane
    merging_car = Vehicle(
        id=4,
        x=70.0,
        y=merge_lane_y,
        vx=18.0,
        vy=-0.3,  # Slight lateral motion towards mainline
        length=4.5,
        width=1.8,
        heading=-0.02,
        vehicle_class=VehicleClass.MERGING
    )
    vehicles.append(merging_car)
    
    # Another car in middle lane
    car_middle = Vehicle(
        id=5,
        x=95.0,
        y=lane_centers[1],
        vx=24.0,
        vy=0.0,
        length=4.5,
        width=1.8,
        heading=0.0,
        vehicle_class=VehicleClass.CAR
    )
    vehicles.append(car_middle)
    
    # Car in leftmost lane
    car_left = Vehicle(
        id=6,
        x=60.0,
        y=lane_centers[0],
        vx=26.0,
        vy=0.0,
        length=4.5,
        width=1.8,
        heading=0.0,
        vehicle_class=VehicleClass.CAR
    )
    vehicles.append(car_left)
    
    return Scenario(
        vehicles=vehicles,
        ego_id=1,
        road_bounds=(0, 200, -2, 18),
        lane_centers=lane_centers,
        merge_lane_y=merge_lane_y,
        merge_end_x=170.0,
        frame_id=0,
        recording_id="sample_merge"
    )


def load_scenario_from_json(json_path: str) -> Optional[Scenario]:
    """Load scenario from exported JSON file (from exiD analysis)."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        vehicles = []
        for agent in data.get('agents', []):
            veh = Vehicle(
                id=agent['id'],
                x=agent['x'],
                y=agent['y'],
                vx=agent.get('vx', 0),
                vy=agent.get('vy', 0),
                length=agent.get('length', 4.5),
                width=agent.get('width', 1.8),
                heading=agent.get('heading', 0),
                vehicle_class=VehicleClass.TRUCK if agent.get('class', 'car') in ['truck', 'bus'] else VehicleClass.CAR,
                is_ego=(agent['id'] == data.get('ego_id'))
            )
            vehicles.append(veh)
        
        # Estimate lane centers from vehicle positions
        y_positions = [v.y for v in vehicles]
        lane_centers = sorted(set([round(y / 3.5) * 3.5 + 1.75 for y in y_positions]))
        
        return Scenario(
            vehicles=vehicles,
            ego_id=data.get('ego_id', vehicles[0].id if vehicles else 0),
            road_bounds=(
                min(v.x for v in vehicles) - 30,
                max(v.x for v in vehicles) + 30,
                min(v.y for v in vehicles) - 5,
                max(v.y for v in vehicles) + 10
            ),
            lane_centers=lane_centers,
            merge_lane_y=max(y_positions) + 3.5 if y_positions else 12.0,
            merge_end_x=max(v.x for v in vehicles) + 50 if vehicles else 150.0,
            frame_id=data.get('frame', 0),
            recording_id=data.get('recording_id', 'loaded')
        )
    except Exception as e:
        print(f"Error loading scenario: {e}")
        return None


# =============================================================================
# Comprehensive Risk Field Visualizer
# =============================================================================

class PDERiskFieldVisualizer:
    """
    Comprehensive visualizer for PDE-based risk field modeling.
    
    Provides:
    1. Risk field profile visualization with distance dynamics
    2. Risk field decomposition (source, occlusion, topology, predator-prey)
    3. Wave propagation animation
    4. Comparative analysis views
    """
    
    def __init__(self, config: PDEConfig = None):
        self.config = config or PDEConfig()
        self.solver = TelegrapherPDESolver(self.config)
        self.scenario: Optional[Scenario] = None
        
        # Color maps
        self.cmap_risk = plt.cm.hot
        self.cmap_urgency = plt.cm.YlOrRd
        self.cmap_occlusion = plt.cm.Blues
        self.cmap_topology = plt.cm.Greens
        self.cmap_predprey = plt.cm.RdYlBu_r
        
    def setup_scenario(self, scenario: Scenario):
        """Initialize solver with scenario data."""
        self.scenario = scenario
        self.solver.reset()
        
        # Adjust grid to scenario bounds
        x_min, x_max, y_min, y_max = scenario.road_bounds
        self.solver.nx = int((x_max - x_min) / self.config.dx)
        self.solver.ny = int((y_max - y_min) / self.config.dx)
        
        # Reinitialize arrays
        self.solver.R = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.R_prev = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.Q_source = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.Q_occlusion = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.Phi_merge = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.U_urgency = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.R_hv = np.zeros((self.solver.nx, self.solver.ny))
        self.solver.D_tensor = np.ones((self.solver.nx, self.solver.ny, 2))
        self.solver.D_tensor[:, :, 0] = self.config.D_longitudinal
        self.solver.D_tensor[:, :, 1] = self.config.D_lateral
        self.solver.v_flow = np.zeros((self.solver.nx, self.solver.ny, 2))
        self.solver.v_topo = np.zeros((self.solver.nx, self.solver.ny, 2))
        
        # Update coordinate grids
        self.solver.x_coords = np.linspace(x_min, x_max, self.solver.nx)
        self.solver.y_coords = np.linspace(y_min, y_max, self.solver.ny)
        self.solver.X, self.solver.Y = np.meshgrid(
            self.solver.x_coords, self.solver.y_coords, indexing='ij'
        )
        
        # Set up merging potential
        if scenario.merge_end_x and scenario.merge_lane_y:
            self.solver.setup_merging_potential(
                scenario.merge_end_x,
                scenario.merge_lane_y
            )
        
        # Compute vehicle sources
        ego = None
        observer = None
        
        for vehicle in scenario.vehicles:
            source = self.solver.compute_vehicle_source(vehicle)
            self.solver.Q_source += source
            
            if vehicle.is_ego:
                ego = vehicle
            elif vehicle.id == 2:  # Observer (car behind)
                observer = vehicle
                
            # Initialize predator-prey fields
            if vehicle.vehicle_class in [VehicleClass.TRUCK, VehicleClass.BUS]:
                self.solver.R_hv += source * 2.0
            elif vehicle.vehicle_class == VehicleClass.MERGING:
                urgency_source = self.solver.compute_vehicle_source(vehicle, 1.5)
                self.solver.U_urgency += urgency_source
        
        # Compute occlusion if ego (truck) exists
        if ego and observer:
            occ_source, shadow_mask = self.solver.compute_occlusion_source(ego, observer)
            self.solver.Q_occlusion = occ_source
            self.solver.compute_diffusion_tensor(shadow_mask)
        
        # Run initial time steps to build up field
        for _ in range(50):
            self.solver.step_telegrapher()
            self.solver.step_predator_prey()
            
    def _draw_road(self, ax, bounds, lane_centers, merge_lane_y=None):
        """Draw road with lanes."""
        x_min, x_max, y_min, y_max = bounds
        
        # Road surface
        road_rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor=self.config.road_color, edgecolor='none', zorder=0
        )
        ax.add_patch(road_rect)
        
        # Lane markings
        for y in lane_centers:
            ax.axhline(y - 1.75, color='white', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
            ax.axhline(y + 1.75, color='white', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
        
        # Merge lane
        if merge_lane_y:
            ax.axhline(merge_lane_y - 1.75, color='yellow', linewidth=1, linestyle='-', alpha=0.7, zorder=1)
            # Merge end indicator
            if self.scenario and self.scenario.merge_end_x:
                ax.axvline(self.scenario.merge_end_x, color='red', linewidth=2, linestyle='-', alpha=0.8, zorder=1)
                ax.annotate('Merge End', (self.scenario.merge_end_x, y_max - 1), 
                           fontsize=8, color='red', ha='center')
                
    def _draw_vehicles(self, ax, vehicles):
        """Draw vehicles as rotated rectangles."""
        colors = {
            VehicleClass.CAR: '#3498DB',
            VehicleClass.TRUCK: '#E74C3C',
            VehicleClass.BUS: '#F39C12',
            VehicleClass.MERGING: '#2ECC71'
        }
        
        for vehicle in vehicles:
            corners = vehicle.corners
            color = colors.get(vehicle.vehicle_class, '#95A5A6')
            
            if vehicle.is_ego:
                edgecolor = 'yellow'
                linewidth = 2
            else:
                edgecolor = 'white'
                linewidth = 0.5
                
            poly = Polygon(corners, facecolor=color, edgecolor=edgecolor, 
                          linewidth=linewidth, alpha=0.8, zorder=10)
            ax.add_patch(poly)
            
            # Vehicle ID label
            ax.text(vehicle.x, vehicle.y, str(vehicle.id), 
                   fontsize=7, color='white', ha='center', va='center', 
                   fontweight='bold', zorder=11)
            
            # Direction arrow
            arrow_len = vehicle.length * 0.4
            dx = arrow_len * np.cos(vehicle.heading)
            dy = arrow_len * np.sin(vehicle.heading)
            ax.arrow(vehicle.x, vehicle.y, dx, dy, head_width=0.5, 
                    head_length=0.3, fc='white', ec='white', alpha=0.6, zorder=11)
            
    def _draw_occlusion_zone(self, ax, occluder: Vehicle, observer: Vehicle):
        """Draw the occlusion shadow zone."""
        corners = occluder.corners
        obs_x, obs_y = observer.x, observer.y
        
        # Compute tangent angles
        angles = []
        for corner in corners:
            dx, dy = corner[0] - obs_x, corner[1] - obs_y
            angles.append((np.arctan2(dy, dx), corner))
        angles.sort(key=lambda x: x[0])
        
        # Get extreme tangent points
        left_pt = angles[-1][1]
        right_pt = angles[0][1]
        
        # Extend shadow rays
        ext_length = 120
        left_dir = (left_pt - np.array([obs_x, obs_y]))
        right_dir = (right_pt - np.array([obs_x, obs_y]))
        left_dir = left_dir / np.linalg.norm(left_dir) * ext_length
        right_dir = right_dir / np.linalg.norm(right_dir) * ext_length
        
        left_far = np.array([obs_x, obs_y]) + left_dir
        right_far = np.array([obs_x, obs_y]) + right_dir
        
        # Shadow polygon
        shadow_verts = np.array([right_pt, left_pt, left_far, right_far])
        shadow_poly = Polygon(shadow_verts, facecolor='gray', alpha=0.3, 
                             edgecolor='red', linestyle='--', linewidth=1, zorder=5)
        ax.add_patch(shadow_poly)
        
        # Tangent lines
        ax.plot([obs_x, left_pt[0]], [obs_y, left_pt[1]], 'r--', linewidth=1, alpha=0.7, zorder=6)
        ax.plot([obs_x, right_pt[0]], [obs_y, right_pt[1]], 'r--', linewidth=1, alpha=0.7, zorder=6)
        
        # Observer marker
        ax.plot(obs_x, obs_y, 'ro', markersize=6, zorder=12)
        ax.annotate('Observer', (obs_x, obs_y - 2), fontsize=7, color='red', ha='center')
        
    def plot_combined_risk_field(self, ax=None, title_suffix=""):
        """Plot the combined risk field with all components."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor(self.config.bg_color)
        
        ax.set_facecolor(self.config.bg_color)
        
        if self.scenario is None:
            ax.text(0.5, 0.5, 'No scenario loaded', ha='center', va='center', 
                   color='white', fontsize=14, transform=ax.transAxes)
            return ax
        
        bounds = self.scenario.road_bounds
        x_min, x_max, y_min, y_max = bounds
        
        # Draw road
        self._draw_road(ax, bounds, self.scenario.lane_centers, self.scenario.merge_lane_y)
        
        # Plot risk field
        extent = [self.solver.x_coords[0], self.solver.x_coords[-1],
                  self.solver.y_coords[0], self.solver.y_coords[-1]]
        
        im = ax.imshow(self.solver.R.T, origin='lower', extent=extent,
                      cmap=self.cmap_risk, alpha=0.7, aspect='auto', zorder=2,
                      vmin=0, vmax=np.percentile(self.solver.R, 99))
        
        # Contour lines
        X, Y = np.meshgrid(self.solver.x_coords, self.solver.y_coords)
        levels = np.linspace(0.1, np.max(self.solver.R) * 0.9, 6)
        if len(levels) > 1 and np.max(self.solver.R) > 0.1:
            ax.contour(X, Y, self.solver.R.T, levels=levels, colors='white', 
                      linewidths=0.5, alpha=0.5, zorder=3)
        
        # Draw occlusion zone
        ego = next((v for v in self.scenario.vehicles if v.is_ego), None)
        observer = next((v for v in self.scenario.vehicles if v.id == 2), None)
        if ego and observer:
            self._draw_occlusion_zone(ax, ego, observer)
        
        # Draw vehicles
        self._draw_vehicles(ax, self.scenario.vehicles)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Risk Density R(x,t)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Longitudinal Position (m)', color='white')
        ax.set_ylabel('Lateral Position (m)', color='white')
        ax.set_title(f'Combined Risk Field (Telegrapher Equation) {title_suffix}', 
                    color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
            
        return ax
    
    def plot_risk_field_decomposition(self, fig=None):
        """
        Plot decomposed risk field components:
        1. Source term Q(x,t)
        2. Occlusion-aware risk
        3. Road topology (merging pressure)
        4. Predator-Prey dynamics
        """
        if fig is None:
            fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.config.bg_color)
        
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)
        
        if self.scenario is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No scenario loaded', ha='center', va='center', 
                   color='white', fontsize=14, transform=ax.transAxes)
            return fig
        
        bounds = self.scenario.road_bounds
        x_min, x_max, y_min, y_max = bounds
        extent = [self.solver.x_coords[0], self.solver.x_coords[-1],
                  self.solver.y_coords[0], self.solver.y_coords[-1]]
        
        # 1. Source Term
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.config.bg_color)
        self._draw_road(ax1, bounds, self.scenario.lane_centers, self.scenario.merge_lane_y)
        
        im1 = ax1.imshow(self.solver.Q_source.T, origin='lower', extent=extent,
                        cmap='Reds', alpha=0.8, aspect='auto', zorder=2)
        self._draw_vehicles(ax1, self.scenario.vehicles)
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
        cbar1.set_label('Q_source', color='white', fontsize=9)
        cbar1.ax.tick_params(colors='white', labelsize=8)
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title('(a) Vehicle Source Term Q(x,t)', color='white', fontsize=11, fontweight='bold')
        ax1.set_xlabel('x (m)', color='white', fontsize=9)
        ax1.set_ylabel('y (m)', color='white', fontsize=9)
        ax1.tick_params(colors='white', labelsize=8)
        for spine in ax1.spines.values():
            spine.set_color('#4A4A6A')
        
        # 2. Occlusion-Aware Risk
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.config.bg_color)
        self._draw_road(ax2, bounds, self.scenario.lane_centers, self.scenario.merge_lane_y)
        
        im2 = ax2.imshow(self.solver.Q_occlusion.T, origin='lower', extent=extent,
                        cmap=self.cmap_occlusion, alpha=0.8, aspect='auto', zorder=2)
        
        # Draw diffraction pattern annotation
        ego = next((v for v in self.scenario.vehicles if v.is_ego), None)
        observer = next((v for v in self.scenario.vehicles if v.id == 2), None)
        if ego and observer:
            self._draw_occlusion_zone(ax2, ego, observer)
        self._draw_vehicles(ax2, self.scenario.vehicles)
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
        cbar2.set_label('Q_occlusion (Diffraction)', color='white', fontsize=9)
        cbar2.ax.tick_params(colors='white', labelsize=8)
        
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_title('(b) Occlusion-Aware Risk (Diffractive Barrier)', color='white', fontsize=11, fontweight='bold')
        ax2.set_xlabel('x (m)', color='white', fontsize=9)
        ax2.set_ylabel('y (m)', color='white', fontsize=9)
        ax2.tick_params(colors='white', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#4A4A6A')
        
        # 3. Road Topology (Merging Pressure)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.config.bg_color)
        self._draw_road(ax3, bounds, self.scenario.lane_centers, self.scenario.merge_lane_y)
        
        # Normalize merging potential for visualization
        phi_viz = self.solver.Phi_merge.copy()
        phi_viz = np.clip(phi_viz, 0, np.percentile(phi_viz[phi_viz > 0], 95) if np.any(phi_viz > 0) else 1)
        
        im3 = ax3.imshow(phi_viz.T, origin='lower', extent=extent,
                        cmap=self.cmap_topology, alpha=0.8, aspect='auto', zorder=2)
        
        # Add merge pressure arrows
        if self.scenario.merge_end_x and self.scenario.merge_lane_y:
            for x_pos in np.linspace(x_min + 20, self.scenario.merge_end_x - 10, 6):
                dist = self.scenario.merge_end_x - x_pos
                arrow_scale = min(5, 100 / (dist + 10))
                ax3.annotate('', xy=(x_pos, self.scenario.merge_lane_y - 2),
                           xytext=(x_pos, self.scenario.merge_lane_y),
                           arrowprops=dict(arrowstyle='->', color='lime', lw=1.5),
                           zorder=8)
                           
        self._draw_vehicles(ax3, self.scenario.vehicles)
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.7)
        cbar3.set_label('Φ_merge (Singular Potential)', color='white', fontsize=9)
        cbar3.ax.tick_params(colors='white', labelsize=8)
        
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_title('(c) Road Topology: Merging Pressure Φ(x) ∝ 1/(x_end-x)²', 
                     color='white', fontsize=11, fontweight='bold')
        ax3.set_xlabel('x (m)', color='white', fontsize=9)
        ax3.set_ylabel('y (m)', color='white', fontsize=9)
        ax3.tick_params(colors='white', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_color('#4A4A6A')
        
        # 4. Predator-Prey Dynamics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.config.bg_color)
        self._draw_road(ax4, bounds, self.scenario.lane_centers, self.scenario.merge_lane_y)
        
        # Combine predator (red) and prey (blue) fields
        pred_norm = self.solver.R_hv / (np.max(self.solver.R_hv) + 1e-6)
        prey_norm = self.solver.U_urgency / (np.max(self.solver.U_urgency) + 1e-6)
        
        # Create RGB image: Red=Predator(HV), Blue=Prey(Urgency), Green=Interaction
        rgb_field = np.zeros((self.solver.nx, self.solver.ny, 4))
        rgb_field[:, :, 0] = pred_norm  # Red channel - HV Risk
        rgb_field[:, :, 2] = prey_norm  # Blue channel - Urgency
        rgb_field[:, :, 1] = pred_norm * prey_norm * 2  # Green - Interaction
        rgb_field[:, :, 3] = np.maximum(pred_norm, prey_norm) * 0.8  # Alpha
        
        ax4.imshow(np.transpose(rgb_field, (1, 0, 2)), origin='lower', extent=extent,
                  aspect='auto', zorder=2)
        
        self._draw_vehicles(ax4, self.scenario.vehicles)
        
        # Legend for predator-prey
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='R_hv (Predator: HV Risk)'),
            Patch(facecolor='blue', alpha=0.7, label='U_mv (Prey: Merge Urgency)'),
            Patch(facecolor='yellow', alpha=0.7, label='Interaction Zone')
        ]
        ax4.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                  facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white')
        
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        ax4.set_title('(d) Predator-Prey Dynamics (Lotka-Volterra)', 
                     color='white', fontsize=11, fontweight='bold')
        ax4.set_xlabel('x (m)', color='white', fontsize=9)
        ax4.set_ylabel('y (m)', color='white', fontsize=9)
        ax4.tick_params(colors='white', labelsize=8)
        for spine in ax4.spines.values():
            spine.set_color('#4A4A6A')
        
        fig.suptitle('Risk Field Decomposition: PDE Components', 
                    color='white', fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_distance_based_risk_profile(self, fig=None):
        """
        Plot risk field profile as function of distance:
        1. Distance between observer and HV
        2. Distance between HV and occluded targets
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(self.config.bg_color)
        
        if self.scenario is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No scenario loaded', ha='center', va='center', 
                   color='white', fontsize=14, transform=ax.transAxes)
            return fig
        
        # Find key vehicles
        ego = next((v for v in self.scenario.vehicles if v.is_ego), None)
        observer = next((v for v in self.scenario.vehicles if v.id == 2), None)
        ahead_car = next((v for v in self.scenario.vehicles if v.id == 3), None)
        merging_car = next((v for v in self.scenario.vehicles 
                           if v.vehicle_class == VehicleClass.MERGING), None)
        
        if not ego:
            return fig
            
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. Longitudinal Risk Profile (along mainline)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(self.config.bg_color)
        
        # Extract risk along the truck's lane
        lane_y = ego.y
        y_idx = np.argmin(np.abs(self.solver.y_coords - lane_y))
        risk_profile = self.solver.R[:, y_idx]
        
        ax1.fill_between(self.solver.x_coords, 0, risk_profile, 
                        color='#E74C3C', alpha=0.5, label='Risk R(x)')
        ax1.plot(self.solver.x_coords, risk_profile, 'r-', linewidth=2)
        
        # Mark vehicle positions
        for v in self.scenario.vehicles:
            if abs(v.y - lane_y) < 4:
                ax1.axvline(v.x, color='white', linestyle='--', alpha=0.5)
                marker = 'v' if v.is_ego else 'o'
                color = '#E74C3C' if v.is_ego else '#3498DB'
                ax1.plot(v.x, risk_profile[np.argmin(np.abs(self.solver.x_coords - v.x))],
                        marker, markersize=10, color=color)
                ax1.annotate(f'V{v.id}', (v.x, risk_profile[np.argmin(np.abs(self.solver.x_coords - v.x))] + 0.1),
                           fontsize=9, color='white', ha='center')
        
        # Distance annotations
        if observer and ego:
            d_obs_ego = ego.x - observer.x
            mid_x = (observer.x + ego.x) / 2
            ax1.annotate(f'd₁={d_obs_ego:.1f}m', (mid_x, np.max(risk_profile) * 0.8),
                        fontsize=10, color='cyan', ha='center',
                        bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='cyan'))
        
        if ego and ahead_car:
            d_ego_ahead = ahead_car.x - ego.x
            mid_x = (ego.x + ahead_car.x) / 2
            ax1.annotate(f'd₂={d_ego_ahead:.1f}m', (mid_x, np.max(risk_profile) * 0.6),
                        fontsize=10, color='lime', ha='center',
                        bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='lime'))
        
        ax1.set_xlabel('Longitudinal Position x (m)', color='white', fontsize=11)
        ax1.set_ylabel('Risk Density R(x)', color='white', fontsize=11)
        ax1.set_title('Longitudinal Risk Profile Along Mainline', 
                     color='white', fontsize=12, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white')
        ax1.grid(True, alpha=0.2, color='white')
        for spine in ax1.spines.values():
            spine.set_color('#4A4A6A')
        
        # 2. Risk vs Distance Behind HV
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(self.config.bg_color)
        
        if observer and ego:
            distances_behind = np.linspace(5, 80, 50)
            risks_behind = []
            
            for d in distances_behind:
                x_pos = ego.x - d
                if x_pos > self.solver.x_coords[0]:
                    x_idx = np.argmin(np.abs(self.solver.x_coords - x_pos))
                    risk_val = self.solver.R[x_idx, y_idx]
                else:
                    risk_val = 0
                risks_behind.append(risk_val)
            
            ax2.plot(distances_behind, risks_behind, 'c-', linewidth=2, label='Risk (Observer → HV)')
            ax2.fill_between(distances_behind, 0, risks_behind, color='cyan', alpha=0.3)
            
            # Mark current observer distance
            current_d = ego.x - observer.x
            ax2.axvline(current_d, color='yellow', linestyle='--', linewidth=2, label=f'Current d₁={current_d:.1f}m')
            
            ax2.set_xlabel('Distance Behind HV (m)', color='white', fontsize=11)
            ax2.set_ylabel('Perceived Risk', color='white', fontsize=11)
            ax2.set_title('Risk Profile: Observer → Heavy Vehicle', 
                         color='white', fontsize=11, fontweight='bold')
            ax2.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white', fontsize=9)
            ax2.grid(True, alpha=0.2, color='white')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_color('#4A4A6A')
        
        # 3. Occlusion-Induced Risk vs Distance Ahead of HV
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(self.config.bg_color)
        
        if ego:
            distances_ahead = np.linspace(5, 80, 50)
            occlusion_risks = []
            
            for d in distances_ahead:
                x_pos = ego.x + d
                if x_pos < self.solver.x_coords[-1]:
                    x_idx = np.argmin(np.abs(self.solver.x_coords - x_pos))
                    occ_val = self.solver.Q_occlusion[x_idx, y_idx]
                else:
                    occ_val = 0
                occlusion_risks.append(occ_val)
            
            ax3.plot(distances_ahead, occlusion_risks, 'b-', linewidth=2, label='Occlusion Risk')
            ax3.fill_between(distances_ahead, 0, occlusion_risks, color='blue', alpha=0.3)
            
            # Mark potential occluded target
            if ahead_car:
                target_d = ahead_car.x - ego.x
                ax3.axvline(target_d, color='lime', linestyle='--', linewidth=2, 
                           label=f'Occluded Target d₂={target_d:.1f}m')
            
            # Diffraction decay annotation
            ax3.annotate('Diffraction\nDecay Zone', (60, np.max(occlusion_risks) * 0.3),
                        fontsize=9, color='white', ha='center',
                        bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='blue'))
            
            ax3.set_xlabel('Distance Ahead of HV (m)', color='white', fontsize=11)
            ax3.set_ylabel('Occlusion-Induced Risk', color='white', fontsize=11)
            ax3.set_title('Occlusion Risk: HV → Ahead (Shadow Zone)', 
                         color='white', fontsize=11, fontweight='bold')
            ax3.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white', fontsize=9)
            ax3.grid(True, alpha=0.2, color='white')
            ax3.tick_params(colors='white')
            for spine in ax3.spines.values():
                spine.set_color('#4A4A6A')
        
        fig.suptitle('Distance-Based Risk Field Analysis', 
                    color='white', fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_wave_propagation_analysis(self, fig=None):
        """
        Visualize the wave-like propagation properties of the risk field.
        Shows finite propagation speed (c) vs infinite (parabolic) diffusion.
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(self.config.bg_color)
        
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
        
        # Simulate wave propagation from a point source
        test_solver = TelegrapherPDESolver(self.config)
        test_solver.reset()
        
        # Point source at center
        cx, cy = test_solver.nx // 2, test_solver.ny // 2
        test_solver.R[cx, cy] = 10.0
        test_solver.R_prev = test_solver.R.copy()
        
        # Record snapshots
        snapshots_hyp = []
        for t in range(60):
            test_solver.step_telegrapher()
            if t % 15 == 0:
                snapshots_hyp.append(test_solver.R.copy())
        
        # Compare with pure diffusion (parabolic)
        R_diff = np.zeros((test_solver.nx, test_solver.ny))
        R_diff[cx, cy] = 10.0
        D = self.config.D_longitudinal
        dt = self.config.dt
        dx = self.config.dx
        
        snapshots_par = [R_diff.copy()]
        for t in range(60):
            laplacian = np.zeros_like(R_diff)
            laplacian[1:-1, 1:-1] = (
                R_diff[2:, 1:-1] + R_diff[:-2, 1:-1] +
                R_diff[1:-1, 2:] + R_diff[1:-1, :-2] -
                4 * R_diff[1:-1, 1:-1]
            ) / dx**2
            R_diff = R_diff + dt * D * laplacian
            R_diff = np.maximum(R_diff, 0)
            if t % 15 == 0:
                snapshots_par.append(R_diff.copy())
        
        # Plot comparison
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(self.config.bg_color)
        
        # Radial profiles at different times
        r_coords = np.arange(0, min(cx, cy))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(snapshots_hyp)))
        
        for i, (snap, color) in enumerate(zip(snapshots_hyp, colors)):
            profile = snap[cx, cy:cy+len(r_coords)]
            t_val = i * 15 * self.config.dt
            ax1.plot(r_coords * dx, profile, color=color, linewidth=2, 
                    label=f't={t_val:.2f}s')
            
            # Mark wavefront (finite speed)
            c = self.config.c
            wave_pos = c * t_val
            if wave_pos < r_coords[-1] * dx:
                ax1.axvline(wave_pos, color=color, linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Radial Distance (m)', color='white', fontsize=11)
        ax1.set_ylabel('Risk Density', color='white', fontsize=11)
        ax1.set_title(f'Hyperbolic (Telegrapher): Finite Speed c={self.config.c} m/s', 
                     color='white', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white', fontsize=9)
        ax1.grid(True, alpha=0.2, color='white')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('#4A4A6A')
        
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(self.config.bg_color)
        
        for i, (snap, color) in enumerate(zip(snapshots_par, colors)):
            if i < len(snapshots_par):
                profile = snap[cx, cy:cy+len(r_coords)]
                t_val = i * 15 * self.config.dt
                ax2.plot(r_coords * dx, profile, color=color, linewidth=2,
                        label=f't={t_val:.2f}s')
        
        ax2.set_xlabel('Radial Distance (m)', color='white', fontsize=11)
        ax2.set_ylabel('Risk Density', color='white', fontsize=11)
        ax2.set_title('Parabolic (Diffusion): Infinite Propagation Speed', 
                     color='white', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white', fontsize=9)
        ax2.grid(True, alpha=0.2, color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#4A4A6A')
        
        fig.suptitle('Wave Propagation: Hyperbolic vs Parabolic PDE', 
                    color='white', fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def create_full_analysis(self, scenario: Scenario = None, output_dir: str = './pde_output'):
        """Generate complete analysis with all visualization components."""
        
        if scenario is None:
            scenario = create_sample_scenario()
        
        self.setup_scenario(scenario)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Combined Risk Field
        fig1 = plt.figure(figsize=(14, 8))
        self.plot_combined_risk_field(fig1.add_subplot(111))
        fig1.tight_layout()
        fig1.savefig(output_path / 'combined_risk_field.png', dpi=150, 
                    facecolor=fig1.get_facecolor(), bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Risk Field Decomposition
        fig2 = self.plot_risk_field_decomposition()
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.savefig(output_path / 'risk_field_decomposition.png', dpi=150,
                    facecolor=fig2.get_facecolor(), bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Distance-Based Risk Profile
        fig3 = self.plot_distance_based_risk_profile()
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        fig3.savefig(output_path / 'distance_risk_profile.png', dpi=150,
                    facecolor=fig3.get_facecolor(), bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Wave Propagation Analysis
        fig4 = self.plot_wave_propagation_analysis()
        fig4.tight_layout()
        fig4.savefig(output_path / 'wave_propagation_analysis.png', dpi=150,
                    facecolor=fig4.get_facecolor(), bbox_inches='tight')
        plt.close(fig4)
        
        print(f"Analysis complete. Outputs saved to: {output_path}")
        return output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function to run the PDE risk field visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDE-Based Risk Field Visualization')
    parser.add_argument('--scenario_json', type=str, default=None,
                       help='Path to scenario JSON file (from exiD analysis)')
    parser.add_argument('--output_dir', type=str, default='./pde_output',
                       help='Output directory for visualizations')
    parser.add_argument('--use_sample', action='store_true',
                       help='Use sample scenario instead of loading from file')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    config = PDEConfig()
    visualizer = PDERiskFieldVisualizer(config)
    
    # Load or create scenario
    scenario = None
    if args.scenario_json and not args.use_sample:
        scenario = load_scenario_from_json(args.scenario_json)
        if scenario is None:
            print("Failed to load scenario, using sample...")
            scenario = create_sample_scenario()
    else:
        scenario = create_sample_scenario()
    
    # Run full analysis
    output_path = visualizer.create_full_analysis(scenario, args.output_dir)
    
    print(f"\nPDE Risk Field Analysis Complete!")
    print(f"Scenario: {scenario.recording_id}, Frame: {scenario.frame_id}")
    print(f"Vehicles: {len(scenario.vehicles)}")
    print(f"Outputs: {output_path}")


if __name__ == '__main__':
    main()
