"""
Field Gradient vs Longitudinal Position Visualization
=======================================================
Generates statistical figures showing how risk field gradients change
with longitudinal vehicle position, and how these align with driver
decision-making behaviors (deceleration, lane change, merging, etc.)

Key visualizations:
1. Field gradient profile along longitudinal axis
2. Risk field cross-section at vehicle trajectory
3. Decision-behavior overlay with gradient peaks
4. Statistical correlation analysis
5. Multi-method comparison panels

Based on Wang et al. (2024) insight:
"If the risk field is robust and informative, the critical region 
(moderate → high risk transition) should align with regions where 
drivers actually execute safety-critical decisions."

Author: Research Implementation for exiD Dataset Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import ndimage, signal, stats
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration and Enums
# =============================================================================

class DecisionType(Enum):
    """Types of driver decisions."""
    DECELERATION = 'decel'
    ACCELERATION = 'accel'
    LANE_CHANGE_LEFT = 'lc_left'
    LANE_CHANGE_RIGHT = 'lc_right'
    MERGE = 'merge'
    OVERTAKE = 'overtake'
    YIELD = 'yield'
    STEADY = 'steady'


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    # Grid settings
    longitudinal_range: Tuple[float, float] = (-40, 80)  # meters relative to ego
    lateral_range: Tuple[float, float] = (-15, 15)
    grid_resolution: float = 1.0  # meters
    
    # Decision detection thresholds
    decel_threshold: float = -1.5  # m/s²
    accel_threshold: float = 1.0   # m/s²
    lateral_vel_threshold: float = 0.3  # m/s
    
    # Risk zones (percentiles)
    high_risk_pct: float = 75.0
    moderate_risk_pct: float = 50.0
    low_risk_pct: float = 25.0
    
    # Visual style
    figsize_main: Tuple[int, int] = (18, 14)
    dpi: int = 150
    dark_theme: bool = True


# =============================================================================
# Color Schemes
# =============================================================================

def get_color_scheme(dark: bool = True) -> Dict[str, str]:
    """Get color scheme for visualizations."""
    if dark:
        return {
            'bg': '#1a1a2e',
            'panel': '#16213e',
            'fg': '#e0e0e0',
            'grid': '#394867',
            'spine': '#394867',
            'risk_high': '#e74c3c',
            'risk_moderate': '#f39c12',
            'risk_low': '#27ae60',
            'gradient_high': '#ff6b6b',
            'gradient_low': '#4ecdc4',
            'decision_decel': '#e74c3c',
            'decision_accel': '#27ae60',
            'decision_lc': '#3498db',
            'decision_merge': '#9b59b6',
            'ego_vehicle': '#00ff88',
            'other_vehicle': '#ff9500'
        }
    else:
        return {
            'bg': '#ffffff',
            'panel': '#f8f9fa',
            'fg': '#2c3e50',
            'grid': '#bdc3c7',
            'spine': '#7f8c8d',
            'risk_high': '#c0392b',
            'risk_moderate': '#d35400',
            'risk_low': '#27ae60',
            'gradient_high': '#e74c3c',
            'gradient_low': '#2980b9',
            'decision_decel': '#c0392b',
            'decision_accel': '#27ae60',
            'decision_lc': '#2980b9',
            'decision_merge': '#8e44ad',
            'ego_vehicle': '#16a085',
            'other_vehicle': '#e67e22'
        }


def get_risk_colormap():
    """Create custom risk field colormap."""
    colors = [
        (0.0, '#2c3e50'),   # Dark blue (low)
        (0.25, '#27ae60'),  # Green
        (0.5, '#f1c40f'),   # Yellow
        (0.75, '#e67e22'),  # Orange
        (1.0, '#c0392b')    # Red (high)
    ]
    positions = [c[0] for c in colors]
    color_list = [c[1] for c in colors]
    return LinearSegmentedColormap.from_list('risk', list(zip(positions, color_list)))


# =============================================================================
# Risk Field Construction (Multiple Methods)
# =============================================================================

class RiskFieldBuilder:
    """Construct risk fields using different methods."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def build_grid(self, ego_x: float = 0, ego_y: float = 0):
        """Build coordinate grid centered on ego."""
        x_min = ego_x + self.config.longitudinal_range[0]
        x_max = ego_x + self.config.longitudinal_range[1]
        y_min = ego_y + self.config.lateral_range[0]
        y_max = ego_y + self.config.lateral_range[1]
        
        nx = int((x_max - x_min) / self.config.grid_resolution) + 1
        ny = int((y_max - y_min) / self.config.grid_resolution) + 1
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        return X, Y, x, y
    
    def gvf_field(self, X: np.ndarray, Y: np.ndarray, 
                  ego: Dict, others: List[Dict]) -> np.ndarray:
        """Gaussian Velocity Field method."""
        R = np.zeros_like(X)
        
        for veh in others:
            dx = X - veh['x']
            dy = Y - veh['y']
            
            # Speed-dependent spread
            speed = veh.get('speed', 10.0)
            sigma_x = 5.0 + 0.5 * speed
            sigma_y = 2.0 + 0.1 * speed
            
            # Heading rotation
            heading = veh.get('heading', 0.0)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx_rot = dx * cos_h + dy * sin_h
            dy_rot = -dx * sin_h + dy * cos_h
            
            # Gaussian contribution
            R += np.exp(-0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2))
        
        return np.clip(R / (R.max() + 1e-10), 0, 1)
    
    def edrf_field(self, X: np.ndarray, Y: np.ndarray,
                   ego: Dict, others: List[Dict]) -> np.ndarray:
        """Elliptic Driving Risk Field method."""
        R = np.zeros_like(X)
        
        for veh in others:
            dx = X - veh['x']
            dy = Y - veh['y']
            
            # Ellipse parameters based on vehicle dimensions
            length = veh.get('length', 4.5)
            width = veh.get('width', 1.8)
            speed = veh.get('speed', 10.0)
            
            # Semi-axes (extend with speed)
            a = length / 2 + 0.3 * speed
            b = width / 2 + 0.1 * speed
            
            heading = veh.get('heading', 0.0)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx_rot = dx * cos_h + dy * sin_h
            dy_rot = -dx * sin_h + dy * cos_h
            
            # Elliptic distance
            d_ell = np.sqrt((dx_rot / a)**2 + (dy_rot / b)**2)
            R += np.exp(-d_ell)
        
        return np.clip(R / (R.max() + 1e-10), 0, 1)
    
    def ada_field(self, X: np.ndarray, Y: np.ndarray,
                  ego: Dict, others: List[Dict]) -> np.ndarray:
        """Asymmetric Driving Aggressiveness field."""
        R = np.zeros_like(X)
        
        for veh in others:
            dx = X - veh['x']
            dy = Y - veh['y']
            
            speed = veh.get('speed', 10.0)
            vx = veh.get('vx', speed)
            vy = veh.get('vy', 0.0)
            
            # Asymmetric extension (larger ahead of vehicle)
            heading = veh.get('heading', 0.0)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx_rot = dx * cos_h + dy * sin_h
            dy_rot = -dx * sin_h + dy * cos_h
            
            # Front/rear asymmetry
            sigma_front = 8.0 + 0.6 * speed
            sigma_rear = 3.0
            sigma_x = np.where(dx_rot > 0, sigma_front, sigma_rear)
            sigma_y = 2.5
            
            R += np.exp(-0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2))
        
        return np.clip(R / (R.max() + 1e-10), 0, 1)
    
    def apf_wang_field(self, X: np.ndarray, Y: np.ndarray,
                       ego: Dict, others: List[Dict]) -> np.ndarray:
        """APF-based field from Wang et al. (2024)."""
        R = np.zeros_like(X)
        
        ego_speed = ego.get('speed', 15.0)
        
        for veh in others:
            dx = X - veh['x']
            dy = Y - veh['y']
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            
            # Relative velocity
            rel_vx = ego.get('vx', ego_speed) - veh.get('vx', veh.get('speed', 10.0))
            rel_vy = ego.get('vy', 0.0) - veh.get('vy', 0.0)
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
            
            # Mass ratio (for heterogeneous vehicles)
            mass = veh.get('mass', 1500)
            ego_mass = ego.get('mass', 15000)
            mass_ratio = mass / ego_mass
            
            # Distance-based risk with velocity influence
            heading = veh.get('heading', 0.0)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            dx_rot = dx * cos_h + dy * sin_h
            dy_rot = -dx * sin_h + dy * cos_h
            
            # Anisotropic scaling
            sigma_x = 10.0 + 0.4 * rel_speed
            sigma_y = 3.0
            
            potential = mass_ratio * np.exp(-0.5 * ((dx_rot/sigma_x)**2 + (dy_rot/sigma_y)**2))
            R += potential
        
        return np.clip(R / (R.max() + 1e-10), 0, 1)
    
    def build_all_methods(self, ego: Dict, others: List[Dict]) -> Dict[str, np.ndarray]:
        """Build fields for all methods."""
        X, Y, x, y = self.build_grid(ego.get('x', 0), ego.get('y', 0))
        
        fields = {
            'gvf': self.gvf_field(X, Y, ego, others),
            'edrf': self.edrf_field(X, Y, ego, others),
            'ada': self.ada_field(X, Y, ego, others),
            'apf': self.apf_wang_field(X, Y, ego, others)
        }
        
        return fields, X, Y, x, y


# =============================================================================
# Decision Detection
# =============================================================================

class DecisionDetector:
    """Detect driver decisions from trajectory data."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def detect_decisions(self, 
                        timestamps: np.ndarray,
                        positions_x: np.ndarray,
                        positions_y: np.ndarray,
                        velocities_x: np.ndarray,
                        velocities_y: np.ndarray,
                        accelerations_x: np.ndarray,
                        accelerations_y: np.ndarray) -> List[Dict]:
        """Detect decision points along trajectory."""
        decisions = []
        n = len(timestamps)
        
        for i in range(1, n - 1):
            ax = accelerations_x[i]
            ay = accelerations_y[i]
            vy = velocities_y[i]
            
            decision = {
                'time': timestamps[i],
                'x': positions_x[i],
                'y': positions_y[i],
                'type': DecisionType.STEADY,
                'intensity': 0.0
            }
            
            # Deceleration detection
            if ax < self.config.decel_threshold:
                decision['type'] = DecisionType.DECELERATION
                decision['intensity'] = abs(ax / self.config.decel_threshold)
            
            # Acceleration detection
            elif ax > self.config.accel_threshold:
                decision['type'] = DecisionType.ACCELERATION
                decision['intensity'] = ax / self.config.accel_threshold
            
            # Lane change detection
            elif abs(vy) > self.config.lateral_vel_threshold:
                if vy > 0:
                    decision['type'] = DecisionType.LANE_CHANGE_LEFT
                else:
                    decision['type'] = DecisionType.LANE_CHANGE_RIGHT
                decision['intensity'] = abs(vy / self.config.lateral_vel_threshold)
            
            if decision['type'] != DecisionType.STEADY:
                decisions.append(decision)
        
        return decisions
    
    def get_decision_regions(self, decisions: List[Dict], 
                            merge_threshold: float = 5.0) -> List[Dict]:
        """Merge nearby decisions into regions."""
        if not decisions:
            return []
        
        # Sort by x position
        sorted_decisions = sorted(decisions, key=lambda d: d['x'])
        
        regions = []
        current_region = {
            'x_start': sorted_decisions[0]['x'],
            'x_end': sorted_decisions[0]['x'],
            'types': [sorted_decisions[0]['type']],
            'intensities': [sorted_decisions[0]['intensity']],
            'decisions': [sorted_decisions[0]]
        }
        
        for d in sorted_decisions[1:]:
            if d['x'] - current_region['x_end'] < merge_threshold:
                current_region['x_end'] = d['x']
                current_region['types'].append(d['type'])
                current_region['intensities'].append(d['intensity'])
                current_region['decisions'].append(d)
            else:
                regions.append(current_region)
                current_region = {
                    'x_start': d['x'],
                    'x_end': d['x'],
                    'types': [d['type']],
                    'intensities': [d['intensity']],
                    'decisions': [d]
                }
        
        regions.append(current_region)
        
        # Compute dominant type and mean intensity for each region
        for region in regions:
            type_counts = {}
            for t in region['types']:
                type_counts[t] = type_counts.get(t, 0) + 1
            region['dominant_type'] = max(type_counts.keys(), key=lambda k: type_counts[k])
            region['mean_intensity'] = np.mean(region['intensities'])
            region['x_center'] = (region['x_start'] + region['x_end']) / 2
        
        return regions


# =============================================================================
# Longitudinal Analysis
# =============================================================================

class LongitudinalAnalyzer:
    """Analyze field along longitudinal axis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def extract_longitudinal_profile(self, 
                                     risk_field: np.ndarray,
                                     X: np.ndarray, Y: np.ndarray,
                                     y_slice: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract risk profile along longitudinal axis at given y."""
        y_idx = np.argmin(np.abs(Y[:, 0] - y_slice))
        x_vals = X[y_idx, :]
        risk_profile = risk_field[y_idx, :]
        return x_vals, risk_profile
    
    def compute_gradient_profile(self,
                                 risk_field: np.ndarray,
                                 X: np.ndarray, Y: np.ndarray,
                                 dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradient magnitude along longitudinal axis."""
        grad_y, grad_x = np.gradient(risk_field, dx, dx)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract center profile
        y_idx = grad_magnitude.shape[0] // 2
        x_vals = X[y_idx, :]
        grad_profile = grad_magnitude[y_idx, :]
        risk_profile = risk_field[y_idx, :]
        
        return x_vals, grad_profile, risk_profile
    
    def find_gradient_peaks(self, 
                           x_vals: np.ndarray, 
                           grad_profile: np.ndarray,
                           min_prominence: float = 0.1) -> List[Dict]:
        """Find peaks in gradient profile."""
        peaks, properties = signal.find_peaks(
            grad_profile, 
            prominence=min_prominence * grad_profile.max(),
            width=2
        )
        
        peak_info = []
        for i, peak in enumerate(peaks):
            peak_info.append({
                'x': x_vals[peak],
                'gradient': grad_profile[peak],
                'prominence': properties['prominences'][i] if 'prominences' in properties else 0,
                'width': properties['widths'][i] if 'widths' in properties else 0
            })
        
        return sorted(peak_info, key=lambda p: p['gradient'], reverse=True)
    
    def compute_alignment_statistics(self,
                                    gradient_peaks: List[Dict],
                                    decision_regions: List[Dict],
                                    alignment_threshold: float = 10.0) -> Dict:
        """Compute alignment between gradient peaks and decision regions."""
        if not gradient_peaks or not decision_regions:
            return {
                'alignment_score': 0.0,
                'matched_peaks': 0,
                'total_peaks': len(gradient_peaks),
                'matched_regions': 0,
                'total_regions': len(decision_regions),
                'mean_distance': float('inf'),
                'correlations': []
            }
        
        matched_peaks = 0
        matched_regions = 0
        distances = []
        correlations = []
        
        for peak in gradient_peaks:
            for region in decision_regions:
                dist = abs(peak['x'] - region['x_center'])
                if dist < alignment_threshold:
                    matched_peaks += 1
                    distances.append(dist)
                    correlations.append({
                        'peak_x': peak['x'],
                        'peak_gradient': peak['gradient'],
                        'region_x': region['x_center'],
                        'region_type': region['dominant_type'],
                        'region_intensity': region['mean_intensity'],
                        'distance': dist
                    })
                    break
        
        for region in decision_regions:
            for peak in gradient_peaks:
                if abs(peak['x'] - region['x_center']) < alignment_threshold:
                    matched_regions += 1
                    break
        
        alignment_score = 0.0
        if gradient_peaks and decision_regions:
            precision = matched_peaks / len(gradient_peaks) if gradient_peaks else 0
            recall = matched_regions / len(decision_regions) if decision_regions else 0
            if precision + recall > 0:
                alignment_score = 2 * precision * recall / (precision + recall)
        
        return {
            'alignment_score': alignment_score,
            'matched_peaks': matched_peaks,
            'total_peaks': len(gradient_peaks),
            'matched_regions': matched_regions,
            'total_regions': len(decision_regions),
            'mean_distance': np.mean(distances) if distances else float('inf'),
            'correlations': correlations,
            'precision': matched_peaks / len(gradient_peaks) if gradient_peaks else 0,
            'recall': matched_regions / len(decision_regions) if decision_regions else 0
        }


# =============================================================================
# Main Visualization Class
# =============================================================================

class FieldLongitudinalVisualizer:
    """Main visualization class for field-longitudinal analysis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.colors = get_color_scheme(self.config.dark_theme)
        self.risk_cmap = get_risk_colormap()
        self.field_builder = RiskFieldBuilder(self.config)
        self.decision_detector = DecisionDetector(self.config)
        self.longitudinal_analyzer = LongitudinalAnalyzer(self.config)
    
    def create_main_figure(self,
                          ego: Dict,
                          others: List[Dict],
                          trajectory: Dict = None,
                          methods: List[str] = None,
                          title: str = "Field Gradient vs Longitudinal Position Analysis") -> plt.Figure:
        """Create comprehensive visualization figure."""
        if methods is None:
            methods = ['gvf', 'edrf', 'ada', 'apf']
        
        # Build all risk fields
        fields, X, Y, x, y = self.field_builder.build_all_methods(ego, others)
        
        # Extract trajectory decisions if available
        decisions = []
        decision_regions = []
        if trajectory is not None:
            decisions = self.decision_detector.detect_decisions(
                trajectory['timestamps'],
                trajectory['positions_x'],
                trajectory['positions_y'],
                trajectory['velocities_x'],
                trajectory['velocities_y'],
                trajectory['accelerations_x'],
                trajectory['accelerations_y']
            )
            decision_regions = self.decision_detector.get_decision_regions(decisions)
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize_main, facecolor=self.colors['bg'])
        
        # Layout: 4 rows
        # Row 1: Risk field heatmaps (4 panels)
        # Row 2: Gradient profiles along x-axis
        # Row 3: Risk profiles with decision overlays
        # Row 4: Statistics and alignment analysis
        
        gs = GridSpec(4, 4, figure=fig, height_ratios=[1.2, 0.8, 0.8, 1.0],
                     hspace=0.35, wspace=0.25)
        
        # Row 1: Risk field heatmaps
        axes_heatmap = []
        for i, method in enumerate(methods[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor(self.colors['panel'])
            self._plot_risk_heatmap(ax, fields[method], X, Y, ego, others, method.upper())
            axes_heatmap.append(ax)
        
        # Row 2: Gradient profiles
        ax_gradient = fig.add_subplot(gs[1, :])
        ax_gradient.set_facecolor(self.colors['panel'])
        self._plot_gradient_profiles(ax_gradient, fields, X, Y, methods, decision_regions)
        
        # Row 3: Risk profiles with decisions
        ax_risk = fig.add_subplot(gs[2, :])
        ax_risk.set_facecolor(self.colors['panel'])
        self._plot_risk_profiles(ax_risk, fields, X, Y, methods, decision_regions)
        
        # Row 4: Statistics panels
        ax_stats_left = fig.add_subplot(gs[3, :2])
        ax_stats_left.set_facecolor(self.colors['panel'])
        
        ax_stats_right = fig.add_subplot(gs[3, 2:])
        ax_stats_right.set_facecolor(self.colors['panel'])
        
        self._plot_alignment_statistics(ax_stats_left, ax_stats_right, 
                                        fields, X, Y, methods, decision_regions)
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', 
                    color=self.colors['fg'], y=0.98)
        
        # Apply styling to all axes
        for ax in fig.get_axes():
            for spine in ax.spines.values():
                spine.set_color(self.colors['spine'])
        
        return fig
    
    def _plot_risk_heatmap(self, ax, field, X, Y, ego, others, title):
        """Plot risk field heatmap."""
        im = ax.pcolormesh(X, Y, field, cmap=self.risk_cmap, shading='auto')
        
        # Plot vehicles
        ego_rect = plt.Rectangle(
            (ego['x'] - ego.get('length', 4)/2, ego['y'] - ego.get('width', 2)/2),
            ego.get('length', 4), ego.get('width', 2),
            angle=np.degrees(ego.get('heading', 0)),
            facecolor=self.colors['ego_vehicle'],
            edgecolor='white',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(ego_rect)
        
        for veh in others:
            rect = plt.Rectangle(
                (veh['x'] - veh.get('length', 4)/2, veh['y'] - veh.get('width', 2)/2),
                veh.get('length', 4), veh.get('width', 2),
                angle=np.degrees(veh.get('heading', 0)),
                facecolor=self.colors['other_vehicle'],
                edgecolor='white',
                linewidth=1,
                alpha=0.7
            )
            ax.add_patch(rect)
        
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.colors['fg'])
        ax.set_xlabel('Longitudinal (m)', fontsize=9, color=self.colors['fg'])
        ax.set_ylabel('Lateral (m)', fontsize=9, color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'], labelsize=8)
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Risk', fontsize=8, color=self.colors['fg'])
        cbar.ax.tick_params(colors=self.colors['fg'], labelsize=7)
    
    def _plot_gradient_profiles(self, ax, fields, X, Y, methods, decision_regions):
        """Plot gradient profiles along longitudinal axis."""
        method_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']
        
        for i, method in enumerate(methods[:4]):
            x_vals, grad_profile, _ = self.longitudinal_analyzer.compute_gradient_profile(
                fields[method], X, Y
            )
            ax.plot(x_vals, grad_profile, color=method_colors[i], 
                   linewidth=2, label=method.upper(), alpha=0.8)
        
        # Overlay decision regions
        ymin, ymax = ax.get_ylim()
        for region in decision_regions:
            color = self._get_decision_color(region['dominant_type'])
            ax.axvspan(region['x_start'], region['x_end'], 
                      alpha=0.2, color=color, zorder=0)
            ax.axvline(region['x_center'], color=color, 
                      linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Longitudinal Position (m)', fontsize=10, color=self.colors['fg'])
        ax.set_ylabel('Gradient Magnitude', fontsize=10, color=self.colors['fg'])
        ax.set_title('Field Gradient vs Longitudinal Position', 
                    fontsize=12, fontweight='bold', color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'], labelsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
    
    def _plot_risk_profiles(self, ax, fields, X, Y, methods, decision_regions):
        """Plot risk profiles with decision overlays."""
        method_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']
        
        for i, method in enumerate(methods[:4]):
            x_vals, risk_profile = self.longitudinal_analyzer.extract_longitudinal_profile(
                fields[method], X, Y, y_slice=0.0
            )
            ax.plot(x_vals, risk_profile, color=method_colors[i],
                   linewidth=2, label=method.upper(), alpha=0.8)
        
        # Risk zone thresholds
        ax.axhline(0.75, color=self.colors['risk_high'], linestyle=':', 
                  linewidth=1.5, alpha=0.7, label='High Risk Threshold')
        ax.axhline(0.5, color=self.colors['risk_moderate'], linestyle=':',
                  linewidth=1.5, alpha=0.7, label='Moderate Threshold')
        
        # Decision markers
        for region in decision_regions:
            color = self._get_decision_color(region['dominant_type'])
            marker = self._get_decision_marker(region['dominant_type'])
            ax.scatter([region['x_center']], [0.1], marker=marker, 
                      s=150, color=color, edgecolor='white', linewidth=1.5,
                      zorder=5, label=region['dominant_type'].value if len(decision_regions) <= 5 else None)
        
        ax.set_xlabel('Longitudinal Position (m)', fontsize=10, color=self.colors['fg'])
        ax.set_ylabel('Risk Value', fontsize=10, color=self.colors['fg'])
        ax.set_title('Risk Profile with Decision Points', 
                    fontsize=12, fontweight='bold', color=self.colors['fg'])
        ax.tick_params(colors=self.colors['fg'], labelsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
    
    def _plot_alignment_statistics(self, ax_left, ax_right, fields, X, Y, 
                                   methods, decision_regions):
        """Plot alignment statistics."""
        # Left panel: Bar chart of alignment scores
        scores = []
        labels = []
        
        for method in methods[:4]:
            x_vals, grad_profile, _ = self.longitudinal_analyzer.compute_gradient_profile(
                fields[method], X, Y
            )
            gradient_peaks = self.longitudinal_analyzer.find_gradient_peaks(x_vals, grad_profile)
            stats = self.longitudinal_analyzer.compute_alignment_statistics(
                gradient_peaks, decision_regions
            )
            scores.append(stats['alignment_score'])
            labels.append(method.upper())
        
        method_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']
        bars = ax_left.bar(labels, scores, color=method_colors[:len(labels)], 
                          edgecolor='white', linewidth=1.5)
        
        for bar, score in zip(bars, scores):
            ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=self.colors['fg'])
        
        ax_left.set_ylabel('Alignment Score (F1)', fontsize=10, color=self.colors['fg'])
        ax_left.set_title('Gradient-Decision Alignment', 
                         fontsize=12, fontweight='bold', color=self.colors['fg'])
        ax_left.tick_params(colors=self.colors['fg'], labelsize=9)
        ax_left.set_ylim(0, 1.1)
        ax_left.grid(True, alpha=0.3, axis='y', color=self.colors['grid'])
        
        # Right panel: Statistics summary
        ax_right.axis('off')
        
        summary_lines = [
            "=" * 40,
            "ALIGNMENT ANALYSIS SUMMARY",
            "=" * 40,
            ""
        ]
        
        for i, method in enumerate(methods[:4]):
            x_vals, grad_profile, _ = self.longitudinal_analyzer.compute_gradient_profile(
                fields[method], X, Y
            )
            gradient_peaks = self.longitudinal_analyzer.find_gradient_peaks(x_vals, grad_profile)
            stats = self.longitudinal_analyzer.compute_alignment_statistics(
                gradient_peaks, decision_regions
            )
            
            summary_lines.extend([
                f">>> {method.upper()} <<<",
                f"  Alignment Score: {stats['alignment_score']:.4f}",
                f"  Precision: {stats['precision']:.4f}",
                f"  Recall: {stats['recall']:.4f}",
                f"  Gradient Peaks: {stats['total_peaks']}",
                f"  Decision Regions: {stats['total_regions']}",
                f"  Mean Distance: {stats['mean_distance']:.2f}m",
                ""
            ])
        
        summary_text = "\n".join(summary_lines)
        ax_right.text(0.05, 0.95, summary_text, transform=ax_right.transAxes,
                     fontsize=9, color=self.colors['fg'], family='monospace',
                     verticalalignment='top')
    
    def _get_decision_color(self, decision_type: DecisionType) -> str:
        """Get color for decision type."""
        color_map = {
            DecisionType.DECELERATION: self.colors['decision_decel'],
            DecisionType.ACCELERATION: self.colors['decision_accel'],
            DecisionType.LANE_CHANGE_LEFT: self.colors['decision_lc'],
            DecisionType.LANE_CHANGE_RIGHT: self.colors['decision_lc'],
            DecisionType.MERGE: self.colors['decision_merge'],
            DecisionType.OVERTAKE: self.colors['decision_merge'],
            DecisionType.YIELD: self.colors['decision_decel'],
            DecisionType.STEADY: self.colors['fg']
        }
        return color_map.get(decision_type, self.colors['fg'])
    
    def _get_decision_marker(self, decision_type: DecisionType) -> str:
        """Get marker for decision type."""
        marker_map = {
            DecisionType.DECELERATION: 'v',
            DecisionType.ACCELERATION: '^',
            DecisionType.LANE_CHANGE_LEFT: '<',
            DecisionType.LANE_CHANGE_RIGHT: '>',
            DecisionType.MERGE: 's',
            DecisionType.OVERTAKE: 'D',
            DecisionType.YIELD: 'o',
            DecisionType.STEADY: '.'
        }
        return marker_map.get(decision_type, 'o')
    
    def create_comparison_figure(self,
                                scenarios: List[Dict],
                                methods: List[str] = None,
                                title: str = "Multi-Scenario Comparison") -> plt.Figure:
        """Create multi-scenario comparison figure."""
        if methods is None:
            methods = ['gvf', 'edrf', 'ada', 'apf']
        
        n_scenarios = len(scenarios)
        fig = plt.figure(figsize=(16, 4 * n_scenarios), facecolor=self.colors['bg'])
        
        gs = GridSpec(n_scenarios, 4, figure=fig, hspace=0.4, wspace=0.25)
        
        for s_idx, scenario in enumerate(scenarios):
            ego = scenario['ego']
            others = scenario['surrounding']
            
            fields, X, Y, x, y = self.field_builder.build_all_methods(ego, others)
            
            for m_idx, method in enumerate(methods[:4]):
                ax = fig.add_subplot(gs[s_idx, m_idx])
                ax.set_facecolor(self.colors['panel'])
                
                x_vals, grad_profile, risk_profile = \
                    self.longitudinal_analyzer.compute_gradient_profile(fields[method], X, Y)
                
                # Twin axis for risk and gradient
                ax.plot(x_vals, risk_profile, color='#3498db', 
                       linewidth=2, label='Risk')
                ax.set_ylabel('Risk', color='#3498db', fontsize=9)
                ax.tick_params(axis='y', labelcolor='#3498db')
                
                ax2 = ax.twinx()
                ax2.plot(x_vals, grad_profile, color='#e74c3c',
                        linewidth=2, label='Gradient', linestyle='--')
                ax2.set_ylabel('Gradient', color='#e74c3c', fontsize=9)
                ax2.tick_params(axis='y', labelcolor='#e74c3c')
                
                if s_idx == 0:
                    ax.set_title(f'{method.upper()}', fontsize=11, 
                               fontweight='bold', color=self.colors['fg'])
                
                if m_idx == 0:
                    ax.text(-0.15, 0.5, f"Scenario {s_idx + 1}",
                           transform=ax.transAxes, fontsize=10,
                           fontweight='bold', color=self.colors['fg'],
                           rotation=90, va='center')
                
                if s_idx == n_scenarios - 1:
                    ax.set_xlabel('Longitudinal (m)', fontsize=9, color=self.colors['fg'])
                
                ax.tick_params(colors=self.colors['fg'], labelsize=8)
                ax.grid(True, alpha=0.3, color=self.colors['grid'])
                
                for spine in ax.spines.values():
                    spine.set_color(self.colors['spine'])
        
        fig.suptitle(title, fontsize=14, fontweight='bold',
                    color=self.colors['fg'], y=0.99)
        
        return fig


# =============================================================================
# Demo Data Generation
# =============================================================================

def create_demo_scenario() -> Dict:
    """Create demo highway merging scenario."""
    ego = {
        'id': 0,
        'x': 0.0,
        'y': 0.0,
        'vx': 20.0,
        'vy': 0.0,
        'speed': 20.0,
        'heading': 0.0,
        'length': 12.0,
        'width': 2.5,
        'mass': 15000,
        'class': 'truck'
    }
    
    others = [
        # Leading car
        {'id': 1, 'x': 35.0, 'y': 0.0, 'vx': 18.0, 'vy': 0.0, 'speed': 18.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Merging car from right
        {'id': 2, 'x': 25.0, 'y': 8.0, 'vx': 22.0, 'vy': -2.0, 'speed': 22.1,
         'heading': -0.09, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Adjacent truck
        {'id': 3, 'x': 15.0, 'y': 3.5, 'vx': 19.0, 'vy': 0.0, 'speed': 19.0,
         'heading': 0.0, 'length': 10.0, 'width': 2.5, 'mass': 12000, 'class': 'truck'},
        # Following car
        {'id': 4, 'x': -20.0, 'y': 0.0, 'vx': 22.0, 'vy': 0.0, 'speed': 22.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
        # Car in left lane
        {'id': 5, 'x': 10.0, 'y': -3.5, 'vx': 25.0, 'vy': 0.0, 'speed': 25.0,
         'heading': 0.0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
    ]
    
    return {'ego': ego, 'surrounding': others}


def create_demo_trajectory() -> Dict:
    """Create demo trajectory with decision points."""
    n = 200
    dt = 0.1
    
    timestamps = np.arange(n) * dt
    
    # Position: start at ego position, move forward
    positions_x = np.cumsum(np.ones(n) * 20 * dt)
    positions_y = np.zeros(n)
    
    # Add lane change between t=10-14s
    lc_start, lc_end = 100, 140
    positions_y[lc_start:lc_end] = np.linspace(0, 3.5, lc_end - lc_start)
    positions_y[lc_end:] = 3.5
    
    # Velocities
    velocities_x = np.gradient(positions_x, dt)
    velocities_y = np.gradient(positions_y, dt)
    
    # Add deceleration at t=5-7s
    velocities_x[50:70] -= np.linspace(0, 4, 20)
    velocities_x[70:100] = velocities_x[69]
    velocities_x[100:] += np.linspace(0, 2, n - 100)
    
    # Accelerations
    accelerations_x = np.gradient(velocities_x, dt)
    accelerations_y = np.gradient(velocities_y, dt)
    
    return {
        'timestamps': timestamps,
        'positions_x': positions_x,
        'positions_y': positions_y,
        'velocities_x': velocities_x,
        'velocities_y': velocities_y,
        'accelerations_x': accelerations_x,
        'accelerations_y': accelerations_y
    }


# =============================================================================
# Main Entry
# =============================================================================

def main():
    """Run demo visualization."""
    print("=" * 60)
    print("Field Gradient vs Longitudinal Position Analysis")
    print("=" * 60)
    
    # Create demo data
    scenario = create_demo_scenario()
    trajectory = create_demo_trajectory()
    
    print(f"\nScenario: Ego={scenario['ego']['class']}, "
          f"{len(scenario['surrounding'])} surrounding vehicles")
    print(f"Trajectory: {len(trajectory['timestamps'])} samples")
    
    # Create visualizer
    config = VisualizationConfig(dark_theme=True)
    visualizer = FieldLongitudinalVisualizer(config)
    
    # Generate main figure
    print("\nGenerating main visualization...")
    fig = visualizer.create_main_figure(
        ego=scenario['ego'],
        others=scenario['surrounding'],
        trajectory=trajectory,
        methods=['gvf', 'edrf', 'ada', 'apf'],
        title="Field Gradient vs Longitudinal Position - Highway Merging Scenario"
    )
    
    # Save figure
    output_path = '/mnt/user-data/outputs/field_longitudinal_analysis.png'
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
               bbox_inches='tight', pad_inches=0.2)
    print(f"\nSaved: {output_path}")
    
    # Create multi-scenario comparison
    print("\nGenerating scenario comparison...")
    scenarios = [
        create_demo_scenario(),
        # Create a second scenario with different configuration
        {
            'ego': {'id': 0, 'x': 0, 'y': 0, 'vx': 15, 'vy': 0, 'speed': 15,
                   'heading': 0, 'length': 5, 'width': 2, 'mass': 2000, 'class': 'car'},
            'surrounding': [
                {'id': 1, 'x': 30, 'y': 0, 'vx': 12, 'vy': 0, 'speed': 12,
                 'heading': 0, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
                {'id': 2, 'x': 20, 'y': 3.5, 'vx': 18, 'vy': -1, 'speed': 18,
                 'heading': -0.05, 'length': 4.5, 'width': 1.8, 'mass': 1500, 'class': 'car'},
            ]
        }
    ]
    
    fig2 = visualizer.create_comparison_figure(
        scenarios=scenarios,
        methods=['gvf', 'edrf', 'ada', 'apf'],
        title="Risk & Gradient Profiles Across Scenarios"
    )
    
    output_path2 = '/mnt/user-data/outputs/field_scenario_comparison.png'
    fig2.savefig(output_path2, dpi=150, facecolor=fig2.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output_path2}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    return fig, fig2


if __name__ == '__main__':
    main()
