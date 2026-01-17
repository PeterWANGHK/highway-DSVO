"""
Integrated Field-Traffic Visualization for exiD Dataset
=========================================================
Combines traffic scenario visualization with field gradient
longitudinal analysis. Loads scenario snapshots from role
analysis JSON exports.

Command Line Examples:
----------------------

# 1. Using scenario JSON from role analysis:
python integrated_field_traffic_viz.py --scenario ./output_roles/rec25_ego123_frame500/scenario_snapshot.json

# 2. Using scenario JSON with specific methods:
python integrated_field_traffic_viz.py --scenario scenario.json --methods gvf edrf ada apf

# 3. Using scenario JSON with light theme:
python integrated_field_traffic_viz.py --scenario scenario.json --light-theme

# 4. Direct exiD data loading (without scenario JSON):
python integrated_field_traffic_viz.py --data_dir ./data --recording 25 --ego_id 123 --frame 500

# 5. Demo mode (no data required):
python integrated_field_traffic_viz.py --demo

# 6. Save output to specific directory:
python integrated_field_traffic_viz.py --scenario scenario.json --output_dir ./my_output

# 7. Generate trajectory analysis with decision detection:
python integrated_field_traffic_viz.py --data_dir ./data --recording 25 --ego_id 123 --with-trajectory

# 8. Export statistics to CSV:
python integrated_field_traffic_viz.py --scenario scenario.json --export-csv

Author: Research Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from scipy import signal, ndimage
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import argparse
import json
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IntegratedConfig:
    """Configuration for integrated visualization."""
    # Grid parameters
    longitudinal_range: Tuple[float, float] = (-50, 100)
    lateral_range: Tuple[float, float] = (-20, 20)
    grid_resolution: float = 1.0
    
    # Decision thresholds
    decel_threshold: float = -2.0
    accel_threshold: float = 1.5
    lateral_vel_threshold: float = 0.5
    
    # Alignment
    alignment_threshold: float = 15.0
    gradient_peak_prominence: float = 0.15
    
    # Vehicle classes
    heavy_vehicle_classes: tuple = ('truck', 'bus', 'trailer', 'van')
    
    # Visualization
    figsize: Tuple[int, int] = (20, 16)
    dpi: int = 150


class ColorScheme:
    """Color scheme for visualization."""
    
    def __init__(self, dark: bool = True):
        self.dark = dark
        if dark:
            self.bg = '#0D1117'
            self.panel = '#1A1A2E'
            self.fg = '#E0E0E0'
            self.grid = '#4A4A6A'
            self.spine = '#4A4A6A'
        else:
            self.bg = '#FFFFFF'
            self.panel = '#F8F9FA'
            self.fg = '#2C3E50'
            self.grid = '#BDC3C7'
            self.spine = '#7F8C8D'
        
        self.methods = ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6']
        self.vehicles = {
            'truck': '#E74C3C',
            'car': '#3498DB',
            'bus': '#F39C12',
            'van': '#9B59B6',
            'trailer': '#E67E22'
        }
        self.roles = {
            'normal_main': '#3498DB',
            'merging': '#E74C3C',
            'exiting': '#F39C12',
            'yielding': '#9B59B6',
            'unknown': '#95A5A6'
        }
        self.risk_cmap = self._create_risk_cmap()
    
    def _create_risk_cmap(self):
        colors = [
            (0.0, '#2C3E50'),
            (0.25, '#27AE60'),
            (0.5, '#F1C40F'),
            (0.75, '#E67E22'),
            (1.0, '#C0392B')
        ]
        return LinearSegmentedColormap.from_list('risk', 
            [(c[0], c[1]) for c in colors])


# =============================================================================
# Scenario Loader
# =============================================================================

class ScenarioLoader:
    """Load scenario from JSON or exiD data."""
    
    def __init__(self, config: IntegratedConfig = None):
        self.config = config or IntegratedConfig()
    
    def load_from_json(self, json_path: str) -> Dict:
        """Load scenario from JSON file exported by role analysis."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded scenario: Recording {data.get('recording_id')}, "
                   f"Frame {data.get('frame')}, Ego {data.get('ego_id')}")
        
        # Convert to internal format
        scenario = {
            'recording_id': data.get('recording_id'),
            'frame': data.get('frame'),
            'ego_id': data.get('ego_id'),
            'ego': None,
            'surrounding': [],
            'occlusions': data.get('occlusions', [])
        }
        
        for agent in data.get('agents', []):
            agent_dict = {
                'id': agent['id'],
                'x': agent['x'],
                'y': agent['y'],
                'vx': agent.get('vx', agent.get('speed', 15)),
                'vy': agent.get('vy', 0),
                'ax': agent.get('ax', 0),
                'ay': agent.get('ay', 0),
                'speed': agent.get('speed', np.sqrt(agent.get('vx', 15)**2 + agent.get('vy', 0)**2)),
                'heading': agent.get('heading', 0),
                'length': agent.get('length', 4.5),
                'width': agent.get('width', 1.8),
                'mass': agent.get('mass', 1500),
                'class': agent.get('vehicle_class', 'car'),
                'role': agent.get('role', 'unknown'),
                'urgency': agent.get('urgency', 0)
            }
            
            if agent['id'] == scenario['ego_id']:
                scenario['ego'] = agent_dict
            else:
                scenario['surrounding'].append(agent_dict)
        
        # If no ego specified, use first heavy vehicle or first agent
        if scenario['ego'] is None and scenario['surrounding']:
            for agent in scenario['surrounding']:
                if agent.get('class', '').lower() in self.config.heavy_vehicle_classes:
                    scenario['ego'] = agent
                    scenario['surrounding'].remove(agent)
                    scenario['ego_id'] = agent['id']
                    break
            
            if scenario['ego'] is None:
                scenario['ego'] = scenario['surrounding'].pop(0)
                scenario['ego_id'] = scenario['ego']['id']
        
        logger.info(f"Ego: ID {scenario['ego_id']}, "
                   f"Surrounding: {len(scenario['surrounding'])} vehicles")
        
        return scenario
    
    def load_from_exid(self, data_dir: str, recording: int, 
                       ego_id: int = None, frame: int = None) -> Dict:
        """Load scenario directly from exiD data files."""
        data_path = Path(data_dir)
        rec_str = f"{recording:02d}" if recording < 100 else str(recording)
        
        tracks_path = data_path / f"{rec_str}_tracks.csv"
        meta_path = data_path / f"{rec_str}_tracksMeta.csv"
        
        if not tracks_path.exists():
            raise FileNotFoundError(f"Tracks file not found: {tracks_path}")
        
        logger.info(f"Loading recording {recording} from {data_dir}")
        tracks_df = pd.read_csv(tracks_path)
        
        meta_df = None
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
        
        # Find ego vehicle
        if ego_id is None:
            if meta_df is not None:
                heavy_mask = meta_df['class'].str.lower().isin(self.config.heavy_vehicle_classes)
                heavy_ids = meta_df[heavy_mask]['trackId'].tolist()
                if heavy_ids:
                    ego_id = heavy_ids[0]
            
            if ego_id is None:
                ego_id = tracks_df['trackId'].iloc[0]
        
        # Find frame
        if frame is None:
            ego_frames = tracks_df[tracks_df['trackId'] == ego_id]['frame'].values
            frame = int(np.median(ego_frames))
        
        logger.info(f"Using ego_id={ego_id}, frame={frame}")
        
        # Extract frame data
        frame_data = tracks_df[tracks_df['frame'] == frame]
        
        scenario = {
            'recording_id': recording,
            'frame': frame,
            'ego_id': ego_id,
            'ego': None,
            'surrounding': [],
            'occlusions': []
        }
        
        x_col = 'x' if 'x' in frame_data.columns else 'xCenter'
        y_col = 'y' if 'y' in frame_data.columns else 'yCenter'
        
        for _, row in frame_data.iterrows():
            vclass = str(row.get('class', 'car')).lower() if 'class' in row else 'car'
            if meta_df is not None and 'class' not in row:
                meta_row = meta_df[meta_df['trackId'] == row['trackId']]
                if len(meta_row) > 0:
                    vclass = str(meta_row.iloc[0].get('class', 'car')).lower()
            
            agent_dict = {
                'id': int(row['trackId']),
                'x': float(row[x_col]),
                'y': float(row[y_col]),
                'vx': float(row.get('xVelocity', 15)),
                'vy': float(row.get('yVelocity', 0)),
                'ax': float(row.get('xAcceleration', 0)),
                'ay': float(row.get('yAcceleration', 0)),
                'speed': np.sqrt(row.get('xVelocity', 15)**2 + row.get('yVelocity', 0)**2),
                'heading': np.radians(float(row.get('heading', 0))),
                'length': float(row.get('length', 4.5)),
                'width': float(row.get('width', 1.8)),
                'mass': 15000 if vclass in self.config.heavy_vehicle_classes else 1500,
                'class': vclass,
                'role': 'unknown',
                'urgency': 0
            }
            
            if row['trackId'] == ego_id:
                scenario['ego'] = agent_dict
            else:
                scenario['surrounding'].append(agent_dict)
        
        return scenario


# =============================================================================
# Risk Field Builders
# =============================================================================

def build_fields(X: np.ndarray, Y: np.ndarray, 
                 ego: Dict, others: List[Dict]) -> Dict[str, np.ndarray]:
    """Build risk fields using multiple methods."""
    fields = {}
    
    # GVF
    R = np.zeros_like(X)
    for veh in others:
        dx, dy = X - veh['x'], Y - veh['y']
        speed = veh.get('speed', 10)
        h = veh.get('heading', 0)
        c, s = np.cos(h), np.sin(h)
        dx_r, dy_r = dx*c + dy*s, -dx*s + dy*c
        sx, sy = 5 + 0.5*speed, 2 + 0.1*speed
        R += np.exp(-0.5*((dx_r/sx)**2 + (dy_r/sy)**2))
    fields['gvf'] = np.clip(R / (R.max() + 1e-10), 0, 1)
    
    # EDRF
    R = np.zeros_like(X)
    for veh in others:
        dx, dy = X - veh['x'], Y - veh['y']
        L, W = veh.get('length', 4.5), veh.get('width', 1.8)
        speed = veh.get('speed', 10)
        a, b = L/2 + 0.3*speed, W/2 + 0.1*speed
        h = veh.get('heading', 0)
        c, s = np.cos(h), np.sin(h)
        dx_r, dy_r = dx*c + dy*s, -dx*s + dy*c
        R += np.exp(-np.sqrt((dx_r/a)**2 + (dy_r/b)**2))
    fields['edrf'] = np.clip(R / (R.max() + 1e-10), 0, 1)
    
    # ADA
    R = np.zeros_like(X)
    for veh in others:
        dx, dy = X - veh['x'], Y - veh['y']
        speed = veh.get('speed', 10)
        h = veh.get('heading', 0)
        c, s = np.cos(h), np.sin(h)
        dx_r, dy_r = dx*c + dy*s, -dx*s + dy*c
        sf, sr = 8 + 0.6*speed, 3
        sx = np.where(dx_r > 0, sf, sr)
        R += np.exp(-0.5*((dx_r/sx)**2 + (dy_r/2.5)**2))
    fields['ada'] = np.clip(R / (R.max() + 1e-10), 0, 1)
    
    # APF
    R = np.zeros_like(X)
    ego_speed = ego.get('speed', 15)
    for veh in others:
        dx, dy = X - veh['x'], Y - veh['y']
        rel_vx = ego.get('vx', ego_speed) - veh.get('vx', veh.get('speed', 10))
        rel_vy = ego.get('vy', 0) - veh.get('vy', 0)
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
        h = veh.get('heading', 0)
        c, s = np.cos(h), np.sin(h)
        dx_r, dy_r = dx*c + dy*s, -dx*s + dy*c
        sx, sy = 10 + 0.4*rel_speed, 3
        R += np.exp(-0.5*((dx_r/sx)**2 + (dy_r/sy)**2))
    fields['apf'] = np.clip(R / (R.max() + 1e-10), 0, 1)
    
    return fields


# =============================================================================
# Integrated Visualizer
# =============================================================================

class IntegratedVisualizer:
    """Creates integrated traffic + field gradient visualizations."""
    
    def __init__(self, config: IntegratedConfig = None, dark_theme: bool = True):
        self.config = config or IntegratedConfig()
        self.colors = ColorScheme(dark_theme)
    
    def _draw_vehicle(self, ax, veh: Dict, is_ego: bool = False):
        """Draw vehicle rectangle."""
        cx, cy = veh['x'], veh['y']
        length, width = veh.get('length', 4.5), veh.get('width', 1.8)
        heading = veh.get('heading', 0)
        
        half_l, half_w = length/2, width/2
        corners = np.array([
            [-half_l, -half_w], [half_l, -half_w],
            [half_l, half_w], [-half_l, half_w]
        ])
        
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T + np.array([cx, cy])
        
        vclass = veh.get('class', 'car').lower()
        
        if is_ego:
            color = self.colors.vehicles.get(vclass, '#E74C3C')
            edgecolor = 'yellow'
            linewidth = 3
            alpha = 1.0
        elif vclass in self.config.heavy_vehicle_classes:
            color = self.colors.vehicles.get(vclass, '#E74C3C')
            edgecolor = 'white'
            linewidth = 2
            alpha = 0.9
        else:
            role = veh.get('role', 'unknown')
            color = self.colors.roles.get(role, '#3498DB')
            edgecolor = 'white'
            linewidth = 1.5
            alpha = 0.85
        
        poly = Polygon(corners, closed=True, facecolor=color,
                      edgecolor=edgecolor, linewidth=linewidth, 
                      alpha=alpha, zorder=10)
        ax.add_patch(poly)
        
        # Label
        label = f"EGO\n{veh['id']}" if is_ego else str(veh['id'])
        text_color = 'yellow' if is_ego else 'white'
        ax.text(cx, cy + width/2 + 1.5, label,
               ha='center', va='bottom', fontsize=7,
               color=text_color, fontweight='bold' if is_ego else 'normal',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.5),
               zorder=11)
    
    def create_integrated_figure(self, scenario: Dict, 
                                methods: List[str] = None,
                                decisions: List[Dict] = None) -> plt.Figure:
        """Create comprehensive integrated figure."""
        if methods is None:
            methods = ['gvf', 'edrf', 'ada', 'apf']
        
        ego = scenario['ego']
        others = scenario['surrounding']
        
        # Build grid
        x_range = (ego['x'] + self.config.longitudinal_range[0],
                  ego['x'] + self.config.longitudinal_range[1])
        y_range = (ego['y'] + self.config.lateral_range[0],
                  ego['y'] + self.config.lateral_range[1])
        
        nx = int((x_range[1] - x_range[0]) / self.config.grid_resolution)
        ny = int((y_range[1] - y_range[0]) / self.config.grid_resolution)
        
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        
        # Build fields
        fields = build_fields(X, Y, ego, others)
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize, facecolor=self.colors.bg)
        
        # Layout: 
        # Row 0: Traffic snapshot (wide) + Legend
        # Row 1: 4 risk field heatmaps
        # Row 2: Gradient profiles (wide)
        # Row 3: Risk profiles + Statistics
        
        gs = GridSpec(4, 4, figure=fig, height_ratios=[1.2, 1, 0.8, 0.8],
                     hspace=0.35, wspace=0.25)
        
        # Row 0: Traffic snapshot
        ax_traffic = fig.add_subplot(gs[0, :3])
        ax_traffic.set_facecolor(self.colors.panel)
        self._plot_traffic_with_field(ax_traffic, scenario, fields[methods[0]], X, Y)
        
        # Row 0: Legend/Info panel
        ax_legend = fig.add_subplot(gs[0, 3])
        ax_legend.set_facecolor(self.colors.panel)
        self._plot_info_panel(ax_legend, scenario)
        
        # Row 1: Risk field heatmaps
        for i, method in enumerate(methods[:4]):
            ax = fig.add_subplot(gs[1, i])
            ax.set_facecolor(self.colors.panel)
            self._plot_field_heatmap(ax, fields[method], X, Y, ego, others, method.upper())
        
        # Row 2: Gradient profiles
        ax_grad = fig.add_subplot(gs[2, :])
        ax_grad.set_facecolor(self.colors.panel)
        self._plot_gradient_profiles(ax_grad, fields, x, X, Y, methods, decisions)
        
        # Row 3: Risk profiles + Statistics
        ax_risk = fig.add_subplot(gs[3, :2])
        ax_risk.set_facecolor(self.colors.panel)
        self._plot_risk_profiles(ax_risk, fields, x, X, Y, methods)
        
        ax_stats = fig.add_subplot(gs[3, 2:])
        ax_stats.set_facecolor(self.colors.panel)
        self._plot_statistics(ax_stats, fields, x, X, Y, methods, decisions)
        
        # Title
        title = (f"Integrated Field-Traffic Analysis | "
                f"Recording: {scenario.get('recording_id', 'N/A')} | "
                f"Frame: {scenario.get('frame', 'N/A')} | "
                f"Ego: {scenario.get('ego_id', 'N/A')} | "
                f"Vehicles: {len(others) + 1}")
        fig.suptitle(title, fontsize=14, fontweight='bold',
                    color=self.colors.fg, y=0.99)
        
        # Apply spine styling
        for ax in fig.get_axes():
            for spine in ax.spines.values():
                spine.set_color(self.colors.spine)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig
    
    def _plot_traffic_with_field(self, ax, scenario, field, X, Y):
        """Plot traffic snapshot with field overlay."""
        ego = scenario['ego']
        others = scenario['surrounding']
        
        # Plot field as background
        im = ax.pcolormesh(X, Y, field, cmap=self.colors.risk_cmap, 
                          shading='auto', alpha=0.6, zorder=1)
        
        # Draw lane markings (approximate)
        y_min, y_max = Y.min(), Y.max()
        for lane_y in np.linspace(y_min, y_max, 6):
            ax.axhline(lane_y, color='white', linestyle='--', 
                      linewidth=0.5, alpha=0.3, zorder=2)
        
        # Draw vehicles
        for veh in others:
            self._draw_vehicle(ax, veh, is_ego=False)
        self._draw_vehicle(ax, ego, is_ego=True)
        
        # Draw occlusion shadows if available
        for occ in scenario.get('occlusions', []):
            if occ.get('shadow_polygon'):
                poly = np.array(occ['shadow_polygon'])
                shadow = Polygon(poly, closed=True, facecolor='gray',
                               edgecolor='darkgray', alpha=0.3, zorder=3)
                ax.add_patch(shadow)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('X (m)', color=self.colors.fg)
        ax.set_ylabel('Y (m)', color=self.colors.fg)
        ax.set_title('Traffic Scenario with Risk Field Overlay',
                    color=self.colors.fg, fontweight='bold')
        ax.tick_params(colors=self.colors.fg)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Risk', color=self.colors.fg, fontsize=9)
        cbar.ax.tick_params(colors=self.colors.fg, labelsize=8)
    
    def _plot_info_panel(self, ax, scenario):
        """Plot info panel with scenario details."""
        ax.axis('off')
        
        ego = scenario['ego']
        others = scenario['surrounding']
        
        # Count vehicle types
        type_counts = {}
        for veh in others + [ego]:
            vclass = veh.get('class', 'car')
            type_counts[vclass] = type_counts.get(vclass, 0) + 1
        
        # Count roles
        role_counts = {}
        for veh in others:
            role = veh.get('role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        
        info_lines = [
            "═" * 25,
            "SCENARIO INFO",
            "═" * 25,
            f"Recording: {scenario.get('recording_id', 'N/A')}",
            f"Frame: {scenario.get('frame', 'N/A')}",
            f"Ego ID: {scenario.get('ego_id', 'N/A')}",
            "",
            "─" * 25,
            "EGO VEHICLE",
            "─" * 25,
            f"Class: {ego.get('class', 'N/A')}",
            f"Speed: {ego.get('speed', 0):.1f} m/s",
            f"Position: ({ego['x']:.1f}, {ego['y']:.1f})",
            "",
            "─" * 25,
            "TRAFFIC COMPOSITION",
            "─" * 25,
        ]
        
        for vclass, count in sorted(type_counts.items()):
            info_lines.append(f"  {vclass}: {count}")
        
        info_lines.extend([
            "",
            "─" * 25,
            "ROLE DISTRIBUTION",
            "─" * 25,
        ])
        
        for role, count in sorted(role_counts.items()):
            info_lines.append(f"  {role}: {count}")
        
        info_text = "\n".join(info_lines)
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=8, color=self.colors.fg, family='monospace',
               verticalalignment='top')
    
    def _plot_field_heatmap(self, ax, field, X, Y, ego, others, title):
        """Plot single field heatmap with vehicles."""
        im = ax.pcolormesh(X, Y, field, cmap=self.colors.risk_cmap, shading='auto')
        
        # Draw vehicles (simplified)
        for veh in others:
            ax.plot(veh['x'], veh['y'], 'o', color='white', 
                   markersize=4, markeredgecolor='black', markeredgewidth=0.5)
        ax.plot(ego['x'], ego['y'], 's', color='yellow',
               markersize=6, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=self.colors.fg)
        ax.set_xlabel('X (m)', fontsize=8, color=self.colors.fg)
        ax.set_ylabel('Y (m)', fontsize=8, color=self.colors.fg)
        ax.tick_params(colors=self.colors.fg, labelsize=7)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.tick_params(colors=self.colors.fg, labelsize=6)
    
    def _plot_gradient_profiles(self, ax, fields, x_vals, X, Y, methods, decisions):
        """Plot gradient profiles along longitudinal axis."""
        dx = self.config.grid_resolution
        
        for i, method in enumerate(methods[:4]):
            grad_y, grad_x = np.gradient(fields[method], dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            y_center = grad_mag.shape[0] // 2
            profile = grad_mag[y_center, :]
            
            ax.plot(X[y_center, :], profile, color=self.colors.methods[i],
                   linewidth=2, label=method.upper(), alpha=0.85)
        
        # Decision points
        if decisions:
            for d in decisions:
                ax.axvline(d['x'], color='#F39C12', linestyle='--',
                          linewidth=1.5, alpha=0.7)
            ax.scatter([d['x'] for d in decisions], [0] * len(decisions),
                      marker='^', s=100, color='#F39C12', edgecolor='white',
                      linewidth=1, zorder=5, label='Decisions')
        
        ax.set_xlabel('Longitudinal Position (m)', fontsize=10, color=self.colors.fg)
        ax.set_ylabel('Gradient Magnitude', fontsize=10, color=self.colors.fg)
        ax.set_title('Field Gradient vs Longitudinal Position',
                    fontsize=11, fontweight='bold', color=self.colors.fg)
        ax.tick_params(colors=self.colors.fg, labelsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, color=self.colors.grid)
    
    def _plot_risk_profiles(self, ax, fields, x_vals, X, Y, methods):
        """Plot risk profiles along centerline."""
        for i, method in enumerate(methods[:4]):
            y_center = fields[method].shape[0] // 2
            profile = fields[method][y_center, :]
            ax.plot(X[y_center, :], profile, color=self.colors.methods[i],
                   linewidth=2, label=method.upper(), alpha=0.85)
        
        ax.axhline(0.7, color='red', linestyle=':', alpha=0.5, label='High')
        ax.axhline(0.4, color='orange', linestyle=':', alpha=0.5, label='Moderate')
        
        ax.set_xlabel('Longitudinal Position (m)', fontsize=10, color=self.colors.fg)
        ax.set_ylabel('Risk Value', fontsize=10, color=self.colors.fg)
        ax.set_title('Risk Profile Along Centerline', fontsize=11,
                    fontweight='bold', color=self.colors.fg)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(colors=self.colors.fg, labelsize=9)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, color=self.colors.grid)
    
    def _plot_statistics(self, ax, fields, x_vals, X, Y, methods, decisions):
        """Plot statistics summary."""
        ax.axis('off')
        
        dx = self.config.grid_resolution
        stats_lines = ["═" * 40, "FIELD GRADIENT STATISTICS", "═" * 40, ""]
        
        headers = f"{'Method':<8} {'GradMax':>8} {'GradMean':>9} {'Peaks':>6}"
        stats_lines.append(headers)
        stats_lines.append("─" * 40)
        
        for method in methods[:4]:
            grad_y, grad_x = np.gradient(fields[method], dx)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            y_center = grad_mag.shape[0] // 2
            profile = grad_mag[y_center, :]
            
            peaks, _ = signal.find_peaks(profile, 
                prominence=self.config.gradient_peak_prominence * profile.max())
            
            line = f"{method.upper():<8} {profile.max():>8.4f} {profile.mean():>9.4f} {len(peaks):>6}"
            stats_lines.append(line)
        
        if decisions:
            stats_lines.extend(["", "─" * 40, "DECISION POINTS", "─" * 40])
            for d in decisions[:5]:
                stats_lines.append(f"  x={d['x']:.1f}m, type={d.get('type', 'N/A')}")
        
        stats_text = "\n".join(stats_lines)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, color=self.colors.fg, family='monospace',
               verticalalignment='top')


# =============================================================================
# Demo Data
# =============================================================================

def create_demo_scenario() -> Dict:
    """Create demo scenario for testing."""
    ego = {
        'id': 100,
        'x': 0, 'y': 0,
        'vx': 20, 'vy': 0,
        'ax': 0, 'ay': 0,
        'speed': 20,
        'heading': 0,
        'length': 12, 'width': 2.5,
        'mass': 15000,
        'class': 'truck',
        'role': 'normal_main',
        'urgency': 0
    }
    
    others = [
        {'id': 1, 'x': 35, 'y': 0, 'vx': 18, 'vy': 0, 'speed': 18,
         'heading': 0, 'length': 4.5, 'width': 1.8, 'class': 'car', 'role': 'normal_main'},
        {'id': 2, 'x': 25, 'y': 8, 'vx': 22, 'vy': -2, 'speed': 22.1,
         'heading': -0.09, 'length': 4.5, 'width': 1.8, 'class': 'car', 'role': 'merging'},
        {'id': 3, 'x': 15, 'y': 3.5, 'vx': 19, 'vy': 0, 'speed': 19,
         'heading': 0, 'length': 10, 'width': 2.5, 'class': 'truck', 'role': 'normal_main'},
        {'id': 4, 'x': -20, 'y': 0, 'vx': 22, 'vy': 0, 'speed': 22,
         'heading': 0, 'length': 4.5, 'width': 1.8, 'class': 'car', 'role': 'normal_main'},
        {'id': 5, 'x': 60, 'y': -3.5, 'vx': 25, 'vy': 0, 'speed': 25,
         'heading': 0, 'length': 4.5, 'width': 1.8, 'class': 'car', 'role': 'normal_main'},
        {'id': 6, 'x': 45, 'y': 7, 'vx': 20, 'vy': -1, 'speed': 20,
         'heading': -0.05, 'length': 4.5, 'width': 1.8, 'class': 'car', 'role': 'merging'},
    ]
    
    decisions = [
        {'x': 15, 'type': 'decel', 'intensity': 0.8},
        {'x': 35, 'type': 'lc_left', 'intensity': 0.6},
        {'x': 55, 'type': 'accel', 'intensity': 0.5}
    ]
    
    return {
        'recording_id': 'demo',
        'frame': 100,
        'ego_id': 100,
        'ego': ego,
        'surrounding': others,
        'occlusions': [],
        'decisions': decisions
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Integrated Field-Traffic Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using scenario JSON from role analysis:
  python integrated_field_traffic_viz.py --scenario ./output_roles/rec25_ego123_frame500/scenario_snapshot.json

  # Using scenario JSON with specific methods:
  python integrated_field_traffic_viz.py --scenario scenario.json --methods gvf edrf ada apf

  # Direct exiD data loading:
  python integrated_field_traffic_viz.py --data_dir ./data --recording 25 --ego_id 123 --frame 500

  # Demo mode:
  python integrated_field_traffic_viz.py --demo
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--scenario', type=str,
                            help='Path to scenario JSON from role analysis')
    input_group.add_argument('--demo', action='store_true',
                            help='Use demo data (no external data required)')
    input_group.add_argument('--data_dir', type=str,
                            help='Path to exiD data directory')
    
    # ExiD options
    parser.add_argument('--recording', type=int, default=25,
                       help='Recording ID (default: 25)')
    parser.add_argument('--ego_id', type=int, default=None,
                       help='Ego vehicle ID')
    parser.add_argument('--frame', type=int, default=None,
                       help='Frame number')
    
    # Methods
    parser.add_argument('--methods', type=str, nargs='+',
                       choices=['gvf', 'edrf', 'ada', 'apf'],
                       default=['gvf', 'edrf', 'ada', 'apf'],
                       help='Field methods to visualize')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output_integrated',
                       help='Output directory')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export statistics to CSV')
    
    # Visualization
    parser.add_argument('--light-theme', action='store_true',
                       help='Use light color theme')
    parser.add_argument('--with-trajectory', action='store_true',
                       help='Include trajectory decision analysis')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.demo and not args.scenario and not args.data_dir:
        parser.print_help()
        print("\nError: Specify --demo, --scenario, or --data_dir")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Integrated Field-Traffic Visualization")
    logger.info("=" * 60)
    
    # Load scenario
    loader = ScenarioLoader()
    
    if args.demo:
        logger.info("Running in DEMO mode")
        scenario = create_demo_scenario()
    elif args.scenario:
        logger.info(f"Loading scenario from: {args.scenario}")
        scenario = loader.load_from_json(args.scenario)
    else:
        logger.info(f"Loading from exiD: {args.data_dir}")
        scenario = loader.load_from_exid(
            args.data_dir, args.recording, 
            args.ego_id, args.frame
        )
    
    # Create visualizer
    visualizer = IntegratedVisualizer(dark_theme=not args.light_theme)
    
    # Generate figure
    logger.info("Generating integrated figure...")
    decisions = scenario.get('decisions', [])
    
    fig = visualizer.create_integrated_figure(
        scenario=scenario,
        methods=args.methods,
        decisions=decisions
    )
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_id = scenario.get('recording_id', 'demo')
    frame_id = scenario.get('frame', 0)
    
    output_file = output_path / f"integrated_rec{rec_id}_frame{frame_id}_{timestamp}.png"
    fig.savefig(output_file, dpi=150, facecolor=fig.get_facecolor(),
               bbox_inches='tight', pad_inches=0.2)
    logger.info(f"Saved: {output_file}")
    
    # Export CSV if requested
    if args.export_csv:
        # TODO: Implement CSV export
        logger.info("CSV export not yet implemented")
    
    logger.info("=" * 60)
    logger.info("Visualization complete!")
    logger.info("=" * 60)
    
    return fig


if __name__ == '__main__':
    main()
