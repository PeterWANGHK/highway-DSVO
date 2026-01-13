"""
ExiD Integration for Risk Field Metrics
=======================================
Integrates the risk field metrics evaluation with the exiD data loader
from exid_gvf_svo_visualization.py

Usage:
    python exid_metrics_integration.py --data_dir /path/to/exiD --recording 25 --ego_id 123 --frame 500
    
    # Or with occlusion log:
    python exid_metrics_integration.py --data_dir /path/to/exiD --occlusion-csv occlusion_log.csv --occlusion-row 0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import argparse
import logging
import sys

# Import metrics module
from risk_field_metrics import (
    SmoothnessAnalyzer, BoundaryAnalyzer, OcclusionAnalyzer,
    RiskFieldConstructor, OcclusionGeometry, RiskFieldMetricsVisualizer,
    AllMetrics, SmoothnessMetrics, BoundaryMetrics, OcclusionMetrics
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for metrics evaluation."""
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus', 'van', 'trailer'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    
    MASS_HV: float = 15000.0
    MASS_PC: float = 3000.0
    
    # Grid parameters
    OBS_RANGE_AHEAD: float = 80.0
    OBS_RANGE_BEHIND: float = 40.0
    OBS_RANGE_LEFT: float = 20.0
    OBS_RANGE_RIGHT: float = 20.0
    GRID_NX: int = 80
    GRID_NY: int = 40


# =============================================================================
# ExiD Data Loader (Simplified from original)
# =============================================================================

class ExiDLoader:
    """Load exiD data for metrics evaluation."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.config = Config()
        self.tracks_df = None
        self.tracks_meta_df = None
    
    def load_recording(self, recording_id: int) -> bool:
        """Load a recording."""
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
            
            logger.info(f"Loaded recording {recording_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading recording: {e}")
            return False
    
    def get_snapshot(self, ego_id: int, frame: int) -> Optional[Dict]:
        """Get snapshot of ego and surrounding vehicles."""
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
            
            other_class = str(row.get('class', 'car')).lower()
            
            # Check if within range
            dx = row['xCenter'] - ego['x']
            dy = row['yCenter'] - ego['y']
            
            if abs(dx) > 100 or abs(dy) > 30:
                continue
            
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
                'width': float(row.get('width', 1.8)),
                'length': float(row.get('length', 4.5)),
                'class': other_class,
                'mass': self.config.MASS_HV if other_class in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            }
            
            surrounding.append(other)
        
        return {'ego': ego, 'surrounding': surrounding, 'frame': frame}
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle IDs."""
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['trackId'].tolist()
    
    def find_best_interaction_frame(self, ego_id: int) -> Optional[int]:
        """Find frame with most surrounding vehicles."""
        ego_data = self.tracks_df[self.tracks_df['trackId'] == ego_id]
        if ego_data.empty:
            return None
        
        frames = ego_data['frame'].values
        best_frame = None
        best_count = 0
        
        for frame in frames[::10]:  # Sample every 10 frames
            snapshot = self.get_snapshot(ego_id, frame)
            if snapshot and len(snapshot['surrounding']) > best_count:
                best_count = len(snapshot['surrounding'])
                best_frame = frame
        
        return best_frame if best_frame else int(np.median(frames))


# =============================================================================
# Comprehensive Evaluation with Multiple Field Types
# =============================================================================

class ComprehensiveRiskFieldEvaluator:
    """Comprehensive evaluation of multiple risk field methods."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.smoothness_analyzer = SmoothnessAnalyzer()
        self.boundary_analyzer = BoundaryAnalyzer()
        self.occlusion_analyzer = OcclusionAnalyzer()
        self.constructor = RiskFieldConstructor()
        self.occ_geometry = OcclusionGeometry()
    
    def evaluate_scenario(self, snapshot: Dict, 
                         occluder_ids: List[int] = None,
                         methods: List[str] = None) -> Dict[str, AllMetrics]:
        """
        Evaluate a scenario with multiple risk field methods.
        
        Args:
            snapshot: Data snapshot with ego and surrounding vehicles
            occluder_ids: IDs of occluding vehicles
            methods: List of methods to evaluate
        
        Returns:
            Dictionary mapping method names to AllMetrics
        """
        if methods is None:
            methods = ['gvf', 'edrf', 'ada']
        
        ego = snapshot['ego']
        others = snapshot['surrounding']
        
        # Identify occluders
        if occluder_ids is None:
            occluders = [v for v in others if v.get('class', '') in self.config.HEAVY_VEHICLE_CLASSES]
        else:
            occluders = [v for v in others if v.get('id') in occluder_ids]
        
        # Setup grid
        x_range = (-self.config.OBS_RANGE_BEHIND, self.config.OBS_RANGE_AHEAD)
        y_range = (-self.config.OBS_RANGE_LEFT, self.config.OBS_RANGE_RIGHT)
        grid_size = (self.config.GRID_NX, self.config.GRID_NY)
        
        dx = (x_range[1] - x_range[0]) / grid_size[0]
        dy = (y_range[1] - y_range[0]) / grid_size[1]
        
        results = {}
        
        for method in methods:
            # Construct field
            if method == 'gvf':
                X, Y, R = self.constructor.construct_gvf_risk(ego, others, x_range, y_range, grid_size)
            elif method == 'edrf':
                X, Y, R = self.constructor.construct_edrf_risk(ego, others, x_range, y_range, grid_size)
            elif method == 'ada':
                X, Y, R = self.constructor.construct_ada_risk(ego, others, x_range, y_range, grid_size)
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            # Compute occlusion geometry
            shadow_mask = self.occ_geometry.compute_shadow_mask(ego, occluders, X, Y)
            visibility_map = self.occ_geometry.compute_visibility_map(ego, occluders, X, Y)
            
            # Critical zone (corridor ahead of ego)
            critical_zone = (X > 0) & (X < 60) & (np.abs(Y) < 6)
            
            # Compute metrics
            smooth = self.smoothness_analyzer.compute_all(R, dx, dy)
            
            threshold = np.percentile(R, 70)
            boundary = self.boundary_analyzer.compute_all(R, threshold, dx, dy)
            
            R_occluded = R * visibility_map
            occlusion = self.occlusion_analyzer.compute_all(
                R, R_occluded, visibility_map, shadow_mask, critical_zone, threshold, dx, dy
            )
            
            results[method] = AllMetrics(smooth, boundary, occlusion)
        
        return results


# =============================================================================
# Enhanced Visualization with Detailed Analysis
# =============================================================================

class DetailedMetricsVisualizer:
    """Create detailed visualization with metric analysis."""
    
    def __init__(self, light_theme: bool = False):
        self.light_theme = light_theme
        if light_theme:
            self.bg_color = 'white'
            self.panel_color = 'white'
            self.fg_color = 'black'
            self.spine_color = '#4A4A4A'
        else:
            self.bg_color = '#0D1117'
            self.panel_color = '#161B22'
            self.fg_color = 'white'
            self.spine_color = '#4A4A6A'
    
    def _apply_axes_style(self, ax):
        ax.set_facecolor(self.panel_color)
        ax.tick_params(colors=self.fg_color)
        for spine in ax.spines.values():
            spine.set_color(self.spine_color)
    
    def _risk_cmap(self):
        if self.light_theme:
            return LinearSegmentedColormap.from_list(
                'risk_light',
                ['#F7F7F7', '#D6EAF8', '#F8C471', '#E67E22', '#C0392B']
            )
        return LinearSegmentedColormap.from_list(
            'risk_dark',
            ['#0D1117', '#1E3A5F', '#F4A261', '#E76F51', '#9B2335']
        )
    
    def create_field_figure(self, snapshot: Dict, metrics: Dict[str, AllMetrics],
                            occluder_ids: List[int] = None,
                            output_path: str = None):
        """Create snapshot figure with modeled fields and occlusion overlay."""
        
        ego = snapshot['ego']
        others = snapshot['surrounding']
        
        if occluder_ids is None:
            occluder_ids = [v['id'] for v in others if v.get('class', '') in ['truck', 'bus', 'trailer']]
        
        methods = list(metrics.keys())
        n_methods = len(methods)
        
        fig = plt.figure(figsize=(8 * max(n_methods, 2), 9))
        fig.patch.set_facecolor(self.bg_color)
        
        gs = GridSpec(2, max(n_methods, 2), figure=fig,
                      height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.3)
        
        config = Config()
        constructor = RiskFieldConstructor()
        occ_geometry = OcclusionGeometry()
        
        x_range = (-config.OBS_RANGE_BEHIND, config.OBS_RANGE_AHEAD)
        y_range = (-config.OBS_RANGE_LEFT, config.OBS_RANGE_RIGHT)
        grid_size = (config.GRID_NX, config.GRID_NY)
        
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
            logger.warning("No fields available for plotting")
            return
        
        X_mesh = fields[methods[0]]['X']
        Y_mesh = fields[methods[0]]['Y']
        occluders = [v for v in others if v.get('id') in occluder_ids]
        shadow_mask = occ_geometry.compute_shadow_mask(ego, occluders, X_mesh, Y_mesh)
        visibility_map = occ_geometry.compute_visibility_map(ego, occluders, X_mesh, Y_mesh)
        
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            self._plot_field(ax, fields[method], ego, others,
                             f"{method.upper()} Risk Field", occluder_ids)
        
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[1, i])
            R_occ = fields[method]['R'] * visibility_map
            self._plot_field_with_occlusion(
                ax, fields[method]['X'], fields[method]['Y'],
                fields[method]['R'], R_occ, shadow_mask,
                ego, others, occluder_ids, f"{method.upper()} + Occlusion"
            )
        
        fig.suptitle(
            f"Modeled Risk Fields\n"
            f"Ego: {ego.get('class', 'vehicle').title()} (ID: {ego.get('id', 'N/A')}) | "
            f"Frame: {snapshot.get('frame', 'N/A')} | "
            f"Vehicles: {len(others)} | Occluders: {len(occluder_ids)}",
            fontsize=13, fontweight='bold', color=self.fg_color, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def create_metrics_figure(self, snapshot: Dict, metrics: Dict[str, AllMetrics],
                              output_path: str = None):
        """Create figure with metric subplots (smoothness/boundary/occlusion/heatmap)."""
        
        methods = list(metrics.keys())
        n_methods = len(methods)
        
        if n_methods > 1:
            fig = plt.figure(figsize=(12, 9))
            gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
            ax_smooth = fig.add_subplot(gs[0, 0], projection='polar')
            ax_boundary = fig.add_subplot(gs[0, 1])
            ax_occ = fig.add_subplot(gs[1, 0])
            ax_heat = fig.add_subplot(gs[1, 1])
            
            self._plot_smoothness_radar(ax_smooth, metrics, methods)
            self._plot_boundary_bars(ax_boundary, metrics, methods)
            self._plot_occlusion_bars(ax_occ, metrics, methods)
            self._plot_metrics_heatmap(ax_heat, metrics, methods)
        else:
            fig = plt.figure(figsize=(10, 9))
            gs = GridSpec(3, 1, figure=fig, hspace=0.4)
            ax_smooth = fig.add_subplot(gs[0, 0], projection='polar')
            ax_boundary = fig.add_subplot(gs[1, 0])
            ax_occ = fig.add_subplot(gs[2, 0])
            
            self._plot_smoothness_radar(ax_smooth, metrics, methods)
            self._plot_boundary_bars(ax_boundary, metrics, methods)
            self._plot_occlusion_bars(ax_occ, metrics, methods)
        
        fig.patch.set_facecolor(self.bg_color)
        ego = snapshot['ego']
        fig.suptitle(
            f"Risk Field Metrics Overview\n"
            f"Ego: {ego.get('class', 'vehicle').title()} (ID: {ego.get('id', 'N/A')}) | "
            f"Frame: {snapshot.get('frame', 'N/A')}",
            fontsize=13, fontweight='bold', color=self.fg_color, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def create_table_figure(self, snapshot: Dict, metrics: Dict[str, AllMetrics],
                            output_path: str = None):
        """Create a standalone metrics table figure."""
        
        fig = plt.figure(figsize=(12, 4.8))
        fig.patch.set_facecolor(self.bg_color)
        ax_table = fig.add_subplot(111)
        self._plot_metrics_table(ax_table, metrics, list(metrics.keys()))
        
        ego = snapshot['ego']
        fig.suptitle(
            f"Detailed Metrics Summary\n"
            f"Ego: {ego.get('class', 'vehicle').title()} (ID: {ego.get('id', 'N/A')}) | "
            f"Frame: {snapshot.get('frame', 'N/A')}",
            fontsize=12, fontweight='bold', color=self.fg_color, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def create_detailed_figure(self, snapshot: Dict, 
                              metrics: Dict[str, AllMetrics],
                              occluder_ids: List[int] = None,
                              output_path: str = None):
        """Create detailed visualization with all metrics."""
        
        ego = snapshot['ego']
        others = snapshot['surrounding']
        
        if occluder_ids is None:
            occluder_ids = [v['id'] for v in others if v.get('class', '') in ['truck', 'bus', 'trailer']]
        
        methods = list(metrics.keys())
        n_methods = len(methods)
        
        # Create figure
        fig = plt.figure(figsize=(8 * max(n_methods, 2), 18))
        fig.patch.set_facecolor(self.bg_color)
        
        gs = GridSpec(5, max(n_methods, 2), figure=fig, 
                     height_ratios=[1.2, 1.2, 0.8, 0.8, 1.0],
                     hspace=0.35, wspace=0.3)
        
        # Setup common field construction
        config = Config()
        constructor = RiskFieldConstructor()
        occ_geometry = OcclusionGeometry()
        
        x_range = (-config.OBS_RANGE_BEHIND, config.OBS_RANGE_AHEAD)
        y_range = (-config.OBS_RANGE_LEFT, config.OBS_RANGE_RIGHT)
        grid_size = (config.GRID_NX, config.GRID_NY)
        
        # Store fields for plotting
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
        
        # Compute occlusion
        X_mesh = fields[methods[0]]['X']
        Y_mesh = fields[methods[0]]['Y']
        occluders = [v for v in others if v.get('id') in occluder_ids]
        shadow_mask = occ_geometry.compute_shadow_mask(ego, occluders, X_mesh, Y_mesh)
        visibility_map = occ_geometry.compute_visibility_map(ego, occluders, X_mesh, Y_mesh)
        
        # Row 1: Risk fields
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            self._plot_field(ax, fields[method], ego, others, 
                           f"{method.upper()} Risk Field", occluder_ids)
        
        # Row 2: Fields with occlusion overlay
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[1, i])
            R_occ = fields[method]['R'] * visibility_map
            self._plot_field_with_occlusion(
                ax, fields[method]['X'], fields[method]['Y'],
                fields[method]['R'], R_occ, shadow_mask,
                ego, others, occluder_ids, f"{method.upper()} + Occlusion"
            )
        
        # Row 3: Smoothness metrics radar chart
        ax_smooth = fig.add_subplot(gs[2, 0], projection='polar')
        self._plot_smoothness_radar(ax_smooth, metrics, methods)
        
        # Row 3: Boundary metrics comparison
        if n_methods > 1:
            ax_boundary = fig.add_subplot(gs[2, 1])
            self._plot_boundary_bars(ax_boundary, metrics, methods)
        
        # Row 4: Occlusion metrics
        ax_occ = fig.add_subplot(gs[3, 0])
        self._plot_occlusion_bars(ax_occ, metrics, methods)
        
        # Row 4: Heat map comparison
        if n_methods > 1:
            ax_heat = fig.add_subplot(gs[3, 1])
            self._plot_metrics_heatmap(ax_heat, metrics, methods)
        
        # Row 5: Detailed metrics table
        ax_table = fig.add_subplot(gs[4, :])
        self._plot_metrics_table(ax_table, metrics, methods)
        
        # Title
        fig.suptitle(
            f"Comprehensive Risk Field Analysis\n"
            f"Ego: {ego.get('class', 'vehicle').title()} (ID: {ego.get('id', 'N/A')}) | "
            f"Frame: {snapshot.get('frame', 'N/A')} | "
            f"Vehicles: {len(others)} | Occluders: {len(occluder_ids)}",
            fontsize=14, fontweight='bold', color=self.fg_color, y=0.99
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved: {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _plot_field(self, ax, field_data, ego, others, title, occluder_ids):
        """Plot a risk field."""
        self._apply_axes_style(ax)
        
        X, Y, R = field_data['X'], field_data['Y'], field_data['R']
        R_norm = R / (np.max(R) + 1e-10)
        
        cmap = self._risk_cmap()
        pcm = ax.pcolormesh(X, Y, R_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Contours
        levels = [0.3, 0.5, 0.7, 0.9]
        contour_color = '#333333' if self.light_theme else 'white'
        ax.contour(X, Y, R_norm, levels=levels, colors=contour_color, alpha=0.3, linewidths=0.5)
        
        # Ego vehicle
        ego_rect = mpatches.FancyBboxPatch(
            (-ego.get('length', 5)/2, -ego.get('width', 2)/2),
            ego.get('length', 5), ego.get('width', 2),
            boxstyle="round,pad=0.02",
            facecolor='#E74C3C', edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # Other vehicles
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            is_occ = other.get('id') in occluder_ids
            color = '#FFD700' if is_occ else '#3498DB'
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other.get('length', 4.5)/2, dy_rel - other.get('width', 1.8)/2),
                other.get('length', 4.5), other.get('width', 1.8),
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=self.fg_color, linewidth=1.5 if is_occ else 1, alpha=0.85
            )
            ax.add_patch(rect)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color, fontsize=9)
        ax.set_ylabel('Lateral (m)', color=self.fg_color, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('Risk', color=self.fg_color, fontsize=8)
        cbar.ax.tick_params(colors=self.fg_color, labelsize=7)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
    
    def _plot_field_with_occlusion(self, ax, X, Y, R_full, R_occ, shadow_mask,
                                   ego, others, occluder_ids, title):
        """Plot field with occlusion overlay."""
        self._apply_axes_style(ax)
        
        R_norm = R_occ / (np.max(R_full) + 1e-10)
        
        cmap = self._risk_cmap()
        pcm = ax.pcolormesh(X, Y, R_norm, cmap=cmap, shading='gouraud', alpha=0.9)
        
        # Shadow overlay
        shadow_overlay = np.ma.masked_where(~shadow_mask, np.ones_like(shadow_mask, dtype=float))
        ax.pcolormesh(X, Y, shadow_overlay, cmap='Greys', alpha=0.5, shading='auto')
        
        # Vehicles
        cos_h = np.cos(-ego.get('heading', 0))
        sin_h = np.sin(-ego.get('heading', 0))
        
        ego_rect = mpatches.FancyBboxPatch(
            (-ego.get('length', 5)/2, -ego.get('width', 2)/2),
            ego.get('length', 5), ego.get('width', 2),
            boxstyle="round,pad=0.02", facecolor='#E74C3C', edgecolor=self.fg_color, linewidth=2
        )
        ax.add_patch(ego_rect)
        
        for other in others:
            dx = other['x'] - ego['x']
            dy = other['y'] - ego['y']
            dx_rel = dx * cos_h - dy * sin_h
            dy_rel = dx * sin_h + dy * cos_h
            
            is_occ = other.get('id') in occluder_ids
            color = '#FFD700' if is_occ else '#3498DB'
            
            rect = mpatches.FancyBboxPatch(
                (dx_rel - other.get('length', 4.5)/2, dy_rel - other.get('width', 1.8)/2),
                other.get('length', 4.5), other.get('width', 1.8),
                boxstyle="round,pad=0.02", facecolor=color, edgecolor=self.fg_color, 
                linewidth=2 if is_occ else 1, alpha=0.9
            )
            ax.add_patch(rect)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('Longitudinal (m)', color=self.fg_color, fontsize=9)
        ax.set_ylabel('Lateral (m)', color=self.fg_color, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label('Visible Risk', color=self.fg_color, fontsize=8)
        cbar.ax.tick_params(colors=self.fg_color, labelsize=7)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
    
    def _plot_smoothness_radar(self, ax, metrics, methods):
        """Plot smoothness metrics as radar chart."""
        ax.set_facecolor(self.panel_color)
        categories = ['1/SNHS', 'TVR', '1-SSI', '1/AR']
        n_cats = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            m = metrics[method].smoothness
            values = [
                1 / (m.SNHS + 0.001),
                1 - min(m.TVR, 5) / 5,
                1 - m.SSI,
                1 / (m.AR + 0.001) if m.AR < 5 else 0.2
            ]
            # Normalize to [0, 1]
            values = [min(max(v, 0), 1) for v in values]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method.upper(), color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color=self.fg_color, fontsize=9)
        ax.set_title('Smoothness Profile', fontsize=11, fontweight='bold', 
                    color=self.fg_color, pad=20)
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(colors=self.fg_color)
        if 'polar' in ax.spines:
            ax.spines['polar'].set_color(self.spine_color)
    
    def _plot_boundary_bars(self, ax, metrics, methods):
        """Plot boundary metrics as grouped bars."""
        self._apply_axes_style(ax)
        
        metric_names = ['ISI', 'PPI', 'GCI', 'DBS']
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            m = metrics[method].boundary
            values = [min(m.ISI * 10, 1), m.PPI, m.GCI, m.DBS]
            ax.bar(x + i * width - width * len(methods) / 2 + width / 2,
                  values, width, label=method.upper(), color=colors[i], edgecolor=self.fg_color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, color=self.fg_color, fontsize=9)
        ax.set_ylabel('Value', color=self.fg_color, fontsize=9)
        ax.set_title('Boundary Metrics', fontsize=11, fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2, axis='y')
    
    def _plot_occlusion_bars(self, ax, metrics, methods):
        """Plot occlusion metrics."""
        self._apply_axes_style(ax)
        
        metric_names = ['OFSR', 'CZU', 'VWRD', 'ITOI_JS']
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            m = metrics[method].occlusion
            values = [m.OFSR, m.CZU, m.VWRD, m.ITOI_JS]
            ax.bar(x + i * width - width * len(methods) / 2 + width / 2,
                  values, width, label=method.upper(), color=colors[i], edgecolor=self.fg_color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, color=self.fg_color, fontsize=9)
        ax.set_ylabel('Value', color=self.fg_color, fontsize=9)
        ax.set_title('Occlusion Metrics (Lower = Better)', fontsize=11, 
                    fontweight='bold', color=self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2, axis='y')
    
    def _plot_metrics_heatmap(self, ax, metrics, methods):
        """Plot metrics as heatmap."""
        self._apply_axes_style(ax)
        
        metric_names = ['SNHS', 'TVR', 'SSI', 'ISI', 'PPI', 'GCI', 'DBS', 'OFSR', 'CZU', 'VWRD']
        
        data = []
        for method in methods:
            m = metrics[method]
            row = [
                m.smoothness.SNHS, m.smoothness.TVR, m.smoothness.SSI,
                m.boundary.ISI, m.boundary.PPI, m.boundary.GCI, m.boundary.DBS,
                m.occlusion.OFSR, m.occlusion.CZU, m.occlusion.VWRD
            ]
            data.append(row)
        
        data = np.array(data)
        
        # Normalize columns
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
        
        im = ax.imshow(data_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(metric_names, color=self.fg_color, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels([m.upper() for m in methods], color=self.fg_color, fontsize=9)
        
        # Add value annotations
        for i in range(len(methods)):
            for j in range(len(metric_names)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', 
                       color='white' if data_norm[i, j] > 0.5 else 'black', fontsize=7)
        
        ax.set_title('Metrics Comparison Heatmap', fontsize=11, fontweight='bold', color=self.fg_color)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Relative Value', color=self.fg_color, fontsize=8)
        cbar.ax.tick_params(colors=self.fg_color, labelsize=7)
        cbar.ax.yaxis.set_tick_params(color=self.fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.fg_color)
    
    def _plot_metrics_table(self, ax, metrics, methods):
        """Plot detailed metrics table."""
        ax.set_facecolor(self.panel_color)
        ax.axis('off')
        
        # Build table data
        columns = ['Metric'] + [m.upper() for m in methods] + ['Interpretation']
        
        rows = [
            ['SNHS', *[f'{metrics[m].smoothness.SNHS:.4f}' for m in methods], 'Lower = smoother'],
            ['TVR', *[f'{metrics[m].smoothness.TVR:.4f}' for m in methods], 'Lower = fewer edges'],
            ['SSI', *[f'{metrics[m].smoothness.SSI:.4f}' for m in methods], 'Lower = less noise'],
            ['AR', *[f'{metrics[m].smoothness.AR:.4f}' for m in methods], '~1 = isotropic'],
            ['ISI', *[f'{metrics[m].boundary.ISI:.4f}' for m in methods], 'Higher = sharper'],
            ['PPI', *[f'{metrics[m].boundary.PPI:.4f}' for m in methods], 'Higher = better'],
            ['GCI', *[f'{metrics[m].boundary.GCI:.4f}' for m in methods], 'Higher = localized'],
            ['DBS', *[f'{metrics[m].boundary.DBS:.4f}' for m in methods], 'Higher = stable'],
            ['OFSR', *[f'{metrics[m].occlusion.OFSR:.4f}' for m in methods], 'Lower = safer'],
            ['CZU', *[f'{metrics[m].occlusion.CZU:.4f}' for m in methods], 'Lower = safer'],
            ['VWRD', *[f'{metrics[m].occlusion.VWRD:.4f}' for m in methods], 'Lower = safer'],
            ['ITOI_JS', *[f'{metrics[m].occlusion.ITOI_JS:.4f}' for m in methods], 'Lower = consistent'],
        ]
        
        table = ax.table(cellText=rows, colLabels=columns,
                        loc='center', cellLoc='center',
                        colWidths=[0.12] + [0.15] * len(methods) + [0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style header
        for j, col in enumerate(columns):
            cell = table[(0, j)]
            if self.light_theme:
                cell.set_facecolor('#E6E6E6')
                cell.set_text_props(color='black', fontweight='bold')
            else:
                cell.set_facecolor('#2E4057')
                cell.set_text_props(color='white', fontweight='bold')
        
        # Alternate row colors
        for i in range(1, len(rows) + 1):
            for j in range(len(columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#1A1A2E' if not self.light_theme else '#F0F0F0')
                else:
                    cell.set_facecolor('#161B22' if not self.light_theme else '#FFFFFF')
                cell.set_text_props(color=self.fg_color)
        
        ax.set_title('Detailed Metrics Summary', fontsize=12, fontweight='bold', 
                    color=self.fg_color, pad=20)


# =============================================================================
# Main Entry Point
# =============================================================================

def main(data_dir: str, recording_id: int, ego_id: Optional[int] = None,
         frame: Optional[int] = None, output_dir: str = './output_metrics',
         methods: List[str] = None, light_theme: bool = True):
    """Main evaluation function."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Risk Field Metrics Evaluation")
    logger.info("=" * 60)
    
    # Load data
    loader = ExiDLoader(data_dir)
    if not loader.load_recording(recording_id):
        return None
    
    # Find ego vehicle
    if ego_id is None:
        heavy_ids = loader.get_heavy_vehicles()
        if heavy_ids:
            ego_id = heavy_ids[0]
            logger.info(f"Auto-selected ego: {ego_id}")
        else:
            logger.error("No heavy vehicles found")
            return None
    
    # Find best frame
    if frame is None:
        frame = loader.find_best_interaction_frame(ego_id)
        if frame is None:
            logger.error("Could not find suitable frame")
            return None
        logger.info(f"Auto-selected frame: {frame}")
    
    # Get snapshot
    snapshot = loader.get_snapshot(ego_id, frame)
    if snapshot is None:
        logger.error("Could not get snapshot")
        return None
    
    logger.info(f"Ego: {snapshot['ego']['class']} (ID: {ego_id})")
    logger.info(f"Surrounding: {len(snapshot['surrounding'])} vehicles")
    
    # Evaluate
    if methods is None:
        methods = ['gvf', 'edrf', 'ada']
    
    evaluator = ComprehensiveRiskFieldEvaluator()
    metrics = evaluator.evaluate_scenario(snapshot, methods=methods)
    
    # Visualize (separate outputs in one subfolder)
    visualizer = DetailedMetricsVisualizer(light_theme=light_theme)
    run_dir = output_path / f'metrics_rec{recording_id}_ego{ego_id}_frame{frame}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {run_dir}")
    
    field_file = run_dir / 'field_snapshot.png'
    metrics_file = run_dir / 'metrics_panels.png'
    table_file = run_dir / 'metrics_table.png'
    
    visualizer.create_field_figure(snapshot, metrics, output_path=str(field_file))
    visualizer.create_metrics_figure(snapshot, metrics, output_path=str(metrics_file))
    visualizer.create_table_figure(snapshot, metrics, output_path=str(table_file))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("METRICS SUMMARY")
    logger.info("=" * 60)
    
    for method, m in metrics.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Smoothness: SNHS={m.smoothness.SNHS:.4f}, TVR={m.smoothness.TVR:.4f}, SSI={m.smoothness.SSI:.4f}")
        logger.info(f"  Boundary:   ISI={m.boundary.ISI:.4f}, PPI={m.boundary.PPI:.4f}, GCI={m.boundary.GCI:.4f}, DBS={m.boundary.DBS:.4f}")
        logger.info(f"  Occlusion:  OFSR={m.occlusion.OFSR:.4f}, CZU={m.occlusion.CZU:.4f}, VWRD={m.occlusion.VWRD:.4f}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Risk Field Metrics Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to exiD data')
    parser.add_argument('--recording', type=int, default=25, help='Recording ID')
    parser.add_argument('--ego_id', type=int, default=None, help='Ego vehicle ID')
    parser.add_argument('--frame', type=int, default=None, help='Frame number')
    parser.add_argument('--output_dir', type=str, default='./output_metrics', help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=['gvf', 'edrf', 'ada'],
                       help='Risk field methods to evaluate')
    parser.add_argument('--light-theme', action='store_true', help='Force light theme')
    parser.add_argument('--dark-theme', action='store_true', help='Use dark theme')
    
    args = parser.parse_args()
    
    light_theme = not args.dark_theme
    
    main(args.data_dir, args.recording, args.ego_id, args.frame, 
         args.output_dir, args.methods, light_theme)
