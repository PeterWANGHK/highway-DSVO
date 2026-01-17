"""
ExiD Dataset Integration for Field-Longitudinal Analysis
==========================================================
Comprehensive integration with exiD dataset for statistical
analysis of risk field gradients vs longitudinal vehicle positions.

Features:
1. Direct exiD data loading and trajectory extraction
2. Multi-vehicle batch analysis
3. Statistical aggregation across recordings
4. Publication-ready figure generation
5. CSV/JSON export of metrics

Author: Research Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal, stats, ndimage
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExiDAnalysisConfig:
    """Configuration for exiD analysis."""
    # exiD data paths
    data_dir: str = './exiD'
    
    # Vehicle filtering
    heavy_vehicle_classes: tuple = ('truck', 'bus', 'trailer')
    min_trajectory_length: int = 50  # minimum frames
    
    # Grid parameters
    longitudinal_range: Tuple[float, float] = (-50, 100)
    lateral_range: Tuple[float, float] = (-20, 20)
    grid_resolution: float = 1.0
    
    # Decision thresholds
    decel_threshold: float = -2.0
    accel_threshold: float = 1.5
    lateral_vel_threshold: float = 0.5
    
    # Analysis settings
    smoothing_window: int = 5
    gradient_peak_prominence: float = 0.15
    alignment_threshold: float = 15.0  # meters


# =============================================================================
# ExiD Data Loader
# =============================================================================

class ExiDLoader:
    """Load and process exiD dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.tracks_df = None
        self.tracks_meta_df = None
        self.recording_meta = None
        self.current_recording = None
    
    def load_recording(self, recording_id: int) -> bool:
        """Load a specific recording."""
        rec_str = f"{recording_id:02d}" if recording_id < 100 else str(recording_id)
        
        tracks_path = self.data_dir / f"{rec_str}_tracks.csv"
        meta_path = self.data_dir / f"{rec_str}_tracksMeta.csv"
        rec_meta_path = self.data_dir / f"{rec_str}_recordingMeta.csv"
        
        if not tracks_path.exists():
            print(f"Recording {recording_id} not found at {tracks_path}")
            return False
        
        print(f"Loading recording {recording_id}...")
        self.tracks_df = pd.read_csv(tracks_path)
        
        if meta_path.exists():
            self.tracks_meta_df = pd.read_csv(meta_path)
        else:
            self.tracks_meta_df = None
        
        if rec_meta_path.exists():
            self.recording_meta = pd.read_csv(rec_meta_path)
        
        self.current_recording = recording_id
        print(f"Loaded {len(self.tracks_df)} track points, "
              f"{self.tracks_df['trackId'].nunique()} vehicles")
        
        return True
    
    def get_vehicle_trajectory(self, track_id: int) -> Optional[Dict]:
        """Extract trajectory for a specific vehicle."""
        if self.tracks_df is None:
            return None
        
        veh_data = self.tracks_df[self.tracks_df['trackId'] == track_id].copy()
        if len(veh_data) == 0:
            return None
        
        veh_data = veh_data.sort_values('frame')
        
        # Standard exiD column names
        trajectory = {
            'track_id': track_id,
            'frames': veh_data['frame'].values,
            'timestamps': veh_data['frame'].values / 25.0,  # 25 Hz
            'positions_x': veh_data['x'].values if 'x' in veh_data else veh_data['xCenter'].values,
            'positions_y': veh_data['y'].values if 'y' in veh_data else veh_data['yCenter'].values,
            'velocities_x': veh_data['xVelocity'].values if 'xVelocity' in veh_data else np.gradient(veh_data['x'].values, 1/25),
            'velocities_y': veh_data['yVelocity'].values if 'yVelocity' in veh_data else np.gradient(veh_data['y'].values, 1/25),
            'headings': veh_data['heading'].values if 'heading' in veh_data else np.zeros(len(veh_data)),
            'lengths': veh_data['length'].values if 'length' in veh_data else np.full(len(veh_data), 4.5),
            'widths': veh_data['width'].values if 'width' in veh_data else np.full(len(veh_data), 1.8),
        }
        
        # Compute accelerations
        dt = 1/25.0
        trajectory['accelerations_x'] = np.gradient(trajectory['velocities_x'], dt)
        trajectory['accelerations_y'] = np.gradient(trajectory['velocities_y'], dt)
        trajectory['speeds'] = np.sqrt(trajectory['velocities_x']**2 + trajectory['velocities_y']**2)
        
        # Get vehicle class if available
        if self.tracks_meta_df is not None:
            meta = self.tracks_meta_df[self.tracks_meta_df['trackId'] == track_id]
            if len(meta) > 0:
                trajectory['class'] = meta.iloc[0].get('class', 'car')
            else:
                trajectory['class'] = 'car'
        else:
            trajectory['class'] = 'car'
        
        return trajectory
    
    def get_snapshot(self, ego_id: int, frame: int) -> Optional[Dict]:
        """Get snapshot at a specific frame."""
        if self.tracks_df is None:
            return None
        
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        if len(frame_data) == 0:
            return None
        
        ego_data = frame_data[frame_data['trackId'] == ego_id]
        if len(ego_data) == 0:
            return None
        
        ego_row = ego_data.iloc[0]
        
        ego = {
            'id': ego_id,
            'x': ego_row['x'] if 'x' in ego_row else ego_row['xCenter'],
            'y': ego_row['y'] if 'y' in ego_row else ego_row['yCenter'],
            'vx': ego_row.get('xVelocity', 15.0),
            'vy': ego_row.get('yVelocity', 0.0),
            'speed': np.sqrt(ego_row.get('xVelocity', 15)**2 + ego_row.get('yVelocity', 0)**2),
            'heading': ego_row.get('heading', 0.0),
            'length': ego_row.get('length', 4.5),
            'width': ego_row.get('width', 1.8),
        }
        
        others = []
        for _, row in frame_data[frame_data['trackId'] != ego_id].iterrows():
            others.append({
                'id': row['trackId'],
                'x': row['x'] if 'x' in row else row['xCenter'],
                'y': row['y'] if 'y' in row else row['yCenter'],
                'vx': row.get('xVelocity', 15.0),
                'vy': row.get('yVelocity', 0.0),
                'speed': np.sqrt(row.get('xVelocity', 15)**2 + row.get('yVelocity', 0)**2),
                'heading': row.get('heading', 0.0),
                'length': row.get('length', 4.5),
                'width': row.get('width', 1.8),
            })
        
        return {
            'ego': ego,
            'surrounding': others,
            'frame': frame,
            'recording': self.current_recording
        }
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get list of heavy vehicle track IDs."""
        if self.tracks_meta_df is None:
            return []
        
        heavy_classes = {'truck', 'bus', 'trailer', 'van'}
        heavy_ids = self.tracks_meta_df[
            self.tracks_meta_df['class'].str.lower().isin(heavy_classes)
        ]['trackId'].tolist()
        
        return heavy_ids


# =============================================================================
# Statistical Analysis Engine
# =============================================================================

class StatisticalAnalyzer:
    """Perform statistical analysis on field-decision alignment."""
    
    def __init__(self, config: ExiDAnalysisConfig = None):
        self.config = config or ExiDAnalysisConfig()
        self.results = []
    
    def compute_longitudinal_metrics(self, 
                                    risk_field: np.ndarray,
                                    X: np.ndarray, Y: np.ndarray,
                                    decisions: List[Dict]) -> Dict:
        """Compute comprehensive longitudinal metrics."""
        # Extract gradient profile at center
        grad_y, grad_x = np.gradient(risk_field, self.config.grid_resolution)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        y_center = grad_mag.shape[0] // 2
        x_vals = X[y_center, :]
        grad_profile = grad_mag[y_center, :]
        risk_profile = risk_field[y_center, :]
        
        # Find gradient peaks
        peaks, props = signal.find_peaks(
            grad_profile,
            prominence=self.config.gradient_peak_prominence * grad_profile.max(),
            width=2
        )
        
        peak_positions = x_vals[peaks]
        peak_values = grad_profile[peaks]
        
        # Decision positions
        decision_positions = np.array([d['x'] for d in decisions]) if decisions else np.array([])
        
        # Alignment metrics
        if len(peak_positions) > 0 and len(decision_positions) > 0:
            distances = np.abs(peak_positions[:, np.newaxis] - decision_positions)
            min_distances = distances.min(axis=1)
            matched_peaks = np.sum(min_distances < self.config.alignment_threshold)
            
            distances_rev = np.abs(decision_positions[:, np.newaxis] - peak_positions)
            min_distances_rev = distances_rev.min(axis=1)
            matched_decisions = np.sum(min_distances_rev < self.config.alignment_threshold)
            
            precision = matched_peaks / len(peak_positions)
            recall = matched_decisions / len(decision_positions)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            mean_distance = np.mean(min_distances)
        else:
            precision, recall, f1, mean_distance = 0, 0, 0, float('inf')
            matched_peaks, matched_decisions = 0, 0
        
        # Gradient statistics
        grad_max = grad_profile.max()
        grad_mean = grad_profile.mean()
        grad_std = grad_profile.std()
        
        # Risk transition zones
        high_thresh = np.percentile(risk_profile, self.config.gradient_peak_prominence * 100 + 50)
        moderate_thresh = np.percentile(risk_profile, 50)
        
        transition_mask = (risk_profile > moderate_thresh) & (risk_profile < high_thresh)
        transition_indices = np.where(transition_mask)[0]
        
        return {
            'n_peaks': len(peak_positions),
            'n_decisions': len(decision_positions),
            'matched_peaks': matched_peaks,
            'matched_decisions': matched_decisions,
            'precision': precision,
            'recall': recall,
            'f1_alignment': f1,
            'mean_alignment_distance': mean_distance,
            'grad_max': grad_max,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'peak_positions': peak_positions.tolist(),
            'decision_positions': decision_positions.tolist(),
            'x_vals': x_vals.tolist(),
            'grad_profile': grad_profile.tolist(),
            'risk_profile': risk_profile.tolist(),
            'n_transition_points': len(transition_indices)
        }
    
    def aggregate_statistics(self, results_list: List[Dict]) -> Dict:
        """Aggregate statistics across multiple analyses."""
        if not results_list:
            return {}
        
        # Collect all metrics
        metrics = {}
        for key in ['precision', 'recall', 'f1_alignment', 'mean_alignment_distance',
                   'grad_max', 'grad_mean', 'grad_std', 'n_peaks', 'n_decisions']:
            values = [r[key] for r in results_list if key in r and not np.isinf(r[key])]
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'n': len(values)
                }
        
        return metrics


# =============================================================================
# Publication-Quality Figure Generator
# =============================================================================

class PublicationFigureGenerator:
    """Generate publication-quality figures."""
    
    def __init__(self, dark_theme: bool = False):
        self.dark = dark_theme
        self._setup_colors()
        self._setup_style()
    
    def _setup_colors(self):
        if self.dark:
            self.colors = {
                'bg': '#1a1a2e', 'panel': '#16213e', 'fg': '#e0e0e0',
                'grid': '#394867', 'spine': '#394867',
                'methods': ['#e74c3c', '#3498db', '#27ae60', '#9b59b6'],
                'decision': '#f39c12'
            }
        else:
            self.colors = {
                'bg': '#ffffff', 'panel': '#f8f9fa', 'fg': '#2c3e50',
                'grid': '#bdc3c7', 'spine': '#7f8c8d',
                'methods': ['#c0392b', '#2980b9', '#27ae60', '#8e44ad'],
                'decision': '#d35400'
            }
    
    def _setup_style(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def create_longitudinal_figure(self,
                                  x_vals: np.ndarray,
                                  profiles: Dict[str, Dict],
                                  decisions: List[Dict] = None,
                                  title: str = None,
                                  figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Create comprehensive longitudinal analysis figure."""
        fig = plt.figure(figsize=figsize, facecolor=self.colors['bg'])
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 0.8],
                     hspace=0.35, wspace=0.25)
        
        methods = list(profiles.keys())[:4]
        
        # Panel A: Gradient profiles
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(self.colors['panel'])
        
        for i, method in enumerate(methods):
            ax1.plot(x_vals, profiles[method]['gradient'], 
                    color=self.colors['methods'][i], linewidth=2,
                    label=method.upper(), alpha=0.85)
        
        if decisions:
            for d in decisions:
                ax1.axvline(d['x'], color=self.colors['decision'],
                          linestyle='--', linewidth=1.5, alpha=0.6)
            ax1.scatter([d['x'] for d in decisions], 
                       [0] * len(decisions),
                       marker='^', s=100, color=self.colors['decision'],
                       edgecolor='white', linewidth=1, zorder=5,
                       label='Decision Points')
        
        ax1.set_xlabel('Longitudinal Position (m)', color=self.colors['fg'])
        ax1.set_ylabel('Field Gradient Magnitude', color=self.colors['fg'])
        ax1.set_title('(a) Field Gradient vs Longitudinal Position',
                     color=self.colors['fg'], fontweight='bold', loc='left')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        ax1.tick_params(colors=self.colors['fg'])
        
        # Panel B: Risk profiles
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(self.colors['panel'])
        
        for i, method in enumerate(methods):
            ax2.plot(x_vals, profiles[method]['risk'],
                    color=self.colors['methods'][i], linewidth=2,
                    label=method.upper(), alpha=0.85)
        
        ax2.axhline(0.7, color='red', linestyle=':', alpha=0.5, label='High Risk')
        ax2.axhline(0.4, color='orange', linestyle=':', alpha=0.5, label='Moderate')
        
        ax2.set_xlabel('Longitudinal Position (m)', color=self.colors['fg'])
        ax2.set_ylabel('Risk Value', color=self.colors['fg'])
        ax2.set_title('(b) Risk Profile', color=self.colors['fg'],
                     fontweight='bold', loc='left')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3, color=self.colors['grid'])
        ax2.tick_params(colors=self.colors['fg'])
        
        # Panel C: Alignment bar chart
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(self.colors['panel'])
        
        alignment_scores = [profiles[m].get('f1_alignment', 0) for m in methods]
        bars = ax3.bar(methods, alignment_scores, color=self.colors['methods'][:len(methods)],
                      edgecolor='white', linewidth=1.5)
        
        for bar, score in zip(bars, alignment_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=self.colors['fg'])
        
        ax3.set_ylabel('Alignment Score (F1)', color=self.colors['fg'])
        ax3.set_title('(c) Gradient-Decision Alignment',
                     color=self.colors['fg'], fontweight='bold', loc='left')
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(colors=self.colors['fg'])
        ax3.grid(True, alpha=0.3, axis='y', color=self.colors['grid'])
        
        # Panel D: Statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_facecolor(self.colors['panel'])
        ax4.axis('off')
        
        # Build statistics table
        table_data = []
        headers = ['Method', 'Grad Max', 'Grad Mean', 'Peaks', 'Precision', 'Recall', 'F1']
        
        for method in methods:
            p = profiles[method]
            row = [
                method.upper(),
                f"{p.get('grad_max', 0):.4f}",
                f"{p.get('grad_mean', 0):.4f}",
                str(p.get('n_peaks', 0)),
                f"{p.get('precision', 0):.4f}",
                f"{p.get('recall', 0):.4f}",
                f"{p.get('f1_alignment', 0):.4f}"
            ]
            table_data.append(row)
        
        table = ax4.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=[self.colors['grid']] * len(headers)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style table
        for key, cell in table.get_celld().items():
            cell.set_text_props(color=self.colors['fg'])
            if key[0] == 0:
                cell.set_text_props(fontweight='bold')
        
        ax4.set_title('(d) Summary Statistics',
                     color=self.colors['fg'], fontweight='bold', loc='left', y=0.95)
        
        # Apply spine styling
        for ax in [ax1, ax2, ax3]:
            for spine in ax.spines.values():
                spine.set_color(self.colors['spine'])
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold',
                        color=self.colors['fg'], y=0.98)
        
        plt.tight_layout()
        return fig
    
    def create_multi_scenario_figure(self,
                                    scenarios_data: List[Dict],
                                    figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Create multi-scenario comparison figure."""
        n_scenarios = len(scenarios_data)
        methods = list(scenarios_data[0]['profiles'].keys())[:4]
        
        fig = plt.figure(figsize=figsize, facecolor=self.colors['bg'])
        gs = GridSpec(n_scenarios, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        for s_idx, scenario in enumerate(scenarios_data):
            x_vals = np.array(scenario['x_vals'])
            profiles = scenario['profiles']
            
            for m_idx, method in enumerate(methods):
                ax = fig.add_subplot(gs[s_idx, m_idx])
                ax.set_facecolor(self.colors['panel'])
                
                # Plot risk and gradient
                ax.plot(x_vals, profiles[method]['risk'],
                       color='#3498db', linewidth=2, label='Risk')
                
                ax2 = ax.twinx()
                ax2.plot(x_vals, profiles[method]['gradient'],
                        color='#e74c3c', linewidth=2, linestyle='--', label='Gradient')
                
                if s_idx == 0:
                    ax.set_title(method.upper(), fontsize=11,
                               fontweight='bold', color=self.colors['fg'])
                
                if m_idx == 0:
                    ax.set_ylabel(f"Scenario {s_idx+1}\nRisk",
                                 fontsize=9, color='#3498db')
                else:
                    ax.set_ylabel('Risk', fontsize=9, color='#3498db')
                
                ax2.set_ylabel('Gradient', fontsize=9, color='#e74c3c')
                
                if s_idx == n_scenarios - 1:
                    ax.set_xlabel('Longitudinal (m)', fontsize=9, color=self.colors['fg'])
                
                ax.tick_params(axis='y', labelcolor='#3498db', labelsize=8)
                ax2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=8)
                ax.tick_params(axis='x', colors=self.colors['fg'], labelsize=8)
                ax.grid(True, alpha=0.3, color=self.colors['grid'])
                
                for spine in ax.spines.values():
                    spine.set_color(self.colors['spine'])
        
        fig.suptitle('Risk and Gradient Profiles Across Methods and Scenarios',
                    fontsize=14, fontweight='bold', color=self.colors['fg'], y=0.99)
        
        return fig


# =============================================================================
# Risk Field Methods
# =============================================================================

def build_risk_field_gvf(X, Y, ego, others):
    """GVF risk field."""
    R = np.zeros_like(X)
    for veh in others:
        dx, dy = X - veh['x'], Y - veh['y']
        speed = veh.get('speed', 10)
        h = veh.get('heading', 0)
        c, s = np.cos(h), np.sin(h)
        dx_r, dy_r = dx*c + dy*s, -dx*s + dy*c
        sx, sy = 5 + 0.5*speed, 2 + 0.1*speed
        R += np.exp(-0.5*((dx_r/sx)**2 + (dy_r/sy)**2))
    return np.clip(R / (R.max() + 1e-10), 0, 1)


def build_risk_field_edrf(X, Y, ego, others):
    """EDRF risk field."""
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
    return np.clip(R / (R.max() + 1e-10), 0, 1)


def build_risk_field_ada(X, Y, ego, others):
    """ADA risk field."""
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
    return np.clip(R / (R.max() + 1e-10), 0, 1)


def build_risk_field_apf(X, Y, ego, others):
    """APF-Wang risk field."""
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
    return np.clip(R / (R.max() + 1e-10), 0, 1)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(scenario: Dict, config: ExiDAnalysisConfig = None) -> Dict:
    """Run complete analysis on a scenario."""
    config = config or ExiDAnalysisConfig()
    
    ego = scenario['ego']
    others = scenario['surrounding']
    
    # Build grid
    x = np.linspace(ego['x'] + config.longitudinal_range[0],
                   ego['x'] + config.longitudinal_range[1],
                   int((config.longitudinal_range[1] - config.longitudinal_range[0]) / config.grid_resolution))
    y = np.linspace(ego['y'] + config.lateral_range[0],
                   ego['y'] + config.lateral_range[1],
                   int((config.lateral_range[1] - config.lateral_range[0]) / config.grid_resolution))
    X, Y = np.meshgrid(x, y)
    
    # Build fields
    fields = {
        'gvf': build_risk_field_gvf(X, Y, ego, others),
        'edrf': build_risk_field_edrf(X, Y, ego, others),
        'ada': build_risk_field_ada(X, Y, ego, others),
        'apf': build_risk_field_apf(X, Y, ego, others)
    }
    
    # Analyze each method
    analyzer = StatisticalAnalyzer(config)
    profiles = {}
    
    decisions = scenario.get('decisions', [])
    
    for method, field in fields.items():
        metrics = analyzer.compute_longitudinal_metrics(field, X, Y, decisions)
        profiles[method] = {
            'risk': np.array(metrics['risk_profile']),
            'gradient': np.array(metrics['grad_profile']),
            'grad_max': metrics['grad_max'],
            'grad_mean': metrics['grad_mean'],
            'n_peaks': metrics['n_peaks'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_alignment': metrics['f1_alignment']
        }
    
    return {
        'x_vals': x.tolist(),
        'profiles': profiles,
        'scenario': scenario
    }


# =============================================================================
# Demo
# =============================================================================

def create_demo_scenario():
    """Create demo scenario."""
    ego = {'id': 0, 'x': 0, 'y': 0, 'vx': 20, 'vy': 0, 'speed': 20,
           'heading': 0, 'length': 12, 'width': 2.5, 'class': 'truck'}
    
    others = [
        {'id': 1, 'x': 35, 'y': 0, 'vx': 18, 'vy': 0, 'speed': 18,
         'heading': 0, 'length': 4.5, 'width': 1.8},
        {'id': 2, 'x': 25, 'y': 8, 'vx': 22, 'vy': -2, 'speed': 22,
         'heading': -0.09, 'length': 4.5, 'width': 1.8},
        {'id': 3, 'x': 15, 'y': 3.5, 'vx': 19, 'vy': 0, 'speed': 19,
         'heading': 0, 'length': 10, 'width': 2.5},
        {'id': 4, 'x': -20, 'y': 0, 'vx': 22, 'vy': 0, 'speed': 22,
         'heading': 0, 'length': 4.5, 'width': 1.8},
        {'id': 5, 'x': 60, 'y': -3.5, 'vx': 25, 'vy': 0, 'speed': 25,
         'heading': 0, 'length': 4.5, 'width': 1.8},
    ]
    
    # Simulated decision points
    decisions = [
        {'x': 15, 'type': 'decel', 'intensity': 0.8},
        {'x': 35, 'type': 'lc_left', 'intensity': 0.6},
        {'x': 55, 'type': 'accel', 'intensity': 0.5}
    ]
    
    return {'ego': ego, 'surrounding': others, 'decisions': decisions}


def main():
    """Run demo analysis."""
    print("=" * 60)
    print("ExiD Field-Longitudinal Analysis")
    print("=" * 60)
    
    # Create demo scenarios
    scenarios = [
        create_demo_scenario(),
        {
            'ego': {'id': 0, 'x': 0, 'y': 0, 'vx': 15, 'vy': 0, 'speed': 15,
                   'heading': 0, 'length': 5, 'width': 2},
            'surrounding': [
                {'id': 1, 'x': 40, 'y': 0, 'vx': 12, 'vy': 0, 'speed': 12,
                 'heading': 0, 'length': 4.5, 'width': 1.8},
                {'id': 2, 'x': 20, 'y': 4, 'vx': 18, 'vy': -1, 'speed': 18,
                 'heading': -0.05, 'length': 4.5, 'width': 1.8},
            ],
            'decisions': [{'x': 25, 'type': 'decel', 'intensity': 0.7}]
        }
    ]
    
    # Run analysis
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"\nAnalyzing scenario {i+1}...")
        result = run_analysis(scenario)
        results.append(result)
        
        # Print metrics
        for method, p in result['profiles'].items():
            print(f"  {method.upper()}: F1={p['f1_alignment']:.4f}, "
                  f"Peaks={p['n_peaks']}, GradMax={p['grad_max']:.4f}")
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = PublicationFigureGenerator(dark_theme=True)
    
    # Single scenario figure
    fig1 = fig_gen.create_longitudinal_figure(
        x_vals=np.array(results[0]['x_vals']),
        profiles=results[0]['profiles'],
        decisions=results[0]['scenario'].get('decisions', []),
        title="Field Gradient vs Longitudinal Position Analysis"
    )
    
    output1 = '/mnt/user-data/outputs/exid_longitudinal_analysis.png'
    fig1.savefig(output1, dpi=150, facecolor=fig1.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output1}")
    
    # Multi-scenario figure
    fig2 = fig_gen.create_multi_scenario_figure(results)
    
    output2 = '/mnt/user-data/outputs/exid_multi_scenario.png'
    fig2.savefig(output2, dpi=150, facecolor=fig2.get_facecolor(),
                bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output2}")
    
    # Export statistics to JSON
    stats_export = {}
    for i, result in enumerate(results):
        stats_export[f'scenario_{i+1}'] = {
            method: {k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
            for method, p in result['profiles'].items()
        }
    
    json_path = '/mnt/user-data/outputs/field_statistics.json'
    with open(json_path, 'w') as f:
        json.dump(stats_export, f, indent=2)
    print(f"Saved: {json_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
