"""
AD4CHE Dataset: Occlusion Scenario Scanner and Analyzer
========================================================
Scans all recordings to find and catalog occlusion scenarios.

Features:
1. Scan all 68 recordings for occlusion events
2. Rank scenarios by occlusion severity and vehicle count
3. Save catalog to CSV for easy browsing
4. Select and analyze specific scenarios
5. Generate batch analysis report

Data structure: C:\field_modeling\data\AD4CHE\DJI_XXXX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import transforms
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import warnings
import argparse
import logging
import json
import re
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VehicleState:
    """Vehicle state."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    speed: float
    length: float
    width: float
    vehicle_class: str
    mass: float
    driving_direction: Optional[int] = None
    is_occluded: bool = False
    occluder_id: Optional[int] = None
    visibility: float = 1.0


@dataclass
class OcclusionScenario:
    """Describes a detected occlusion scenario."""
    recording_id: int
    recording_folder: str
    frame: int
    ego_id: int
    ego_class: str
    ego_speed: float
    n_surrounding: int
    n_occluded: int
    n_occluders: int
    occluded_vehicle_ids: List[int]
    occluder_vehicle_ids: List[int]
    avg_visibility: float
    min_visibility: float
    occlusion_score: float  # Combined metric for ranking
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Config:
    """Configuration."""
    HEAVY_VEHICLE_CLASSES: Set[str] = field(default_factory=lambda: {'truck', 'bus'})
    CAR_CLASSES: Set[str] = field(default_factory=lambda: {'car'})
    MAX_SURROUNDING: int = 6
    OBS_RANGE: float = 60.0
    OCCLUSION_RANGE: float = 80.0
    FRAME_SAMPLE_STEP: int = 25  # Sample every N frames for scanning
    MIN_OCCLUSION_SCORE: float = 0.1  # Minimum score to record
    
    # Physical
    MASS_HV: float = 15000.0
    MASS_PC: float = 1500.0
    
    # Visualization
    GRID_NX: int = 100
    GRID_NY: int = 60
    REPULSION_AMPLITUDE: float = 100.0
    REPULSION_DECAY_LONG: float = 8.0
    REPULSION_DECAY_LAT: float = 3.0
    VELOCITY_RISK_WEIGHT: float = 2.0
    TTC_CRITICAL: float = 3.0
    TTC_WEIGHT: float = 50.0
    SHADOW_UNCERTAINTY_BASE: float = 0.5
    SHADOW_UNCERTAINTY_GROWTH: float = 0.02
    
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'truck': '#E74C3C', 'car': '#3498DB', 'bus': '#F39C12',
        'ego': '#2ECC71', 'occluded': '#95A5A6',
    })
    BG_DARK: str = '#0D1117'
    BG_PANEL: str = '#161B22'
    SHADOW_COLOR: str = '#4A4A4A'
    LANE_WIDTH: float = 3.5
    LANES_LEFT: int = 2
    LANES_RIGHT: int = 2
    ROAD_COLOR: str = '#1F2937'
    LANE_MARK_COLOR: str = '#9CA3AF'
    CENTERLINE_COLOR: str = '#FBBF24'


# =============================================================================
# Data Loader
# =============================================================================

class AD4CHELoader:
    """Load AD4CHE data from folder structure."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.config = Config()
        self.tracks_df = None
        self.tracks_meta_df = None
        self.current_recording_id = None
        self.current_folder = None
        self.background_image = None
        self.background_extent = None
    
    def get_recording_folders(self) -> List[Tuple[int, Path]]:
        """Get all recording folders with their IDs."""
        folders = []
        
        # Look for DJI_XXXX folders
        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and item.name.startswith('DJI_'):
                try:
                    # Extract recording ID from folder name
                    rec_id = int(item.name.split('_')[1])
                    folders.append((rec_id, item))
                except (ValueError, IndexError):
                    continue
        
        return folders
    
    def load_recording(self, recording_id: int, folder_path: Path = None) -> bool:
        """Load a specific recording."""
        if folder_path is None:
            # Find folder for this recording ID
            folders = self.get_recording_folders()
            folder_path = next((f for rid, f in folders if rid == recording_id), None)
            if folder_path is None:
                return False
        
        self.current_recording_id = recording_id
        self.current_folder = folder_path
        
        prefix = f"{recording_id:02d}_"
        
        try:
            tracks_path = folder_path / f"{prefix}tracks.csv"
            meta_path = folder_path / f"{prefix}tracksMeta.csv"
            
            if not tracks_path.exists() or not meta_path.exists():
                return False
            
            self.tracks_df = pd.read_csv(tracks_path)
            self.tracks_meta_df = pd.read_csv(meta_path)
            
            # Merge
            self.tracks_df = self.tracks_df.merge(
                self.tracks_meta_df[['id', 'class', 'width', 'height', 'drivingDirection']],
                on='id', how='left', suffixes=('', '_meta')
            )
            
            if 'width_meta' in self.tracks_df.columns:
                self.tracks_df['length'] = self.tracks_df['width_meta']
                self.tracks_df['veh_width'] = self.tracks_df['height']
            
            # Load background image if present and compute extent from data
            img_path = folder_path / f"{recording_id:02d}_highway.png"
            if img_path.exists():
                try:
                    self.background_image = plt.imread(img_path)
                    x_min, x_max = self.tracks_df['x'].min(), self.tracks_df['x'].max()
                    y_min, y_max = self.tracks_df['y'].min(), self.tracks_df['y'].max()
                    padding_x = 15
                    padding_y = 10
                    self.background_extent = (x_min - padding_x, x_max + padding_x,
                                              y_min - padding_y, y_max + padding_y)
                    logger.info(f"Loaded road layout: {img_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load background image {img_path}: {e}")
                    self.background_image = None
                    self.background_extent = None
            else:
                self.background_image = None
                self.background_extent = None
            
            return True
            
        except Exception as e:
            logger.debug(f"Error loading recording {recording_id}: {e}")
            return False
    
    def get_heavy_vehicles(self) -> List[int]:
        """Get heavy vehicle IDs."""
        if self.tracks_meta_df is None:
            return []
        mask = self.tracks_meta_df['class'].str.lower().isin(self.config.HEAVY_VEHICLE_CLASSES)
        return self.tracks_meta_df[mask]['id'].tolist()
    
    def get_frames_for_vehicle(self, vehicle_id: int) -> np.ndarray:
        """Get all frames where vehicle appears."""
        if self.tracks_df is None:
            return np.array([])
        mask = self.tracks_df['id'] == vehicle_id
        return self.tracks_df[mask]['frame'].values
    
    def get_vehicle_state(self, vehicle_id: int, frame: int) -> Optional[VehicleState]:
        """Get vehicle state at specific frame."""
        if self.tracks_df is None:
            return None
        
        row = self.tracks_df[(self.tracks_df['id'] == vehicle_id) & 
                            (self.tracks_df['frame'] == frame)]
        if row.empty:
            return None
        
        row = row.iloc[0]
        vclass = str(row.get('class', 'car')).lower()
        driving_direction = row.get('drivingDirection', None)
        try:
            driving_direction = int(driving_direction) if pd.notna(driving_direction) else None
        except Exception:
            driving_direction = None
        
        return VehicleState(
            id=vehicle_id,
            x=float(row['x']),
            y=float(row['y']),
            vx=float(row.get('xVelocity', 0)),
            vy=float(row.get('yVelocity', 0)),
            heading=float(row.get('orientation', 0)),
            speed=np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
            length=float(row.get('length', row.get('width', 5.0))),
            width=float(row.get('veh_width', row.get('height', 2.0))),
            vehicle_class=vclass,
            mass=self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC,
            driving_direction=driving_direction
        )
    
    def get_surrounding_vehicles(self, ego: VehicleState, frame: int, 
                                  max_count: int = 6) -> List[VehicleState]:
        """Get closest surrounding vehicles."""
        if self.tracks_df is None:
            return []
        
        frame_data = self.tracks_df[self.tracks_df['frame'] == frame]
        candidates = []
        
        for _, row in frame_data.iterrows():
            if row['id'] == ego.id:
                continue
            
            # Only consider same driving direction when available
            ego_dir = ego.driving_direction
            other_dir = row.get('drivingDirection', None)
            if ego_dir is not None and pd.notna(other_dir):
                try:
                    other_dir_int = int(other_dir)
                except Exception:
                    other_dir_int = None
                if other_dir_int is not None and other_dir_int != ego_dir:
                    continue
            
            dx = row['x'] - ego.x
            dy = row['y'] - ego.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > self.config.OBS_RANGE:
                continue
            
            vclass = str(row.get('class', 'car')).lower()
            vehicle = VehicleState(
                id=int(row['id']),
                x=float(row['x']),
                y=float(row['y']),
                vx=float(row.get('xVelocity', 0)),
                vy=float(row.get('yVelocity', 0)),
                heading=float(row.get('orientation', 0)),
                speed=np.sqrt(row.get('xVelocity', 0)**2 + row.get('yVelocity', 0)**2),
                length=float(row.get('length', row.get('width', 4.5))),
                width=float(row.get('veh_width', row.get('height', 1.8))),
                vehicle_class=vclass,
                mass=self.config.MASS_HV if vclass in self.config.HEAVY_VEHICLE_CLASSES else self.config.MASS_PC
            )
            candidates.append((dist, vehicle))
        
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:max_count]]


# =============================================================================
# Occlusion Detector
# =============================================================================

class OcclusionDetector:
    """Detect occlusion scenarios."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def analyze_frame(self, ego: VehicleState, 
                      surrounding: List[VehicleState]) -> Dict:
        """Analyze occlusions in a single frame."""
        occluded_ids = []
        occluder_ids = []
        visibilities = []
        
        # Find potential occluders (heavy vehicles)
        occluders = [v for v in surrounding 
                    if v.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES]
        
        for occluder in occluders:
            # Check what this occluder blocks
            blocked = self._find_occluded_by(ego, occluder, surrounding)
            
            if blocked:
                occluder_ids.append(occluder.id)
                for v, vis in blocked:
                    if v.id not in occluded_ids:
                        occluded_ids.append(v.id)
                        visibilities.append(vis)
                        v.is_occluded = True
                        v.occluder_id = occluder.id
                        v.visibility = vis
        
        # Compute occlusion score
        if occluded_ids:
            avg_vis = np.mean(visibilities)
            min_vis = min(visibilities)
            # Score: more occluded vehicles + lower visibility = higher score
            score = len(occluded_ids) * (1 - avg_vis) + (1 - min_vis)
        else:
            avg_vis = 1.0
            min_vis = 1.0
            score = 0.0
        
        return {
            'n_occluded': len(occluded_ids),
            'n_occluders': len(set(occluder_ids)),
            'occluded_ids': occluded_ids,
            'occluder_ids': list(set(occluder_ids)),
            'avg_visibility': avg_vis,
            'min_visibility': min_vis,
            'occlusion_score': score
        }
    
    def _find_occluded_by(self, ego: VehicleState, occluder: VehicleState,
                          all_vehicles: List[VehicleState]) -> List[Tuple[VehicleState, float]]:
        """Find vehicles occluded by a specific occluder."""
        blocked = []
        
        # Work in ego frame so "front view" is aligned to +X
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        dx_occ = occluder.x - ego.x
        dy_occ = occluder.y - ego.y
        occ_x = dx_occ * cos_h - dy_occ * sin_h
        occ_y = dx_occ * sin_h + dy_occ * cos_h
        dist_occ = np.sqrt(occ_x**2 + occ_y**2)
        
        if dist_occ < 1.0:
            return blocked
        
        # Only consider occluders in front of ego
        if occ_x <= 0:
            return blocked
        
        occ_angle = np.arctan2(occ_y, occ_x)
        half_diag = np.sqrt(occluder.length**2 + occluder.width**2) / 2
        occ_angular_width = np.arctan2(half_diag, dist_occ)
        
        for v in all_vehicles:
            if v.id == occluder.id or v.id == ego.id:
                continue
            
            dx_v = v.x - ego.x
            dy_v = v.y - ego.y
            v_x = dx_v * cos_h - dy_v * sin_h
            v_y = dx_v * sin_h + dy_v * cos_h
            
            dist_v = np.sqrt(v_x**2 + v_y**2)
            
            # Must be behind occluder
            if dist_v <= dist_occ:
                continue
            
            # Only consider vehicles in front of ego
            if v_x <= 0:
                continue
            
            v_angle = np.arctan2(v_y, v_x)
            angle_diff = abs(np.arctan2(np.sin(v_angle - occ_angle), 
                                        np.cos(v_angle - occ_angle)))
            
            # Check if in shadow cone
            if angle_diff < occ_angular_width:
                # Compute visibility
                v_angular_width = np.arctan2(v.width/2, dist_v)
                total_width = occ_angular_width + v_angular_width
                
                if angle_diff >= total_width:
                    visibility = 1.0
                elif angle_diff <= occ_angular_width - v_angular_width:
                    visibility = 0.0
                else:
                    overlap = total_width - angle_diff
                    visibility = 1.0 - (overlap / (2 * v_angular_width))
                
                if visibility < 0.95:  # Consider occluded if < 95% visible
                    blocked.append((v, visibility))
        
        return blocked


# =============================================================================
# Scenario Scanner
# =============================================================================

class OcclusionScanner:
    """Scan all recordings for occlusion scenarios."""
    
    def __init__(self, base_dir: str, config: Config = None):
        self.base_dir = Path(base_dir)
        self.config = config or Config()
        self.loader = AD4CHELoader(base_dir)
        self.detector = OcclusionDetector(config)
        self.scenarios: List[OcclusionScenario] = []
    
    def scan_all_recordings(self, progress: bool = True) -> pd.DataFrame:
        """Scan all recordings and find occlusion scenarios."""
        folders = self.loader.get_recording_folders()
        
        if not folders:
            logger.error(f"No recording folders found in {self.base_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(folders)} recordings to scan")
        
        iterator = tqdm(folders, desc="Scanning recordings") if progress else folders
        
        for recording_id, folder_path in iterator:
            self._scan_recording(recording_id, folder_path)
        
        # Create DataFrame
        if self.scenarios:
            df = pd.DataFrame([s.to_dict() for s in self.scenarios])
            df = df.sort_values('occlusion_score', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    def _scan_recording(self, recording_id: int, folder_path: Path):
        """Scan a single recording for occlusion scenarios."""
        if not self.loader.load_recording(recording_id, folder_path):
            return
        
        heavy_ids = self.loader.get_heavy_vehicles()
        if not heavy_ids:
            return
        
        # For each heavy vehicle as potential ego
        for ego_id in heavy_ids:
            frames = self.loader.get_frames_for_vehicle(ego_id)
            if len(frames) == 0:
                continue
            
            # Sample frames
            sampled_frames = frames[::self.config.FRAME_SAMPLE_STEP]
            
            for frame in sampled_frames:
                ego = self.loader.get_vehicle_state(ego_id, frame)
                if ego is None:
                    continue
                
                surrounding = self.loader.get_surrounding_vehicles(ego, frame, 
                                                                   self.config.MAX_SURROUNDING)
                if len(surrounding) < 2:
                    continue
                
                # Analyze occlusions
                result = self.detector.analyze_frame(ego, surrounding)
                
                if result['occlusion_score'] >= self.config.MIN_OCCLUSION_SCORE:
                    scenario = OcclusionScenario(
                        recording_id=recording_id,
                        recording_folder=folder_path.name,
                        frame=int(frame),
                        ego_id=ego_id,
                        ego_class=ego.vehicle_class,
                        ego_speed=ego.speed,
                        n_surrounding=len(surrounding),
                        n_occluded=result['n_occluded'],
                        n_occluders=result['n_occluders'],
                        occluded_vehicle_ids=result['occluded_ids'],
                        occluder_vehicle_ids=result['occluder_ids'],
                        avg_visibility=result['avg_visibility'],
                        min_visibility=result['min_visibility'],
                        occlusion_score=result['occlusion_score']
                    )
                    self.scenarios.append(scenario)
    
    def save_catalog(self, output_path: str) -> pd.DataFrame:
        """Save scenario catalog to CSV."""
        if not self.scenarios:
            logger.warning("No scenarios to save")
            return pd.DataFrame()
        
        df = pd.DataFrame([s.to_dict() for s in self.scenarios])
        df = df.sort_values('occlusion_score', ascending=False)
        
        # Convert lists to strings for CSV
        df['occluded_vehicle_ids'] = df['occluded_vehicle_ids'].apply(str)
        df['occluder_vehicle_ids'] = df['occluder_vehicle_ids'].apply(str)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} scenarios to {output_path}")
        
        return df


# =============================================================================
# Interactive Scenario Selector
# =============================================================================

class ScenarioSelector:
    """Interactive scenario selection and analysis."""
    
    def __init__(self, catalog_path: str, base_dir: str):
        self.catalog = pd.read_csv(catalog_path)
        self.base_dir = Path(base_dir)
        self.config = Config()
    
    def show_top_scenarios(self, n: int = 20) -> pd.DataFrame:
        """Show top N scenarios by occlusion score."""
        top = self.catalog.nlargest(n, 'occlusion_score')
        
        print("\n" + "="*80)
        print("TOP OCCLUSION SCENARIOS")
        print("="*80)
        
        display_cols = ['recording_id', 'frame', 'ego_id', 'ego_class', 
                       'n_surrounding', 'n_occluded', 'min_visibility', 'occlusion_score']
        
        for idx, (_, row) in enumerate(top.iterrows()):
            print(f"\n[{idx}] Recording {row['recording_id']:02d} ({row['recording_folder']}) "
                  f"| Frame {row['frame']} | Ego: {row['ego_class']} #{row['ego_id']}")
            print(f"    Surrounding: {row['n_surrounding']} | Occluded: {row['n_occluded']} | "
                  f"Min Vis: {row['min_visibility']:.2f} | Score: {row['occlusion_score']:.3f}")
        
        print("\n" + "="*80)
        return top
    
    def get_scenario(self, index: int) -> Optional[Dict]:
        """Get scenario details by index in top list."""
        if index >= len(self.catalog):
            return None
        
        row = self.catalog.iloc[index]
        return {
            'recording_id': int(row['recording_id']),
            'recording_folder': row['recording_folder'],
            'frame': int(row['frame']),
            'ego_id': int(row['ego_id']),
            'base_dir': str(self.base_dir),
        }
    
    def filter_scenarios(self, min_occluded: int = 1, 
                        min_score: float = 0.5,
                        ego_class: str = None) -> pd.DataFrame:
        """Filter scenarios by criteria."""
        filtered = self.catalog[
            (self.catalog['n_occluded'] >= min_occluded) &
            (self.catalog['occlusion_score'] >= min_score)
        ]
        
        if ego_class:
            filtered = filtered[filtered['ego_class'] == ego_class]
        
        return filtered.sort_values('occlusion_score', ascending=False)
    
    def get_recording_summary(self) -> pd.DataFrame:
        """Get summary statistics by recording."""
        summary = self.catalog.groupby(['recording_id', 'recording_folder']).agg({
            'frame': 'count',
            'n_occluded': 'sum',
            'occlusion_score': ['mean', 'max']
        }).reset_index()
        
        summary.columns = ['recording_id', 'folder', 'n_scenarios', 
                          'total_occluded', 'avg_score', 'max_score']
        
        return summary.sort_values('max_score', ascending=False)


# =============================================================================
# Interaction Field Analysis (from previous artifact)
# =============================================================================

class OcclusionShadow:
    """Shadow cone data."""
    def __init__(self, occluder_id, occluded_ids, polygon, cone_near, cone_far):
        self.occluder_id = occluder_id
        self.occluded_ids = occluded_ids
        self.polygon = polygon
        self.cone_near_dist = cone_near
        self.cone_far_dist = cone_far


class InteractionFieldAnalyzer:
    """Analyze selected occlusion scenarios."""
    
    def __init__(self, base_dir: str, config: Config = None):
        self.base_dir = Path(base_dir)
        self.config = config or Config()
        self.loader = AD4CHELoader(base_dir)
        self.detector = OcclusionDetector(config)
        self.road_image = None
        self.road_image_key = None
    
    def _load_road_image(self, recording_id: int, recording_folder: str):
        """Load cached road layout image if available for this recording."""
        key = (recording_id, recording_folder)
        if self.road_image_key == key and self.road_image is not None:
            return self.road_image
        
        folder_path = self.base_dir / recording_folder
        img_path = folder_path / f"{recording_id:02d}_highway.png"
        if not img_path.exists():
            logger.debug(f"No road layout image found at {img_path}")
            self.road_image = None
            self.road_image_key = key
            return None
        
        try:
            self.road_image = plt.imread(img_path)
            self.road_image_key = key
            logger.info(f"Loaded road layout: {img_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load road layout image {img_path}: {e}")
            self.road_image = None
            self.road_image_key = key
        
        return self.road_image
    
    def analyze_scenario(self, recording_id: int, recording_folder: str,
                        frame: int, ego_id: int,
                        output_dir: str = './output_interaction',
                        create_animation: bool = False,
                        animation_frames: int = 60,
                        animation_step: int = 2) -> Dict:
        """Run full analysis on a selected scenario (optionally with animation)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load recording
        folder_path = self.base_dir / recording_folder
        if not self.loader.load_recording(recording_id, folder_path):
            logger.error(f"Failed to load recording {recording_id}")
            return {}
        
        # Load road layout image for this recording if available
        self._load_road_image(recording_id, recording_folder)
        
        # Get ego and surrounding
        ego = self.loader.get_vehicle_state(ego_id, frame)
        if ego is None:
            logger.error(f"Ego vehicle {ego_id} not found at frame {frame}")
            return {}
        
        surrounding = self.loader.get_surrounding_vehicles(ego, frame, self.config.MAX_SURROUNDING)
        
        # Analyze occlusions
        result = self.detector.analyze_frame(ego, surrounding)
        
        # Compute shadow polygons
        shadows = self._compute_shadows(ego, surrounding)
        
        # Compute fields
        fields = self._compute_fields(ego, surrounding, shadows)
        
        # Create visualization
        fig_path = output_path / f'occlusion_rec{recording_id}_ego{ego_id}_frame{frame}.png'
        self._create_visualization(ego, surrounding, shadows, fields, frame, 
                                   recording_folder, str(fig_path))
        
        # Save metrics
        metrics = {
            'recording_id': recording_id,
            'recording_folder': recording_folder,
            'frame': frame,
            'ego_id': ego_id,
            'ego_class': ego.vehicle_class,
            'ego_speed_kmh': ego.speed * 3.6,
            'n_surrounding': len(surrounding),
            **result,
            'output_figure': str(fig_path)
        }
        
        animation_path = None
        if create_animation:
            animation_path = self.create_animation(
                recording_id, recording_folder, ego_id, frame,
                output_dir, animation_frames, animation_step
            )
            if animation_path:
                metrics['animation'] = str(animation_path)
        
        metrics_path = output_path / f'metrics_rec{recording_id}_ego{ego_id}_frame{frame}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Analysis complete: {fig_path}")
        return metrics
    
    def _compute_shadows(self, ego: VehicleState, 
                        surrounding: List[VehicleState]) -> List[OcclusionShadow]:
        """Compute shadow cones for visualization."""
        shadows = []
        
        occluders = [v for v in surrounding 
                    if v.vehicle_class in self.config.HEAVY_VEHICLE_CLASSES]
        
        for occluder in occluders:
            dx = occluder.x - ego.x
            dy = occluder.y - ego.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 1.0:
                continue
            
            center_angle = np.arctan2(dy, dx)
            half_diag = np.sqrt(occluder.length**2 + occluder.width**2) / 2
            angular_width = np.arctan2(half_diag, dist)
            
            # Find occluded vehicles
            occluded = []
            for v in surrounding:
                if v.is_occluded and v.occluder_id == occluder.id:
                    occluded.append(v.id)
            
            if not occluded:
                continue
            
            # Create polygon
            far_dist = self.config.OCCLUSION_RANGE
            polygon = np.array([
                [ego.x + dist * np.cos(center_angle + angular_width),
                 ego.y + dist * np.sin(center_angle + angular_width)],
                [ego.x + dist * np.cos(center_angle - angular_width),
                 ego.y + dist * np.sin(center_angle - angular_width)],
                [ego.x + far_dist * np.cos(center_angle - angular_width),
                 ego.y + far_dist * np.sin(center_angle - angular_width)],
                [ego.x + far_dist * np.cos(center_angle + angular_width),
                 ego.y + far_dist * np.sin(center_angle + angular_width)],
            ])
            
            shadows.append(OcclusionShadow(occluder.id, occluded, polygon, dist, far_dist))
        
        return shadows
    
    def _compute_fields(self, ego: VehicleState, surrounding: List[VehicleState],
                       shadows: List[OcclusionShadow]) -> Dict:
        """Compute interaction fields."""
        # Grid in ego frame
        x = np.linspace(-30, 60, self.config.GRID_NX)
        y = np.linspace(-20, 20, self.config.GRID_NY)
        X, Y = np.meshgrid(x, y)
        
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        repulsion = np.zeros_like(X)
        velocity_risk = np.zeros_like(X)
        occlusion_uncertainty = np.zeros_like(X)
        
        for other in surrounding:
            # Transform to ego frame
            dx = other.x - ego.x
            dy = other.y - ego.y
            x_rel = dx * cos_h - dy * sin_h
            y_rel = dx * sin_h + dy * cos_h
            
            # Repulsion
            sigma_x = self.config.REPULSION_DECAY_LONG
            sigma_y = self.config.REPULSION_DECAY_LAT
            mass_factor = np.sqrt(other.mass / self.config.MASS_PC)
            
            repulsion += self.config.REPULSION_AMPLITUDE * mass_factor * np.exp(
                -0.5 * ((X - x_rel)**2 / sigma_x**2 + (Y - y_rel)**2 / sigma_y**2)
            )
            
            # Velocity risk
            dvx = other.vx - ego.vx
            dvy = other.vy - ego.vy
            vx_rel = dvx * cos_h - dvy * sin_h
            vy_rel = dvx * sin_h + dvy * cos_h
            rel_speed = np.sqrt(vx_rel**2 + vy_rel**2)
            
            if rel_speed > 0.5:
                dist_grid = np.sqrt((X - x_rel)**2 + (Y - y_rel)**2) + 0.1
                velocity_risk += self.config.VELOCITY_RISK_WEIGHT * rel_speed * np.exp(-dist_grid / 15.0)
        
        # Occlusion uncertainty
        for shadow in shadows:
            # Transform polygon to ego frame
            poly_ego = []
            for pt in shadow.polygon:
                dx = pt[0] - ego.x
                dy = pt[1] - ego.y
                poly_ego.append([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
            poly_ego = np.array(poly_ego)
            
            # Check points in polygon (simplified)
            from matplotlib.path import Path as MplPath
            path = MplPath(poly_ego)
            points = np.column_stack([X.ravel(), Y.ravel()])
            inside = path.contains_points(points).reshape(X.shape)
            
            # Distance from shadow start
            shadow_depth = np.sqrt(X**2 + Y**2) - shadow.cone_near_dist
            shadow_depth = np.maximum(shadow_depth, 0)
            
            uncertainty = np.where(
                inside,
                self.config.SHADOW_UNCERTAINTY_BASE + self.config.SHADOW_UNCERTAINTY_GROWTH * shadow_depth,
                0
            )
            occlusion_uncertainty = np.maximum(occlusion_uncertainty, uncertainty)
        
        # Combined risk
        combined = (
            0.4 * repulsion / (repulsion.max() + 1e-10) +
            0.3 * velocity_risk / (velocity_risk.max() + 1e-10) +
            0.3 * occlusion_uncertainty / (occlusion_uncertainty.max() + 1e-10)
        )
        
        return {
            'X': X, 'Y': Y,
            'repulsion': repulsion,
            'velocity_risk': velocity_risk,
            'occlusion_uncertainty': occlusion_uncertainty,
            'combined_risk': combined
        }
    
    def _create_visualization(self, ego: VehicleState, surrounding: List[VehicleState],
                             shadows: List[OcclusionShadow], fields: Dict,
                             frame: int, folder: str, output_path: str):
        """Create visualization figure."""
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(self.config.BG_DARK)
        
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Traffic scene
        ax2 = fig.add_subplot(gs[0, 1])  # Repulsion
        ax3 = fig.add_subplot(gs[0, 2])  # Velocity risk
        ax4 = fig.add_subplot(gs[1, 0])  # Occlusion uncertainty
        ax5 = fig.add_subplot(gs[1, 1])  # Combined risk
        ax6 = fig.add_subplot(gs[1, 2])  # Summary
        
        X, Y = fields['X'], fields['Y']
        cos_h = np.cos(-ego.heading)
        sin_h = np.sin(-ego.heading)
        
        # 1. Traffic scene
        self._plot_scene(ax1, ego, surrounding, shadows, X, Y, cos_h, sin_h)
        
        # 2. Repulsion field
        self._plot_field(ax2, X, Y, fields['repulsion'], 'Repulsion Field', 'hot_r',
                        ego, surrounding, cos_h, sin_h)
        
        # 3. Velocity risk
        self._plot_field(ax3, X, Y, fields['velocity_risk'], 'Velocity Risk Field', 'YlOrRd',
                        ego, surrounding, cos_h, sin_h)
        
        # 4. Occlusion uncertainty
        self._plot_occlusion(ax4, X, Y, fields['occlusion_uncertainty'], 
                            ego, surrounding, shadows, cos_h, sin_h)
        
        # 5. Combined risk
        self._plot_combined(ax5, X, Y, fields['combined_risk'],
                           ego, surrounding, shadows, cos_h, sin_h)
        
        # 6. Summary
        self._plot_summary(ax6, ego, surrounding, shadows, fields)
        
        # Title
        n_occluded = sum(1 for v in surrounding if v.is_occluded)
        fig.suptitle(
            f"Occlusion Scenario Analysis | {folder} | Frame {frame} | "
            f"Ego: {ego.vehicle_class} #{ego.id} | Occluded: {n_occluded}",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
    
    def create_animation(self, recording_id: int, recording_folder: str,
                        ego_id: int, center_frame: int, output_dir: str,
                        num_frames: int = 60, step: int = 2) -> Optional[Path]:
        """Create animation of traffic and combined risk evolution around a frame."""
        folder_path = self.base_dir / recording_folder
        if self.loader.current_recording_id != recording_id:
            if not self.loader.load_recording(recording_id, folder_path):
                logger.error(f"Failed to load recording {recording_id} for animation")
                return None
        
        # Ensure road layout is available if present
        self._load_road_image(recording_id, recording_folder)
        
        frames_available = sorted(self.loader.tracks_df['frame'].unique()) if self.loader.tracks_df is not None else []
        if not frames_available:
            logger.warning("No frames available for animation")
            return None
        
        half_window = num_frames // 2
        frame_min = center_frame - half_window * step
        frame_max = center_frame + half_window * step
        frames = [f for f in frames_available if frame_min <= f <= frame_max]
        frames = frames[::step]
        
        if not frames:
            logger.warning("No frames in requested window for animation")
            return None
        
        # Fix surrounding vehicle IDs based on center frame to avoid switching neighbors
        center_ego = self.loader.get_vehicle_state(ego_id, center_frame)
        if center_ego is None:
            logger.warning(f"Ego vehicle {ego_id} missing at frame {center_frame} for animation")
            return None
        anchor_surrounding = self.loader.get_surrounding_vehicles(center_ego, center_frame, self.config.MAX_SURROUNDING)
        anchor_ids = [v.id for v in anchor_surrounding]
        
        precomputed = []
        for frame_id in frames:
            ego = self.loader.get_vehicle_state(ego_id, frame_id)
            if ego is None:
                continue
            surrounding = []
            for vid in anchor_ids:
                v_state = self.loader.get_vehicle_state(vid, frame_id)
                if v_state is not None:
                    surrounding.append(v_state)
            self.detector.analyze_frame(ego, surrounding)
            shadows = self._compute_shadows(ego, surrounding)
            fields = self._compute_fields(ego, surrounding, shadows)
            precomputed.append({
                'frame': frame_id,
                'ego': ego,
                'surrounding': surrounding,
                'shadows': shadows,
                'fields': fields
            })
        
        if not precomputed:
            logger.warning("No valid frames for animation after filtering")
            return None
        
        max_risk = max(np.max(item['fields']['combined_risk']) for item in precomputed)
        vmax = max(max_risk, 1e-6)
        
        fig, (ax_scene, ax_comb) = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor(self.config.BG_DARK)
        
        pcm_ref = {'obj': None}
        
        def update(idx):
            ax_scene.clear()
            ax_comb.clear()
            data = precomputed[idx]
            ego = data['ego']
            surrounding = data['surrounding']
            shadows = data['shadows']
            fields = data['fields']
            X, Y = fields['X'], fields['Y']
            cos_h = np.cos(-ego.heading)
            sin_h = np.sin(-ego.heading)
            
            # Use center-frame ego to stabilize background alignment across frames
            road_ref = center_ego
            self._plot_scene(ax_scene, ego, surrounding, shadows, X, Y, cos_h, sin_h,
                             road_ref_ego=road_ref)
            ax_scene.set_title(f"Traffic Scene | Frame {data['frame']}", fontsize=11, fontweight='bold', color='white')
            
            pcm = self._plot_combined(
                ax_comb, X, Y, fields['combined_risk'], ego, surrounding, shadows, cos_h, sin_h,
                show_colorbar=False, vmin=0, vmax=vmax, road_ref_ego=road_ref
            )
            ax_comb.set_title(f"Combined Risk Field | Frame {data['frame']}", fontsize=11, fontweight='bold', color='white')
            
            if pcm_ref['obj'] is None:
                pcm_ref['obj'] = pcm
                cbar = fig.colorbar(pcm, ax=ax_comb, shrink=0.8)
                cbar.set_label('Risk', color='white', fontsize=9)
                cbar.ax.tick_params(colors='white', labelsize=7)
            
            return []
        
        anim = FuncAnimation(fig, update, frames=len(precomputed), interval=150, blit=False, repeat=False)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        gif_path = output_path / f'occlusion_anim_rec{recording_id}_ego{ego_id}_frame{center_frame}.gif'
        
        try:
            writer = PillowWriter(fps=10)
            anim.save(str(gif_path), writer=writer, dpi=100,
                      savefig_kwargs={'facecolor': fig.get_facecolor()})
            logger.info(f"Saved animation: {gif_path}")
            return gif_path
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
            return None
        finally:
            plt.close(fig)
    
    def _setup_ax(self, ax, title):
        ax.set_facecolor(self.config.BG_PANEL)
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('X (m)', color='white', fontsize=9)
        ax.set_ylabel('Y (m)', color='white', fontsize=9)
        for spine in ax.spines.values():
            spine.set_color('#30363D')
    
    def _draw_road_layout(self, ax, ego: VehicleState, X, Y, draw_fill: bool = True):
        """Draw road layout using dataset image when available, otherwise procedural lanes.
        Background image is aligned by translating to ego at origin and rotating so ego faces +X."""
        if self.loader and self.loader.background_image is not None and self.loader.background_extent is not None:
            x_min, x_max, y_min, y_max = self.loader.background_extent
            trans = transforms.Affine2D().translate(-ego.x, -ego.y).rotate(-ego.heading) + ax.transData
            ax.imshow(self.loader.background_image, extent=(x_min, x_max, y_min, y_max),
                      origin='lower', alpha=0.6, zorder=0, transform=trans, aspect='equal')
            return
        if getattr(self, 'road_image', None) is not None:
            ax.imshow(self.road_image, extent=(X.min(), X.max(), Y.min(), Y.max()),
                      origin='lower', alpha=0.6, zorder=0)
            return
        lane_w = self.config.LANE_WIDTH
        n_left = self.config.LANES_LEFT
        n_right = self.config.LANES_RIGHT
        x_min, x_max = X.min(), X.max()
        road_bottom = -n_right * lane_w
        road_top = n_left * lane_w
        
        if draw_fill:
            road_rect = mpatches.Rectangle(
                (x_min, road_bottom), x_max - x_min, road_top - road_bottom,
                facecolor=self.config.ROAD_COLOR, edgecolor='none', alpha=0.25, zorder=0
            )
            ax.add_patch(road_rect)
        
        # Lane markings
        for i in range(-n_right, n_left + 1):
            y = i * lane_w
            is_center = i == 0
            ax.axhline(
                y, color=self.config.CENTERLINE_COLOR if is_center else self.config.LANE_MARK_COLOR,
                linestyle='-' if is_center else '--',
                linewidth=1.5 if is_center else 1.0,
                alpha=0.6, zorder=2
            )
        
        # Shoulders
        ax.axhline(road_bottom, color=self.config.LANE_MARK_COLOR, linewidth=1.2, alpha=0.8, zorder=2)
        ax.axhline(road_top, color=self.config.LANE_MARK_COLOR, linewidth=1.2, alpha=0.8, zorder=2)
    
    def _draw_vehicle(self, ax, veh, ego, cos_h, sin_h, is_ego=False):
        if is_ego:
            x_rel, y_rel = 0, 0
            color = self.config.COLORS['ego']
            ec = 'yellow'
            lw = 3
        else:
            dx = veh.x - ego.x
            dy = veh.y - ego.y
            x_rel = dx * cos_h - dy * sin_h
            y_rel = dx * sin_h + dy * cos_h
            color = self.config.COLORS['occluded'] if veh.is_occluded else self.config.COLORS.get(veh.vehicle_class, '#3498DB')
            ec = 'red' if veh.is_occluded else 'white'
            lw = 2 if veh.is_occluded else 1.5
        
        rect = mpatches.FancyBboxPatch(
            (x_rel - veh.length/2, y_rel - veh.width/2),
            veh.length, veh.width,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor=ec, linewidth=lw, alpha=0.9, zorder=3
        )
        ax.add_patch(rect)
        
        label = 'EGO' if is_ego else f'{veh.id}'
        if veh.is_occluded:
            label += f'\nvis:{veh.visibility:.1f}'
        ax.text(x_rel, y_rel + veh.width/2 + 1.2, label,
               ha='center', va='bottom', fontsize=8,
               color='yellow' if is_ego else ('red' if veh.is_occluded else 'white'),
               fontweight='bold' if is_ego else 'normal', zorder=4)
    
    def _plot_scene(self, ax, ego, surrounding, shadows, X, Y, cos_h, sin_h,
                    road_ref_ego: Optional[VehicleState] = None):
        self._setup_ax(ax, 'Traffic Scene with Occlusion Shadows')
        layout_ego = road_ref_ego or ego
        self._draw_road_layout(ax, layout_ego, X, Y, draw_fill=True)
        
        # Draw shadows
        for shadow in shadows:
            poly_ego = []
            for pt in shadow.polygon:
                dx = pt[0] - ego.x
                dy = pt[1] - ego.y
                poly_ego.append([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
            
            patch = plt.Polygon(poly_ego, facecolor=self.config.SHADOW_COLOR, alpha=0.5,
                               edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(patch)
            
            center = np.mean(poly_ego, axis=0)
            ax.text(center[0], center[1], f'SHADOW\n(by #{shadow.occluder_id})',
                   ha='center', va='center', fontsize=8, color='red', alpha=0.8)
        
        # Draw vehicles
        self._draw_vehicle(ax, ego, ego, cos_h, sin_h, is_ego=True)
        for v in surrounding:
            self._draw_vehicle(ax, v, ego, cos_h, sin_h)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal')
        
        # Legend
        handles = [
            mpatches.Patch(fc=self.config.COLORS['ego'], ec='yellow', label='Ego'),
            mpatches.Patch(fc=self.config.COLORS['car'], ec='white', label='Visible'),
            mpatches.Patch(fc=self.config.COLORS['occluded'], ec='red', label='Occluded'),
            mpatches.Patch(fc=self.config.SHADOW_COLOR, ec='red', ls='--', alpha=0.5, label='Shadow'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=7,
                 facecolor=self.config.BG_PANEL, edgecolor='white', labelcolor='white')
    
    def _plot_field(self, ax, X, Y, field, title, cmap, ego, surrounding, cos_h, sin_h,
                    road_ref_ego: Optional[VehicleState] = None):
        self._setup_ax(ax, title)
        
        vmax = np.percentile(field, 98) if field.max() > 0 else 1
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading='gouraud', vmin=0, vmax=vmax)
        
        if field.max() > 0:
            ax.contour(X, Y, field, levels=5, colors='white', linewidths=0.5, alpha=0.5)
        
        layout_ego = road_ref_ego or ego
        self._draw_road_layout(ax, layout_ego, X, Y, draw_fill=False)
        
        self._draw_vehicle(ax, ego, ego, cos_h, sin_h, is_ego=True)
        for v in surrounding:
            self._draw_vehicle(ax, v, ego, cos_h, sin_h)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal')
        plt.colorbar(pcm, ax=ax, shrink=0.8).ax.tick_params(colors='white', labelsize=7)
    
    def _plot_occlusion(self, ax, X, Y, field, ego, surrounding, shadows, cos_h, sin_h,
                        road_ref_ego: Optional[VehicleState] = None):
        self._setup_ax(ax, 'Occlusion Uncertainty Field')
        
        cmap = LinearSegmentedColormap.from_list('unc', ['#161B22', '#4A4A6A', '#8B5CF6', '#EC4899'])
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading='gouraud')
        layout_ego = road_ref_ego or ego
        self._draw_road_layout(ax, layout_ego, X, Y, draw_fill=False)
        
        # Shadow outlines
        for shadow in shadows:
            poly_ego = [[
                (pt[0] - ego.x) * cos_h - (pt[1] - ego.y) * sin_h,
                (pt[0] - ego.x) * sin_h + (pt[1] - ego.y) * cos_h
            ] for pt in shadow.polygon]
            ax.plot([p[0] for p in poly_ego] + [poly_ego[0][0]],
                   [p[1] for p in poly_ego] + [poly_ego[0][1]], 'r--', lw=2)
        
        self._draw_vehicle(ax, ego, ego, cos_h, sin_h, is_ego=True)
        for v in surrounding:
            self._draw_vehicle(ax, v, ego, cos_h, sin_h)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal')
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
        cbar.set_label('Uncertainty', color='white', fontsize=9)
        cbar.ax.tick_params(colors='white', labelsize=7)
    
    def _plot_combined(self, ax, X, Y, field, ego, surrounding, shadows, cos_h, sin_h,
                      show_colorbar: bool = True, vmin: float = None, vmax: float = None,
                      road_ref_ego: Optional[VehicleState] = None):
        self._setup_ax(ax, 'Combined Risk Field')
        
        cmap = LinearSegmentedColormap.from_list('risk', 
            ['#0D1117', '#1E3A5F', '#22C55E', '#EAB308', '#EF4444'])
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading='gouraud',
                            vmin=vmin, vmax=vmax)
        layout_ego = road_ref_ego or ego
        self._draw_road_layout(ax, layout_ego, X, Y, draw_fill=False)
        
        # Gradient arrows
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        grad_x = -np.gradient(field, dx, axis=1)
        grad_y = -np.gradient(field, dy, axis=0)
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        
        skip = 6
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 grad_x[::skip, ::skip]/mag[::skip, ::skip],
                 grad_y[::skip, ::skip]/mag[::skip, ::skip],
                 color='white', alpha=0.5, scale=25)
        
        # Shadow outlines
        for shadow in shadows:
            poly_ego = [[
                (pt[0] - ego.x) * cos_h - (pt[1] - ego.y) * sin_h,
                (pt[0] - ego.x) * sin_h + (pt[1] - ego.y) * cos_h
            ] for pt in shadow.polygon]
            ax.plot([p[0] for p in poly_ego] + [poly_ego[0][0]],
                   [p[1] for p in poly_ego] + [poly_ego[0][1]], 'r--', lw=1.5, alpha=0.7)
        
        self._draw_vehicle(ax, ego, ego, cos_h, sin_h, is_ego=True)
        for v in surrounding:
            self._draw_vehicle(ax, v, ego, cos_h, sin_h)
        
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal')
        if show_colorbar:
            cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
            cbar.set_label('Risk', color='white', fontsize=9)
            cbar.ax.tick_params(colors='white', labelsize=7)
        return pcm
    
    def _plot_summary(self, ax, ego, surrounding, shadows, fields):
        ax.set_facecolor(self.config.BG_PANEL)
        ax.axis('off')
        
        n_occ = sum(1 for v in surrounding if v.is_occluded)
        max_risk = fields['combined_risk'].max()
        
        if max_risk > 0.7:
            level, color = "HIGH", "#EF4444"
        elif max_risk > 0.4:
            level, color = "MODERATE", "#EAB308"
        else:
            level, color = "LOW", "#22C55E"
        
        lines = [
            "═══ OCCLUSION ANALYSIS ═══",
            "",
            f"Ego: {ego.vehicle_class.upper()} #{ego.id}",
            f"Speed: {ego.speed * 3.6:.1f} km/h",
            "",
            "─── Vehicles ───",
            f"Surrounding: {len(surrounding)}",
            f"Occluded: {n_occ}",
            f"Shadow cones: {len(shadows)}",
            "",
            "─── Occluded Details ───",
        ]
        
        for v in surrounding:
            if v.is_occluded:
                lines.append(f"  #{v.id}: vis={v.visibility:.2f} by #{v.occluder_id}")
        
        lines.extend([
            "",
            "─── Risk ───",
            f"Max combined: {max_risk:.3f}",
            f"Level: {level}",
        ])
        
        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
               fontsize=9, color='white', family='monospace', va='top')
        
        ax.add_patch(mpatches.Rectangle(
            (0.65, 0.05), 0.3, 0.12, transform=ax.transAxes,
            facecolor=color, edgecolor='white', linewidth=2
        ))
        ax.text(0.8, 0.11, level, transform=ax.transAxes,
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')


# =============================================================================
# Main Functions
# =============================================================================

def scan_dataset(base_dir: str, output_path: str):
    """Scan entire dataset for occlusion scenarios."""
    scanner = OcclusionScanner(base_dir)
    catalog = scanner.scan_all_recordings(progress=True)
    
    if not catalog.empty:
        scanner.save_catalog(output_path)
        
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)
        print(f"Total scenarios found: {len(catalog)}")
        print(f"Recordings with occlusions: {catalog['recording_id'].nunique()}")
        print(f"Catalog saved to: {output_path}")
        
        print("\nTop 10 scenarios by occlusion score:")
        top10 = catalog.nlargest(10, 'occlusion_score')
        for _, row in top10.iterrows():
            print(f"  Rec {row['recording_id']:02d} | Frame {row['frame']:5d} | "
                  f"Ego #{row['ego_id']:5d} | Occluded: {row['n_occluded']} | "
                  f"Score: {row['occlusion_score']:.3f}")
    
    return catalog


def analyze_from_catalog(catalog_path: str, base_dir: str, output_dir: str,
                        top_n: int = None, index: int = None):
    """Analyze scenarios from catalog."""
    selector = ScenarioSelector(catalog_path, base_dir)
    analyzer = InteractionFieldAnalyzer(base_dir)
    
    if index is not None:
        # Analyze specific scenario by index
        top = selector.catalog.nlargest(100, 'occlusion_score')
        if index >= len(top):
            print(f"Index {index} out of range")
            return
        
        row = top.iloc[index]
        folder_path = Path(base_dir) / row['recording_folder']
        
        analyzer.analyze_scenario(
            int(row['recording_id']),
            row['recording_folder'],
            int(row['frame']),
            int(row['ego_id']),
            output_dir
        )
    
    elif top_n is not None:
        # Analyze top N scenarios
        top = selector.catalog.nlargest(top_n, 'occlusion_score')
        
        for _, row in tqdm(top.iterrows(), total=len(top), desc="Analyzing scenarios"):
            analyzer.analyze_scenario(
                int(row['recording_id']),
                row['recording_folder'],
                int(row['frame']),
                int(row['ego_id']),
                output_dir
            )


def interactive_select(catalog_path: str, base_dir: str, output_dir: str):
    """Interactive scenario selection."""
    selector = ScenarioSelector(catalog_path, base_dir)
    analyzer = InteractionFieldAnalyzer(base_dir)
    
    while True:
        print("\n" + "="*60)
        print("OCCLUSION SCENARIO SELECTOR")
        print("="*60)
        print("Commands:")
        print("  top [N]    - Show top N scenarios (default 20)")
        print("  rec        - Show recording summary")
        print("  analyze N  - Analyze scenario at index N")
        print("  batch N    - Analyze top N scenarios")
        print("  filter     - Filter scenarios")
        print("  quit       - Exit")
        print("="*60)
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd.startswith('top'):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 20
            selector.show_top_scenarios(n)
        
        elif cmd == 'rec':
            summary = selector.get_recording_summary()
            print("\nRecording Summary:")
            print(summary.to_string())
        
        elif cmd.startswith('analyze'):
            parts = cmd.split()
            if len(parts) < 2:
                print("Usage: analyze N")
                continue
            idx = int(parts[1])
            scenario = selector.get_scenario(idx)
            if scenario:
                analyzer.analyze_scenario(
                    scenario['recording_id'],
                    scenario['recording_folder'],
                    scenario['frame'],
                    scenario['ego_id'],
                    output_dir
                )
        
        elif cmd.startswith('batch'):
            parts = cmd.split()
            if len(parts) < 2:
                print("Usage: batch N")
                continue
            n = int(parts[1])
            analyze_from_catalog(catalog_path, base_dir, output_dir, top_n=n)
        
        elif cmd == 'filter':
            min_occ = int(input("Min occluded vehicles (default 1): ") or 1)
            min_score = float(input("Min score (default 0.5): ") or 0.5)
            filtered = selector.filter_scenarios(min_occ, min_score)
            print(f"\nFiltered {len(filtered)} scenarios:")
            print(filtered.head(20).to_string())
        
        elif cmd in ['quit', 'exit', 'q']:
            break
        
        else:
            print("Unknown command")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AD4CHE Occlusion Scenario Scanner')
    parser.add_argument('--base_dir', type=str, default='C:/field_modeling/data/AD4CHE',
                       help='Base directory containing DJI_XXXX folders')
    parser.add_argument('--output_dir', type=str, default='./output_ad4che_occlusion',
                       help='Output directory for analysis results')
    parser.add_argument('--catalog', type=str, default='./occlusion_catalog.csv',
                       help='Path to occlusion catalog CSV')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan dataset for occlusion scenarios')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze scenarios from catalog')
    analyze_parser.add_argument('--top', type=int, default=None, help='Analyze top N scenarios')
    analyze_parser.add_argument('--index', type=int, default=None, help='Analyze specific index')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive scenario selection')
    
    # Single scenario command
    single_parser = subparsers.add_parser('single', help='Analyze single scenario')
    single_parser.add_argument('--recording', type=int, required=True)
    single_parser.add_argument('--frame', type=int, required=True)
    single_parser.add_argument('--ego_id', type=int, required=True)
    single_parser.add_argument('--animate', action='store_true',
                               help='Generate animation around the selected frame')
    single_parser.add_argument('--animation_frames', type=int, default=60,
                               help='Number of frames to include in animation window')
    single_parser.add_argument('--animation_step', type=int, default=2,
                               help='Frame stride for animation window')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.command == 'scan':
        scan_dataset(args.base_dir, args.catalog)
    
    elif args.command == 'analyze':
        if not Path(args.catalog).exists():
            print(f"Catalog not found: {args.catalog}")
            print("Run 'scan' first to create the catalog.")
        else:
            analyze_from_catalog(args.catalog, args.base_dir, args.output_dir,
                               args.top, args.index)
    
    elif args.command == 'interactive':
        if not Path(args.catalog).exists():
            print(f"Catalog not found: {args.catalog}")
            print("Run 'scan' first to create the catalog.")
        else:
            interactive_select(args.catalog, args.base_dir, args.output_dir)
    
    elif args.command == 'single':
        # Find folder for recording
        loader = AD4CHELoader(args.base_dir)
        folders = loader.get_recording_folders()
        folder = next((f for rid, f in folders if rid == args.recording), None)
        
        if folder is None:
            print(f"Recording {args.recording} not found")
        else:
            analyzer = InteractionFieldAnalyzer(args.base_dir)
            analyzer.analyze_scenario(args.recording, folder.name, args.frame,
                                     args.ego_id, args.output_dir,
                                     create_animation=args.animate,
                                     animation_frames=args.animation_frames,
                                     animation_step=args.animation_step)
    
    else:
        parser.print_help()
