"""
Behavioral Alignment Metrics for Risk Field Evaluation
=======================================================
Complementary metrics to assess how well risk field critical transitions
align with actual driver decision-making behavior in safety-critical scenarios.

Core Insight (from Wang et al. 2024):
"If the risk field is robust and informative, the critical region 
(moderate → high risk transition) should align with regions where 
drivers actually execute safety-critical decisions."

Metrics Categories:
1. Spatial Alignment Metrics - Do risk transitions occur where decisions happen?
2. Temporal Coherence Metrics - Do risk changes precede/coincide with actions?
3. Behavioral Validity Metrics - Do risk zones predict distinct behavioral phases?
4. Decision Sensitivity Metrics - Does risk gradient signal maneuver necessity?

Reference: Wang et al. (2024) "Modeling risk potential fields for mandatory 
lane changes in intelligent connected vehicle environment"

Author: Research Implementation
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy import ndimage, signal, stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import silhouette_score, adjusted_rand_score
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Enums and Configuration
# =============================================================================

class ManeuverType(Enum):
    """Types of safety-critical maneuvers."""
    LANE_CHANGE_LEFT = 'lc_left'
    LANE_CHANGE_RIGHT = 'lc_right'
    DECELERATION = 'decel'
    ACCELERATION = 'accel'
    EMERGENCY_BRAKE = 'emergency_brake'
    MERGE = 'merge'
    OVERTAKE = 'overtake'
    YIELD = 'yield'
    NONE = 'none'


class RiskZone(Enum):
    """Risk zone classifications (following Wang et al.)."""
    HIGH = 'H-TPF'      # High risk - preparatory phase
    MODERATE = 'M-TPF'  # Moderate risk - decision/execution phase
    LOW = 'L-TPF'       # Low risk - post-maneuver phase


@dataclass
class BehavioralAlignmentConfig:
    """Configuration for behavioral alignment analysis."""
    # Risk zone thresholds (percentiles)
    high_risk_percentile: float = 75.0
    moderate_risk_percentile: float = 50.0
    
    # Temporal analysis
    temporal_window_before: float = 3.0  # seconds before maneuver
    temporal_window_after: float = 2.0   # seconds after maneuver
    sampling_rate: float = 10.0          # Hz
    
    # Spatial analysis
    longitudinal_resolution: float = 5.0  # meters
    lateral_resolution: float = 0.5       # meters
    
    # Decision detection
    decel_threshold: float = -1.5         # m/s² for deceleration detection
    accel_threshold: float = 1.0          # m/s² for acceleration detection
    lateral_threshold: float = 0.3        # m/s lateral velocity for LC
    
    # Statistical thresholds
    min_samples_per_zone: int = 10
    significance_level: float = 0.05


# =============================================================================
# Data Classes for Metrics Results
# =============================================================================

@dataclass
class SpatialAlignmentMetrics:
    """Container for spatial alignment metrics."""
    CTAI: float = 0.0           # Critical Transition Alignment Index
    DZOC: float = 0.0           # Decision Zone Overlap Coefficient
    LPDA: float = 0.0           # Longitudinal Position Decision Accuracy
    TBD: float = 0.0            # Transition Boundary Distance (normalized)
    GMDC: float = 0.0           # Gradient Maximum Decision Correspondence
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'CTAI': self.CTAI,
            'DZOC': self.DZOC,
            'LPDA': self.LPDA,
            'TBD': self.TBD,
            'GMDC': self.GMDC
        }


@dataclass
class TemporalCoherenceMetrics:
    """Container for temporal coherence metrics."""
    DTC: float = 0.0            # Decision Timing Concordance
    RATC: float = 0.0           # Risk-Action Temporal Coherence
    RATL: float = 0.0           # Risk-Action Temporal Lag (seconds)
    PRPT: float = 0.0           # Peak Risk Precedence Time
    TGSC: float = 0.0           # Temporal Gradient-Signal Correlation
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'DTC': self.DTC,
            'RATC': self.RATC,
            'RATL': self.RATL,
            'PRPT': self.PRPT,
            'TGSC': self.TGSC
        }


@dataclass
class BehavioralValidityMetrics:
    """Container for behavioral validity metrics."""
    RZBV: float = 0.0           # Risk Zone Behavioral Validity
    ZCPS: float = 0.0           # Zone Cluster Purity Score
    BPSR: float = 0.0           # Behavioral Phase Separation Ratio
    MZAC: float = 0.0           # Maneuver-Zone Association Consistency
    ARI: float = 0.0            # Adjusted Rand Index (clustering vs behavior)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'RZBV': self.RZBV,
            'ZCPS': self.ZCPS,
            'BPSR': self.BPSR,
            'MZAC': self.MZAC,
            'ARI': self.ARI
        }


@dataclass
class DecisionSensitivityMetrics:
    """Container for decision sensitivity metrics."""
    MODR_precision: float = 0.0  # Maneuver Onset Detection Rate - Precision
    MODR_recall: float = 0.0     # Maneuver Onset Detection Rate - Recall
    MODR_f1: float = 0.0         # Maneuver Onset Detection Rate - F1
    BRC: float = 0.0             # Behavioral Response Consistency
    SCAC: float = 0.0            # Safe-Critical Action Correspondence
    DGS: float = 0.0             # Decision Gradient Sensitivity
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MODR_precision': self.MODR_precision,
            'MODR_recall': self.MODR_recall,
            'MODR_f1': self.MODR_f1,
            'BRC': self.BRC,
            'SCAC': self.SCAC,
            'DGS': self.DGS
        }


@dataclass
class AllBehavioralMetrics:
    """Combined behavioral alignment metrics container."""
    spatial: SpatialAlignmentMetrics = field(default_factory=SpatialAlignmentMetrics)
    temporal: TemporalCoherenceMetrics = field(default_factory=TemporalCoherenceMetrics)
    validity: BehavioralValidityMetrics = field(default_factory=BehavioralValidityMetrics)
    sensitivity: DecisionSensitivityMetrics = field(default_factory=DecisionSensitivityMetrics)
    
    # Aggregate score
    overall_alignment_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spatial': self.spatial.to_dict(),
            'temporal': self.temporal.to_dict(),
            'validity': self.validity.to_dict(),
            'sensitivity': self.sensitivity.to_dict(),
            'overall_alignment_score': self.overall_alignment_score
        }
    
    def compute_overall_score(self, weights: Dict[str, float] = None):
        """Compute weighted overall alignment score."""
        if weights is None:
            weights = {
                'CTAI': 0.15, 'DZOC': 0.10, 'LPDA': 0.10,
                'DTC': 0.10, 'RATC': 0.10,
                'RZBV': 0.15, 'ARI': 0.10,
                'MODR_f1': 0.10, 'SCAC': 0.10
            }
        
        score = 0.0
        total_weight = 0.0
        
        all_values = {
            **self.spatial.to_dict(),
            **self.temporal.to_dict(),
            **self.validity.to_dict(),
            **self.sensitivity.to_dict()
        }
        
        for metric, weight in weights.items():
            if metric in all_values and not np.isnan(all_values[metric]):
                score += weight * all_values[metric]
                total_weight += weight
        
        if total_weight > 0:
            self.overall_alignment_score = score / total_weight
        
        return self.overall_alignment_score


# =============================================================================
# Driver Behavior Data Structure
# =============================================================================

@dataclass
class DriverBehaviorSequence:
    """Container for driver behavior trajectory data."""
    timestamps: np.ndarray              # Time sequence
    positions_x: np.ndarray             # Longitudinal positions
    positions_y: np.ndarray             # Lateral positions
    velocities_x: np.ndarray            # Longitudinal velocities
    velocities_y: np.ndarray            # Lateral velocities
    accelerations_x: np.ndarray         # Longitudinal accelerations
    accelerations_y: np.ndarray         # Lateral accelerations
    headings: np.ndarray                # Vehicle headings
    lane_ids: np.ndarray                # Lane assignments
    maneuver_labels: np.ndarray = None  # Ground truth maneuver labels
    risk_values: np.ndarray = None      # Corresponding risk field values
    
    def __post_init__(self):
        """Validate and process data."""
        n = len(self.timestamps)
        assert all(len(arr) == n for arr in [
            self.positions_x, self.positions_y,
            self.velocities_x, self.velocities_y
        ]), "All arrays must have same length"
        
        if self.maneuver_labels is None:
            self.maneuver_labels = np.array([ManeuverType.NONE.value] * n)
    
    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.velocities_x**2 + self.velocities_y**2)
    
    @property
    def accel_magnitude(self) -> np.ndarray:
        return np.sqrt(self.accelerations_x**2 + self.accelerations_y**2)


@dataclass
class ManeuverEvent:
    """Represents a detected maneuver event."""
    maneuver_type: ManeuverType
    start_time: float
    end_time: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    peak_intensity: float  # e.g., max deceleration, max lateral velocity
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def longitudinal_span(self) -> float:
        return self.end_position[0] - self.start_position[0]


# =============================================================================
# Spatial Alignment Analyzer
# =============================================================================

class SpatialAlignmentAnalyzer:
    """
    Analyze spatial alignment between risk field transitions and decision zones.
    
    Key insight: The moderate→high risk transition boundary should coincide
    with regions where drivers initiate safety-critical maneuvers.
    """
    
    def __init__(self, config: BehavioralAlignmentConfig = None):
        self.config = config or BehavioralAlignmentConfig()
        self.epsilon = 1e-10
    
    def compute_all(self, 
                    risk_field: np.ndarray,
                    x_coords: np.ndarray,
                    y_coords: np.ndarray,
                    decision_positions: List[Tuple[float, float]],
                    decision_types: List[ManeuverType] = None) -> SpatialAlignmentMetrics:
        """
        Compute all spatial alignment metrics.
        
        Args:
            risk_field: 2D risk field array (ny, nx)
            x_coords: X coordinates of grid (longitudinal)
            y_coords: Y coordinates of grid (lateral)
            decision_positions: List of (x, y) positions where decisions were made
            decision_types: Optional list of maneuver types at each position
        
        Returns:
            SpatialAlignmentMetrics object
        """
        metrics = SpatialAlignmentMetrics()
        
        if len(decision_positions) == 0:
            return metrics
        
        # Compute risk field statistics
        R = risk_field.copy()
        R_flat = R.flatten()
        
        # Define risk zone thresholds
        high_thresh = np.percentile(R_flat, self.config.high_risk_percentile)
        mod_thresh = np.percentile(R_flat, self.config.moderate_risk_percentile)
        
        # Compute gradient magnitude
        grad_y, grad_x = np.gradient(R)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 1. Critical Transition Alignment Index (CTAI)
        metrics.CTAI = self._compute_ctai(R, x_coords, y_coords, 
                                          decision_positions, high_thresh, mod_thresh)
        
        # 2. Decision Zone Overlap Coefficient (DZOC)
        metrics.DZOC = self._compute_dzoc(R, x_coords, y_coords,
                                          decision_positions, mod_thresh, high_thresh)
        
        # 3. Longitudinal Position Decision Accuracy (LPDA)
        metrics.LPDA = self._compute_lpda(R, x_coords, decision_positions)
        
        # 4. Transition Boundary Distance (TBD)
        metrics.TBD = self._compute_tbd(R, x_coords, y_coords,
                                        decision_positions, mod_thresh)
        
        # 5. Gradient Maximum Decision Correspondence (GMDC)
        metrics.GMDC = self._compute_gmdc(grad_mag, x_coords, y_coords,
                                          decision_positions)
        
        return metrics
    
    def _compute_ctai(self, R: np.ndarray, x_coords: np.ndarray, 
                      y_coords: np.ndarray, decision_positions: List[Tuple[float, float]],
                      high_thresh: float, mod_thresh: float) -> float:
        """
        Critical Transition Alignment Index.
        
        Measures overlap between the M→H risk transition zone and 
        the region containing actual decision points.
        
        CTAI = |Transition_Zone ∩ Decision_Region| / |Decision_Region|
        """
        # Create transition zone mask (between moderate and high thresholds)
        transition_mask = (R >= mod_thresh) & (R <= high_thresh)
        
        if not np.any(transition_mask):
            return 0.0
        
        # Create decision region (Gaussian KDE around decision points)
        decision_density = self._create_decision_density(
            x_coords, y_coords, decision_positions
        )
        
        # Normalize
        decision_density = decision_density / (np.max(decision_density) + self.epsilon)
        
        # Threshold decision density to create decision region
        decision_mask = decision_density > 0.1
        
        if not np.any(decision_mask):
            return 0.0
        
        # Compute overlap
        overlap = np.sum(transition_mask & decision_mask)
        decision_area = np.sum(decision_mask)
        
        return overlap / (decision_area + self.epsilon)
    
    def _compute_dzoc(self, R: np.ndarray, x_coords: np.ndarray,
                      y_coords: np.ndarray, decision_positions: List[Tuple[float, float]],
                      mod_thresh: float, high_thresh: float) -> float:
        """
        Decision Zone Overlap Coefficient.
        
        Jaccard-like coefficient measuring bidirectional alignment.
        
        DZOC = |Transition ∩ Decision| / |Transition ∪ Decision|
        """
        transition_mask = (R >= mod_thresh) & (R <= high_thresh)
        
        decision_density = self._create_decision_density(
            x_coords, y_coords, decision_positions
        )
        decision_mask = decision_density > 0.1 * np.max(decision_density)
        
        intersection = np.sum(transition_mask & decision_mask)
        union = np.sum(transition_mask | decision_mask)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_lpda(self, R: np.ndarray, x_coords: np.ndarray,
                      decision_positions: List[Tuple[float, float]]) -> float:
        """
        Longitudinal Position Decision Accuracy.
        
        How well does the TPF minimum (in the moderate zone) predict
        the actual longitudinal decision position?
        
        Based on Wang et al.: "the TPF accurately depicts vehicle 
        lane-changing behaviors" through gradient minimum positions.
        """
        if len(decision_positions) == 0:
            return 0.0
        
        # Get actual decision x-positions
        actual_x = np.array([p[0] for p in decision_positions])
        
        # Find risk field gradient in x direction
        grad_x = np.gradient(np.mean(R, axis=0))  # Average across y
        
        # Find local minima (potential decision points)
        local_min_indices = signal.argrelmin(grad_x)[0]
        
        if len(local_min_indices) == 0:
            # Use global minimum
            local_min_indices = [np.argmin(grad_x)]
        
        predicted_x = x_coords[local_min_indices] if len(x_coords.shape) == 1 else x_coords[0, local_min_indices]
        
        # Compute accuracy based on nearest predicted position
        total_error = 0.0
        x_range = np.max(actual_x) - np.min(actual_x) if len(actual_x) > 1 else abs(np.mean(actual_x))
        x_range = max(x_range, 1.0)
        
        for ax in actual_x:
            min_dist = np.min(np.abs(predicted_x - ax))
            total_error += min_dist
        
        mean_error = total_error / len(actual_x)
        
        # Convert to accuracy score (0-1)
        return np.exp(-mean_error / x_range)
    
    def _compute_tbd(self, R: np.ndarray, x_coords: np.ndarray,
                     y_coords: np.ndarray, decision_positions: List[Tuple[float, float]],
                     threshold: float) -> float:
        """
        Transition Boundary Distance (normalized).
        
        Average distance from decision points to the nearest 
        risk transition boundary (iso-contour at threshold).
        
        Lower is better → We return 1 - normalized_distance
        """
        from skimage import measure
        
        try:
            contours = measure.find_contours(R, threshold)
            if not contours:
                return 0.0
            
            # Combine all contour points
            boundary_points = np.vstack(contours)
            
            # Convert to world coordinates
            dx = x_coords[0, 1] - x_coords[0, 0] if len(x_coords.shape) > 1 else x_coords[1] - x_coords[0]
            dy = y_coords[1, 0] - y_coords[0, 0] if len(y_coords.shape) > 1 else y_coords[1] - y_coords[0]
            
            x_min = x_coords.min()
            y_min = y_coords.min()
            
            boundary_world = np.column_stack([
                boundary_points[:, 1] * dx + x_min,
                boundary_points[:, 0] * dy + y_min
            ])
            
            # Compute distances from decision points to boundary
            decision_arr = np.array(decision_positions)
            distances = cdist(decision_arr, boundary_world)
            min_distances = np.min(distances, axis=1)
            
            mean_distance = np.mean(min_distances)
            
            # Normalize by field extent
            field_extent = np.sqrt((x_coords.max() - x_coords.min())**2 + 
                                   (y_coords.max() - y_coords.min())**2)
            
            normalized_dist = mean_distance / (field_extent + self.epsilon)
            
            return 1 - np.clip(normalized_dist, 0, 1)
        
        except Exception:
            return 0.0
    
    def _compute_gmdc(self, grad_mag: np.ndarray, x_coords: np.ndarray,
                      y_coords: np.ndarray, 
                      decision_positions: List[Tuple[float, float]]) -> float:
        """
        Gradient Maximum Decision Correspondence.
        
        Do gradient maxima (sharp risk changes) correspond to decision points?
        
        Based on Wang et al. Fig 15: "distinct mutations in TPF are observed 
        within specific longitudinal position ranges"
        """
        # Find gradient maximum positions
        threshold = np.percentile(grad_mag.flatten(), 90)
        high_grad_mask = grad_mag > threshold
        
        if not np.any(high_grad_mask):
            return 0.0
        
        # Get coordinates of high gradient regions
        high_grad_coords = np.column_stack([
            x_coords[high_grad_mask] if len(x_coords.shape) > 1 else x_coords[np.where(high_grad_mask)[1]],
            y_coords[high_grad_mask] if len(y_coords.shape) > 1 else y_coords[np.where(high_grad_mask)[0]]
        ])
        
        # Compute correspondence
        decision_arr = np.array(decision_positions)
        
        if len(decision_arr) == 0 or len(high_grad_coords) == 0:
            return 0.0
        
        # For each decision point, find nearest high-gradient point
        distances = cdist(decision_arr, high_grad_coords)
        min_distances = np.min(distances, axis=1)
        
        # Score based on proximity
        reference_scale = self.config.longitudinal_resolution * 5
        correspondence_scores = np.exp(-min_distances / reference_scale)
        
        return np.mean(correspondence_scores)
    
    def _create_decision_density(self, x_coords: np.ndarray, 
                                  y_coords: np.ndarray,
                                  decision_positions: List[Tuple[float, float]]) -> np.ndarray:
        """Create a density map from decision positions using Gaussian KDE."""
        if len(x_coords.shape) == 1:
            X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords if len(y_coords.shape) == 1 else y_coords[:, 0])
        else:
            X_mesh, Y_mesh = x_coords, y_coords
        
        density = np.zeros_like(X_mesh)
        
        sigma_x = self.config.longitudinal_resolution * 2
        sigma_y = self.config.lateral_resolution * 4
        
        for dx, dy in decision_positions:
            contribution = np.exp(-((X_mesh - dx)**2 / (2 * sigma_x**2) +
                                    (Y_mesh - dy)**2 / (2 * sigma_y**2)))
            density += contribution
        
        return density


# =============================================================================
# Temporal Coherence Analyzer
# =============================================================================

class TemporalCoherenceAnalyzer:
    """
    Analyze temporal coherence between risk evolution and driver actions.
    
    Key insight: Risk field changes should precede or coincide with 
    driver safety actions, not lag behind them.
    """
    
    def __init__(self, config: BehavioralAlignmentConfig = None):
        self.config = config or BehavioralAlignmentConfig()
        self.epsilon = 1e-10
    
    def compute_all(self,
                    risk_sequence: np.ndarray,
                    timestamps: np.ndarray,
                    behavior: DriverBehaviorSequence,
                    maneuver_events: List[ManeuverEvent] = None) -> TemporalCoherenceMetrics:
        """
        Compute all temporal coherence metrics.
        
        Args:
            risk_sequence: Time series of risk values at ego position
            timestamps: Corresponding timestamps
            behavior: Driver behavior sequence
            maneuver_events: List of detected maneuver events
        
        Returns:
            TemporalCoherenceMetrics object
        """
        metrics = TemporalCoherenceMetrics()
        
        if len(risk_sequence) < 3 or maneuver_events is None or len(maneuver_events) == 0:
            return metrics
        
        # 1. Decision Timing Concordance (DTC)
        metrics.DTC = self._compute_dtc(risk_sequence, timestamps, maneuver_events)
        
        # 2. Risk-Action Temporal Coherence (RATC)
        metrics.RATC = self._compute_ratc(risk_sequence, timestamps, behavior)
        
        # 3. Risk-Action Temporal Lag (RATL)
        metrics.RATL = self._compute_ratl(risk_sequence, timestamps, maneuver_events)
        
        # 4. Peak Risk Precedence Time (PRPT)
        metrics.PRPT = self._compute_prpt(risk_sequence, timestamps, maneuver_events)
        
        # 5. Temporal Gradient-Signal Correlation (TGSC)
        metrics.TGSC = self._compute_tgsc(risk_sequence, timestamps, behavior)
        
        return metrics
    
    def _compute_dtc(self, risk_sequence: np.ndarray, timestamps: np.ndarray,
                     maneuver_events: List[ManeuverEvent]) -> float:
        """
        Decision Timing Concordance.
        
        Correlation between risk gradient peaks and maneuver onset times.
        
        DTC = corr(risk_gradient_peaks, maneuver_onsets)
        """
        # Compute risk gradient
        dt = np.mean(np.diff(timestamps))
        risk_gradient = np.gradient(risk_sequence, dt)
        
        # Find risk gradient peaks
        peak_indices = signal.argrelmax(np.abs(risk_gradient))[0]
        
        if len(peak_indices) == 0:
            return 0.0
        
        risk_peak_times = timestamps[peak_indices]
        
        # Maneuver onset times
        maneuver_times = np.array([e.start_time for e in maneuver_events])
        
        # Compute temporal proximity score
        total_score = 0.0
        time_window = self.config.temporal_window_before + self.config.temporal_window_after
        
        for mt in maneuver_times:
            # Find nearest risk peak
            time_diffs = np.abs(risk_peak_times - mt)
            min_diff = np.min(time_diffs)
            
            # Score based on temporal proximity
            score = np.exp(-min_diff / (time_window / 2))
            total_score += score
        
        return total_score / len(maneuver_times) if len(maneuver_times) > 0 else 0.0
    
    def _compute_ratc(self, risk_sequence: np.ndarray, timestamps: np.ndarray,
                      behavior: DriverBehaviorSequence) -> float:
        """
        Risk-Action Temporal Coherence.
        
        Cross-correlation between risk evolution and action intensity.
        """
        # Create action intensity signal
        action_intensity = self._compute_action_intensity(behavior)
        
        # Ensure same length
        min_len = min(len(risk_sequence), len(action_intensity))
        risk_seq = risk_sequence[:min_len]
        action_seq = action_intensity[:min_len]
        
        if min_len < 3:
            return 0.0
        
        # Normalize
        risk_norm = (risk_seq - np.mean(risk_seq)) / (np.std(risk_seq) + self.epsilon)
        action_norm = (action_seq - np.mean(action_seq)) / (np.std(action_seq) + self.epsilon)
        
        # Cross-correlation
        correlation = np.correlate(risk_norm, action_norm, mode='full')
        max_corr = np.max(np.abs(correlation)) / len(risk_norm)
        
        return np.clip(max_corr, 0, 1)
    
    def _compute_ratl(self, risk_sequence: np.ndarray, timestamps: np.ndarray,
                      maneuver_events: List[ManeuverEvent]) -> float:
        """
        Risk-Action Temporal Lag.
        
        Average time difference between risk peak and maneuver onset.
        Positive = risk leads action (good), Negative = action leads risk (bad)
        """
        dt = np.mean(np.diff(timestamps))
        risk_gradient = np.gradient(risk_sequence, dt)
        
        # Find risk gradient peaks
        peak_indices = signal.argrelmax(np.abs(risk_gradient))[0]
        
        if len(peak_indices) == 0:
            return 0.0
        
        risk_peak_times = timestamps[peak_indices]
        
        lags = []
        for event in maneuver_events:
            # Find nearest peak before maneuver onset
            prior_peaks = risk_peak_times[risk_peak_times <= event.start_time]
            if len(prior_peaks) > 0:
                lag = event.start_time - prior_peaks[-1]
                lags.append(lag)
        
        if len(lags) == 0:
            return 0.0
        
        return np.mean(lags)
    
    def _compute_prpt(self, risk_sequence: np.ndarray, timestamps: np.ndarray,
                      maneuver_events: List[ManeuverEvent]) -> float:
        """
        Peak Risk Precedence Time.
        
        Fraction of maneuvers where risk peaked before action.
        """
        # Find global risk peaks
        peak_indices = signal.argrelmax(risk_sequence)[0]
        
        if len(peak_indices) == 0:
            return 0.0
        
        peak_times = timestamps[peak_indices]
        
        precedence_count = 0
        for event in maneuver_events:
            # Check if any peak occurred before maneuver onset
            prior_peaks = peak_times[peak_times < event.start_time]
            if len(prior_peaks) > 0:
                # Check if within reasonable window
                latest_prior = prior_peaks[-1]
                if event.start_time - latest_prior < self.config.temporal_window_before:
                    precedence_count += 1
        
        return precedence_count / len(maneuver_events) if len(maneuver_events) > 0 else 0.0
    
    def _compute_tgsc(self, risk_sequence: np.ndarray, timestamps: np.ndarray,
                      behavior: DriverBehaviorSequence) -> float:
        """
        Temporal Gradient-Signal Correlation.
        
        Correlation between risk gradient and driver response gradient.
        """
        dt = np.mean(np.diff(timestamps))
        
        # Risk gradient
        risk_grad = np.gradient(risk_sequence, dt)
        
        # Driver response: deceleration (negative accel_x = braking = response to risk)
        min_len = min(len(risk_grad), len(behavior.accelerations_x))
        
        if min_len < 3:
            return 0.0
        
        risk_grad = risk_grad[:min_len]
        driver_response = -behavior.accelerations_x[:min_len]  # Negative sign: decel = response
        
        # Correlation
        correlation, p_value = stats.pearsonr(risk_grad, driver_response)
        
        # Return absolute correlation (direction matters based on sign)
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_action_intensity(self, behavior: DriverBehaviorSequence) -> np.ndarray:
        """
        Compute action intensity signal from driver behavior.
        
        Action intensity combines:
        - Absolute acceleration (longitudinal + lateral)
        - Steering rate (lateral velocity changes)
        """
        # Longitudinal action intensity
        long_intensity = np.abs(behavior.accelerations_x)
        
        # Lateral action intensity
        lat_intensity = np.abs(behavior.accelerations_y)
        
        # Combined with weighting
        action_intensity = long_intensity + 0.5 * lat_intensity
        
        return action_intensity


# =============================================================================
# Behavioral Validity Analyzer
# =============================================================================

class BehavioralValidityAnalyzer:
    """
    Validate that risk zone clusters correspond to distinct behavioral phases.
    
    Key insight (from Wang et al.): 
    - H-TPF zone → Preparatory phase (before lane change)
    - M-TPF zone → Decision/execution phase (during lane change)
    - L-TPF zone → Post-maneuver phase (after lane change)
    """
    
    def __init__(self, config: BehavioralAlignmentConfig = None):
        self.config = config or BehavioralAlignmentConfig()
        self.epsilon = 1e-10
    
    def compute_all(self,
                    risk_sequence: np.ndarray,
                    behavior: DriverBehaviorSequence,
                    n_clusters: int = 3) -> BehavioralValidityMetrics:
        """
        Compute all behavioral validity metrics.
        
        Args:
            risk_sequence: Time series of risk values
            behavior: Driver behavior sequence
            n_clusters: Number of risk clusters (default 3: H, M, L)
        
        Returns:
            BehavioralValidityMetrics object
        """
        metrics = BehavioralValidityMetrics()
        
        min_len = min(len(risk_sequence), len(behavior.timestamps))
        if min_len < self.config.min_samples_per_zone * n_clusters:
            return metrics
        
        risk_seq = risk_sequence[:min_len].reshape(-1, 1)
        
        # Cluster risk values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        risk_labels = kmeans.fit_predict(risk_seq)
        
        # Order clusters by centroid value (H, M, L)
        centroids = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centroids)[::-1]  # Descending
        
        # Remap labels
        label_map = {old: new for new, old in enumerate(sorted_indices)}
        risk_labels = np.array([label_map[l] for l in risk_labels])
        
        # Create behavioral phase labels
        behavior_labels = self._create_behavioral_labels(behavior)[:min_len]
        
        # 1. Risk Zone Behavioral Validity (RZBV)
        metrics.RZBV = self._compute_rzbv(risk_labels, behavior_labels)
        
        # 2. Zone Cluster Purity Score (ZCPS)
        metrics.ZCPS = self._compute_zcps(risk_labels, behavior_labels)
        
        # 3. Behavioral Phase Separation Ratio (BPSR)
        metrics.BPSR = self._compute_bpsr(risk_seq.flatten(), behavior_labels)
        
        # 4. Maneuver-Zone Association Consistency (MZAC)
        metrics.MZAC = self._compute_mzac(risk_labels, behavior)
        
        # 5. Adjusted Rand Index
        metrics.ARI = self._compute_ari(risk_labels, behavior_labels)
        
        return metrics
    
    def _create_behavioral_labels(self, behavior: DriverBehaviorSequence) -> np.ndarray:
        """
        Create behavioral phase labels based on driver actions.
        
        Phases:
        0 = Pre-maneuver (normal driving)
        1 = Active maneuver (lane change, braking, etc.)
        2 = Post-maneuver (stabilization)
        """
        n = len(behavior.timestamps)
        labels = np.zeros(n, dtype=int)
        
        # Detect active maneuvers based on thresholds
        active_decel = behavior.accelerations_x < self.config.decel_threshold
        active_accel = behavior.accelerations_x > self.config.accel_threshold
        active_lateral = np.abs(behavior.velocities_y) > self.config.lateral_threshold
        
        # Combined active maneuver
        active_maneuver = active_decel | active_accel | active_lateral
        labels[active_maneuver] = 1
        
        # Post-maneuver: after active phase
        in_post = False
        for i in range(n):
            if labels[i] == 1:
                in_post = False
            elif in_post:
                labels[i] = 2
            else:
                # Check if we just exited active phase
                if i > 0 and labels[i-1] == 1:
                    in_post = True
                    labels[i] = 2
        
        return labels
    
    def _compute_rzbv(self, risk_labels: np.ndarray, 
                      behavior_labels: np.ndarray) -> float:
        """
        Risk Zone Behavioral Validity.
        
        How well do risk zones align with behavioral phases?
        Uses normalized mutual information.
        """
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(risk_labels, behavior_labels)
    
    def _compute_zcps(self, risk_labels: np.ndarray,
                      behavior_labels: np.ndarray) -> float:
        """
        Zone Cluster Purity Score.
        
        For each risk zone, what fraction belongs to the dominant behavior phase?
        """
        unique_risk = np.unique(risk_labels)
        total_purity = 0.0
        total_samples = 0
        
        for zone in unique_risk:
            zone_mask = risk_labels == zone
            zone_behaviors = behavior_labels[zone_mask]
            
            if len(zone_behaviors) == 0:
                continue
            
            # Find dominant behavior
            unique, counts = np.unique(zone_behaviors, return_counts=True)
            dominant_count = np.max(counts)
            
            total_purity += dominant_count
            total_samples += len(zone_behaviors)
        
        return total_purity / total_samples if total_samples > 0 else 0.0
    
    def _compute_bpsr(self, risk_values: np.ndarray,
                      behavior_labels: np.ndarray) -> float:
        """
        Behavioral Phase Separation Ratio.
        
        How well-separated are risk distributions for different behavioral phases?
        """
        unique_phases = np.unique(behavior_labels)
        
        if len(unique_phases) < 2:
            return 0.0
        
        # Compute between-phase variance / within-phase variance
        overall_mean = np.mean(risk_values)
        
        between_var = 0.0
        within_var = 0.0
        
        for phase in unique_phases:
            phase_mask = behavior_labels == phase
            phase_values = risk_values[phase_mask]
            
            if len(phase_values) == 0:
                continue
            
            phase_mean = np.mean(phase_values)
            n_phase = len(phase_values)
            
            between_var += n_phase * (phase_mean - overall_mean)**2
            within_var += np.sum((phase_values - phase_mean)**2)
        
        if within_var < self.epsilon:
            return 1.0
        
        # F-ratio normalized to [0, 1]
        f_ratio = between_var / (within_var + self.epsilon)
        return f_ratio / (1 + f_ratio)
    
    def _compute_mzac(self, risk_labels: np.ndarray,
                      behavior: DriverBehaviorSequence) -> float:
        """
        Maneuver-Zone Association Consistency.
        
        Are specific maneuver types consistently associated with specific zones?
        """
        if behavior.maneuver_labels is None:
            return 0.0
        
        # Create contingency table
        unique_zones = np.unique(risk_labels)
        unique_maneuvers = np.unique(behavior.maneuver_labels[:len(risk_labels)])
        
        # Chi-square test of independence
        try:
            contingency = np.zeros((len(unique_zones), len(unique_maneuvers)))
            
            for i, zone in enumerate(unique_zones):
                for j, maneuver in enumerate(unique_maneuvers):
                    contingency[i, j] = np.sum(
                        (risk_labels == zone) & 
                        (behavior.maneuver_labels[:len(risk_labels)] == maneuver)
                    )
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Cramér's V
            n = np.sum(contingency)
            min_dim = min(len(unique_zones), len(unique_maneuvers)) - 1
            
            if min_dim == 0 or n == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim + self.epsilon))
            return np.clip(cramers_v, 0, 1)
        
        except Exception:
            return 0.0
    
    def _compute_ari(self, risk_labels: np.ndarray,
                     behavior_labels: np.ndarray) -> float:
        """
        Adjusted Rand Index.
        
        Measures agreement between risk clustering and behavioral phases,
        adjusted for chance.
        """
        return adjusted_rand_score(behavior_labels, risk_labels)


# =============================================================================
# Decision Sensitivity Analyzer
# =============================================================================

class DecisionSensitivityAnalyzer:
    """
    Analyze sensitivity of risk field to driver decision points.
    
    Key insight: A good risk field should detect/predict maneuver onsets
    and show consistent driver responses to similar risk levels.
    """
    
    def __init__(self, config: BehavioralAlignmentConfig = None):
        self.config = config or BehavioralAlignmentConfig()
        self.epsilon = 1e-10
    
    def compute_all(self,
                    risk_sequence: np.ndarray,
                    timestamps: np.ndarray,
                    behavior: DriverBehaviorSequence,
                    maneuver_events: List[ManeuverEvent] = None,
                    risk_threshold: float = None) -> DecisionSensitivityMetrics:
        """
        Compute all decision sensitivity metrics.
        
        Args:
            risk_sequence: Time series of risk values
            timestamps: Corresponding timestamps
            behavior: Driver behavior sequence
            maneuver_events: List of detected maneuver events
            risk_threshold: Threshold for high-risk detection
        
        Returns:
            DecisionSensitivityMetrics object
        """
        metrics = DecisionSensitivityMetrics()
        
        min_len = min(len(risk_sequence), len(behavior.timestamps))
        if min_len < 10:
            return metrics
        
        risk_seq = risk_sequence[:min_len]
        
        if risk_threshold is None:
            risk_threshold = np.percentile(risk_seq, self.config.moderate_risk_percentile)
        
        # Detect maneuver events if not provided
        if maneuver_events is None:
            maneuver_events = self._detect_maneuver_events(behavior)
        
        # 1-3. Maneuver Onset Detection Rate (MODR)
        modr = self._compute_modr(risk_seq, timestamps, maneuver_events, risk_threshold)
        metrics.MODR_precision = modr['precision']
        metrics.MODR_recall = modr['recall']
        metrics.MODR_f1 = modr['f1']
        
        # 4. Behavioral Response Consistency (BRC)
        metrics.BRC = self._compute_brc(risk_seq, behavior)
        
        # 5. Safe-Critical Action Correspondence (SCAC)
        metrics.SCAC = self._compute_scac(risk_seq, behavior, risk_threshold)
        
        # 6. Decision Gradient Sensitivity (DGS)
        metrics.DGS = self._compute_dgs(risk_seq, timestamps, maneuver_events)
        
        return metrics
    
    def _detect_maneuver_events(self, behavior: DriverBehaviorSequence) -> List[ManeuverEvent]:
        """Detect maneuver events from driver behavior."""
        events = []
        n = len(behavior.timestamps)
        
        # State machine for event detection
        in_maneuver = False
        start_idx = 0
        current_type = ManeuverType.NONE
        peak_intensity = 0.0
        
        for i in range(n):
            # Check for maneuver initiation
            is_decel = behavior.accelerations_x[i] < self.config.decel_threshold
            is_accel = behavior.accelerations_x[i] > self.config.accel_threshold
            is_lateral = abs(behavior.velocities_y[i]) > self.config.lateral_threshold
            
            current_intensity = abs(behavior.accelerations_x[i]) + abs(behavior.velocities_y[i])
            
            if not in_maneuver:
                if is_decel:
                    in_maneuver = True
                    start_idx = i
                    current_type = ManeuverType.DECELERATION
                    peak_intensity = abs(behavior.accelerations_x[i])
                elif is_lateral:
                    in_maneuver = True
                    start_idx = i
                    current_type = ManeuverType.LANE_CHANGE_LEFT if behavior.velocities_y[i] > 0 else ManeuverType.LANE_CHANGE_RIGHT
                    peak_intensity = abs(behavior.velocities_y[i])
            else:
                peak_intensity = max(peak_intensity, current_intensity)
                
                # Check for maneuver end
                if not (is_decel or is_accel or is_lateral):
                    events.append(ManeuverEvent(
                        maneuver_type=current_type,
                        start_time=behavior.timestamps[start_idx],
                        end_time=behavior.timestamps[i],
                        start_position=(behavior.positions_x[start_idx], behavior.positions_y[start_idx]),
                        end_position=(behavior.positions_x[i], behavior.positions_y[i]),
                        peak_intensity=peak_intensity
                    ))
                    in_maneuver = False
        
        return events
    
    def _compute_modr(self, risk_seq: np.ndarray, timestamps: np.ndarray,
                      maneuver_events: List[ManeuverEvent],
                      threshold: float) -> Dict[str, float]:
        """
        Maneuver Onset Detection Rate.
        
        How well does high risk predict maneuver onsets?
        """
        dt = np.mean(np.diff(timestamps))
        
        # Create ground truth: 1 at maneuver onset windows
        y_true = np.zeros(len(timestamps))
        window_samples = int(self.config.temporal_window_before / dt)
        
        for event in maneuver_events:
            onset_idx = np.searchsorted(timestamps, event.start_time)
            start_idx = max(0, onset_idx - window_samples)
            end_idx = min(len(timestamps), onset_idx + window_samples // 2)
            y_true[start_idx:end_idx] = 1
        
        # Predictions: high risk indicates potential maneuver
        y_pred = (risk_seq > threshold).astype(int)
        
        # Metrics
        if np.sum(y_true) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _compute_brc(self, risk_seq: np.ndarray, 
                     behavior: DriverBehaviorSequence) -> float:
        """
        Behavioral Response Consistency.
        
        How consistent are driver responses to similar risk levels?
        Low variance = consistent response = good risk representation
        """
        min_len = min(len(risk_seq), len(behavior.accelerations_x))
        risk_seq = risk_seq[:min_len]
        
        # Bin risk levels
        n_bins = 10
        risk_bins = np.digitize(risk_seq, np.percentile(risk_seq, np.linspace(0, 100, n_bins + 1)))
        
        response_variances = []
        
        for bin_idx in range(1, n_bins + 1):
            bin_mask = risk_bins == bin_idx
            if np.sum(bin_mask) < 3:
                continue
            
            # Response: acceleration (braking/throttle)
            bin_responses = behavior.accelerations_x[:min_len][bin_mask]
            response_variances.append(np.var(bin_responses))
        
        if len(response_variances) == 0:
            return 0.0
        
        # Consistency = 1 / (1 + normalized_variance)
        mean_variance = np.mean(response_variances)
        reference_variance = np.var(behavior.accelerations_x[:min_len])
        
        if reference_variance < self.epsilon:
            return 1.0
        
        normalized_variance = mean_variance / reference_variance
        return 1 / (1 + normalized_variance)
    
    def _compute_scac(self, risk_seq: np.ndarray,
                      behavior: DriverBehaviorSequence,
                      threshold: float) -> float:
        """
        Safe-Critical Action Correspondence.
        
        Correlation between high-risk zones and safety-critical actions.
        """
        min_len = min(len(risk_seq), len(behavior.accelerations_x))
        
        # High risk indicator
        high_risk = (risk_seq[:min_len] > threshold).astype(float)
        
        # Safety-critical action indicator
        safety_action = (
            (behavior.accelerations_x[:min_len] < self.config.decel_threshold) |
            (np.abs(behavior.velocities_y[:min_len]) > self.config.lateral_threshold)
        ).astype(float)
        
        # Correlation
        if np.std(high_risk) < self.epsilon or np.std(safety_action) < self.epsilon:
            return 0.0
        
        correlation, _ = stats.pearsonr(high_risk, safety_action)
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_dgs(self, risk_seq: np.ndarray, timestamps: np.ndarray,
                     maneuver_events: List[ManeuverEvent]) -> float:
        """
        Decision Gradient Sensitivity.
        
        How strongly does risk gradient change near decision points?
        """
        if len(maneuver_events) == 0:
            return 0.0
        
        dt = np.mean(np.diff(timestamps))
        risk_gradient = np.abs(np.gradient(risk_seq, dt))
        
        # Compare gradient magnitude near decisions vs. elsewhere
        decision_mask = np.zeros(len(timestamps), dtype=bool)
        window_samples = int(self.config.temporal_window_before / dt)
        
        for event in maneuver_events:
            onset_idx = np.searchsorted(timestamps, event.start_time)
            start_idx = max(0, onset_idx - window_samples)
            end_idx = min(len(timestamps), onset_idx + window_samples)
            decision_mask[start_idx:end_idx] = True
        
        if not np.any(decision_mask) or not np.any(~decision_mask):
            return 0.0
        
        gradient_at_decisions = np.mean(risk_gradient[decision_mask])
        gradient_elsewhere = np.mean(risk_gradient[~decision_mask])
        
        # Sensitivity ratio
        if gradient_elsewhere < self.epsilon:
            return 1.0 if gradient_at_decisions > self.epsilon else 0.0
        
        ratio = gradient_at_decisions / gradient_elsewhere
        return np.clip(ratio / (1 + ratio), 0, 1)


# =============================================================================
# Main Behavioral Alignment Evaluator
# =============================================================================

class BehavioralAlignmentEvaluator:
    """
    Main class for comprehensive behavioral alignment evaluation.
    
    Integrates all analyzers to provide a complete assessment of
    how well a risk field aligns with actual driver decision-making.
    """
    
    def __init__(self, config: BehavioralAlignmentConfig = None):
        self.config = config or BehavioralAlignmentConfig()
        
        self.spatial_analyzer = SpatialAlignmentAnalyzer(self.config)
        self.temporal_analyzer = TemporalCoherenceAnalyzer(self.config)
        self.validity_analyzer = BehavioralValidityAnalyzer(self.config)
        self.sensitivity_analyzer = DecisionSensitivityAnalyzer(self.config)
    
    def evaluate_full(self,
                      risk_field: np.ndarray,
                      x_coords: np.ndarray,
                      y_coords: np.ndarray,
                      risk_sequence: np.ndarray,
                      timestamps: np.ndarray,
                      behavior: DriverBehaviorSequence,
                      decision_positions: List[Tuple[float, float]] = None,
                      maneuver_events: List[ManeuverEvent] = None) -> AllBehavioralMetrics:
        """
        Perform comprehensive behavioral alignment evaluation.
        
        Args:
            risk_field: 2D risk field array (for spatial analysis)
            x_coords, y_coords: Grid coordinates
            risk_sequence: Time series of risk values at ego position
            timestamps: Timestamps for risk sequence
            behavior: Driver behavior sequence
            decision_positions: List of decision (x, y) positions
            maneuver_events: List of maneuver events
        
        Returns:
            AllBehavioralMetrics object with all metrics
        """
        results = AllBehavioralMetrics()
        
        # Extract decision positions from behavior if not provided
        if decision_positions is None:
            decision_positions = self._extract_decision_positions(behavior)
        
        # Detect maneuvers if not provided
        if maneuver_events is None:
            maneuver_events = self.sensitivity_analyzer._detect_maneuver_events(behavior)
        
        # 1. Spatial Alignment
        results.spatial = self.spatial_analyzer.compute_all(
            risk_field, x_coords, y_coords, decision_positions
        )
        
        # 2. Temporal Coherence
        results.temporal = self.temporal_analyzer.compute_all(
            risk_sequence, timestamps, behavior, maneuver_events
        )
        
        # 3. Behavioral Validity
        results.validity = self.validity_analyzer.compute_all(
            risk_sequence, behavior
        )
        
        # 4. Decision Sensitivity
        results.sensitivity = self.sensitivity_analyzer.compute_all(
            risk_sequence, timestamps, behavior, maneuver_events
        )
        
        # Compute overall score
        results.compute_overall_score()
        
        return results
    
    def evaluate_from_trajectory(self,
                                 risk_field_func,  # Function (x, y, t) -> risk
                                 behavior: DriverBehaviorSequence) -> AllBehavioralMetrics:
        """
        Evaluate behavioral alignment from a trajectory and risk field function.
        
        Args:
            risk_field_func: Callable that returns risk value for (x, y, t)
            behavior: Driver behavior sequence
        
        Returns:
            AllBehavioralMetrics object
        """
        # Sample risk along trajectory
        risk_sequence = np.array([
            risk_field_func(
                behavior.positions_x[i],
                behavior.positions_y[i],
                behavior.timestamps[i]
            )
            for i in range(len(behavior.timestamps))
        ])
        
        # Create grid for spatial analysis
        x_range = (np.min(behavior.positions_x) - 20, np.max(behavior.positions_x) + 20)
        y_range = (np.min(behavior.positions_y) - 10, np.max(behavior.positions_y) + 10)
        
        nx, ny = 60, 30
        x_coords = np.linspace(x_range[0], x_range[1], nx)
        y_coords = np.linspace(y_range[0], y_range[1], ny)
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
        
        # Sample risk field at mid-trajectory time
        t_mid = np.median(behavior.timestamps)
        risk_field = np.array([
            [risk_field_func(x, y, t_mid) for x in x_coords]
            for y in y_coords
        ])
        
        return self.evaluate_full(
            risk_field, X_mesh, Y_mesh,
            risk_sequence, behavior.timestamps, behavior
        )
    
    def _extract_decision_positions(self, behavior: DriverBehaviorSequence) -> List[Tuple[float, float]]:
        """Extract decision positions from behavior based on action intensity."""
        positions = []
        
        # Detect high-intensity action points
        action_intensity = (
            np.abs(behavior.accelerations_x) + 
            0.5 * np.abs(behavior.velocities_y)
        )
        
        threshold = np.percentile(action_intensity, 75)
        high_action = action_intensity > threshold
        
        # Group consecutive high-action points
        labels, n_features = ndimage.label(high_action)
        
        for i in range(1, n_features + 1):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                # Take centroid of action region
                mid_idx = indices[len(indices) // 2]
                positions.append((
                    behavior.positions_x[mid_idx],
                    behavior.positions_y[mid_idx]
                ))
        
        return positions
    
    def compare_methods(self,
                        risk_fields: Dict[str, np.ndarray],
                        x_coords: np.ndarray,
                        y_coords: np.ndarray,
                        risk_sequences: Dict[str, np.ndarray],
                        timestamps: np.ndarray,
                        behavior: DriverBehaviorSequence) -> Dict[str, AllBehavioralMetrics]:
        """
        Compare behavioral alignment across multiple risk field methods.
        
        Args:
            risk_fields: Dict mapping method name to 2D risk field
            risk_sequences: Dict mapping method name to risk time series
            Other args as in evaluate_full
        
        Returns:
            Dict mapping method name to AllBehavioralMetrics
        """
        results = {}
        
        decision_positions = self._extract_decision_positions(behavior)
        maneuver_events = self.sensitivity_analyzer._detect_maneuver_events(behavior)
        
        for method_name in risk_fields.keys():
            results[method_name] = self.evaluate_full(
                risk_fields[method_name],
                x_coords, y_coords,
                risk_sequences[method_name],
                timestamps, behavior,
                decision_positions, maneuver_events
            )
        
        return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_behavioral_alignment_summary(
    metrics: AllBehavioralMetrics,
    title: str = "Behavioral Alignment Metrics",
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Create comprehensive visualization of behavioral alignment metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fg_color = '#e0e0e0'
    
    # 1. Spatial Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    spatial_data = metrics.spatial.to_dict()
    names = list(spatial_data.keys())
    values = list(spatial_data.values())
    bars = ax1.barh(names, values, color='#4ecca3')
    ax1.set_xlim(0, 1)
    ax1.set_title('Spatial Alignment', color=fg_color, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=fg_color)
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', color=fg_color, fontsize=9)
    
    # 2. Temporal Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    temporal_data = metrics.temporal.to_dict()
    names = list(temporal_data.keys())
    values = [min(1, max(0, v)) if v != metrics.temporal.RATL else 0 for v in temporal_data.values()]  # Normalize
    bars = ax2.barh(names, values, color='#f39c12')
    ax2.set_xlim(0, 1)
    ax2.set_title('Temporal Coherence', color=fg_color, fontsize=12, fontweight='bold')
    ax2.tick_params(colors=fg_color)
    for bar, val in zip(bars, values):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color=fg_color, fontsize=9)
    
    # 3. Behavioral Validity
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#16213e')
    validity_data = metrics.validity.to_dict()
    names = list(validity_data.keys())
    values = [min(1, max(-1, v)) for v in validity_data.values()]  # ARI can be negative
    colors = ['#e74c3c' if v < 0 else '#4ecca3' for v in values]
    bars = ax3.barh(names, values, color=colors)
    ax3.set_xlim(-0.5, 1)
    ax3.axvline(x=0, color=fg_color, linewidth=0.5, linestyle='--')
    ax3.set_title('Behavioral Validity', color=fg_color, fontsize=12, fontweight='bold')
    ax3.tick_params(colors=fg_color)
    for bar, val in zip(bars, values):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color=fg_color, fontsize=9)
    
    # 4. Decision Sensitivity
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    sensitivity_data = metrics.sensitivity.to_dict()
    names = list(sensitivity_data.keys())
    values = list(sensitivity_data.values())
    bars = ax4.barh(names, values, color='#9b59b6')
    ax4.set_xlim(0, 1)
    ax4.set_title('Decision Sensitivity', color=fg_color, fontsize=12, fontweight='bold')
    ax4.tick_params(colors=fg_color)
    for bar, val in zip(bars, values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color=fg_color, fontsize=9)
    
    # Overall title with score
    fig.suptitle(f'{title}\nOverall Alignment Score: {metrics.overall_alignment_score:.3f}',
                 color=fg_color, fontsize=14, fontweight='bold', y=0.98)
    
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color('#394867')
    
    plt.tight_layout()
    return fig


def plot_method_comparison(
    all_metrics: Dict[str, AllBehavioralMetrics],
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """Compare behavioral alignment metrics across multiple methods."""
    import matplotlib.pyplot as plt
    
    methods = list(all_metrics.keys())
    
    # Collect key metrics
    metric_names = ['CTAI', 'DZOC', 'DTC', 'RATC', 'RZBV', 'MODR_f1', 'SCAC', 'Overall']
    
    data = []
    for method in methods:
        m = all_metrics[method]
        row = [
            m.spatial.CTAI,
            m.spatial.DZOC,
            m.temporal.DTC,
            m.temporal.RATC,
            m.validity.RZBV,
            m.sensitivity.MODR_f1,
            m.sensitivity.SCAC,
            m.overall_alignment_score
        ]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, (method, row) in enumerate(zip(methods, data)):
        ax.bar(x + i * width - width * len(methods) / 2 + width / 2,
               row, width, label=method.upper(), color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color='#e0e0e0', fontsize=10)
    ax.set_ylabel('Score', color='#e0e0e0', fontsize=11)
    ax.set_title('Behavioral Alignment Comparison', color='#e0e0e0', 
                fontsize=14, fontweight='bold')
    ax.tick_params(colors='#e0e0e0')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')
    
    for spine in ax.spines.values():
        spine.set_color('#394867')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demo / Testing
# =============================================================================

def create_demo_behavior() -> DriverBehaviorSequence:
    """Create demo driver behavior data for testing."""
    n = 200
    dt = 0.1  # 10 Hz
    timestamps = np.arange(n) * dt
    
    # Simulated lane change scenario
    # Phase 1: Normal driving (0-5s)
    # Phase 2: Deceleration and preparation (5-8s)
    # Phase 3: Lane change execution (8-12s)
    # Phase 4: Stabilization (12-20s)
    
    positions_x = np.cumsum(np.ones(n) * 20 * dt)  # ~20 m/s
    positions_y = np.zeros(n)
    
    # Lane change between t=8-12s
    lc_start = 80
    lc_end = 120
    positions_y[lc_start:lc_end] = np.linspace(0, 3.5, lc_end - lc_start)
    positions_y[lc_end:] = 3.5
    
    # Velocities
    velocities_x = np.gradient(positions_x, dt)
    velocities_y = np.gradient(positions_y, dt)
    
    # Add some speed reduction during preparation phase
    velocities_x[50:80] -= np.linspace(0, 3, 30)
    velocities_x[80:] = velocities_x[79]
    
    # Accelerations
    accelerations_x = np.gradient(velocities_x, dt)
    accelerations_y = np.gradient(velocities_y, dt)
    
    # Headings
    headings = np.arctan2(velocities_y, velocities_x)
    
    # Lane IDs
    lane_ids = np.zeros(n)
    lane_ids[lc_end:] = 1  # Changed to lane 1 after LC
    
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


def create_demo_risk_field(behavior: DriverBehaviorSequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create demo risk field and sequence."""
    # Grid
    x_range = (0, np.max(behavior.positions_x) + 50)
    y_range = (-10, 10)
    nx, ny = 60, 30
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    
    # Create risk field with higher risk in merging zone
    risk_field = 0.2 + 0.3 * np.exp(-((X - 150)**2 / 1000 + Y**2 / 20))
    risk_field += 0.4 * np.exp(-((X - 200)**2 / 500 + (Y - 2)**2 / 10))
    
    # Risk sequence along trajectory
    risk_sequence = np.array([
        0.3 + 0.4 * np.exp(-((behavior.positions_x[i] - 150)**2 / 1000))
        for i in range(len(behavior.timestamps))
    ])
    
    return risk_field, X, Y, risk_sequence


def main():
    """Run demo evaluation."""
    print("=" * 60)
    print("Behavioral Alignment Metrics Demo")
    print("=" * 60)
    
    # Create demo data
    behavior = create_demo_behavior()
    risk_field, X, Y, risk_sequence = create_demo_risk_field(behavior)
    
    print(f"\nBehavior sequence: {len(behavior.timestamps)} samples")
    print(f"Risk field shape: {risk_field.shape}")
    
    # Evaluate
    evaluator = BehavioralAlignmentEvaluator()
    metrics = evaluator.evaluate_full(
        risk_field, X, Y,
        risk_sequence, behavior.timestamps, behavior
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nSpatial Alignment:")
    for k, v in metrics.spatial.to_dict().items():
        print(f"  {k}: {v:.4f}")
    
    print("\nTemporal Coherence:")
    for k, v in metrics.temporal.to_dict().items():
        print(f"  {k}: {v:.4f}")
    
    print("\nBehavioral Validity:")
    for k, v in metrics.validity.to_dict().items():
        print(f"  {k}: {v:.4f}")
    
    print("\nDecision Sensitivity:")
    for k, v in metrics.sensitivity.to_dict().items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n{'=' * 60}")
    print(f"OVERALL ALIGNMENT SCORE: {metrics.overall_alignment_score:.4f}")
    print("=" * 60)
    
    # Create visualization
    try:
        fig = plot_behavioral_alignment_summary(metrics)
        fig.savefig('./behavioral_alignment_demo.png', 
                   facecolor=fig.get_facecolor(), dpi=150, bbox_inches='tight')
        print("\nVisualization saved to: behavioral_alignment_demo.png")
    except Exception as e:
        print(f"\nCould not create visualization: {e}")
    
    return metrics


if __name__ == '__main__':
    main()
