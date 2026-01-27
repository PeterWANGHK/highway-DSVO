"""
PDE Risk Field Master Dashboard
================================
Creates a comprehensive visualization dashboard combining all components
of the Hyperbolic-Parabolic PDE framework for traffic risk assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Import from main module
from pde_risk_field_visualizer import (
    PDEConfig, PDERiskFieldVisualizer, TelegrapherPDESolver,
    create_sample_scenario, Vehicle, VehicleClass, Scenario
)


def create_master_dashboard(output_path: str = '/home/claude/pde_output'):
    """Create a comprehensive master dashboard visualization."""
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = PDEConfig()
    visualizer = PDERiskFieldVisualizer(config)
    scenario = create_sample_scenario()
    visualizer.setup_scenario(scenario)
    
    # Create master figure
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor('#0D1117')
    
    # Create grid layout
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25,
                          height_ratios=[1.2, 1, 1, 0.8])
    
    # ========================================================================
    # Row 1: Main Combined Risk Field (spans 2 columns) + PDE Formula Panel
    # ========================================================================
    
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_facecolor('#0D1117')
    visualizer.plot_combined_risk_field(ax_main, title_suffix=f'\nScenario: {scenario.recording_id}')
    
    # PDE Formula Panel
    ax_formula = fig.add_subplot(gs[0, 2])
    ax_formula.set_facecolor('#1A1A2E')
    ax_formula.axis('off')
    
    formula_text = """
    Telegrapher's Equation
    (Damped Wave Equation)
    
    τ ∂²R/∂t² + ∂R/∂t + ∇·(v_eff R)
         = ∇·(𝔻∇R) + Q(x,t) - λR
    
    Parameters:
    ─────────────────────────
    τ = {:.2f} s   (Relaxation time)
    c = {:.1f} m/s (Wave speed)
    λ = {:.2f} /s  (Dissipation)
    
    Merging Potential:
    ─────────────────────────
    Φ(x) = k/(x_end - x)^γ
    
    k = {:.0f}
    γ = {:.1f}
    
    Key Properties:
    ─────────────────────────
    ✓ Finite propagation speed
    ✓ Causal risk dynamics
    ✓ Wave-like behavior
    ✓ Hyperbolic stability
    """.format(
        config.tau, config.c, config.lambda_decay,
        config.k_topo, config.gamma
    )
    
    ax_formula.text(0.1, 0.95, formula_text, transform=ax_formula.transAxes,
                   fontsize=10, color='white', family='monospace',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='#4A4A6A', pad=0.5))
    ax_formula.set_title('PDE Framework', color='white', fontsize=12, fontweight='bold', pad=10)
    
    # ========================================================================
    # Row 2: Risk Field Decomposition (4 panels across 3 columns)
    # ========================================================================
    
    # 2a. Source Term
    ax_source = fig.add_subplot(gs[1, 0])
    ax_source.set_facecolor('#0D1117')
    bounds = scenario.road_bounds
    extent = [visualizer.solver.x_coords[0], visualizer.solver.x_coords[-1],
              visualizer.solver.y_coords[0], visualizer.solver.y_coords[-1]]
    
    visualizer._draw_road(ax_source, bounds, scenario.lane_centers, scenario.merge_lane_y)
    im_src = ax_source.imshow(visualizer.solver.Q_source.T, origin='lower', extent=extent,
                              cmap='Reds', alpha=0.8, aspect='auto', zorder=2)
    visualizer._draw_vehicles(ax_source, scenario.vehicles)
    ax_source.set_xlim(bounds[0], bounds[1])
    ax_source.set_ylim(bounds[2], bounds[3])
    ax_source.set_title('(a) Source Term Q(x,t)', color='white', fontsize=10, fontweight='bold')
    ax_source.set_xlabel('x (m)', color='white', fontsize=9)
    ax_source.set_ylabel('y (m)', color='white', fontsize=9)
    ax_source.tick_params(colors='white', labelsize=8)
    cbar_src = plt.colorbar(im_src, ax=ax_source, shrink=0.6, pad=0.02)
    cbar_src.ax.tick_params(colors='white', labelsize=7)
    for spine in ax_source.spines.values():
        spine.set_color('#4A4A6A')
    
    # 2b. Occlusion Field
    ax_occ = fig.add_subplot(gs[1, 1])
    ax_occ.set_facecolor('#0D1117')
    visualizer._draw_road(ax_occ, bounds, scenario.lane_centers, scenario.merge_lane_y)
    im_occ = ax_occ.imshow(visualizer.solver.Q_occlusion.T, origin='lower', extent=extent,
                           cmap='Blues', alpha=0.8, aspect='auto', zorder=2)
    
    # Draw occlusion zone
    ego = next((v for v in scenario.vehicles if v.is_ego), None)
    observer = next((v for v in scenario.vehicles if v.id == 2), None)
    if ego and observer:
        visualizer._draw_occlusion_zone(ax_occ, ego, observer)
    visualizer._draw_vehicles(ax_occ, scenario.vehicles)
    ax_occ.set_xlim(bounds[0], bounds[1])
    ax_occ.set_ylim(bounds[2], bounds[3])
    ax_occ.set_title('(b) Occlusion Risk (Diffraction)', color='white', fontsize=10, fontweight='bold')
    ax_occ.set_xlabel('x (m)', color='white', fontsize=9)
    ax_occ.set_ylabel('y (m)', color='white', fontsize=9)
    ax_occ.tick_params(colors='white', labelsize=8)
    cbar_occ = plt.colorbar(im_occ, ax=ax_occ, shrink=0.6, pad=0.02)
    cbar_occ.ax.tick_params(colors='white', labelsize=7)
    for spine in ax_occ.spines.values():
        spine.set_color('#4A4A6A')
    
    # 2c. Topology (Merge Pressure)
    ax_topo = fig.add_subplot(gs[1, 2])
    ax_topo.set_facecolor('#0D1117')
    visualizer._draw_road(ax_topo, bounds, scenario.lane_centers, scenario.merge_lane_y)
    
    phi_viz = visualizer.solver.Phi_merge.copy()
    phi_viz = np.clip(phi_viz, 0, np.percentile(phi_viz[phi_viz > 0], 95) if np.any(phi_viz > 0) else 1)
    im_topo = ax_topo.imshow(phi_viz.T, origin='lower', extent=extent,
                             cmap='Greens', alpha=0.8, aspect='auto', zorder=2)
    
    # Merge pressure arrows
    if scenario.merge_end_x and scenario.merge_lane_y:
        for x_pos in np.linspace(bounds[0] + 20, scenario.merge_end_x - 10, 5):
            ax_topo.annotate('', xy=(x_pos, scenario.merge_lane_y - 2.5),
                           xytext=(x_pos, scenario.merge_lane_y),
                           arrowprops=dict(arrowstyle='->', color='lime', lw=1.5), zorder=8)
    
    visualizer._draw_vehicles(ax_topo, scenario.vehicles)
    ax_topo.set_xlim(bounds[0], bounds[1])
    ax_topo.set_ylim(bounds[2], bounds[3])
    ax_topo.set_title('(c) Topology: Merge Pressure Φ(x)', color='white', fontsize=10, fontweight='bold')
    ax_topo.set_xlabel('x (m)', color='white', fontsize=9)
    ax_topo.set_ylabel('y (m)', color='white', fontsize=9)
    ax_topo.tick_params(colors='white', labelsize=8)
    cbar_topo = plt.colorbar(im_topo, ax=ax_topo, shrink=0.6, pad=0.02)
    cbar_topo.ax.tick_params(colors='white', labelsize=7)
    for spine in ax_topo.spines.values():
        spine.set_color('#4A4A6A')
    
    # ========================================================================
    # Row 3: Distance-Based Profiles + Predator-Prey
    # ========================================================================
    
    # 3a. Longitudinal Risk Profile
    ax_long = fig.add_subplot(gs[2, 0])
    ax_long.set_facecolor('#0D1117')
    
    lane_y = ego.y if ego else scenario.lane_centers[-1]
    y_idx = np.argmin(np.abs(visualizer.solver.y_coords - lane_y))
    risk_profile = visualizer.solver.R[:, y_idx]
    
    ax_long.fill_between(visualizer.solver.x_coords, 0, risk_profile, 
                        color='#E74C3C', alpha=0.5)
    ax_long.plot(visualizer.solver.x_coords, risk_profile, 'r-', linewidth=2)
    
    # Mark vehicles
    for v in scenario.vehicles:
        if abs(v.y - lane_y) < 4:
            x_idx = np.argmin(np.abs(visualizer.solver.x_coords - v.x))
            marker = 'v' if v.is_ego else 'o'
            color = '#E74C3C' if v.is_ego else '#3498DB'
            ax_long.plot(v.x, risk_profile[x_idx], marker, markersize=8, color=color)
            ax_long.annotate(f'V{v.id}', (v.x, risk_profile[x_idx] + 0.15),
                           fontsize=8, color='white', ha='center')
    
    ax_long.set_xlabel('x (m)', color='white', fontsize=9)
    ax_long.set_ylabel('Risk R(x)', color='white', fontsize=9)
    ax_long.set_title('Longitudinal Risk Profile', color='white', fontsize=10, fontweight='bold')
    ax_long.tick_params(colors='white', labelsize=8)
    ax_long.grid(True, alpha=0.2, color='white')
    for spine in ax_long.spines.values():
        spine.set_color('#4A4A6A')
    
    # 3b. Distance-Risk Curves
    ax_dist = fig.add_subplot(gs[2, 1])
    ax_dist.set_facecolor('#0D1117')
    
    if observer and ego:
        distances = np.linspace(5, 80, 50)
        risks_behind = []
        for d in distances:
            x_pos = ego.x - d
            if x_pos > visualizer.solver.x_coords[0]:
                x_idx = np.argmin(np.abs(visualizer.solver.x_coords - x_pos))
                risks_behind.append(visualizer.solver.R[x_idx, y_idx])
            else:
                risks_behind.append(0)
        
        ax_dist.plot(distances, risks_behind, 'c-', linewidth=2, label='Observer → HV')
        ax_dist.fill_between(distances, 0, risks_behind, color='cyan', alpha=0.3)
        
        current_d = ego.x - observer.x
        ax_dist.axvline(current_d, color='yellow', linestyle='--', linewidth=2)
        ax_dist.annotate(f'd₁={current_d:.0f}m', (current_d + 2, max(risks_behind) * 0.8),
                        fontsize=9, color='yellow')
    
    ax_dist.set_xlabel('Distance Behind HV (m)', color='white', fontsize=9)
    ax_dist.set_ylabel('Perceived Risk', color='white', fontsize=9)
    ax_dist.set_title('Risk vs Distance (Observer)', color='white', fontsize=10, fontweight='bold')
    ax_dist.tick_params(colors='white', labelsize=8)
    ax_dist.grid(True, alpha=0.2, color='white')
    ax_dist.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A', 
                  labelcolor='white', fontsize=8)
    for spine in ax_dist.spines.values():
        spine.set_color('#4A4A6A')
    
    # 3c. Predator-Prey Dynamics
    ax_pp = fig.add_subplot(gs[2, 2])
    ax_pp.set_facecolor('#0D1117')
    visualizer._draw_road(ax_pp, bounds, scenario.lane_centers, scenario.merge_lane_y)
    
    # Normalize fields
    pred_max = np.max(visualizer.solver.R_hv) if np.max(visualizer.solver.R_hv) > 0 else 1
    prey_max = np.max(visualizer.solver.U_urgency) if np.max(visualizer.solver.U_urgency) > 0 else 1
    
    pred_norm = np.clip(visualizer.solver.R_hv / pred_max, 0, 1)
    prey_norm = np.clip(visualizer.solver.U_urgency / prey_max, 0, 1)
    
    # RGB composite
    rgb_field = np.zeros((visualizer.solver.nx, visualizer.solver.ny, 4))
    rgb_field[:, :, 0] = pred_norm  # Red = Predator (HV)
    rgb_field[:, :, 2] = prey_norm  # Blue = Prey (MV Urgency)
    rgb_field[:, :, 1] = np.clip(pred_norm * prey_norm * 2, 0, 1)  # Green = Interaction
    rgb_field[:, :, 3] = np.clip(np.maximum(pred_norm, prey_norm) * 0.8, 0, 1)  # Alpha
    
    ax_pp.imshow(np.transpose(rgb_field, (1, 0, 2)), origin='lower', extent=extent,
                aspect='auto', zorder=2)
    visualizer._draw_vehicles(ax_pp, scenario.vehicles)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='HV Risk (Predator)'),
        Patch(facecolor='blue', alpha=0.7, label='MV Urgency (Prey)'),
        Patch(facecolor='yellow', alpha=0.7, label='Interaction')
    ]
    ax_pp.legend(handles=legend_elements, loc='upper left', fontsize=7,
                facecolor='#1A1A2E', edgecolor='#4A4A6A', labelcolor='white')
    
    ax_pp.set_xlim(bounds[0], bounds[1])
    ax_pp.set_ylim(bounds[2], bounds[3])
    ax_pp.set_title('(d) Predator-Prey (Lotka-Volterra)', color='white', fontsize=10, fontweight='bold')
    ax_pp.set_xlabel('x (m)', color='white', fontsize=9)
    ax_pp.set_ylabel('y (m)', color='white', fontsize=9)
    ax_pp.tick_params(colors='white', labelsize=8)
    for spine in ax_pp.spines.values():
        spine.set_color('#4A4A6A')
    
    # ========================================================================
    # Row 4: Wave Propagation Comparison + Scenario Summary
    # ========================================================================
    
    # Wave propagation comparison
    ax_wave = fig.add_subplot(gs[3, :2])
    ax_wave.set_facecolor('#0D1117')
    
    # Simulate both models
    test_solver = TelegrapherPDESolver(config)
    test_solver.reset()
    cx, cy = test_solver.nx // 2, test_solver.ny // 2
    test_solver.R[cx, cy] = 10.0
    test_solver.R_prev = test_solver.R.copy()
    
    # Telegrapher snapshots
    snapshots_hyp = []
    for t in range(60):
        test_solver.step_telegrapher()
        if t % 20 == 0:
            snapshots_hyp.append(test_solver.R.copy())
    
    # Diffusion snapshots
    R_diff = np.zeros((test_solver.nx, test_solver.ny))
    R_diff[cx, cy] = 10.0
    D = config.D_longitudinal
    dt, dx = config.dt, config.dx
    
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
        if t % 20 == 0:
            snapshots_par.append(R_diff.copy())
    
    # Plot radial profiles
    r_coords = np.arange(0, min(cx, cy))
    colors_hyp = ['#FF6B6B', '#FFE66D', '#4ECDC4']
    colors_par = ['#FF6B6B', '#FFE66D', '#4ECDC4']
    
    for i, (snap_h, snap_p, ch, cp) in enumerate(zip(snapshots_hyp, snapshots_par[1:], colors_hyp, colors_par)):
        profile_h = snap_h[cx, cy:cy+len(r_coords)]
        profile_p = snap_p[cx, cy:cy+len(r_coords)]
        t_val = (i + 1) * 20 * config.dt
        
        ax_wave.plot(r_coords * dx, profile_h, color=ch, linewidth=2, linestyle='-',
                    label=f'Hyperbolic t={t_val:.1f}s')
        ax_wave.plot(r_coords * dx, profile_p, color=cp, linewidth=2, linestyle='--',
                    alpha=0.6)
        
        # Mark wavefront
        wave_pos = config.c * t_val
        if wave_pos < r_coords[-1] * dx:
            ax_wave.axvline(wave_pos, color=ch, linestyle=':', alpha=0.5)
    
    ax_wave.set_xlabel('Radial Distance (m)', color='white', fontsize=10)
    ax_wave.set_ylabel('Risk Density', color='white', fontsize=10)
    ax_wave.set_title('Wave Propagation: Hyperbolic (solid) vs Parabolic (dashed)', 
                     color='white', fontsize=11, fontweight='bold')
    ax_wave.legend(loc='upper right', facecolor='#1A1A2E', edgecolor='#4A4A6A',
                  labelcolor='white', fontsize=8, ncol=3)
    ax_wave.grid(True, alpha=0.2, color='white')
    ax_wave.tick_params(colors='white', labelsize=8)
    
    # Annotate finite speed
    ax_wave.annotate(f'Finite Speed c={config.c} m/s\n(Causal Propagation)',
                    xy=(0.7, 0.7), xycoords='axes fraction',
                    fontsize=9, color='lime',
                    bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='lime'))
    
    for spine in ax_wave.spines.values():
        spine.set_color('#4A4A6A')
    
    # Scenario Summary Panel
    ax_summary = fig.add_subplot(gs[3, 2])
    ax_summary.set_facecolor('#1A1A2E')
    ax_summary.axis('off')
    
    # Gather statistics
    total_risk = np.sum(visualizer.solver.R)
    max_risk = np.max(visualizer.solver.R)
    occlusion_area = np.sum(visualizer.solver.Q_occlusion > 0.1) * config.dx**2
    
    summary_text = f"""
    Scenario Summary
    ════════════════════════
    
    Recording: {scenario.recording_id}
    Frame: {scenario.frame_id}
    Vehicles: {len(scenario.vehicles)}
    
    Vehicle Types:
      • Heavy Vehicles: {sum(1 for v in scenario.vehicles if v.vehicle_class in [VehicleClass.TRUCK, VehicleClass.BUS])}
      • Passenger Cars: {sum(1 for v in scenario.vehicles if v.vehicle_class == VehicleClass.CAR)}
      • Merging: {sum(1 for v in scenario.vehicles if v.vehicle_class == VehicleClass.MERGING)}
    
    Risk Field Statistics:
      • Total Risk: {total_risk:.1f}
      • Max Risk: {max_risk:.2f}
      • Occlusion Area: {occlusion_area:.0f} m²
    
    Key Distances:
      • Observer → HV: {ego.x - observer.x:.1f} m
      • Merge End: x = {scenario.merge_end_x} m
    
    PDE Model Properties:
      ✓ Hyperbolic (wave-like)
      ✓ Finite propagation speed
      ✓ Diffraction at barriers
      ✓ Singular merge potential
      ✓ Predator-prey coupling
    """
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, color='white', family='monospace',
                   verticalalignment='top')
    ax_summary.set_title('Analysis Summary', color='white', fontsize=11, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('PDE-Based Risk Field Analysis: Heavy Vehicle Occlusion Scenario',
                color='white', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / 'master_dashboard.png'
    fig.savefig(output_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    
    print(f"Master dashboard saved to: {output_file}")
    return output_file


def create_animation_frames(output_path: str = '/home/claude/pde_output', n_frames: int = 30):
    """Create animation frames showing temporal evolution of risk field."""
    
    output_dir = Path(output_path) / 'animation_frames'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = PDEConfig()
    visualizer = PDERiskFieldVisualizer(config)
    scenario = create_sample_scenario()
    
    # Initialize solver
    visualizer.setup_scenario(scenario)
    
    bounds = scenario.road_bounds
    extent = [visualizer.solver.x_coords[0], visualizer.solver.x_coords[-1],
              visualizer.solver.y_coords[0], visualizer.solver.y_coords[-1]]
    
    for frame_idx in range(n_frames):
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#0D1117')
        ax.set_facecolor('#0D1117')
        
        visualizer._draw_road(ax, bounds, scenario.lane_centers, scenario.merge_lane_y)
        
        # Plot current risk field
        im = ax.imshow(visualizer.solver.R.T, origin='lower', extent=extent,
                      cmap='hot', alpha=0.7, aspect='auto', zorder=2,
                      vmin=0, vmax=2.0)
        
        visualizer._draw_vehicles(ax, scenario.vehicles)
        
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_title(f'Risk Field Evolution - Frame {frame_idx} (t={frame_idx*config.dt:.2f}s)',
                    color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        
        fig.tight_layout()
        fig.savefig(output_dir / f'frame_{frame_idx:03d}.png', dpi=100,
                   facecolor=fig.get_facecolor())
        plt.close(fig)
        
        # Advance simulation
        for _ in range(5):
            visualizer.solver.step_telegrapher()
            visualizer.solver.step_predator_prey()
    
    print(f"Animation frames saved to: {output_dir}")
    return output_dir


if __name__ == '__main__':
    # Create master dashboard
    dashboard_path = create_master_dashboard()
    print(f"\nDashboard created: {dashboard_path}")
