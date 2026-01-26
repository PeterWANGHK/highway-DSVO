# PINN Framework for Occlusion-Aware Risk Field Modeling

## Overview

This package implements a Physics-Informed Neural Network (PINN) framework for learning occlusion-aware risk potential fields around heavy trucks. The approach models risk propagation using a convection-diffusion-reaction PDE:

$$\frac{\partial R}{\partial t} + \mathbf{v}_{field} \cdot \nabla R = D(x,y,t) \nabla^2 R + Q(x,y,t) - \lambda R$$

Where:
- **R(x,y,t)**: Risk potential field
- **v_field**: Velocity field (advection by moving agents)
- **D**: Diffusion coefficient (elevated in occluded regions)
- **Q**: Source term (from known agents + occlusion-induced uncertainty)
- **λ**: Decay rate

## Features

- **Occlusion-Aware Modeling**: Occluded regions contribute probabilistic risk sources
- **Physics-Constrained Learning**: No ground truth labels needed; learns from PDE constraints
- **Fourier Feature Encoding**: Better representation of high-frequency patterns
- **Adaptive Loss Weighting**: Automatic balancing of loss components
- **Integration with exiD Dataset**: Direct interface with occlusion analysis pipeline

## Installation

```bash
# Create virtual environment (recommended)
python -m venv pinn_env
source pinn_env/bin/activate  # Linux/Mac
# pinn_env\Scripts\activate   # Windows

# Install dependencies
pip install torch numpy matplotlib

# Optional: GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Files

- `pinn_risk_field.py`: Main PINN framework implementation
- `demo_pinn_training.py`: Demonstration script with synthetic scenario
- `exid_role_occlusion_analysis.py`: Dataset processing (provides scenarios)

## Quick Start

### 1. Using Synthetic Scenario (Testing)

```python
from pinn_risk_field import train_pinn_synthetic, PINNConfig

# Configure
config = PINNConfig(
    num_epochs=3000,
    use_fourier_features=True,
    hidden_layers=6,
    hidden_dim=128
)

# Train
model, history = train_pinn_synthetic(config, num_epochs=3000)
```

### 2. Using exiD Dataset Scenarios

```python
from pinn_risk_field import train_pinn_from_scenario, PINNConfig

# Train from exported scenario JSON
model, history = train_pinn_from_scenario(
    'path/to/scenario_snapshot.json',
    num_epochs=5000
)
```

### 3. Command Line

```bash
# Synthetic scenario
python pinn_risk_field.py --synthetic --epochs 3000

# From scenario file
python pinn_risk_field.py --scenario path/to/scenario.json --epochs 5000

# Quick test
python demo_pinn_training.py --quick
```

## Configuration Options

### PDE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_diffusion` | 5.0 | Base diffusion coefficient D₀ |
| `occlusion_diffusion_boost` | 3.0 | Multiplier for D in occluded regions |
| `decay_rate` | 0.5 | Risk decay rate λ |
| `vehicle_source_weight` | 1.0 | Weight for known agent sources |
| `occlusion_source_weight` | 0.8 | Weight for occlusion-induced sources |
| `vehicle_source_sigma` | 5.0 | Spatial spread of vehicle risk (m) |

### Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 6 | Number of hidden layers |
| `hidden_dim` | 128 | Neurons per hidden layer |
| `use_fourier_features` | True | Use Fourier feature encoding |
| `num_fourier_features` | 64 | Number of Fourier features |
| `activation` | 'tanh' | Activation function |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_collocation_points` | 10000 | PDE residual sampling points |
| `num_boundary_points` | 1000 | Boundary condition points |
| `learning_rate` | 1e-3 | Initial learning rate |
| `weight_pde` | 1.0 | PDE loss weight |
| `weight_bc` | 10.0 | Boundary condition weight |

## PDE Formulation Details

### 1. Convection Term

The velocity field advects risk with moving agents:

```python
v_field(x,y,t) = Σᵢ wᵢ(x,y) * vᵢ
```

Where weights decay with distance from each agent.

### 2. Diffusion Term

Spatially varying diffusion coefficient:

```python
D(x,y,t) = D₀ * (1 + α * p_occ(x,y,t))
```

Where `p_occ` is the occlusion probability and `α` is the boost factor.

### 3. Source Terms

Combined from known agents and occlusions:

```python
Q(x,y,t) = Q_veh(x,y,t) + Q_occ(x,y,t)

# Vehicle sources (Gaussian around each agent)
Q_veh_i = w_i * exp(-|x - x_i|² / (2σ²))

# Occlusion sources (probabilistic)
Q_occ = w_occ * p_occ(x,y,t)
```

### 4. Decay Term

Linear decay prevents risk accumulation:

```python
-λ * R(x,y,t)
```

## Loss Function

The PINN is trained to minimize:

```python
L = L_PDE + β_BC * L_BC + β_IC * L_IC + β_pos * L_pos
```

Where:
- `L_PDE`: Mean squared PDE residual
- `L_BC`: Boundary condition violation (R=0 at far boundaries)
- `L_IC`: Initial condition matching
- `L_pos`: Soft positivity constraint (R ≥ 0)

## Integration with exiD Pipeline

The PINN framework integrates with the `exid_role_occlusion_analysis.py` script:

```python
from exid_role_occlusion_analysis import analyze_recording
from pinn_risk_field import ScenarioData, RiskFieldPINN, PINNConfig

# 1. Run exiD analysis (generates scenario JSON)
metadata = analyze_recording(
    data_dir='./data',
    recording_id=25,
    export_scenario=True
)

# 2. Load scenario into PINN
config = PINNConfig()
scenario = ScenarioData(config)
scenario.load_from_json(metadata['files']['scenario_json'])

# 3. Train PINN
model = RiskFieldPINN(config, scenario)
trainer = PINNTrainer(model, config)
history = trainer.train(5000)
```

## Output Files

Training produces:
- `training_history.png`: Loss curves
- `risk_field_t*.png`: Risk field snapshots at different times
- `risk_evolution.png`: Multi-panel temporal evolution
- `pde_residual.png`: PDE satisfaction visualization
- `risk_animation.gif`: Animated risk evolution
- `summary.png`: Combined overview figure
- `trained_model.pt`: Saved model checkpoint

## Expected Results

After training, you should observe:

1. **High risk near known vehicles**: Gaussian-like risk distribution around other agents
2. **Elevated risk in occlusion zones**: Risk "fills in" behind the truck where hidden hazards may exist
3. **Risk advection**: Risk field moves with vehicles over time
4. **Smooth transitions**: Diffusion creates smooth risk gradients
5. **Decay in clear areas**: Risk diminishes where no threats persist

## Troubleshooting

### High PDE Residual
- Increase `num_collocation_points`
- Use more hidden layers or wider networks
- Enable Fourier features if disabled

### Unstable Training
- Reduce learning rate
- Increase `weight_bc` and `weight_ic`
- Use gradient clipping (already enabled)

### Slow Convergence
- Enable adaptive weight balancing
- Increase Fourier scale
- Use SIREN activation (`activation='sin'`)

## References

1. Raissi, M., et al. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.

2. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.

3. Wang, S., et al. (2021). When and why PINNs fail to train: A neural tangent kernel perspective.

## License

MIT License - See LICENSE file for details.
