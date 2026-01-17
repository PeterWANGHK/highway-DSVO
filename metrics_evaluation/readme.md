## Implemented Metrics
### 1. Gradient Smoothness Metrics
- SNHS
- ASD_s/ASD_d
- AR
- TVR
- SSI
### 2. Safe/Unsafe Boundary Metrics
- ISI
- PPI
- BCI
- GCI
- DBS
### 3. Occlusion/Uncertainty Metrics
- MU
- CZU
- OFSR

risk_field_metrics.py — Core metrics implementation with standalone demo

exid_metrics_integration.py — Integration with exiD data loader

### Example outputs:

![The example evaluation outputs based on exiD merging scenarios](metrics_panels.png)

### Usage with exiD
```shell
python exid_metrics_integration.py --data_dir /path/to/exiD --recording 25 --ego_id 123 --frame 500 --methods gvf edrf ada
```
### for unified evaluation
```shell
# Demo mode (no data required)
python unified_metrics_integration.py --demo --methods gvf edrf ada apf

# Only field metrics on specific methods
python unified_metrics_integration.py --demo --methods gvf ada --metrics field

# Only behavioral metrics
python unified_metrics_integration.py --demo --methods gvf apf --metrics behavioral

# With exiD data
python unified_metrics_integration.py --data_dir ./exiD --recording 25 --ego_id 123

# Save to JSON
python unified_metrics_integration.py --demo --output results.json
```

### integrated field traffic analysis:
```shell
# Using scenario JSON from your role analysis:
python integrated_field_traffic_viz.py --scenario ./output_roles/rec25_ego123_frame500/scenario_snapshot.json

# With specific field methods:
python integrated_field_traffic_viz.py --scenario scenario_snapshot.json --methods gvf edrf ada apf

# Direct exiD data loading (no pre-exported JSON needed):
python integrated_field_traffic_viz.py --data_dir ./data --recording 25 --ego_id 123 --frame 500

# Auto-select ego and frame:
python integrated_field_traffic_viz.py --data_dir ./data --recording 25

# Demo mode (no data required):
python integrated_field_traffic_viz.py --demo

# Light theme output:
python integrated_field_traffic_viz.py --scenario scenario.json --light-theme

# Custom output directory:
python integrated_field_traffic_viz.py --scenario scenario.json --output_dir ./my_figures
```
### evaluation together with the traffic snapshot:
```shell
# Step 1: Generate scenario snapshot with role analysis
python exid_role_occlusion_analysis.py --data_dir ./data --recording 25 --ego_id 123 --frame 500 --output_dir ./output_roles

# Step 2: Use the exported JSON in field visualization
python integrated_field_traffic_viz.py --scenario ./output_roles/rec25_ego123_frame500/scenario_snapshot.json --methods gvf edrf ada apf --output_dir ./output_integrated
```

