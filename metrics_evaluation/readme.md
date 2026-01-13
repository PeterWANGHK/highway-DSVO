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

!(metrics_panel.png)
### Usage with exiD
```shell
python exid_metrics_integration.py --data_dir /path/to/exiD --recording 25 --ego_id 123 --frame 500 --methods gvf edrf ada
```
