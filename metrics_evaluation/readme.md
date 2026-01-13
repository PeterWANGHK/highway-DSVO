## Implemented Metrics
### 1. Gradient Smoothness Metrics
MetricFormula BasisInterpretationSNHS∫∥HR∥F2dp∫R2dp\frac{\int \|H_R\|_F^2 d\mathbf{p}}{\int R^2 d\mathbf{p}}
∫R2dp∫∥HR​∥F2​dp​Scale-normalized; lower = smootherASD_s/ASD_d$\frac{1}{\OmegaARASDs/ASDd\text{ASD}_s / \text{ASD}_d
ASDs​/ASDd​Anisotropy ratio; ~1 = isotropicTVR$\frac{\text{TV}(R)}{R_{\max}\sqrt{\OmegaSSIHigh-freq FFT energy ratioSpectral smoothness; lower = less noise
### 2. Safe/Unsafe Boundary Metrics
MetricPurposeIdeal ValueISIInterface sharpness at boundariesHigher = sharperPPIBimodality of risk distribution→1 = good phase separationBCIIsoperimetric boundary complexityModerate (1-3)GCIGradient concentration at boundaries→1 = localizedDBSStability under perturbations→1 = robust
### 3. Occlusion/Uncertainty Metrics
MetricMeasuresIdeal ValueMUMaximum risk underestimationLowCZUCritical zone underestimationLowOFSRFalse safety rate due to occlusionLowVWRDVisibility-weighted risk discrepancyLowITOI_JSInformation-theoretic impact (JS divergence)LowSZRI_normShadow zone risk intensity~1
Files Provided

risk_field_metrics.py — Core metrics implementation with standalone demo
exid_metrics_integration.py — Integration with your exiD data loader
Visualizations — Comprehensive comparison of GVF, EDRF, and ADA methods

Usage with exiD Data
# With your existing exiD data
'''shell
python exid_metrics_integration.py --data_dir /path/to/exiD --recording 25 --ego_id 123 --frame 500 --methods gvf edrf ada
'''
