How the PDE Describes Risk Under Uncertainty
The Governing Equation
τ∂2R∂t2+∂R∂t+∇⋅(vR)=D∇2R+Q−λR\tau \frac{\partial^2 R}{\partial t^2} + \frac{\partial R}{\partial t} + \nabla \cdot (\mathbf{v} R) = D\nabla^2 R + Q - \lambda Rτ∂t2∂2R​+∂t∂R​+∇⋅(vR)=D∇2R+Q−λR
Why This Captures Uncertainty Well
PDE TermUncertainty InterpretationVisible in FiguresQ_occ (Occlusion Source)Injects "phantom risk" where we CANNOT see. Models P(hazard exists | occluded)Red region inside blue dashed occlusion boundaryD increased in occlusionUncertainty about WHERE the hazard is → risk spreads widerBroader, more diffuse risk in blind zoneτ ∂²R/∂t² (Inertia)Finite propagation speed c=√(D/τ) ≈ 9 m/sWave propagation figure shows gradual buildup, not instantλR (Decay)Risk diminishes when uncertainty resolvesRisk lower at domain edges
Key Observations from the Figures
1. Standard vs Occlusion-Aware Comparison (Top Row):

Standard field: Risk only where visible vehicles exist
Occlusion-aware: Additional risk in the blind zone (between truck and hidden car)
The hidden car (red, x≈78m) shows elevated risk even though observer (orange) can't see it

2. Longitudinal Profile (Bottom Left):

Blue curve (standard): Peaks only at OBS, EGO, and HID vehicle locations
Red curve (occlusion-aware): Shows elevated risk between EGO (x≈50m) and HID (x≈78m)
The area under the red curve in the occlusion zone represents epistemic uncertainty

3. Wave Propagation (Second Figure):

Risk builds up gradually over time (t=0 → t=2.8s)
This is the HYPERBOLIC character: information travels at finite speed
Contrast with parabolic (heat equation) where risk would appear instantly everywhere
