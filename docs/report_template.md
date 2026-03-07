# EE5311 CA2 Report Template

Use this as a 2-page template. Keep the final version compact and figure-driven.

## Title

Physics-Guided Differentiable MAP Estimation for Seabed Sensor Array Localization

## 1. Problem Setup

Suggested paragraph:

This work formulates the CA2 task as a differentiable inverse problem. The objective is to recover the 2D locations of 1926 seabed sensors using a coarse ship track prior and noisy acoustic arrival times from 12 transmissions made at 6 known transmitter locations. Because the absolute emission times are unknown and the timing measurements contain significant outliers, the inference problem must combine physical constraints with a robust probabilistic observation model.

## 2. Geometry Model

Suggested paragraph:

The array was modeled as a smooth deviation from the ship track rather than as 1926 independent points. First, the ship track was converted into an arclength-parameterized baseline. The sensor locations were then represented as positions on this baseline plus a lateral offset along the local normal direction. To reduce dimensionality and enforce smoothness, the lateral offset field was parameterized with a cubic B-spline using a small number of control coefficients.

Equation to include:

`p_i = gamma(s_i) + d_i n_i`

## 3. Observation and Noise Model

Suggested paragraph:

For each transmission, the predicted arrival time at each sensor was computed from the 3D propagation distance using the known sound speed and the fixed 18 m vertical separation between transmitter and seabed. The unknown emission time of each transmission was included as a nuisance parameter and estimated jointly with the array geometry. To account for corrupted timing measurements, a Student-t likelihood was used instead of ordinary least squares. In addition, repeated transmissions from the same location were used to derive reliability weights for each sensor observation.

Equation to include:

`t_ij_pred = tau_j + ||p_i - a_j|| / c`

## 4. Physics-Guided Priors and Inference

Suggested paragraph:

The final objective was the negative log posterior, consisting of the robust data term and several geometry priors. These priors penalized large deviations from the ship track, rapid changes in lateral offset, and violations of the maximum inter-sensor spacing constraint. Inference was carried out by MAP estimation using PyTorch automatic differentiation. A coarse outer search was used to place the array along the track, while gradient-based optimization refined the low-dimensional shape parameters and transmission time offsets.

## 5. Results

Include:

- Figure 1: Ship track, transmitter positions, and estimated array geometry
- Figure 2: Adjacent sensor spacing distribution or residual distribution by shot

Suggested paragraph:

The estimated array followed the overall ship track while exhibiting smooth lateral deviations consistent with current-driven drift. The inferred geometry satisfied the inter-sensor spacing constraint and reduced the timing residuals across the repeated transmissions. The robust likelihood prevented heavily corrupted observations from dominating the solution, especially for the noisier transmission pairs.

## 6. Discussion

Suggested paragraph:

The proposed method is aligned with the themes of differentiable and probabilistic computing: the forward acoustic model is fully differentiable, the noise model is probabilistic and robust, and physical structure is injected through the geometry parameterization and spacing priors. A limitation of the current implementation is that it uses MAP estimation rather than full posterior inference, but this tradeoff keeps the method computationally tractable while still capturing the dominant sources of uncertainty.

## Short Checklist Before Submission

- Keep to 2 pages.
- State clearly that the ship track is a prior, not ground truth.
- Explain why the emission times are estimated jointly.
- Mention why a robust likelihood was necessary.
- Show at least one geometry plot and one diagnostic plot.
- Match the filenames required by the assignment exactly.
