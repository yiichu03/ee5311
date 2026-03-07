# Draft Report Text

## Title

Physics-Guided Differentiable MAP Estimation for Seabed Sensor Array Localization

## Problem Setup

This work addresses the localization of a 1926-sensor seabed acoustic array from a
coarse ship track prior and noisy arrival-time measurements collected from 12
transmissions at 6 known transmitter locations. The task is naturally formulated as
an inverse problem: the unknown array geometry must be inferred from the acoustic
propagation times while remaining consistent with the physical structure of the
cable. The ship track is informative but cannot be treated as ground truth, because
the array can drift laterally as it settles on the seabed. In addition, the absolute
transmission times are unknown and the timing data contain outliers, so a purely
geometric least-squares fit is not sufficient.

## Geometry Model

Rather than optimizing all 1926 sensor coordinates independently, the array is
modeled as a smooth deformation of a baseline sampled from the ship track. First,
the track is converted into a spline and resampled at equal spline arc-length
intervals of the target sensor spacing. This resampling step is essential: the raw
GPS track contains segments that vary in length by more than 20x (e.g., 4.8 m and
114.9 m segments in quick succession), and a naive arc-length parameterization from
GPS piecewise distances causes the cubic spline to travel at up to 1.66x the nominal
speed in those regions, producing baseline points that are already 1.66 m apart
before any lateral offset is applied. The spline arc-length resampling corrects this.

The final array geometry is represented as a lateral displacement of the corrected
baseline along the local normal direction:

    p_i = b_i + d_i * n_i

where b_i is the baseline location of sensor i, n_i is the corresponding unit
normal, and d_i is the unknown lateral offset. To reduce dimensionality and enforce
smoothness, the offset field is parameterized using a cubic B-spline with 40 control
coefficients. This gives a compact, physically interpretable shape model with roughly
48 sensors per control point.

## Observation Model

For each sensor i and each transmission j, the predicted arrival time is:

    t_ij_pred = tau_j + sqrt( (x_i - X_j)^2 + (y_i - Y_j)^2 + 18^2 ) / 1540

where (X_j, Y_j) is the known transmitter 2D position, the 18 m vertical term
accounts for the fixed depth difference between transmitter (2 m) and seabed (20 m),
and tau_j is the unknown emission time for transmission j. Estimating tau_j jointly
with the geometry avoids forming all pairwise TDOA differences, which would amplify
noise, and introduces only 12 extra nuisance parameters.

The timing measurements contain clear outliers (particularly transmitters 4 and 5),
so the observation model uses a Student-t likelihood (nu = 4) instead of ordinary
Gaussian least squares. This heavy-tailed choice is more robust to corrupted
observations and aligns naturally with the probabilistic-computing framing of the
course. Repeated transmissions from the same location are further used to estimate
reliability weights: after removing the median time offset between the repeated
shots, sensors with larger pairwise discrepancies are down-weighted via a Cauchy
kernel applied to the residual magnitude.

## Priors and Inference

The final objective is the negative log posterior, composed of the robust data term
and four geometry priors:

  - Offset magnitude penalty    (lambda=2.0):  array stays near the ship track
  - Smoothness penalty          (lambda=8.0):  offset profile varies smoothly
  - Spacing excess penalty      (lambda=1000): hard-ish ceiling at 1.0213 m
  - Spacing anchor penalty      (lambda=25):   average spacing stays near 1.0 m

Inference is carried out by MAP estimation using PyTorch automatic differentiation.
An outer search over baseline start position (5 m grid) and five spacing candidates
(1.000 to 1.020 m) selects the initial geometry. For each candidate, gradient-based
optimization refines the 40 spline coefficients, 12 transmission offsets, and 12
per-shot noise scales (Adam 1500 steps, then L-BFGS strong-Wolfe 150 steps).

## Results

Best candidate: start_s = 40.0 m, spacing = 1.020 m
Final MAP objective: -2.579
Physical constraint: PASS — 0 spacing violations out of 1925 adjacent pairs
Maximum adjacent spacing: 1.0206 m (limit 1.0213 m)

Per-shot timing residuals (after optimization):

  Shot   Mean abs (s)   Median abs (s)   RMSE (s)   Inferred sigma (s)
  t11    0.049          0.020            0.087       0.021
  t12    0.046          0.019            0.084       0.021
  t21    0.039          0.020            0.064       0.026  <- cleanest shots
  t22    0.033          0.018            0.056       0.026
  t31    0.036          0.031            0.048       0.031
  t32    0.036          0.031            0.050       0.031
  t41    0.107          0.063            0.156       0.074  <- noisiest shots
  t42    0.123          0.068            0.179       0.079
  t51    0.069          0.037            0.096       0.059
  t52    0.071          0.038            0.105       0.058
  t61    0.043          0.026            0.067       0.029
  t62    0.045          0.025            0.075       0.029

The inferred per-shot sigma values confirm the data quality pattern: transmitters
2 and 3 are cleanest (sigma ~0.026-0.031 s), transmitters 4 and 5 are noisiest
(sigma ~0.059-0.079 s). The Student-t model and reliability weights successfully
down-weight the noisy shots without discarding them entirely.

## Discussion

The proposed method is well aligned with the three main themes of the course:

  1. Differentiable programming — the full forward model (spline geometry,
     acoustic propagation, Student-t NLL) is implemented as a differentiable
     PyTorch computation graph. Gradients with respect to all parameters are
     computed automatically via backpropagation.

  2. Probabilistic modeling — the Student-t observation model explicitly
     represents heavy-tailed measurement noise. Per-shot noise scales sigma_j
     are learned jointly, giving a calibrated uncertainty estimate for each
     transmission.

  3. Physics-guided learning — the cable physics appears at three levels:
     the geometry parameterization (smooth deviation from known ship track),
     the acoustic forward model (exact 3D propagation distance with fixed
     depth offset), and the spacing priors (enforcing the 1.0213 m cable
     constraint).

The main limitation is that inference is limited to the MAP point estimate
rather than the full posterior. Full posterior inference via MCMC or variational
inference would provide uncertainty quantification on the sensor positions, which
could further identify poorly constrained regions of the array.
