# EE5311 CA2 Analysis Framework

## 1. Problem Statement

This assignment is best framed as a physics-guided inverse problem. The unknown quantity is the full seabed array geometry, represented by the 2D positions of 1926 sensors. The known inputs are:

- The ship GPS track, which acts as a geometric prior rather than ground truth.
- The 6 transmitter locations, each used twice.
- The 12 columns of noisy arrival times recorded at the sensors.

The task is therefore to infer a cable shape that explains the arrival-time data while remaining physically plausible.

## 2. Key Physical Assumptions

- The transmitter depth is 2 m and the seabed depth is 20 m, so the vertical offset in propagation distance is fixed at 18 m.
- The speed of sound is 1540 m/s.
- Adjacent sensors are connected by 1 m cables and the total separation cannot exceed 1.0213 m.
- The array generally follows the laying ship track, but can drift laterally because of currents.

These assumptions should not be treated as informal intuition; they should appear directly in the model.

## 3. Geometry Parameterization

Let `gamma(s)` denote the ship track after arclength parameterization. The array geometry is modeled as:

`p_i = gamma(s_i) + d_i * n_i`

where:

- `s_i` is the arclength coordinate of the `i`-th sensor along the track baseline.
- `n_i` is the unit normal direction of the track at `s_i`.
- `d_i` is the lateral offset caused by current-driven drift.

To avoid overfitting, the offset field is not optimized pointwise. Instead, `d = Bc`, where `B` is a B-spline basis and `c` is a small vector of control coefficients.

This gives a low-dimensional, physically interpretable shape model.

### Important: spline arc-length resampling

`CubicSpline` fitted to GPS arc-length is NOT an arc-length parameterized curve.
The GPS track contains segments whose lengths vary by more than 20x (e.g. a 4.8 m
segment followed immediately by a 114.9 m segment). In such regions, the cubic
spline's actual speed |dr/ds| deviates far from 1 — reaching ~1.66 in the worst
case — so consecutive baseline points sampled at 1 m GPS-arc intervals end up
1.66 m apart in Euclidean space, violating the cable constraint before any
optimization.

Fix implemented in `track_utils.build_baseline_geometry`: densely sample the spline
(50 pts/m), compute its true cumulative Euclidean arc length, then invert to find
the GPS parameter values corresponding to equal spline arc-length intervals. After
this fix, baseline spacing max = 1.000 m with zero violations.

## 4. Observation Model

For sensor `i` and transmission `j`, the predicted arrival time is:

`t_ij_pred = tau_j + sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + 18^2) / 1540`

where:

- `(x_i, y_i)` is the unknown sensor position.
- `(x_j, y_j)` is the known transmitter position.
- `tau_j` is the unknown emission time for transmission `j`.

This is preferable to forming all pairwise TDOAs explicitly because it keeps the forward model simple and introduces only 12 extra nuisance parameters.

## 5. Noise and Robustness

The timing data contain outliers, so a Gaussian least-squares model is not ideal. A more suitable choice is a Student-t observation model:

- It is robust to large residuals.
- It matches the repeated-shot data better.
- It fits naturally into the probabilistic-computing framing of the course.

Repeated transmissions at the same location are used to estimate reliability weights. After removing the median pair offset, sensors with large pairwise residuals receive lower observation weights.

## 6. Optimization Objective

The solution is obtained by MAP estimation:

`negative log posterior = data term + geometry priors`

The geometry priors include:

- Offset magnitude penalty: the array should stay reasonably close to the track.
- Smoothness penalty: the lateral offset should vary smoothly.
- Spacing penalty: adjacent sensors should not exceed 1.0213 m separation.
- Spacing anchor: average adjacent spacing should stay close to 1 m.

This is the most natural way to combine differentiable computing, probabilistic modeling, and physics-guided learning without making the method unnecessarily complicated.

## 7. Inference Strategy

Use a two-level approach:

1. Outer search over:
   - baseline start position along the ship track
   - candidate average sensor spacing values
2. Inner differentiable optimization over:
   - B-spline control coefficients
   - transmission start times `tau_j`
   - optional per-shot noise scales

PyTorch autograd provides the gradients for the inner optimization.

## 8. Validation Checklist

A high-scoring submission should not rely on one final plot alone. Validate:

- Estimated array versus ship track
- Adjacent sensor spacing histogram
- Residual distribution for each transmission
- Consistency between repeated transmissions
- Stability of the inferred shape under small hyperparameter changes

## 9. Recommended Reporting Angle

The report should present the work as:

- a differentiable inverse problem
- with a probabilistic observation model
- and a physics-guided low-dimensional geometry prior

That framing is aligned with the course while still remaining faithful to the assignment itself.
