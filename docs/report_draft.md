# Draft Report Text

## Title

Physics-Guided Differentiable MAP Estimation for Seabed Sensor Array Localization

## Problem Setup

This work addresses the localization of a 1926-sensor seabed acoustic array from a coarse ship track prior and noisy arrival-time measurements collected from 12 transmissions at 6 known transmitter locations. The task is naturally formulated as an inverse problem: the unknown array geometry must be inferred from the acoustic propagation times while remaining consistent with the physical structure of the cable. The ship track is informative but cannot be treated as ground truth, because the array can drift laterally as it settles on the seabed. In addition, the absolute transmission times are unknown and the timing data contain outliers, so a purely geometric least-squares fit is not sufficient.

## Geometry Model

Rather than optimizing all 1926 sensor coordinates independently, the array is modeled as a smooth deformation of a baseline sampled from the ship track. First, the track is converted into an arclength-parameterized curve. A candidate baseline is then formed by selecting a start position along the track and sampling points at an approximately constant sensor spacing. The final array geometry is represented as a lateral displacement of this baseline along the local normal direction:

`p_i = b_i + d_i n_i`

where `b_i` is the baseline location of sensor `i`, `n_i` is the corresponding unit normal, and `d_i` is the unknown lateral offset. To reduce dimensionality and enforce smoothness, the offset field is parameterized using a cubic B-spline with a small number of control coefficients. This yields a compact geometry model that is expressive enough to capture current-driven drift while remaining physically interpretable.

## Observation Model

For each sensor and each transmission, the predicted arrival time is given by the acoustic travel time plus an unknown transmission offset:

`t_ij^pred = tau_j + ||p_i - a_j|| / c`

where `a_j` is the known transmitter position, `c = 1540 m/s` is the sound speed, and the propagation distance includes the fixed 18 m vertical separation between the transmitter and the seabed. The unknown transmission offsets `tau_j` are estimated jointly with the geometry. This avoids the need to form all pairwise TDOA measurements explicitly and keeps the forward model simple and differentiable.

The timing measurements contain clear outliers, so the observation model uses a Student-t likelihood instead of an ordinary Gaussian least-squares loss. This heavy-tailed choice is more robust to corrupted observations and aligns naturally with the probabilistic-computing framing of the course. Repeated transmissions from the same location are further used to estimate reliability weights: after removing the median time offset between the repeated shots, sensors with larger pairwise discrepancies are down-weighted in the objective.

## Priors and Inference

The final objective is the negative log posterior, composed of the robust data term and several geometry priors. The priors penalize large lateral offsets from the ship-track baseline, rapid changes in the offset profile, and violations of the maximum allowed inter-sensor spacing. An additional weak anchor term keeps the average adjacent spacing close to the nominal cable spacing. These terms encode the known physical structure of the array without over-constraining the solution.

Inference is carried out by maximum a posteriori estimation using PyTorch automatic differentiation. A coarse outer search is used to choose the baseline start position and average spacing along the ship track. For each candidate baseline, gradient-based optimization refines the spline coefficients, transmission offsets, and per-shot noise scales. This combination of coarse geometric search and differentiable inner optimization provides a practical balance between physical realism and computational tractability.

## Results and Discussion

The estimated array should be presented together with the ship track and the transmitter positions. In addition to the geometry plot, residual diagnostics are important: the residual distribution by shot and the histogram of adjacent sensor distances show whether the final estimate is both acoustically consistent and physically plausible. In the final report, these figures can be used to demonstrate three points: the recovered array broadly follows the ship track while exhibiting smooth lateral drift, the spacing constraint is respected, and the robust observation model prevents noisy measurements from dominating the fit.

Overall, the method is well aligned with the themes of differentiable and probabilistic computing. The forward acoustic model is differentiable, the inference is performed with gradient-based optimization, and uncertainty in the measurements is handled explicitly through a heavy-tailed likelihood and repeated-shot weighting. The main limitation is that the current implementation uses MAP estimation rather than full posterior inference, but this tradeoff keeps the method lightweight and directly suited to the scale of the assignment.
