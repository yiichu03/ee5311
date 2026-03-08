from __future__ import annotations

import numpy as np

from .types import AssignmentData, WeightDiagnostics


def build_reliability_weights(data: AssignmentData, kappa: float = 3.0) -> WeightDiagnostics:
    """Down-weight sensors whose repeated-shot timing disagreement is unusually large."""
    n_sensors, n_shots = data.timings.shape
    weights = np.ones((n_sensors, n_shots), dtype=np.float64)
    pair_offsets = []
    pair_scales = []

    for shot_a, shot_b in data.shot_pairs:
        delta = data.timings[:, shot_b] - data.timings[:, shot_a]
        pair_offset = np.nanmedian(delta)
        residual = np.abs(delta - pair_offset)
        # Robust scale estimate so a small number of bad sensors does not set the weight scale.
        pair_scale = 1.4826 * np.nanmedian(residual) + 1e-6
        pair_weight = 1.0 / (1.0 + (residual / (kappa * pair_scale + 1e-6)) ** 2)
        weights[:, shot_a] = pair_weight
        weights[:, shot_b] = pair_weight
        pair_offsets.append(pair_offset)
        pair_scales.append(pair_scale)

    return WeightDiagnostics(
        weights=weights,
        pair_offsets=np.asarray(pair_offsets, dtype=np.float64),
        pair_scales=np.asarray(pair_scales, dtype=np.float64),
    )
