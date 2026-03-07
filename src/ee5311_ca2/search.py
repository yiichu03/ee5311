from __future__ import annotations

import math

from .fit import fit_single_candidate
from .track_utils import feasible_start_positions, build_baseline_geometry
from .types import AssignmentData, FitConfig, FitResult
from .weights import build_reliability_weights


def run_candidate_search(data: AssignmentData, config: FitConfig) -> FitResult:
    weights = build_reliability_weights(data)
    best_result: FitResult | None = None

    for spacing in config.spacing_candidates:
        starts = feasible_start_positions(
            data.track_xy,
            spacing=spacing,
            n_sensors=len(data.sensor_ids),
            step=config.start_step,
        )
        for start_s in starts:
            geom = build_baseline_geometry(
                data.track_xy,
                start_s=float(start_s),
                spacing=float(spacing),
                n_sensors=len(data.sensor_ids),
            )
            result = fit_single_candidate(data, geom, weights, config)
            if best_result is None or result.final_objective < best_result.final_objective:
                best_result = result

    if best_result is None:
        raise RuntimeError("Candidate search produced no valid result.")
    if not math.isfinite(best_result.final_objective):
        raise RuntimeError("Best result is not finite.")
    return best_result
