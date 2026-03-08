from __future__ import annotations

import math

from .fit import fit_single_candidate
from .track_utils import feasible_start_positions, build_baseline_geometry
from .types import AssignmentData, FitConfig, FitResult
from .weights import build_reliability_weights


def _iter_candidate_specs(data: AssignmentData, config: FitConfig):
    """Yield all coarse baseline candidates to be refined by the inner optimizer."""
    n_sensors = len(data.sensor_ids)
    for spacing in config.spacing_candidates:
        starts = feasible_start_positions(
            data.track_xy,
            spacing=spacing,
            n_sensors=n_sensors,
            step=config.start_step,
        )
        for start_s in starts:
            yield float(start_s), float(spacing)


def run_candidate_search(data: AssignmentData, config: FitConfig) -> FitResult:
    """Coarse search over start position and spacing, then refine each candidate."""
    weights = build_reliability_weights(data)
    best_result: FitResult | None = None
    skipped_candidates: list[dict[str, float | str]] = []
    num_evaluated = 0

    for start_s, spacing in _iter_candidate_specs(data, config):
        try:
            # Some candidates can still be invalid after spline resampling;
            # skip them and keep the coarse search moving.
            geom = build_baseline_geometry(
                data.track_xy,
                start_s=start_s,
                spacing=spacing,
                n_sensors=len(data.sensor_ids),
            )
            result = fit_single_candidate(data, geom, weights, config)
        except Exception as exc:
            skipped_candidates.append(
                {
                    "start_s": start_s,
                    "spacing": spacing,
                    "reason": str(exc),
                }
            )
            continue

        num_evaluated += 1
        if best_result is None or result.final_objective < best_result.final_objective:
            best_result = result

    if best_result is None:
        raise RuntimeError("Candidate search produced no valid result.")
    if not math.isfinite(best_result.final_objective):
        raise RuntimeError("Best result is not finite.")
    best_result.metadata["skipped_candidates"] = skipped_candidates
    best_result.metadata["num_evaluated_candidates"] = num_evaluated
    best_result.metadata["num_skipped_candidates"] = len(skipped_candidates)
    return best_result
