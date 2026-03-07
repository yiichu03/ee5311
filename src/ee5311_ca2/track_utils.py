from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .types import BaselineGeometry


def cumulative_arclength(track_xy: np.ndarray) -> np.ndarray:
    diffs = np.diff(track_xy, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def build_baseline_geometry(
    track_xy: np.ndarray,
    start_s: float,
    spacing: float,
    n_sensors: int,
) -> BaselineGeometry:
    s_track = cumulative_arclength(track_xy)
    total_length = float(s_track[-1])
    needed = start_s + spacing * (n_sensors - 1)
    if needed > total_length:
        raise ValueError(
            f"Requested baseline exceeds track length: need {needed:.3f} m, "
            f"track length is {total_length:.3f} m."
        )

    sx = CubicSpline(s_track, track_xy[:, 0], bc_type="natural")
    sy = CubicSpline(s_track, track_xy[:, 1], bc_type="natural")
    arc_s = start_s + spacing * np.arange(n_sensors, dtype=np.float64)

    baseline_x = sx(arc_s)
    baseline_y = sy(arc_s)
    tangent = np.column_stack([sx(arc_s, 1), sy(arc_s, 1)])
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-8)
    tangent = tangent / tangent_norm
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    return BaselineGeometry(
        arc_s=arc_s,
        baseline_xy=np.column_stack([baseline_x, baseline_y]),
        normals_xy=normals,
        spacing=spacing,
        start_s=start_s,
    )


def feasible_start_positions(track_xy: np.ndarray, spacing: float, n_sensors: int, step: float) -> np.ndarray:
    s_track = cumulative_arclength(track_xy)
    total_length = float(s_track[-1])
    max_start = total_length - spacing * (n_sensors - 1)
    if max_start < 0:
        raise ValueError("Track is too short for the requested number of sensors and spacing.")
    if step <= 0:
        raise ValueError("Search step must be positive.")
    return np.arange(0.0, max_start + 0.5 * step, step, dtype=np.float64)
