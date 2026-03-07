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

    # Resample at equal spline arc-length intervals.
    # CubicSpline is NOT arc-length parameterized: in regions where the GPS
    # track has extreme segment-length variation, |dr/ds| can deviate
    # significantly from 1, causing consecutive baseline points to be far
    # apart even though they are only `spacing` apart in GPS arc-length.
    # Fix: densely sample the spline to recover its true arc length, then
    # find the GPS-parameter values that correspond to equal spline arc-length
    # intervals of `spacing` metres.
    n_dense = max(10000, int((total_length - start_s) * 50))
    s_dense = np.linspace(start_s, total_length, n_dense)
    x_dense = sx(s_dense)
    y_dense = sy(s_dense)
    spline_arc = np.concatenate([
        [0.0],
        np.cumsum(np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2)),
    ])
    target_arc = spacing * np.arange(n_sensors, dtype=np.float64)
    arc_s = np.interp(target_arc, spline_arc, s_dense)

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
    return np.arange(0.0, max_start + 1e-9, step, dtype=np.float64)
