from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ValidationSummary:
    expected_sensors: int
    actual_sensors: int
    ids_consecutive: bool
    coords_finite: bool
    num_edges: int
    spacing_min: float
    spacing_median: float
    spacing_mean: float
    spacing_max: float
    spacing_violation_count: int
    spacing_violation_fraction: float
    total_array_length: float
    passes_format_checks: bool
    passes_physical_constraints: bool


def validate_sensor_geometry(
    sensor_ids: np.ndarray,
    sensor_positions: np.ndarray,
    expected_sensors: int,
    max_spacing: float,
) -> ValidationSummary:
    actual_sensors = int(len(sensor_ids))
    ids_consecutive = np.array_equal(sensor_ids, np.arange(1, actual_sensors + 1, dtype=sensor_ids.dtype))
    coords_finite = bool(np.isfinite(sensor_positions).all())

    if actual_sensors >= 2:
        spacing = np.linalg.norm(np.diff(sensor_positions, axis=0), axis=1)
        spacing_min = float(np.min(spacing))
        spacing_median = float(np.median(spacing))
        spacing_mean = float(np.mean(spacing))
        spacing_max = float(np.max(spacing))
        violation_mask = spacing > max_spacing
        spacing_violation_count = int(np.sum(violation_mask))
        num_edges = int(len(spacing))
        spacing_violation_fraction = float(spacing_violation_count / num_edges)
        total_array_length = float(np.sum(spacing))
    else:
        spacing_min = float("nan")
        spacing_median = float("nan")
        spacing_mean = float("nan")
        spacing_max = float("nan")
        spacing_violation_count = 0
        num_edges = 0
        spacing_violation_fraction = 0.0
        total_array_length = 0.0

    passes_format_checks = actual_sensors == expected_sensors and ids_consecutive and coords_finite
    passes_physical_constraints = passes_format_checks and spacing_violation_count == 0

    return ValidationSummary(
        expected_sensors=expected_sensors,
        actual_sensors=actual_sensors,
        ids_consecutive=ids_consecutive,
        coords_finite=coords_finite,
        num_edges=num_edges,
        spacing_min=spacing_min,
        spacing_median=spacing_median,
        spacing_mean=spacing_mean,
        spacing_max=spacing_max,
        spacing_violation_count=spacing_violation_count,
        spacing_violation_fraction=spacing_violation_fraction,
        total_array_length=total_array_length,
        passes_format_checks=passes_format_checks,
        passes_physical_constraints=passes_physical_constraints,
    )


def load_results_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    sensor_ids = arr["sensor"].astype(np.int64)
    sensor_positions = np.column_stack([arr["x"], arr["y"]]).astype(np.float64)
    return sensor_ids, sensor_positions


def validation_report_lines(summary: ValidationSummary, max_spacing: float) -> list[str]:
    return [
        f"Format checks: {'PASS' if summary.passes_format_checks else 'FAIL'}",
        f"Physical constraints: {'PASS' if summary.passes_physical_constraints else 'FAIL'}",
        f"Expected sensors: {summary.expected_sensors}",
        f"Actual sensors: {summary.actual_sensors}",
        f"Sensor IDs consecutive: {summary.ids_consecutive}",
        f"Coordinates finite: {summary.coords_finite}",
        f"Adjacent edges checked: {summary.num_edges}",
        f"Spacing min: {summary.spacing_min:.6f} m",
        f"Spacing median: {summary.spacing_median:.6f} m",
        f"Spacing mean: {summary.spacing_mean:.6f} m",
        f"Spacing max: {summary.spacing_max:.6f} m",
        f"Allowed max spacing: {max_spacing:.6f} m",
        f"Spacing violations: {summary.spacing_violation_count}",
        f"Violation fraction: {summary.spacing_violation_fraction:.4%}",
        f"Total array length: {summary.total_array_length:.6f} m",
    ]


def write_validation_report(path: Path, summary: ValidationSummary, max_spacing: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(validation_report_lines(summary, max_spacing)) + "\n", encoding="utf-8")
