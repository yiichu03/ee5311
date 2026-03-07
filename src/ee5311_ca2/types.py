from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AssignmentData:
    track_xy: np.ndarray
    tx_xy: np.ndarray
    sensor_ids: np.ndarray
    timings: np.ndarray
    shot_names: tuple[str, ...]
    shot_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BaselineGeometry:
    arc_s: np.ndarray
    baseline_xy: np.ndarray
    normals_xy: np.ndarray
    spacing: float
    start_s: float


@dataclass(frozen=True)
class WeightDiagnostics:
    weights: np.ndarray
    pair_offsets: np.ndarray
    pair_scales: np.ndarray


@dataclass(frozen=True)
class FitConfig:
    num_controls: int = 40
    spline_degree: int = 3
    nominal_spacing: float = 1.0
    max_spacing: float = 1.0213
    z_offset: float = 18.0
    sound_speed: float = 1540.0
    student_nu: float = 4.0
    init_sigma: float = 0.02
    adam_steps: int = 1500
    adam_lr: float = 3e-2
    lbfgs_steps: int = 150
    lbfgs_lr: float = 0.5
    offset_penalty: float = 2.0
    smoothness_penalty: float = 8.0
    spacing_penalty: float = 1000.0
    spacing_anchor_penalty: float = 25.0
    sigma_penalty: float = 1e-3
    start_step: float = 5.0
    spacing_candidates: tuple[float, ...] = (1.0, 1.005, 1.01, 1.015, 1.02)
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class FitResult:
    start_s: float
    spacing: float
    final_objective: float
    sensor_positions: np.ndarray
    offsets: np.ndarray
    tau: np.ndarray
    sigma: np.ndarray
    loss_terms: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
