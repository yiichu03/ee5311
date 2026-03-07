from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    if n_ctrl <= degree:
        raise ValueError("Number of control points must exceed the spline degree.")
    interior = np.linspace(0.0, 1.0, n_ctrl - degree + 1, dtype=np.float64)[1:-1]
    return np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )


def bspline_basis_matrix(n_samples: int, n_ctrl: int, degree: int) -> np.ndarray:
    knots = open_uniform_knots(n_ctrl, degree)
    u = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    basis = np.zeros((n_samples, n_ctrl), dtype=np.float64)
    for idx in range(n_ctrl):
        coeff = np.zeros(n_ctrl, dtype=np.float64)
        coeff[idx] = 1.0
        basis[:, idx] = BSpline(knots, coeff, degree, extrapolate=False)(u)
    return basis
