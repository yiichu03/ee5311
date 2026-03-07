from __future__ import annotations

import numpy as np
import torch

from .basis import bspline_basis_matrix
from .model import ArrayShapeModel
from .types import AssignmentData, BaselineGeometry, FitConfig, FitResult, WeightDiagnostics


def initial_tau_estimate(
    baseline_xy: np.ndarray,
    tx_xy: np.ndarray,
    timings: np.ndarray,
    z_offset: float,
    sound_speed: float,
) -> np.ndarray:
    dx = baseline_xy[:, None, 0] - tx_xy[None, :, 0]
    dy = baseline_xy[:, None, 1] - tx_xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + z_offset ** 2)
    travel = dist / sound_speed
    return np.nanmedian(timings - travel, axis=0)


def fit_single_candidate(
    data: AssignmentData,
    geom: BaselineGeometry,
    weights: WeightDiagnostics,
    config: FitConfig,
) -> FitResult:
    basis = bspline_basis_matrix(len(data.sensor_ids), config.num_controls, config.spline_degree)
    tau_init = initial_tau_estimate(
        geom.baseline_xy,
        data.tx_xy,
        data.timings,
        z_offset=config.z_offset,
        sound_speed=config.sound_speed,
    )
    model = ArrayShapeModel(
        baseline_xy=geom.baseline_xy,
        normals_xy=geom.normals_xy,
        basis_matrix=basis,
        tx_xy=data.tx_xy,
        timings=data.timings,
        obs_weights=weights.weights,
        config=config,
        tau_init=tau_init,
    )
    device = torch.device(config.device)
    model.to(device)

    history: list[dict[str, float]] = []
    adam = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    for step in range(config.adam_steps):
        adam.zero_grad()
        total, terms = model.loss()
        total.backward()
        adam.step()
        if step % 100 == 0 or step == config.adam_steps - 1:
            history.append({name: float(value.cpu().item()) for name, value in terms.items()})

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            total, _ = model.loss()
            total.backward()
            return total

        lbfgs.step(closure)

    final_loss, final_terms = model.loss()
    with torch.no_grad():
        positions, offsets = model.sensor_positions()
        sigma = torch.nn.functional.softplus(model.log_sigma) + 1e-4

    return FitResult(
        start_s=geom.start_s,
        spacing=geom.spacing,
        final_objective=float(final_loss.cpu().item()),
        sensor_positions=positions.cpu().numpy(),
        offsets=offsets.cpu().numpy(),
        tau=model.tau.detach().cpu().numpy(),
        sigma=sigma.cpu().numpy(),
        loss_terms={name: float(value.cpu().item()) for name, value in final_terms.items()},
        history=history,
        metadata={
            "baseline_xy": geom.baseline_xy.copy(),
            "normals_xy": geom.normals_xy.copy(),
            "arc_s": geom.arc_s.copy(),
            "pair_offsets": weights.pair_offsets.copy(),
            "pair_scales": weights.pair_scales.copy(),
        },
    )
