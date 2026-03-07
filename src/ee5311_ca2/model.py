from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import FitConfig


class ArrayShapeModel(nn.Module):
    def __init__(
        self,
        baseline_xy,
        normals_xy,
        basis_matrix,
        tx_xy,
        timings,
        obs_weights,
        config: FitConfig,
        tau_init,
    ) -> None:
        super().__init__()
        dtype = getattr(torch, config.dtype)

        self.register_buffer("baseline_xy", torch.as_tensor(baseline_xy, dtype=dtype))
        self.register_buffer("normals_xy", torch.as_tensor(normals_xy, dtype=dtype))
        self.register_buffer("basis_matrix", torch.as_tensor(basis_matrix, dtype=dtype))
        self.register_buffer("tx_xy", torch.as_tensor(tx_xy, dtype=dtype))
        self.register_buffer("timings", torch.as_tensor(timings, dtype=dtype))
        self.register_buffer("obs_weights", torch.as_tensor(obs_weights, dtype=dtype))
        self.register_buffer("obs_mask", torch.isfinite(torch.as_tensor(timings, dtype=dtype)).to(dtype))

        self.config = config
        n_ctrl = basis_matrix.shape[1]
        n_shots = timings.shape[1]

        self.ctrl = nn.Parameter(torch.zeros(n_ctrl, dtype=dtype))
        self.tau = nn.Parameter(torch.as_tensor(tau_init, dtype=dtype))
        sigma0 = torch.full((n_shots,), float(config.init_sigma), dtype=dtype)
        self.log_sigma = nn.Parameter(torch.log(torch.expm1(sigma0)))

    def sensor_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = self.basis_matrix @ self.ctrl
        positions = self.baseline_xy + offsets[:, None] * self.normals_xy
        return positions, offsets

    def predicted_times(self, positions: torch.Tensor) -> torch.Tensor:
        dx = positions[:, None, 0] - self.tx_xy[None, :, 0]
        dy = positions[:, None, 1] - self.tx_xy[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy + self.config.z_offset ** 2)
        return self.tau[None, :] + dist / self.config.sound_speed

    def student_t_nll(self, residual: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma) + 1e-4
        scaled = residual / sigma[None, :]
        nu = torch.as_tensor(float(self.config.student_nu), dtype=residual.dtype, device=residual.device)
        nll = 0.5 * (nu + 1.0) * torch.log1p((scaled * scaled) / nu) + torch.log(sigma)[None, :]
        return nll

    def loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positions, offsets = self.sensor_positions()
        pred = self.predicted_times(positions)
        residual = self.timings - pred

        data_nll = self.student_t_nll(residual)
        valid_weight = self.obs_weights * self.obs_mask
        data_loss = (valid_weight * data_nll).sum() / torch.clamp(valid_weight.sum(), min=1.0)

        smoothness = (offsets[2:] - 2.0 * offsets[1:-1] + offsets[:-2]).pow(2).mean()
        offset_mag = offsets.pow(2).mean()

        deltas = positions[1:] - positions[:-1]
        neighbor_dist = torch.linalg.norm(deltas, dim=1)
        spacing_excess = F.relu(neighbor_dist - self.config.max_spacing).pow(2).mean()
        spacing_anchor = (neighbor_dist - self.config.nominal_spacing).pow(2).mean()

        sigma = F.softplus(self.log_sigma) + 1e-4
        sigma_reg = sigma.pow(2).mean()

        total = (
            data_loss
            + self.config.offset_penalty * offset_mag
            + self.config.smoothness_penalty * smoothness
            + self.config.spacing_penalty * spacing_excess
            + self.config.spacing_anchor_penalty * spacing_anchor
            + self.config.sigma_penalty * sigma_reg
        )
        terms = {
            "total": total.detach(),
            "data": data_loss.detach(),
            "offset": offset_mag.detach(),
            "smoothness": smoothness.detach(),
            "spacing_excess": spacing_excess.detach(),
            "spacing_anchor": spacing_anchor.detach(),
            "sigma_reg": sigma_reg.detach(),
        }
        return total, terms
