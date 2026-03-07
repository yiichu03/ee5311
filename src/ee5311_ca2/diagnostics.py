from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .types import AssignmentData, FitConfig, FitResult


def predict_times(sensor_positions: np.ndarray, tx_xy: np.ndarray, tau: np.ndarray, config: FitConfig) -> np.ndarray:
    dx = sensor_positions[:, None, 0] - tx_xy[None, :, 0]
    dy = sensor_positions[:, None, 1] - tx_xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + config.z_offset ** 2)
    return tau[None, :] + dist / config.sound_speed


def residual_matrix(data: AssignmentData, result: FitResult, config: FitConfig) -> np.ndarray:
    return data.timings - predict_times(result.sensor_positions, data.tx_xy, result.tau, config)


def residual_summary_rows(data: AssignmentData, result: FitResult, config: FitConfig) -> list[dict[str, float | str]]:
    residuals = residual_matrix(data, result, config)
    rows: list[dict[str, float | str]] = []
    for idx, shot_name in enumerate(data.shot_names):
        shot_res = residuals[:, idx]
        shot_abs = np.abs(shot_res)
        rows.append(
            {
                "shot": shot_name,
                "mean_abs_residual": float(np.nanmean(shot_abs)),
                "median_abs_residual": float(np.nanmedian(shot_abs)),
                "rmse": float(np.sqrt(np.nanmean(shot_res * shot_res))),
                "sigma": float(result.sigma[idx]),
                "tau": float(result.tau[idx]),
            }
        )
    return rows


def write_residual_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shot", "mean_abs_residual", "median_abs_residual", "rmse", "sigma", "tau"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]


def _svg_footer(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _scale_points(x: np.ndarray, y: np.ndarray, width: int, height: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xspan = max(xmax - xmin, 1e-8)
    yspan = max(ymax - ymin, 1e-8)
    sx = pad + (x - xmin) * (width - 2 * pad) / xspan
    sy = height - pad - (y - ymin) * (height - 2 * pad) / yspan
    return sx, sy


def _polyline(points: np.ndarray, color: str, width: float, opacity: float = 1.0) -> str:
    coord = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}" points="{coord}"/>'
    )


def _circle(x: float, y: float, radius: float, color: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}"/>'


def _text(x: float, y: float, text: str, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" fill="#222">{text}</text>'
    )


def plot_geometry(path: Path, data: AssignmentData, result: FitResult) -> None:
    width, height, pad = 900, 650, 60
    baseline_xy = result.metadata.get("baseline_xy")

    x_all = [data.track_xy[:, 0], result.sensor_positions[:, 0], data.tx_xy[:, 0]]
    y_all = [data.track_xy[:, 1], result.sensor_positions[:, 1], data.tx_xy[:, 1]]
    if baseline_xy is not None:
        x_all.append(baseline_xy[:, 0])
        y_all.append(baseline_xy[:, 1])
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    sx, sy = _scale_points(x, y, width, height, pad)

    track_n = len(data.track_xy)
    est_n = len(result.sensor_positions)
    tx_n = len(data.tx_xy)

    idx = 0
    track_pts = np.column_stack([sx[idx : idx + track_n], sy[idx : idx + track_n]])
    idx += track_n
    est_pts = np.column_stack([sx[idx : idx + est_n], sy[idx : idx + est_n]])
    idx += est_n
    tx_pts = np.column_stack([sx[idx : idx + tx_n], sy[idx : idx + tx_n]])
    idx += tx_n
    base_pts = None
    if baseline_xy is not None:
        base_n = len(baseline_xy)
        base_pts = np.column_stack([sx[idx : idx + base_n], sy[idx : idx + base_n]])

    parts = _svg_header(width, height)
    parts.append(_text(30, 35, "Track, transmitters, and estimated array", size=18))
    parts.append(_polyline(track_pts, "#9aa0a6", 2.0))
    if base_pts is not None:
        parts.append(_polyline(base_pts, "#4e79a7", 1.5, opacity=0.9))
    parts.append(_polyline(est_pts, "#d62728", 1.8))
    for tx_x, tx_y in tx_pts:
        parts.append(_circle(float(tx_x), float(tx_y), 4.0, "#2ca02c"))
    parts.append(_text(width - 190, 40, "gray: track", size=12))
    parts.append(_text(width - 190, 58, "blue: baseline", size=12))
    parts.append(_text(width - 190, 76, "red: estimate", size=12))
    parts.append(_text(width - 190, 94, "green: transmitters", size=12))
    _svg_footer(parts, path)


def plot_spacing(path: Path, result: FitResult, config: FitConfig) -> None:
    width, height, pad = 900, 450, 55
    d = np.diff(result.sensor_positions, axis=0)
    spacing = np.linalg.norm(d, axis=1)
    counts, edges = np.histogram(spacing, bins=50)

    xmin, xmax = float(edges[0]), float(edges[-1])
    ymax = max(int(np.max(counts)), 1)

    def sx(v: float) -> float:
        return pad + (v - xmin) * (width - 2 * pad) / max(xmax - xmin, 1e-8)

    def sy(v: float) -> float:
        return height - pad - v * (height - 2 * pad) / ymax

    parts = _svg_header(width, height)
    parts.append(_text(30, 35, "Adjacent spacing distribution", size=18))
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>')

    for idx, count in enumerate(counts):
        x0 = sx(float(edges[idx]))
        x1 = sx(float(edges[idx + 1]))
        y1 = sy(float(count))
        parts.append(
            f'<rect x="{x0:.2f}" y="{y1:.2f}" width="{max(x1 - x0 - 1.0, 1.0):.2f}" '
            f'height="{height - pad - y1:.2f}" fill="#4e79a7" opacity="0.85"/>'
        )

    for value, color, label in (
        (config.nominal_spacing, "#2ca02c", "nominal"),
        (config.max_spacing, "#d62728", "max"),
    ):
        xline = sx(float(value))
        parts.append(
            f'<line x1="{xline:.2f}" y1="{pad}" x2="{xline:.2f}" y2="{height-pad}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>'
        )
        parts.append(_text(xline + 4.0, pad + 16.0 if label == "nominal" else pad + 34.0, label, size=12))

    parts.append(_text(width / 2.0, height - 15, "Adjacent sensor distance (m)", size=13, anchor="middle"))
    _svg_footer(parts, path)


def plot_residuals(path: Path, data: AssignmentData, result: FitResult, config: FitConfig) -> None:
    width, height, pad = 980, 460, 55
    residuals = residual_matrix(data, result, config)
    clip = np.nanquantile(np.abs(residuals), 0.95)
    clip = float(max(clip, 1e-6))

    def sy(v: float) -> float:
        return pad + (clip - v) * (height - 2 * pad) / (2 * clip)

    parts = _svg_header(width, height)
    parts.append(_text(30, 35, "Arrival-time residuals by shot", size=18))
    y_zero = sy(0.0)
    parts.append(f'<line x1="{pad}" y1="{y_zero:.2f}" x2="{width-pad}" y2="{y_zero:.2f}" stroke="#666"/>')

    x_positions = np.linspace(pad + 45, width - pad - 45, residuals.shape[1])
    box_width = 28.0

    for idx, x in enumerate(x_positions):
        values = residuals[:, idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        whisk_low, whisk_high = np.percentile(values, [10, 90])
        q1 = float(np.clip(q1, -clip, clip))
        med = float(np.clip(med, -clip, clip))
        q3 = float(np.clip(q3, -clip, clip))
        whisk_low = float(np.clip(whisk_low, -clip, clip))
        whisk_high = float(np.clip(whisk_high, -clip, clip))

        parts.append(
            f'<line x1="{x:.2f}" y1="{sy(whisk_low):.2f}" x2="{x:.2f}" y2="{sy(whisk_high):.2f}" '
            f'stroke="#333" stroke-width="1.2"/>'
        )
        parts.append(
            f'<rect x="{x - box_width/2:.2f}" y="{sy(q3):.2f}" width="{box_width:.2f}" '
            f'height="{max(sy(q1) - sy(q3), 1.0):.2f}" fill="#f28e2b" opacity="0.8" stroke="#333"/>'
        )
        parts.append(
            f'<line x1="{x - box_width/2:.2f}" y1="{sy(med):.2f}" x2="{x + box_width/2:.2f}" y2="{sy(med):.2f}" '
            f'stroke="#111" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{x - 10:.2f}" y1="{sy(whisk_low):.2f}" x2="{x + 10:.2f}" y2="{sy(whisk_low):.2f}" '
            f'stroke="#333" stroke-width="1.1"/>'
        )
        parts.append(
            f'<line x1="{x - 10:.2f}" y1="{sy(whisk_high):.2f}" x2="{x + 10:.2f}" y2="{sy(whisk_high):.2f}" '
            f'stroke="#333" stroke-width="1.1"/>'
        )
        parts.append(_text(x, height - 18, data.shot_names[idx], size=12, anchor="middle"))

    parts.append(_text(25, pad + 10, f"+/- {clip:.3f} s", size=12))
    _svg_footer(parts, path)


def save_diagnostic_artifacts(output_dir: Path, data: AssignmentData, result: FitResult, config: FitConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = residual_summary_rows(data, result, config)
    write_residual_summary(output_dir / "residual_summary.csv", rows)
    plot_geometry(output_dir / "geometry.svg", data, result)
    plot_spacing(output_dir / "spacing_hist.svg", result, config)
    plot_residuals(output_dir / "residual_boxplot.svg", data, result, config)
