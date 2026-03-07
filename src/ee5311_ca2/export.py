from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_results_csv(path: Path, sensor_ids: np.ndarray, sensor_positions: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "x", "y"])
        for sensor_id, xy in zip(sensor_ids, sensor_positions):
            writer.writerow([int(sensor_id), f"{float(xy[0]):.10f}", f"{float(xy[1]):.10f}"])
