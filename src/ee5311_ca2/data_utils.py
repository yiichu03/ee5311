from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import AssignmentData


def _load_xy_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return np.column_stack([arr["x"], arr["y"]]).astype(np.float64)


def load_assignment_data(data_dir: Path) -> AssignmentData:
    data_dir = Path(data_dir)
    track_xy = _load_xy_csv(data_dir / "ca2-track.csv")
    tx_locations = _load_xy_csv(data_dir / "ca2-transmissions.csv")

    timing_arr = np.genfromtxt(data_dir / "ca2-timings.csv", delimiter=",", names=True)
    shot_names = tuple(name for name in timing_arr.dtype.names if name != "sensor")
    timings = np.column_stack([timing_arr[name] for name in shot_names]).astype(np.float64)
    sensor_ids = timing_arr["sensor"].astype(np.int64)
    shot_pairs = tuple((2 * i, 2 * i + 1) for i in range(len(shot_names) // 2))
    tx_xy = np.repeat(tx_locations, repeats=2, axis=0)

    if tx_xy.shape[0] != timings.shape[1]:
        raise ValueError(
            "Transmission geometry does not match timing columns: "
            f"{tx_xy.shape[0]} shot positions for {timings.shape[1]} timing columns."
        )

    return AssignmentData(
        track_xy=track_xy,
        tx_xy=tx_xy,
        sensor_ids=sensor_ids,
        timings=timings,
        shot_names=shot_names,
        shot_pairs=shot_pairs,
    )
