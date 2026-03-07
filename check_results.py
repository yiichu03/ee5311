from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ee5311_ca2.validation import (
    load_results_csv,
    validate_sensor_geometry,
    validation_report_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EE5311 CA2 results.csv against basic physical checks.")
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results.csv",
        help="Path to results.csv.",
    )
    parser.add_argument(
        "--expected-sensors",
        type=int,
        default=1926,
        help="Expected number of sensors.",
    )
    parser.add_argument(
        "--max-spacing",
        type=float,
        default=1.0213,
        help="Maximum allowed adjacent sensor spacing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_ids, sensor_positions = load_results_csv(args.results)
    summary = validate_sensor_geometry(
        sensor_ids=sensor_ids,
        sensor_positions=sensor_positions,
        expected_sensors=args.expected_sensors,
        max_spacing=args.max_spacing,
    )
    for line in validation_report_lines(summary, args.max_spacing):
        print(line)
    raise SystemExit(0 if summary.passes_physical_constraints else 1)


if __name__ == "__main__":
    main()
