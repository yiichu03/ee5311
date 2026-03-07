from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ee5311_ca2.data_utils import load_assignment_data
from ee5311_ca2.diagnostics import save_diagnostic_artifacts
from ee5311_ca2.export import write_results_csv
from ee5311_ca2.search import run_candidate_search
from ee5311_ca2.types import FitConfig
from ee5311_ca2.validation import validate_sensor_geometry, write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EE5311 CA2 array localization scaffold.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "EE5311 CA2 data",
        help="Directory containing the three assignment CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--num-controls",
        type=int,
        default=40,
        help="Number of B-spline control points for the lateral offset model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--adam-steps",
        type=int,
        default=1500,
        help="Number of Adam steps for the inner optimization.",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=150,
        help="Max LBFGS iterations for the polishing stage.",
    )
    parser.add_argument(
        "--student-nu",
        type=float,
        default=4.0,
        help="Degrees of freedom in the Student-t observation model.",
    )
    parser.add_argument(
        "--start-step",
        type=float,
        default=5.0,
        help="Grid spacing in meters for the baseline start search.",
    )
    parser.add_argument(
        "--spacing-candidates",
        type=float,
        nargs="+",
        default=[1.0, 1.005, 1.01, 1.015, 1.02],
        help="Candidate average spacings to try during the outer search.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for plots and residual summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_assignment_data(args.data_dir)
    config = FitConfig(
        num_controls=args.num_controls,
        student_nu=args.student_nu,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        device=args.device,
        start_step=args.start_step,
        spacing_candidates=tuple(args.spacing_candidates),
    )
    result = run_candidate_search(data, config)
    write_results_csv(args.output, data.sensor_ids, result.sensor_positions)
    save_diagnostic_artifacts(args.diagnostics_dir, data, result, config)
    validation = validate_sensor_geometry(
        sensor_ids=data.sensor_ids,
        sensor_positions=result.sensor_positions,
        expected_sensors=len(data.sensor_ids),
        max_spacing=config.max_spacing,
    )
    write_validation_report(args.diagnostics_dir / "physical_check.txt", validation, config.max_spacing)

    print(f"Best candidate start_s={result.start_s:.3f} m spacing={result.spacing:.6f} m")
    print(f"Final objective={result.final_objective:.6f}")
    print(f"Wrote results to {args.output}")
    print(f"Wrote diagnostics to {args.diagnostics_dir}")
    print(f"Format checks: {'PASS' if validation.passes_format_checks else 'FAIL'}")
    print(f"Physical constraints: {'PASS' if validation.passes_physical_constraints else 'FAIL'}")
    print(
        "Spacing violations: "
        f"{validation.spacing_violation_count}/{validation.num_edges} "
        f"(max {validation.spacing_max:.6f} m, limit {config.max_spacing:.6f} m)"
    )

    skipped = result.metadata.get("skipped_candidates", [])
    if skipped:
        print(f"Skipped {len(skipped)} invalid candidates during outer search")


if __name__ == "__main__":
    main()
