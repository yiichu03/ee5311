EE5311 CA2 — Seabed Sensor Array Localization
==============================================

Approach
--------
Physics-guided differentiable MAP estimation.
The array is modeled as a smooth lateral deviation from the ship track.
Inference uses PyTorch autograd (Adam + L-BFGS) with a Student-t observation
model and repeated-shot reliability weights.

Key design choices:
  - Geometry: p_i = baseline_i + d_i * normal_i  (B-spline lateral offset)
  - Observation model: Student-t NLL (robust to outlier timing measurements)
  - tau_j estimated jointly as nuisance parameters (avoids noisy TDOA differencing)
  - Outer search over baseline start position and sensor spacing candidates
  - Baseline sampled at equal spline arc-length intervals (not GPS arc-length)
    to avoid spacing errors near track segments with large length variation

Environment setup
-----------------
  conda create -n 5311 python=3.11 -y
  conda activate 5311
  pip install numpy scipy "torch" --extra-index-url https://download.pytorch.org/whl/cpu

Reproducing results
-------------------
  conda activate 5311
  python main.py \
      --data-dir "EE5311 CA2 data" \
      --output results.csv \
      --diagnostics-dir artifacts \
      --start-step 5 \
      --spacing-candidates 1.0 1.005 1.01 1.015 1.02 \
      --adam-steps 1500 \
      --lbfgs-steps 150

Expected output:
  Best candidate start_s=40.000 m spacing=1.020000 m
  Final objective=-2.578593
  Physical constraints: PASS
  Spacing violations: 0/1925 (max ~1.021 m, limit 1.021300 m)

Verifying results independently
--------------------------------
  python check_results.py --results results.csv

Diagnostics (written to --diagnostics-dir)
------------------------------------------
  geometry.svg          — estimated array vs ship track vs transmitters
  residual_boxplot.svg  — TOA residual distribution per shot
  spacing_hist.svg      — adjacent sensor spacing profile along the array
  residual_summary.csv  — per-shot mean/median residual, RMSE, sigma, tau

Source layout
-------------
  main.py                          entry point
  check_results.py                 standalone format + physics checker
  src/ee5311_ca2/
    data_utils.py                  CSV loading
    track_utils.py                 track arclength, spline arc-length resampling
    basis.py                       B-spline basis matrix
    weights.py                     repeated-shot reliability weights
    model.py                       differentiable probabilistic forward model
    fit.py                         inner Adam + L-BFGS optimization
    search.py                      outer search over start_s and spacing
    export.py                      results.csv writer
    diagnostics.py                 SVG plots and residual CSV
    validation.py                  physical constraint checks
    types.py                       dataclasses (AssignmentData, FitConfig, ...)
  docs/
    analysis_framework.md          problem analysis and modeling rationale
    report_draft.md                draft report text with actual results
    report_template.md             2-page report structure checklist

CLI reference
-------------
  --data-dir DIR           directory containing the three assignment CSV files
                           (default: "EE5311 CA2 data")
  --output PATH            path to write results.csv  (default: results.csv)
  --diagnostics-dir DIR    write SVGs and residual_summary.csv here
  --num-controls N         B-spline control points for lateral offset (default: 40)
  --spacing-candidates S…  candidate average spacings in metres (default: 1.0 1.005 1.01 1.015 1.02)
  --start-step M           grid step in metres for start-position search (default: 5.0)
  --adam-steps N           Adam iterations per candidate (default: 1500)
  --lbfgs-steps N          L-BFGS iterations for polishing (default: 150)
  --student-nu NU          Student-t degrees of freedom (default: 4.0)
  --device STR             torch device, e.g. cpu or cuda (default: cpu)
