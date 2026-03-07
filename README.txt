EE5311 CA2 Scaffold

This repository now contains a starter scaffold for the assignment based on:
- NumPy for data handling
- SciPy for track interpolation and B-spline basis construction
- PyTorch autograd for differentiable MAP optimization

Files
- main.py: entrypoint
- src/ee5311_ca2/data_utils.py: CSV loading
- src/ee5311_ca2/track_utils.py: track arclength parametrization and baseline sampling
- src/ee5311_ca2/weights.py: repeated-shot reliability weights
- src/ee5311_ca2/basis.py: B-spline basis matrix
- src/ee5311_ca2/model.py: differentiable probabilistic model
- src/ee5311_ca2/fit.py: inner optimization
- src/ee5311_ca2/search.py: outer search over baseline start and spacing
- src/ee5311_ca2/diagnostics.py: SVG geometry plots and residual diagnostics
- docs/analysis_framework.md: assignment analysis outline
- docs/report_template.md: 2-page report template
- docs/report_draft.md: draft English report text

Suggested environment
- Python 3.10+
- numpy
- scipy
- torch

Example command
python main.py --data-dir "EE5311 CA2 data" --output results.csv --diagnostics-dir artifacts

Suggested workflow
1. Run the scaffold with a small search first:
   python main.py --start-step 20 --adam-steps 300 --lbfgs-steps 50 --spacing-candidates 1.0 1.01
2. Inspect the best candidate and loss terms.
3. Inspect artifacts/geometry.svg, artifacts/spacing_hist.svg, and artifacts/residual_boxplot.svg.
4. Tune the penalties in src/ee5311_ca2/types.py.
5. Re-run with a denser outer search and longer optimization.

Notes
- The current scaffold treats arrival times with a Student-t observation model.
- Transmission start times are estimated jointly as nuisance parameters tau_j.
- The array is represented as a baseline sampled from the ship track plus a smooth lateral offset field.
- Diagnostics are written as SVG files, so no plotting package is required.
