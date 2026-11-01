# Archived verified paper figures

These files are committed copies of SVGs/CSV from [successful manuscript plotting CI #41281229086](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41281229086), artifact `paper-figures-artifact-derived`. The generator is `scripts/manuscript_figures.py`; it retrieves previously successful experimental JSON from the frozen climatology-adapted and source-only workflows.

- `adapted_hot.svg` — frozen climatology-adapted CNN vs matching source-selected blend, observed-hot RMSE
- `source_only_hot.svg` — source-trained no-climatology CNN vs 24-hour persistence, observed-hot RMSE
- `comparison.csv` — unrounded annual/hot RMSE underlying both figures

**Do not directly compare baselines across protocols.** Observed-hot scoring is conditional on test truth (>=305 K); not an operational warning detection result.
