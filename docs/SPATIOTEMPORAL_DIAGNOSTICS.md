# Spatial and temporal diagnostics — evidence boundary

`scripts/frozen_spatial_eval.py` now accepts **optional** `--diagnostics-output path/to/fields.npz`. When a genuine 2022 target ERA5 file and matching saved source checkpoint are available, this writes actual forecast errors and time stamps, plus monthly and per-grid CSVs.

- `.npz`: forecast-valid times, truth temperature, CNN/reference errors, observed-hot mask and fixed threshold.
- `.monthly.csv`: each UTC month, six-hourly sample count, observed-hot sample count, CNN and reference annual-style/all-observation RMSE, and conditional hot RMSE.
- `.grid.csv`: grid *indices*, all-observation errors, observed-hot sample count and conditional hot errors. The grid indices need lat/lon coordinates from the exact matched ERA5 file before mapping.

The exported arrays are paired, not bootstrap data. Existing scalar cross-region artifacts **cannot** be reversed into spatial or monthly fields. This feature is not a claim that such additional 2022 diagnostics have already been computed; a successful full-data workflow using the saved checkpoint is required to produce science-grade maps. No additional model fitting is performed by this optional export.

NaN monthly/grid hot RMSE means no observed t2m ≥305 K for that slice, not zero model error.
