# Figure 4 — Baseline-relative scorecard

This compact scorecard is computed only from the six rows of `figures/manuscript/comparison.csv`. Each entry is `100 × (1 − CNN RMSE / reference RMSE)`; a positive percentage means the CNN has smaller RMSE than the protocol-specific reference, negative means worse. Values are displayed to two decimal places and should not be interpreted as effect estimates of climatology.

Protocol separation is essential: adapted transfer uses a source-selected persistence/climatology blend and pre-2022 target climatology, while source-only transfer uses 24-hour persistence and no target historical climatology. The observed-hot metric is defined by observed t2m ≥305 K, not a forecast alert.

Presentation motivation: [WeatherBench 2 (Rasp et al., 2024)](https://doi.org/10.1029/2023MS004019) uses compact headline scorecards and relative baseline performance, in addition to lead-dependent curves and other diagnostics. Our figure adopts the scorecard **principle only**, not their data or their model comparisons.

The existing archive lacks the gridded paired forecast/error arrays needed for defensible geographical error maps, seasonal breakdowns, forecast/observation snapshots, error distributions, or lead-time skill curves. These should be generated only after verifying the required stored outputs or running a new audit-capable evaluation.
