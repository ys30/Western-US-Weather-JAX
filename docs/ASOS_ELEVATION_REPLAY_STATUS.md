# Southwest 2022 ASOS elevation-bootstrap reproducibility

The full 72-row public result is stored at [results/southwest_2022_asos_elevation_bootstrap_full_72.csv](../results/southwest_2022_asos_elevation_bootstrap_full_72.csv).

**Independent replay CI: [successful run 41872606350](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41872606350).** Its reproducibility output artifact ID is `13344887550`.

The workflow downloads original ASOS run `41862347926` and elevation-audit run `41774740647`, reproduces the matched 2-site × 3-seed × 4-lapse × 3-block-length bootstrap results using [scripts/recompute_asos_elevation_bootstrap.py](../scripts/recompute_asos_elevation_bootstrap.py), then verifies all 72 unique scenario records against the public CSV with absolute tolerance below 1e-9 K. Each comparison uses 2,000 paired time-block draws.

The 72-row CSV is public experimental evidence. The manuscript, submission materials and v2.84 DOCX remain local/private. These temperature and elevation analyses describe retrospective agreement, not independent operational forecasts.
