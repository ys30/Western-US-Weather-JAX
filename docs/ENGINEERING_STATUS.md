# Engineering status and verification gates

Updated: 2026-10-31.

## Verified
- GitHub Actions Python unit tests and synthetic smoke.
- Public ARCO-ERA5 real-data fetch, preflight, automatic JAX CPU CNN training, artifact upload.
- Real ERA5 2020–2021 data: 6-hourly 2m temperature, 49 × 49 regional grid.
- Chronological holdout: train 2020, validate 2021 Jan–Jun, test 2021 Jul–Dec.
- Single-frame CNN and four-frame CNN (zero-initialized residual + target UTC-hour cyclic inputs).
- Held-out monthly and exploratory test-temperature P95 error breakdown.
- Paired circular moving-block bootstrap (4 time steps = 24 hours, 400 draws) for overall RMSE difference.

## Latest verified holdout (GitHub Actions run 41231631347)
| Method | Test RMSE K | CNN minus persistence K | Exploratory 95% block-bootstrap interval K |
| --- | ---: | ---: | --- |
| Persistence | 2.94224 | — | — |
| Single-frame CNN | 2.98605 | +0.04381 | [+0.01730, +0.07113] |
| Four-frame CNN | 2.91864 | −0.02360 | [−0.05016, −0.00147] |

The four-frame CNN's overall RMSE is about 0.80% lower than persistence on this single temporal holdout.
**Limitation:** The time-block bootstrap assumes the selected block length adequately captures forecast-error temporal dependence; spatial dependence is preserved within each sampled timestamp. No claim of cross-year/region significance is made.

### Important extreme-heat caveat
The four-frame CNN's exploratory test-truth P95-temperature RMSE is **2.12745 K**, versus **1.97670 K** for persistence (~7.63% worse). The P95 threshold, 307.234 K, was calculated from the test-set truth and is descriptive, **not** a fixed operational threshold or a pre-registered heatwave definition. Overall skill must not be presented as superior extreme-heat forecasting.

## Not verified
- An actual CUDA-backed JAX GPU training run. The GPU training entry point and its unit-level hardware checks exist, but hosted GitHub Actions currently run on CPU.
- Reproducible CPU vs GPU throughput/latency comparison.
- Independent evaluation over additional held-out years, regions or random seeds.
- U-Net performance, any operational weather forecasting claim, or publication-ready evidence.

## Acceptance gates
1. CPU CI and real ERA5 holdout metrics: **verified**.
2. Real-data evaluation grouped by month and hot-temperature tail: **verified**.
3. GPU execution with real GPU-resident JAX weights and saved hardware audit: **not yet verified**.
4. U-Net or advanced architecture: **deferred until GPU verification and robust baselines**.

Artifacts: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41231631347
