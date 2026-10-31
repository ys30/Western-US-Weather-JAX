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

## Temporal block-size sensitivity (GitHub Actions run 41233159820)

Paired circular block bootstrap with 1,000 resamples for each block length. The four-frame model's CNN-minus-persistence test RMSE difference is −0.02360 K:

| Temporal block | 95% bootstrap interval (K) | Includes zero? |
| --- | --- | --- |
| 1 day | [−0.04863, −0.00010] | No |
| 3 days | [−0.04917, +0.00238] | **Yes** |
| 7 days | [−0.04257, −0.00426] | No |
| 14 days | [−0.04209, −0.00323] | No |

The 3-day sensitivity interval crosses zero; improvement is **not** robust across all block lengths. Interpretation remains exploratory, not proof of statistical significance across future years.

## Independent calendar-year evaluation: 2022 (run 41233856494)

Data: real public ERA5 2019–2022; train 2019–2020, validation 2021, independent holdout 2022. Four-frame residual CNN with known UTC-hour features.

| Method | 2022 test RMSE (K) |
| --- | ---: |
| Persistence | 3.22027 |
| Four-frame CNN | 3.18849 |

CNN is 0.03178 K (~0.99%) better in aggregate. Paired moving-block bootstrap (1,000 draws per setting):

| Block | 95% RMSE difference CI (K) |
| --- | --- |
| 1 day | [−0.05163, −0.01172] |
| 3 days | [−0.05450, −0.00791] |
| 7 days | [−0.05263, −0.00996] |
| 14 days | [−0.05077, −0.00954] |

All four exploratory intervals are below zero on the 2022 holdout. This is encouraging cross-year generalization evidence but is **not** a claim of operational forecasting skill or universal significance.

**Heat-tail limitation persists:** test-truth P95 subset CNN RMSE 2.22494 K versus persistence 2.06231 K (~7.9% worse). The P95 cutoff is test-derived exploratory diagnostics, not a predeclared heatwave criterion.

Action: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41233856494

## Fixed-threshold heat-weighted experiment (run 41234785888)

On the 2022 independent holdout, both runs succeeded with the same temporal split and seed. Baseline 4-frame CNN: overall RMSE **3.18849 K**, exploratory P95 RMSE **2.22494 K**. The 305 K heat-weighted model (additional weight 2, total hot weight 3): overall RMSE **3.19116 K**, P95 RMSE **2.12150 K**. Thus P95 error improved by about 4.65%, while aggregate RMSE degraded by 0.00267 K. Persistence P95 RMSE remains better at **2.06231 K**.

Important: 305 K is a fixed training-temperature threshold selected before this comparison; P95 remains test-derived exploratory stratification. The next run explicitly exports test-set RMSE at the fixed 305 K threshold as well.

## Fixed 305 K high-temperature evaluation (run 41235424599)

For the independent 2022 test, the same 210,130 gridcell-times have target temperature >= 305 K.

| Forecast | Fixed-threshold high-temperature RMSE (K) |
| --- | ---: |
| Persistence | **2.08005** |
| Original four-frame CNN | 2.23366 |
| Heat-weighted four-frame CNN | 2.13841 |

The weighted model improves over the original CNN by ~4.26% on this heat subset but remains ~2.81% worse than persistence. Its whole-year RMSE is also slightly worse than the unweighted CNN. Do not portray this as achieving heat-tail skill.

## Not verified
- An actual CUDA-backed JAX GPU training run. The GPU training entry point and its unit-level hardware checks exist, but hosted GitHub Actions currently run on CPU.
- Reproducible CPU vs GPU throughput/latency comparison.
- Generalization to additional independent years beyond 2022, other regions, or random seeds.
- U-Net performance, any operational weather forecasting claim, or publication-ready evidence.

## Acceptance gates
1. CPU CI and real ERA5 holdout metrics: **verified**.
2. Real-data evaluation grouped by month and hot-temperature tail: **verified**.
3. GPU execution with real GPU-resident JAX weights and saved hardware audit: **not yet verified**.
4. U-Net or advanced architecture: **deferred until GPU verification and robust baselines**.

Artifacts: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41231631347

## Validation-only high-temperature weight selection (run 41238730272)

Prespecified additional target-temperature weights 0/1/2 at 305 K were evaluated on the **2021 validation** data:

| Additional heat weight | 2021 validation fixed-heat RMSE (K) | 2022 holdout overall RMSE (K) |
| --- | ---: | ---: |
| 0 | 2.22467 | **3.18849** |
| 1 | 2.15295 | 3.18994 |
| 2 (selected) | **2.13242** | 3.19116 |

The selected weight is **2**, corresponding to 3x total weight for hot training targets. On the independent 2022 test, its fixed 305 K heat RMSE is **2.13841 K**, still worse than persistence **2.08005 K**. No model offers verified improvement over persistence on this hot subset. The selection script reads **only validation metrics**; a separate reporting script evaluates the frozen choice on the test set.

Action: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41238730272
