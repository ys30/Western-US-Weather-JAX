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
- Repeated CPU timings and paired CPU/GPU speedup uncertainty; 3 repeated GPU runs are now verified, but CPU timing is still from one earlier run.
- Generalization to additional independent years beyond 2022, other regions, or random seeds.
- U-Net performance, any operational weather forecasting claim, or publication-ready evidence.

## Acceptance gates
1. CPU CI and real ERA5 holdout metrics: **verified**.
2. Real-data evaluation grouped by month and hot-temperature tail: **verified**.
3. GPU execution with real GPU-resident JAX weights and saved hardware audit: **verified on Colab Tesla T4 (one run)**.
4. U-Net or advanced architecture: **deferred pending repeatable GPU benchmarks and stronger scientific baselines**.

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

## Frozen validation selection and independently reported 2022 test (run 41239658291)

The complete workflow **passed**: Python tests, real ERA5 CPU regression, weight selection, independent holdout report, and artifact upload. Selection was based on **2021 validation only**, choosing additional heat weight **2** (total 3x weight above fixed 305 K). The subsequent frozen-model 2022 report gives **3.191155 K** overall RMSE versus **3.220267 K** persistence (**+0.9040%** skill) and fixed-305 K subset **2.138412 K** versus **2.080046 K** persistence (**−2.8060%** skill) across **210,130** gridcell-times. High-temperature skill remains negative, despite improved heat RMSE against unweighted CNN. This is an engineering baseline, not an operationally validated forecast.

Verified run: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41239658291

**Remaining gate:** actual GPU-backed JAX training (hardware audit and reproducible GPU/CPU performance comparison); successful CPU CI does not establish GPU execution.

## Real Colab NVIDIA GPU acceptance — 2026-10-31

**Hardware-backed GPU training verified from user-supplied Colab run artifacts and executed notebook** (not from a GitHub-hosted GPU runner). NVIDIA **Tesla T4**, 15,360 MiB GPU RAM; NVIDIA driver 580.82.07; `nvidia-smi` reported CUDA driver capability 13.0. JAX 0.12.0.1, Python 3.13.16. JAX hardware audit reports `jax_backend="gpu"`, `parameter_platforms=["gpu"]`, device `cuda:0`, and `training_verified_on_gpu=true`.

- Data: real ERA5 2m temperature, 49×49 grid at 6-hour intervals, 2019-01-01 through 2022-12-31 (5,844 time steps).
- Train 2019–2020; validate 2021; independent test 2022; same four-frame CNN; 40 epochs; extra hot-target weight 2 at a fixed 305 K threshold.
- CPU and GPU SHA-256 are identical: `ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6`.
- Both CPU and GPU best epoch: **34**.
- Independent 2022 test RMSE: CPU **3.191155195 K**, GPU **3.191155195 K**, Persistence **3.220266819 K**. Relative CPU/GPU RMSE difference **0.00%**; automatic parity check **passed**.
- Measured training elapsed time: CPU **258.5146 s**, T4 GPU **33.2946 s**, **7.7645×** CPU/GPU elapsed-time ratio for this **single Colab run**. Compilations/overheads included; this is not a multi-run statistically characterized GPU benchmark.
- Accuracy caveat: the validation-selected heat-weighted model has **negative fixed-305-K high-temperature skill** relative to persistence on the independent 2022 test, despite overall RMSE improvement. Hardware success does not solve this scientific limitation.
- Colab notebook emitted a nonblocking NumPy/Numba dependency warning and a CUDA plugin already-exists message; all training/acceptance subprocesses completed with exit code 0. Environment is not claimed warning-free.

**Acceptance verdict:** Real hardware GPU execution, audit, identical input, model parity, and one-run descriptive speedup are **verified**. Repeated timing, other GPU hardware, and operational forecast skill **not yet verified**. No GitHub self-hosted runner was configured or needed for this Colab run.

## Repeated real Tesla T4 GPU benchmark — 2026-10-31

User-supplied executed Colab notebook and ZIP with three independently launched GPU training processes were inspected. All three runs audited `jax_backend=gpu`, GPU-resident parameters on `cuda:0`, JAX 0.12.0.1, Python 3.13.16, identical ERA5 file SHA-256 `ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6`, 40 epochs, best epoch 34, and independent 2022 test RMSE **3.191155195 K**.

| GPU repetition | Training elapsed seconds | Test RMSE K |
| --- | ---: | ---: |
| 1 | 35.62996 | 3.191155195 |
| 2 | 32.71881 | 3.191155195 |
| 3 | 32.01255 | 3.191155195 |

GPU elapsed mean **33.45378 s**, sample standard deviation **1.91743 s**, minimum **32.01255 s**, maximum **35.62996 s**, and test RMSE range **0.0 K**. Separate Python processes include compilation. Relative sample SD / mean ≈ **5.73%**.

Using the *earlier single* same-configuration CPU run (258.5146 s), the GPU repeat-average implies a descriptive **~7.73×** elapsed-time ratio. **No repeated CPU timings were collected in this ZIP**; avoid labeling it a paired repeated speedup distribution or hardware-independent benchmark.

Evidence scope: 3 repeated real GPU measurements and stable test accuracy **verified**; repeated CPU baseline and multi-hardware scaling **not verified**. Exact hardware T4 from the prior Colab acceptance, not independently re-queried from this ZIP's `gpu_metrics.json`.


## Stronger 2022 climatology-blend baseline (verified run 41255803026)

A leakage-controlled **per-gridcell, month-of-year × UTC-hour climatology** was learned only from **2019–2020 training ERA5**. A persistence/climatology convex blend was chosen **only using 2021 validation RMSE** from prespecified weights {0, 0.25, 0.50, 0.75, 1.00}. Validation selected **0.25 climatology + 0.75 persistence**.

| 2022 independent-test method | RMSE (K) |
| --- | ---: |
| Persistence | 3.22027 |
| Four-frame CNN (unweighted) | 3.18849 |
| Validation-selected heat-weighted CNN | 3.19116 |
| Training-only climatology | 4.40352 |
| **Validation-selected persistence/climatology blend** | **2.99462** |

The blend improves over persistence by **~7.01%**, and beats the unweighted CNN's RMSE by **~6.08% relative to CNN** (CNN error is ~6.47% higher than blend). This **supersedes any suggestion that the existing CNN is the best benchmark**; the scientific target is now outperforming the stronger, leakage-controlled blend. This is a simple empirical baseline, not a statement of operational forecast skill.

Source: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41255803026


## Climatology-aware residual CNN: first 2022 holdout (run 41256841210)

The model uses **training-only** (2019–2020) per-gridcell month/UTC-hour climatology, a **2021 validation-selected** persistence/climatology blend (25% climatology), and a JAX residual CNN correcting that blend. Independent test: full 2022. Real ERA5 CPU run and unit tests passed.

| Metric | Strong blended baseline | Climatology-aware CNN |
| --- | ---: | ---: |
| 2022 overall RMSE | 2.994618 K | **2.983654 K** |
| 2022 fixed 305 K heat RMSE | **2.159303 K** | 2.178752 K |

Overall improvement over blended baseline: **0.010965 K (~0.366%)**. Heat-subset performance is **worse** than the blend. Best checkpoint was epoch **40/40**, so saturation has not been established. This is **one seed, one independent year**, with no paired uncertainty interval for the new model; it is premature to claim robust improvement. Previous claims about the existing CNN being inferior to the blend refer to the **earlier plain CNN architecture**, not this new residual model.

Source: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41256841210


## Three-seed CNN robustness and block-bootstrap audit (run 41258489704)

All GitHub CI jobs passed on the real-ERA5, independent-calendar-year 2022 holdout. The stronger **validation-only selected** blend remains **2.994618 K** overall RMSE and **2.159303 K** fixed-305-K heat RMSE. Identical chronological training/validation/test setup and 40 requested epochs across random seeds.

| CNN random seed | 2022 overall RMSE (K) | CNN minus blend (K) | 305 K CNN RMSE (K) | 305 K blend RMSE (K) |
| --- | ---: | ---: | ---: | ---: |
| 42 | 2.983654 | -0.010965 | 2.178752 | 2.159303 |
| 7 | 2.996587 | +0.001968 | 2.097889 | 2.159303 |
| 123 | 2.988817 | -0.005801 | 2.123850 | 2.159303 |

**Conclusion:** Only 2/3 seeds improve annual RMSE over the strong blend. In contrast, 2/3 improve the fixed hot-temperature subset. **Do not claim robust superiority for either overall or hot-subset skill.** The performance tradeoff depends on initialization.

Paired circular moving-block bootstrap over target time, preserving spatial grids within timestamps, used 1000 draws for blocks of 1, 3, 7 and 14 days. At 14 days, the 95% intervals for annual **CNN-minus-blend** RMSE (K) were:

- Seed 42: **[-0.022932, +0.001660]**
- Seed 7: **[-0.006482, +0.011137]**
- Seed 123: **[-0.019357, +0.008926]**

**Every 14-day interval includes zero.** This is exploratory uncertainty conditional on one test year and cannot establish spatially/temporally transferable significance. Future work: analyze validation-selected heat-loss tradeoffs and a seed-robust selection rule; consider multi-year and region tests before escalating architecture complexity.

Evidence: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41258489704


## Validation-selected 305 K heat-weight sensitivity (verified run 41259276907)

Nine climatology-aware CNN trainings (seeds 42, 7, 123 × additional 305 K training loss weights 0, 1, 2) completed successfully. For **each seed independently**, the lowest **2021 validation-set fixed-305-K RMSE** selected **extra weight 2**, implying a 3× multiplier for eligible hot targets in the weighted loss. The 2022 independent-test year was not used for weight selection.

| Seed | Selected weight | 2021 validation heat RMSE K | 2022 annual RMSE K | 2022 fixed-305-K heat RMSE K |
| --- | ---: | ---: | ---: | ---: |
| 42 | 2 | 2.082155 | 2.984499 | 2.054921 |
| 7 | 2 | 2.071333 | 2.997407 | 2.045063 |
| 123 | 2 | 2.061141 | 2.992431 | 2.039816 |
| Strong blend reference | — | — | 2.994618 | 2.159303 |

**Interpretation:** All three selected models improve 2022 fixed-305-K heat RMSE compared with the strong blend by approximately **4.83%**, **5.29%**, and **5.53%**, respectively. But only **two of three** improve annual RMSE. These are three random seeds in one held-out year, **not** an independent multi-year sample. The per-seed heat improvement has not yet been tested using a heat-specific time-block bootstrap, and selection was explicitly optimized for validation heat performance. Next gate: predeclared threshold-aligned bootstrap and additional independent year/region tests.

Source: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41259276907


## Fixed-305-K high-temperature robustness (verified run 41265640777)

The corrected test suite, CPU real-ERA5 workflow, and independent-year evaluation all **passed** after commit `ffbbbf3`. Heat-weighted climatology-aware residual CNN, validation-selected extra loss weight **2**, seeds **42, 7, 123**, independent 2022 test.

Paired circular time-block bootstrap of **truth >=305 K gridcells**, preserving grid dependence within each valid time, 1000 resamples per block length. Negative CNN-minus-blend RMSE differences favor CNN. Verified **14-day-block** 95% percentile intervals in kelvin:

| Seed | 14-day block 95% CI of heat RMSE difference K |
| --- | --- |
| 42 | [-0.051923, -0.020463] |
| 7 | [-0.140386, -0.063754] |
| 123 | [-0.168172, -0.076555] |

**All 1-, 3-, 7-, and 14-day intervals lie entirely below zero for all three seeds.** Each seed also shows lower heat-subset monthly RMSE than the blended baseline in **all eight months with eligible hot observations (March–October 2022)**. January, February, November and December have no qualifying truth-hot observations; monthly hot RMSE is undefined, **not zero**.

This strengthens **within-2022, within-region, seed- and block-length sensitivity** evidence of hot-subset gains. It does **not** establish cross-year generalization, independent spatial-region significance or event-level reliability. Monthly results are descriptive and correlated. Further gates: additional independent calendar years/regions, event-based episode metrics, and potential selection/temperature-conditioning audits.

Evidence: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41265640777


## Earlier 2021 retrospective backtest — validated run 41267651677

A separately rerun chronological experiment used **2018–2019 training**, **2020 validation**, **2021 test** (real ERA5 6-hourly, fixed 24-hour lead). The per-gridcell month/hour climatology came strictly from training, with 2020 validation selecting a 25% climatology / 75% persistence blend. The heat-weighted climatology-aware residual CNN was evaluated at additional fixed-305-K training weight 2 with seeds 42, 7 and 123, as already explored in the earlier 2022 experiments.

| 2021 test metric (K) | Strong blend | Seed 42 | Seed 7 | Seed 123 |
| --- | ---: | ---: | ---: | ---: |
| Annual RMSE | 2.904840 | 2.890040 | 2.900441 | 2.889897 |
| Fixed-305-K hot RMSE | 2.169510 | 2.062780 | 2.004501 | 2.046343 |

All three seeds were lower than the blend both overall and on the fixed hot subset. Relative hot RMSE improvements were about 4.92%, 7.61%, and 5.68%, respectively.

**Evidence boundary:** 2021 was previously involved in earlier model development, including the original 2022 experiments' validation window; this is a **retrospective temporal sensitivity check**, not an untouched external/prospective test. The three models also use fixed weight 2 carried over from development, and no independent high-temperature event-level validation is claimed. The 2022 annual results were mixed across seeds, so this does **not** establish stable annual superiority across settings. A second region or genuinely untouched year remains needed.

Source: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41267651677

