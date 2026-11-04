# Western-US-Weather-JAX

**Reproducible ERA5 regional temperature forecasting, JAX CNN engineering, and validated CPU/GPU execution.**

[Portfolio case study](https://ys30.github.io/projects/western-us-weather-jax.html) · [Recruiter evidence brief](docs/RECRUITER_EVIDENCE_BRIEF.md) · [Experiment provenance and full archives](docs/EXPERIMENT_REPRODUCIBILITY_MANIFEST.md) · [GitHub Actions](../../actions)

## At a glance

| Scope | Verified evidence |
| --- | --- |
| Task | **24-hour-ahead** ERA5 2 m temperature regression; four six-hourly inputs, **49 × 49** regional fields |
| Implementation | Python, **JAX**, residual CNNs, chronological train/validation/test splits, spatial climatology, multi-seed evaluation |
| GPU engineering | Real **NVIDIA Tesla T4**: 258.51 s single CPU reference vs 33.29 s initial GPU training (**7.76× descriptive ratio**); three additional GPU runs averaged 33.45 ± 1.92 s |
| Evaluation | Source training 2019–2020, validation 2021; target tests **Southwest 2022**, **Southwest 2023**, **Southern Plains 2022** |
| Statistical sensitivity | Three seeds (42, 7, 123), direct paired moving-block bootstrap (3, 7, 14, 28 days) for matched channel arms |
| Reproducibility | All **three** archival reruns preserved **54 files each**, with **53/53** independently verified file digests per ZIP |

### Why this work is useful

- Built and tested a real-data scientific ML pipeline, not a synthetic weather-model demo.
- Compared CNNs with meaningful persistence/climatology baselines, including cases where the CNN **did not** outperform the comparator.
- Tested model sensitivity to climatology features across a second test year and a geographically distinct target.
- Audited GPU execution, prediction parity, workflow success, model checkpoints, gridded predictions, and SHA256 checksums.

## Expanded ten-seed cross-target findings (research update)

The later research analysis extends the matched-feature comparison beyond the original three-seed showcase. These are **exploratory, retrospective ERA5 results**, not operational heat-warning scores. The primary contrast is source-mean minus historical-channel observed-hot RMSE (K): a positive difference favors the historical feature channel.

| Frozen-source target | Historical channel better | Median paired ΔRMSE (K) | Across-seed IQR (K) |
| --- | ---: | ---: | --- |
| Southwest 2022 | 6/10 seeds | +0.0017 | [-0.0079, +0.0173] |
| Southwest 2023 | 6/10 seeds | +0.0063 | [-0.0109, +0.0175] |
| Southern Plains 2022 | 6/10 seeds | +0.0063 | [-0.0057, +0.0186] |
| Southern Plains 2023 | 7/10 seeds | +0.0096 | [-0.0017, +0.0252] |

[Download the aggregate ten-seed and factorial-interaction CSV](results/ten_seed_four_case_research_summary.csv). This is a curated numeric summary transcribed from the private research draft, not a newly rerun raw-data experiment. These figures describe an across-initialization distribution, **not** a cross-seed confidence interval or proof of general superiority. Earlier three-seed charts below are historical subsets. The full four-case ten-seed data, archives and factorial interaction analyses require experiment-specific provenance; avoid conflating this with the separately regionally retrained experiments.

### Verified airport-station observations — Southwest 2022 (new empirical result)

The archive-backed ASOS verification **completed successfully** in [GitHub Actions run 41862347926](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41862347926), artifact `13340986074`. The full artifact was independently read and audited against source SHA256s; RMSE was recomputed from each matched station×seed CSV, and all three seeds used exactly the same observation-time samples. This is **retrospective airport/reanalysis agreement** and is not an independent prospective or operational forecast benchmark.

| Station | Seed | Matched / expected | Historical RMSE (K) | Source-mean RMSE (K) | Δ source − historical (K) | 14-day block 95% CI (K) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Albuquerque (ABQ) | 42 | 1,457 / 1,460 | 3.3640 | 3.3558 | −0.0082 | [−0.0236, +0.0056] |
| Albuquerque (ABQ) | 7 | 1,457 / 1,460 | 3.3706 | 3.3738 | +0.0032 | [−0.0012, +0.0076] |
| Albuquerque (ABQ) | 123 | 1,457 / 1,460 | 3.3467 | 3.3461 | −0.0006 | [−0.0064, +0.0050] |
| Roswell (ROW) | 42 | 1,459 / 1,460 | 4.3043 | 4.3205 | +0.0162 | [+0.0070, +0.0258] |
| Roswell (ROW) | 7 | 1,459 / 1,460 | 4.3129 | 4.3068 | −0.0061 | [−0.0125, +0.0003] |
| Roswell (ROW) | 123 | 1,459 / 1,460 | 4.3241 | 4.3260 | +0.0019 | [−0.0049, +0.0085] |

ABQ had **99.8%** target-time coverage and ROW **99.9%**. The across-seed median annual ΔRMSE was **−0.0006 K (ABQ)** and **+0.0019 K (ROW)**; all six absolute contrasts were below 0.017 K, with no seed-invariant direction. The main temporal uncertainty intervals used **14-day calendar blocks and 2,000 paired resamples**; 7/28-day-block sensitivity files are present in the artifact.

**Observed-hot (station temperature ≥305 K):** ABQ had 42 paired hot times across 35 UTC days, ROW 180 across 112 days. The hot-subset source-minus-historical RMSE differences across seeds 42 / 7 / 123 were **ABQ +0.1155 / +0.0098 / +0.0041 K** and **ROW +0.0663 / +0.0307 / −0.0275 K**. These are exploratory, threshold-conditioned, seed-dependent diagnostics, *not* heat-event detection. Station temperatures may already inform ERA5 assimilation; grid-point representativeness, elevation and local-site biases are unresolved.

[Exact numeric CSV](results/southwest_2022_asos_run41862347926_verified.csv) · [Methods / QA protocol](docs/STATION_VALIDATION_ANALYSIS_PROTOCOL.md) · [Original Actions artifact](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41862347926).

![Verified Southwest 2022 ABQ/ROW paired annual RMSE contrasts](figures/southwest_2022_asos_station_contrasts_verified.svg)

**Evidence boundaries:** the two-station/three-seed comparison supplements—but does not replace—the separate four-case ten-seed ERA5-only analysis; intervals describe within-seed temporal resampling, not across-seed confidence or independent model deployment. Private manuscript text and submission figures remain local.

### ASOS station–grid elevation sensitivity (real audited inputs)

The independent [ERA5 surface geopotential and IEM station-elevation audit](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41774740647) passed. It found ABQ **1,620.0 m station vs 1,775.1 m ERA5 bilinear elevation** (station minus grid **−155.08 m**) and ROW **1,118.0 m vs 1,158.8 m** (**−40.76 m**). Re-evaluating the *same* observed station times and three archived CNN seed predictions with a common illustrative dry-environment lapse adjustment `T_adjusted = T_grid − lapse_rate × (z_station − z_grid) / 1000` changes the 6.5 K/km temperatures by **+1.008 K at ABQ** and **+0.265 K at ROW**.

| Site | Seed | Raw source − historical annual ΔRMSE (K) | 6.5 K/km adjusted ΔRMSE (K) |
| --- | ---: | ---: | ---: |
| ABQ | 42 | −0.00819 | −0.00242 |
| ABQ | 7 | +0.00320 | −0.00266 |
| ABQ | 123 | −0.00058 | +0.00056 |
| ROW | 42 | +0.01619 | +0.01436 |
| ROW | 7 | −0.00610 | −0.00725 |
| ROW | 123 | +0.00191 | +0.00218 |

The [full 24-row audited scenario CSV](results/southwest_2022_asos_elevation_sensitivity_verified.csv) includes **0 / 3.0 / 6.5 / 9.8 K/km** corrections. Two ABQ seeds switch ranking at 6.5 K/km; no ROW seed does. These are **assumption-driven sensitivity scenarios** applied identically to all three forecast fields, *not* fitted or meteorologically time-varying station corrections, not a validation of lapse rate in all weather, and not proof of improved forecast skill. Recomputed **14-day, 2,000-draw paired temporal bootstrap intervals** for the 6.5 K/km scenario are available in [the adjusted-bootstrap CSV](results/southwest_2022_asos_elevation_bootstrap_14day.csv). For ABQ, all three adjusted within-seed intervals still include zero. For ROW seed 42 the interval is [+0.00514, +0.02391] K, and for ROW seed 7 it is [−0.01341, −0.00109] K; the two seeds disagree in direction. The full local audit also evaluates 7- and 28-day block lengths. These are conditional temporal, not cross-seed, intervals.

### New cross-case elevation robustness (Southwest 2023 / Southern Plains 2022)

Using the **previously verified four-station elevation audit** (run 41774740647) and the accepted Southwest 2023 / Southern Plains 2022 matched ASOS artifacts (run 41875075661), both historical and source-mean CNN fields were evaluated after an identical **6.5 K/km illustrative height correction**. No station calibration or new model training was performed.

| Site-year | ERA5-to-station height difference (station − ERA5) | Annual ΔRMSE (K), 6.5 K/km, seeds 42 / 7 / 123 |
| --- | ---: | --- |
| Southwest 2023 ABQ | −155.08 m | −0.00787 / +0.00134 / +0.00178 |
| Southwest 2023 ROW | −40.76 m | +0.01355 / −0.00152 / +0.00437 |
| Southern Plains 2022 ABI | −19.30 m | +0.01608 / −0.00117 / −0.00261 |
| Southern Plains 2022 OKC | +22.63 m | +0.01149 / −0.00570 / −0.00215 |

[Exact 12-pair CSV](results/additional_asos_elevation_6p5K_12_verified.csv) · [144-scenario reproducibility code](scripts/recompute_additional_asos_elevation.py) · [144-scenario workflow](.github/workflows/additional-asos-elevation-replay.yml).

Across the additional cases, ABQ seed 123 and ROW seed 7 switch the direction of the annual feature-channel contrast after this common adjustment; ABI and OKC do not. The **144-scenario** factorial includes two region-year targets, four site-years, three seeds, four lapse rates (0/3/6.5/9.8 K/km) and three temporal block sizes (7/14/28 days). All **144 scenario rows are publicly archived**, with independent source-artifact replay and GitHub publication confirmed by [successful Actions run 41878154087](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41878154087). [Download the verified complete CSV](results/additional_asos_elevation_144_scenarios.csv). Across the 12 new site/seed pairs, 14-day within-seed intervals excluding zero number **5, 5, 3 and 3** for lapse assumptions 0, 3.0, 6.5 and 9.8 K/km respectively; neither of the two 6.5 K/km sign-reversal pairs has an interval excluding zero. These are hypothetical station–grid representativeness scenarios, not optimized physical height corrections, independent forecasts, or evidence of a reliable channel winner.

### Independent archive-backed ASOS replication: Southwest 2023 and Southern Plains 2022

The matrix [Actions run 41875075661](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41875075661) **passed both cases**, including strict parsing of all generated JSON, archive SHA256 verification, year-and-region-specific IEM ASOS retrieval, matched-observation identity across three model initializations, primary 14-day/2,000-draw paired block bootstrap and 7/28-day sensitivities. Both full artifacts were downloaded and the RMSE was independently recomputed from their matched prediction/observation CSVs.

| Site-year | Matched / 1,460 | Observed-hot n (≥305 K) | Annual source-mean − historical ΔRMSE across seeds 42 / 7 / 123 (K) | Median ΔRMSE (K) |
| --- | ---: | ---: | --- | ---: |
| Southwest 2023 · ABQ | 1,458 | 88 | −0.01611 / +0.00652 / −0.00093 | −0.00093 |
| Southwest 2023 · ROW | 1,457 | 184 | +0.01463 / +0.00019 / +0.00340 | +0.00340 |
| Southern Plains 2022 · ABI | 1,456 | 234 | +0.01596 / −0.00088 / −0.00289 | −0.00088 |
| Southern Plains 2022 · OKC | 1,453 | 123 | +0.01127 / −0.00593 / −0.00242 | −0.00242 |

[All 12 new site/seed metrics, actual RMSE and 14-day CIs](results/additional_asos_southwest2023_plains2022_verified.csv).

![Verified 18 station-year × seed paired contrasts with 14-day block intervals](figures/asos_three_case_18_contrast_verified.svg)

Together with Southwest 2022 ABQ and ROW above, the validated station dataset covers **three region-year cases, six site-years and 18 fitted-seed/site-year comparisons**, with 9 positive and 9 negative annual channel differences. Five of the 12 new within-seed 14-day 95% intervals exclude zero, **but not in a consistent direction across seeds and sites**; these are not multiplicity-adjusted and are not independent across model seeds. Each case uses **retrospective airport observations potentially overlapping with ERA5 assimilation**, not independent operational 24-hour forecast skill. New regional comparisons are from the archived source-trained CNN predictions without target-specific retraining.

## Key results — do not mix the two experimental protocols

**Protocol A: heat-weighted, regionally retrained models vs a regional validation-selected persistence–climatology blend.** Lower RMSE is better; observed-hot means verifying ERA5 2 m temperature ≥305 K.

| 2022 test / metric | Blend RMSE (K) | CNN seeds 42 / 7 / 123 RMSE (K) |
| --- | ---: | --- |
| Original region · annual | 2.99462 | 2.98450 / 2.99741 / 2.99243 |
| Original region · observed-hot | 2.15930 | 2.05492 / 2.04506 / 2.03982 |
| Second Southwest region · annual | 3.80252 | 3.77695 / 3.79943 / 3.78322 |
| Second Southwest region · observed-hot | 2.40826 | 2.31885 / 2.31365 / 2.32763 |

All three seeds improved observed-hot RMSE in both target regions under this **local retraining** protocol. Annual improvement was not uniform in the original region. These numbers **do not imply frozen-weight zero-shot transfer**. See [verified runs and GPU audit](docs/RECRUITER_EVIDENCE_BRIEF.md).

**Protocol B: frozen source-trained matched climatology-channel models**, evaluated without target CNN retraining (historical channel vs source-mean channel). Both arms still receive a shared baseline carrying historical climatology. The comparison isolates a **specific feature-channel substitution**, not complete removal of target history.

| Target / metric | Historical feature channel (seeds 42 / 7 / 123) | Source-mean feature channel (same seeds) |
| --- | --- | --- |
| Southwest 2022 · observed-hot RMSE (K) | 2.3474 / 2.3308 / 2.3228 | 2.3876 / 2.3524 / 2.3074 |
| Southwest 2023 · observed-hot RMSE (K) | 2.0932 / 2.0724 / 2.0568 | 2.1144 / 2.1022 / 2.0457 |
| Southern Plains 2022 · observed-hot RMSE (K) | 2.5704 / 2.5645 / 2.5347 | 2.6020 / 2.5960 / 2.5291 |

In each target, historical-channel training helped **two of three** seeds but not the third. No seed-invariant advantage is established. The directly paired time-block CIs are seed- and block-dependent. Source-model transfer is **not** purely zero-history transfer.

- [Southwest 2022 matched evaluation](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41356348903)
- [Southwest 2023 temporal replication](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41362832891)
- [Southern Plains 2022 geographical replication](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41386676606)

### Journal-style study-domain and uncertainty figures

![Coordinate-verified ERA5 study regions](figures/paper_study_domains.svg)

**Study domains:** Source 31–43°N, 121–109°W; Southwest 30–36°N, 108–102°W; Southern Plains 32–38°N, 101–95°W. Region rectangles are drawn from coordinates rather than an unverified illustrated basemap.

![Matched 14-day paired-bootstrap intervals](figures/paper_bootstrap_14day.svg)

The error bars are 14-day paired-bootstrap exploratory 95% intervals; unlike the earlier simplified web graphic they preserve uncertainty bounds. The final submission figures remain private.

## Manuscript-derived visual evidence (public adaptations)

The following accessible SVG figures **adapt the research structure and verified numbers** from the private manuscript; they are **not the original private PNG plates**. The third graphic shows point differences only (the complete 14-day confidence intervals remain in the unpublished manuscript and CI JSON).

![Source training to independent target evaluation protocol](figures/manuscript_protocol_public.svg)

![Matched-channel observed-hot RMSE by target and seed](figures/manuscript_performance_public.svg)

![Source-mean minus historical paired hot-RMSE differences](figures/manuscript_bootstrap_public.svg)

Source-mean minus historical **positive** differences favor the historical-feature channel; this occurred in **two of three** seeds for each target. These are retrospective error diagnostics, not heat-alert detection or a demonstration of universal improvement.

## Earlier public engineering charts

![Real Tesla T4 versus one CPU training-time reference](figures/gpu_training_time.svg)

![Regional observed-hot RMSE under local retraining](figures/heat_rmse_regions.svg)

![Retrospectively selected multiday hot episode comparisons](figures/hot_episode_coverage.svg)

The regional and episode figures depict **Protocol A**, not the frozen-weight matched-channel experiments in Protocol B. Truth-defined high-temperature subsets and episodes are retrospective diagnostics, **not forecast event detection or operational warning skill**.

## Reproduce and inspect

```bash
python -m pip install -e '.[test,zarr]' gcsfs
pytest -q
python -m weatherjax --epochs 4
```

The three full-data archives each contain **2 ERA5 NetCDF inputs, 6 checkpoint NPZ, 6 prediction/error NPZ, 12 diagnostics CSV, 24 experiment JSON, one SHA256 manifest and 3 environment/provenance files**.

| Full archival rerun | GitHub Actions | Artifact |
| --- | --- | --- |
| Southwest 2022 | [41400003722](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41400003722) | `13122165252` |
| Southwest 2023 | [41402185866](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41402185866) | `13122890221` |
| Southern Plains 2022 | [41402189567](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41402189567) | `13121977420` |

After downloading any ZIP, run `python scripts/verify_full_archive.py archive.zip`. See [hashes, full run metadata, and retention deadlines](docs/EXPERIMENT_REPRODUCIBILITY_MANIFEST.md). **GitHub artifacts expire on 2027-01-31 UTC; independent permanent storage is not yet established.**

## Scientific and engineering boundaries

- ERA5 reanalysis evaluations are retrospective; this is **not** an operational weather forecasting system or evidence of forecast-time availability.
- The ≥305 K *observed*-hot RMSE metric conditions on the verifying temperature: it is **not event-detection accuracy**.
- The 2021 window was part of model development; multiple validations are **development-stage exploratory evidence**, not pre-registered independent prospective tests.
- The channel substitution retains historical climatology through the common blend; no causal attribution to historical features or general cross-seed superiority is claimed.
- CPU/GPU comparisons use a single earlier CPU timing; the 7.76× figure is a **descriptive elapsed-time ratio**, not a paired GPU speedup distribution.
- The archived inputs and weights belong to **new archival reruns**; the original historical experiment bytes were not proven identical. Dependencies and upstream ARCO snapshots were not pinned immutably.
- Draft manuscripts, submission figures, and private supplementary material are **not published here**.

**Further reading:** [Recruiter brief](docs/RECRUITER_EVIDENCE_BRIEF.md) · [Engineering status](docs/ENGINEERING_STATUS.md) · [Reproducibility manifest](docs/EXPERIMENT_REPRODUCIBILITY_MANIFEST.md) · [Portfolio](https://ys30.github.io/projects/western-us-weather-jax.html).
