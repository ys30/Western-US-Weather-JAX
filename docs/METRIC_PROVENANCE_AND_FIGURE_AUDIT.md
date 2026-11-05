# Metric provenance and manuscript figure audit (verified source contracts)

This is public analysis documentation, not the manuscript itself. The purpose is to prevent inappropriate cross-protocol comparisons of RMSE numbers.

## Three distinct reported error populations

| Metric family | Evaluation population and target | Metric label | Representative published numbers | Source | Correct interpretation |
| --- | --- | --- | --- | --- | --- |
| **A. Source-domain Blend selection** | All ERA5 temperature grid cells at valid times in the **2021 source validation** window, using 2019–2020 source training statistics | `source_validation_blend_rmse_K` | Five weights 0/.25/.5/.75/1 → **3.15880 / 2.91271 / 3.00109 / 3.39793 / 4.01275 K** | [Five-row recovered grid](../results/source_blend_validation_weight_grid_verified.csv), [passed CI 41961414371](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41961414371) | An internal baseline selection, **not** a target-region test result or external airport validation |
| **B1. Frozen-source CNN target ERA5** | All ERA5 grid cells × valid times in a held-out **target region-year**, with frozen source-trained CNN weights and target pre-test climatology where used | `target_CNN_RMSE_K`, `target_source_weight_blend_RMSE_K` | Source-selected 0.25 weight; target region-year scores vary by target, mode and seed | [18 archived records](../results/archived_blend_weights_18_reports_verified.csv), [passed CI 41960278995](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41960278995) | Retrospective target-grid performance; not local CNN retraining and not station validation |
| **B2. Regional retraining experiment** | ERA5 2022 held-out regional test, with **regionally retrained, heat-weighted CNN** and its regionally validation-selected persistence–climatology comparator | Annual and observed-hot region RMSE | Original region annual CNN seeds 2.98450 / 2.99741 / 2.99243 K; comparator 2.99462 K; heat subset CNN 2.05492 / 2.04506 / 2.03982 K vs comparator 2.15930 K | [Recruiter evidence brief](RECRUITER_EVIDENCE_BRIEF.md), original successful CI runs linked therein | Different CNN training and comparator-selection protocol from B1 |
| **C. External matched-station agreement** | Temperature at **IEM ASOS airport stations** paired within ±30 min to retrospective ERA5-grid model predictions; each year ~1453–1459 6-hourly records per site | `historical_rmse_K`, `source_mean_rmse_K`, `baseline_rmse_K`, `delta_rmse_K` | ABQ/ROW 2022, ABQ/ROW 2023, ABI/OKC 2022; absolute RMSE around 3–5 K | [Southwest 2022 station rows](../results/southwest_2022_asos_run41862347926_verified.csv), [other four site-year rows](../results/additional_asos_southwest2023_plains2022_verified.csv) | Site-level retrospective match, not all-grid target ERA5 RMSE or fully independent operational forecast skill |
| **D. ASOS mean-error decomposition** | Same external matched station-year × seed data, separated into prediction arms | `mean_error_K`, `centered_rmse_K`, `mae_K`, `squared_bias_fraction` | ABQ historical-channel annual mean forecast-minus-ASOS error medians −1.03 K (2022) and −1.19 K (2023) | [54-row error audit](../results/asos_station_error_decomposition_54.csv), [passed CI 41907423671](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41907423671) | Descriptive squared-bias decomposition, not causal evidence that elevation explains the error |

### Important experimental and statistical distinctions

1. **Do not compare source validation 2.91271 K with target annual RMSE as though they are competing test models.** The former selected `w=0.25` once using 2021 source observations; the latter is a distinct target data population.
2. **Frozen source-trained versus regionally retrained is not an ablation of just climatology.** Separately label each training procedure and its reference.
3. **Do not place station RMSE in the same unqualified bar chart as gridded ERA5 RMSE.** Station errors incorporate spatial representativeness, exposure, matching, and potentially ERA5 observation-assimilation dependence.
4. **A paired channel contrast** (`source_mean − historical`) is a difference between two model RMSEs computed on the same observation records. Positive favors historical. It is **not** equivalent to the signed pointwise forecast-minus-observation mean error, nor is it a climate mean bias.
5. Within-seed block bootstrap intervals quantify temporal sampling variability **conditional on model initialization**. They are not automatically independent across seeds, stations, thresholds or lapse-rate scenarios, and not multiplicity corrected.
6. **ASOS comparisons are retrospective agreements**, not evidence of prospective operational 24-h forecasts. Observational independence is not established when ERA5 may assimilate related station observations.
7. The 72 + 144 = 216 altitude/block-duration combinations are **sensitivity scenarios, not 216 independent experiments**.
8. The published 10-seed/four-case summary uses a **different sample and statistic** than the three-seed ASOS station-year results; do not use the former's medians or seed votes to describe the latter.

## Minimum figure/caption requirements

- Each figure should identify: `source/target`, `year`, `ERA5 grid versus station`, `CNN retrained versus frozen`, `evaluation subset (annual/305K hot)`, `RMSE units K`, and the comparator (regional Blend, source-selected target Blend, or paired channel).
- A figure showing errors on ASOS matched records should report station coverage and, where inference is shown, block length/number of bootstrap draws and the absence of multiplicity adjustment.
- For hot-subset plots, a target defined by verifying ERA5 or ASOS truth must be identified as **observed-hot** rather than future-event prediction skill.
- Source/validation/test periods should be explicit in the methods table and legends, not inferred from repository paths.

## Acceptance evidence and remaining gates

The [five-source-weight validation grid](../results/source_blend_validation_weight_grid_verified.csv) and [18 archival records](../results/archived_blend_weights_18_reports_verified.csv) are independently rechecked in successful Actions runs 41961414371 and 41960278995. The [144 ASOS elevation scenarios](../results/additional_asos_elevation_144_scenarios.csv) passed independent replay run 41878154087; the [54-row bias decomposition](../results/asos_station_error_decomposition_54.csv) passed run 41907423671. **Do not call the supplementary manuscript submission-ready:** the original complete main manuscript, a figure-to-text cross-reference audit and bibliography validation still require direct review. Manuscript drafts remain private.
