# Validation-selected Blend: exact code contract and reporting limits

This is **public methods documentation**, not an unpublished manuscript.

Implementation: [scripts/climatology_baselines.py](../scripts/climatology_baselines.py), used by [scripts/climatology_aware_cnn.py](../scripts/climatology_aware_cnn.py). The [matched-channel workflow](../.github/workflows/era5-matched-climatology.yml) invokes the latter for both historical and source-mean arms.

## Exact implementation

For six-hourly valid index `i` and lead `L=4` time steps (24 hours), define pointwise persistence `P_i(x)=T_{i-L}(x)` and training-only climatology `C_{m(i),h(i)}(x)`, the mean of training observations in the same month and UTC hour at grid cell `x`. The baseline is

`B_i(x; w) = (1-w) P_i(x) + w C_{m(i),h(i)}(x)`.

The code searches the climatology weight `w ∈ {0, 0.25, 0.5, 0.75, 1}`, selecting the first grid value with minimum **all-grid-cell validation RMSE**, using `np.argmin`. In the referenced source training workflow the split boundaries are training before **2021-01-01**, validation **2021-01-01 to 2021-12-31**, and target evaluation from **2022-01-01**. Test values do not select `w`.

**Month–hour bins:** `month * 24 + UTC_hour` defines a 12×24 *possible index space*; the code builds a climatology only for observed training bins. For a complete 6-hourly 00/06/12/18 UTC archive, there are **12×4 = 48 occupied bins**, not 288 occupied bins. If a requested bin was not present in training, the code falls back to a grid-cell mean over all training timestamps. The actual occupied-bin count and selected weight must be read from each input's source data and archived run; do not infer an experiment-wide selected value from the source code.

## Protocol boundaries and avoidable claims

- A locally retrained heat-weighted CNN versus its validation-selected Blend comparator is **not** the same experiment as a frozen source-trained historical-versus-source-mean feature-channel contrast.
- Both matched-channel model arms may use the same baseline; the paired comparison identifies a feature-channel difference, not the value of climate information in general.
- `scripts/climatology_baselines.py` establishes the *implemented selection rule*. It does **not**, on its own, establish the actual winning weight or the entire grid/time population in every successful archived run.
- Target-region station scores are retrospective agreement, not prospective and independent forecast-skill estimates.

Source-code and run data, rather than this document, are the provenance of numerical performance claims.

## Verified source-trained weights from all three full archives (4 Nov 2026)

The original archived model source JSONs for **Southwest 2022 (Actions 41400003722), Southwest 2023 (41402185866), and Southern Plains 2022 (41402189567)** have now been independently read, and the complete input SHA256 manifest checked for each archive (53 digests × 3 archives = **159 passed**). The successful [independent replay run 41960278995](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41960278995) re-downloaded the archives and compared every published record.

[Exact 18-record audited source and target metrics](../results/archived_blend_weights_18_reports_verified.csv) show that for each archive, mode (`historical`, `source_mean`), and seed (42, 7, 123), **`selected_blend_climatology_weight = 0.25`**, with the same value propagated into the corresponding target prediction report as `source_validation_blend_weight`. Thus the archived source-selected reference has **75% persistence and 25% source-trained month/UTC-hour climatology weight**. The target evaluation applies that source-selected weight without tuning on held-out target test observations. The target protocol uses **pre-test target-history climatology**, however, so it is *not strict zero-target-history transfer*.

All three archive runs use the same source-domain 2019–2020 training and 2021 validation timeline; the repeated 0.25 values do **not** represent 18 independent validation datasets. Some source diagnostics and training timing differ across reruns, so identical checkpoints must not be asserted without checking their own hashes.

**Metric naming warning:** `best_validation_rmse_kelvin` in the source report is the **best CNN validation RMSE**, not the validation RMSE of the five Blend candidates. The complete candidate `validation_rmse_k` vector exists as a value returned by the baseline helper, but it is **not recorded in the source-training report JSONs found in these three archives**. Therefore this archive audit verifies the chosen weight but cannot provide the five candidate RMSEs or the margin over the next-best Blend weight. An actual occupied month-hour training-bin count is likewise not exported in those reports; 48 bins is the value expected for a complete 00/06/12/18 UTC training calendar, not an observed bin-count claim.

## Recovered and independently replayed 2021 Blend validation RMSE grid

The absent five-candidate validation-RMSE vector has now been **recovered from the original 56,182,643-byte source NetCDF** embedded in the full Southwest 2022 archival ZIP, rather than inferred from the CNN report. Original input SHA256: `ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6`. The same source SHA occurs in all three full experiment archives. The independent [CI run 41961414371](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41961414371) **passed** using [the recovery script](../scripts/recover_archived_blend_validation_grid.py). Numerical comparison tolerance is 1e-6 K to accommodate sub-microkelvin NumPy float32 rounding differences (initial stricter run 41961283651 failed solely at 1.27e-7 K for weight 1.0).

| Climatology weight | 2021 source validation RMSE (K) |
| ---: | ---: |
| 0.00 | 3.15880251 |
| **0.25** | **2.91271019** |
| 0.50 | 3.00108528 |
| 0.75 | 3.39793062 |
| 1.00 | 4.01274681 |

Selected `w=0.25` minimizes validation RMSE across these five candidates, beating the second-best grid value `w=0.50` by **0.08837509 K**. The actual source array shape is **5,844 times × 49 × 49**, with **2,924** six-hour samples for training (2019–2020), **1,460** for validation (2021), and **1,460** for 2022 test. There are **48 occupied training month×UTC-hour bins**, precisely 00/06/12/18 UTC for each of 12 months, and none of the 2021 validation timestamps requires the fallback climatology. These bins and the selected weight have been measured, not assumed.

[Public five-row source validation CSV](../results/source_blend_validation_weight_grid_verified.csv) · [Recovery workflow](../.github/workflows/recover-archived-blend-validation-grid.yml).

**Important separation:** this is the source-domain 2021 **Blend validation** RMSE, not the CNN best-validation RMSE, and not a target 2022/2023 benchmark. All three target experiments share the same byte-identical source input, so these five values should be reported **once** rather than counted as multiple independent selections. `w=0.25` was then applied to source-selected matched model comparisons, with pre-test target-history climatology where documented.
