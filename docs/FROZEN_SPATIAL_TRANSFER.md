# Frozen-weights cross-region evaluation protocol

**Status:** Source checkpoint export and frozen-target evaluation **verified successfully** in [CI run 41273562903](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41273562903). The current implementation uses target **pre-test historical climatology adaptation**, not strict no-target-history transfer.

The original and Southwest ERA5 regions have different grid shapes, and the residual CNN's two extra channels are **local month/UTC-hour climatology** and a **locally weighted persistence/climatology blend**. A valid frozen network experiment must distinguish:

1. **Network weights**: train on original area 2019–2020 only, choose epoch on 2021 validation, export the exact JAX parameter arrays; freeze without any target optimization.
2. **Normalization**: use original-area training mean and standard deviation in target inference; never compute them using 2022 test truth. The checkpoint now records these values.
3. **Baseline weights**: use source validation's climatology blend coefficient, not target 2022 selection; checkpoint records the source coefficient.
4. **Spatially varying climatology**: cannot directly paste the source grid into the disjoint target region. Two distinct experiments are needed: (a) adapt local climatology from **target historical pre-2022 training dates**, with frozen CNN parameters, explicitly labeled **target-historical-feature adaptation**; (b) a stricter no-target-history experiment using a prespecified climatology-free compatible feature construction and separately trained source CNN.
5. **Truth masking**: select >=305 K hot test cells only for retrospective evaluation, not inference.
6. **Comparison**: report target local-retraining CNN versus region-specific blend separately; frozen network vs source-weighted blend must be computed with same inputs. Keep region split and timestamp/lead consistency.

Source-region checkpoint export is available via:
`python scripts/climatology_aware_cnn.py --data input/era5_west_2019_2022.nc --train-until 2021-01-01 --validate-until 2022-01-01 --seed 42 --hot-weight 2 --checkpoint-path outputs/source/model-42.npz --output outputs/source/result-42.json`

Note: this command uses the standard 2022 test to produce an output report, but **selection remains training/validation only**. A checkpoint file stores selected weights plus metadata, not the target region dataset.

**Do not label region-specific retraining as frozen spatial transfer.**

## Verified numerical findings

## Verified frozen-weight source-to-Southwest transfer (2022)

[Real ERA5 CI run 41273562903](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41273562903) **succeeded**: train original-region CNN on 2019–2020, select checkpoints using 2021 validation, freeze weights and source mean/std plus source validation-selected blending weight, then infer in geographically non-overlapping Southwest in 2022. **No CNN weight update, target checkpoint selection, or target test-label tuning.** Target 2019–2021 historical ERA5 is used **only to construct spatial climatology features** and the corresponding persistence/climatology blend. Not strict no-target-history zero-shot transfer.

| Southwest 2022 RMSE (K) | Source-weight blend | Frozen CNN seed 42 | Frozen CNN seed 7 | Frozen CNN seed 123 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 3.799870 | 3.790533 | 3.798109 | 3.794606 |
| Observed t2m >=305 K | 2.478656 | 2.347359 | 2.330832 | 2.322750 |

Heat RMSE decreases versus the **source-weight blend comparator** by about **5.30%, 5.96%, 6.29%**. Overall improvements are smaller. All hot-subset results condition retrospectively on **observed** test temperatures, and thus do not measure event detection. This experiment does not establish formal statistical significance or transfer without target historical climatology.

This source-weight comparator is **different** from the Southwest **locally validation-selected blend** (overall 3.802520 K, hot 2.408255 K). Do not conflate these baselines or make unpaired superiority statements across protocols. The locally retrained Southwest CNNs are a separate experiment.


## Frozen CNN transfer: paired time-block uncertainty (verified)

[Real ERA5 frozen Southwest bootstrap CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41274429714) completed successfully, along with unit tests. Using 1000 paired circular time-block resamples at **1, 3, 7 and 14 days**, all fixed-305-K observed-hot RMSE CNN-minus-source-blend 95% percentile intervals were **entirely negative**, for each original-area-trained frozen network seed (42, 7, 123). The corresponding **annual RMSE** intervals all **included zero** at every time-block length.

| Frozen model | 14-day heat RMSE difference CI, K | 14-day annual RMSE difference CI, K |
| --- | --- | --- |
| Seed 42 | [-0.170609, -0.095481] | [-0.024538, 0.006258] |
| Seed 7 | [-0.180439, -0.114352] | [-0.008488, 0.005789] |
| Seed 123 | [-0.191826, -0.119412] | [-0.019666, 0.009477] |

Negative values favor CNN over the **source-weighted persistence+target-history climatology blend**. Confidence intervals are within-2022, within-one-Southwest-region and conditioned on retrospectively observed true-hot grid cells. They do **not** establish operational heat event detection, external-year robustness, independent spatial replication, or strict zero-target-history transfer.

