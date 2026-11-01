# Frozen source CNN: no-target-historical-climatology ablation

Evaluate original-region-trained CNN parameters, source-training mean/std, and source-validation blend coefficient on disjoint Southwest ERA5 2022, without retraining. Replace the target historical climatology feature by the **source global training mean** everywhere. This avoids target-region historical climatology/statistics but retains the necessary **four recent target temperature input frames** that are available at forecast issuance.

This is a **feature-ablation experiment**, not a full zero-target-observation or robust external-generalization demonstration. A constant climatology feature differs substantially from the source training feature distribution. Compare with frozen weights plus target pre-test historical climatology; no 2022 label selection is permitted. Verified results now available in CI run 41275083685; interpretation below.

## Verified numerical findings

## No-target-historical-climatology frozen-CNN ablation — verified

[Real ERA5 successful CI run 41275083685](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41275083685) compared **identical frozen original-region CNN checkpoint weights** in Southwest 2022 with (a) target **2019–2021 historical spatial/month/hour climatology**, versus (b) replacing the entire climatology input with a constant **source-training global mean**. The four recent target temperature frames are still available for each forecast. This is a **feature ablation**, not proof of general zero-shot generalization or failure thereof.

| 2022 Southwest model / K | Adapted annual RMSE | No-target-climatology annual RMSE | Adapted hot RMSE | No-target-climatology hot RMSE |
| --- | ---: | ---: | ---: | ---: |
| Scheme-specific source-weight blend | 3.799870 | 4.555326 | 2.478656 | 6.001585 |
| Frozen seed 42 | 3.790533 | 4.606510 | 2.347359 | 6.185947 |
| Frozen seed 7 | 3.798109 | 4.537596 | 2.330832 | 5.924195 |
| Frozen seed 123 | 3.794606 | 4.540868 | 2.322750 | 5.741222 |

The feature substitution strongly degrades both the **baseline and CNN**, not just the CNN. Without target historical climatology, CNN seed 42 is **worse than its same-scheme blend** on annual and hot subsets, while seeds 7 and 123 improve hot RMSE relative to that scheme's blend. Thus do **not** claim all seeds transfer without adaptation. This is a distribution-shift sensitivity result; it does not quantify the isolated causal contribution of target climatology because baseline features also change.

Uncertainty for the adapted case is documented in [frozen spatial transfer](FROZEN_SPATIAL_TRANSFER.md); do **not** transfer those intervals to the constant-feature ablation. Formal no-history uncertainty and source-only transferable feature alternatives remain future validation needs.

