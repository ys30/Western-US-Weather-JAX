# Source-verified architecture and training contract (matched-channel model)

This is public **implementation documentation**, not manuscript text. Provenance: [model definition](../weatherjax/model.py), [data assembly](../weatherjax/data.py), [climatology-aware training](../scripts/climatology_aware_cnn.py), and [matched-channel workflow](../.github/workflows/era5-matched-climatology.yml).

| Implementation feature | Exact source-code contract |
| --- | --- |
| Input history | Four six-hourly 2-m temperature fields, four lead steps before prediction (24 h), plus two known UTC cyclical time features |
| Matched-channel features | One replaceable climatology feature: training-derived grid-cell month×UTC-hour mean (`historical`) or spatial/temporal source training mean (`source_mean`); both use a common persistence–climatology Blend as the **last** input feature |
| Total channels | 4 temperature + 2 UTC time + 1 candidate feature + 1 shared Blend = **8** input features |
| Hidden network | 3×3 conv (8→12) + ReLU, 3×3 conv (12→12) + ReLU, and 1×1 conv (12→1) |
| Residual output | Final prediction is the **last input feature plus 1×1 conv output**. In normalized input units, the last feature is the Blend. Output 1×1 residual weights initialized to **zero** |
| Initialization | JAX random key by seed, scaling weights by sqrt(2 / fan-in), biases zero; final residual weights zero |
| Training loss | Weighted normalized MSE, weight = `1 + hot_weight × 1[y ≥ 305 K]`. Matching source workflow uses `hot_weight=2` |
| Source training command | `--epochs 40 --hot-weight 2 --seed 42/7/123 --climatology-feature-mode historical/source_mean` |
| Default numerical settings | batch size **8**, fixed gradient step `lr=0.001`, hidden channels **12**. **No Adam/AdamW** is implemented in this matched trainer: parameters updated as `p - lr * grad` |
| Early selection | All epochs run; the best full-validation RMSE epoch is selected, so describe this as **best-epoch checkpoint selection**, not early stopping |
| Normalization | Source training mean and standard deviation computed using the source training segment only; recovered denormalized temperature in K |
| Validation | Source 2019–2020 training, source 2021 validation, frozen target region-years 2022/2023 with no target CNN weight updates |
| Region | Source 49×49 ERA5 grid; 24-h prediction lead. Not a full global weather forecasting model |
| Limits | Observed-hot threshold is defined using verifying truth and cannot be presented as advance heat-event classification; ASOS station matching is retrospective and may not be assimilation-independent |

## Naming warning

This model is a **three-convolution residual correction CNN**, not the multilayer residual block architecture normally meant by “ResNet.” The `source_mean` condition still retains climate information through the common Blend; the contrast measures substitution of the candidate climatology feature under a shared reference. The wider repository also contains **separate local retraining experiments**; their optimization and evaluation contracts must not be inferred from this frozen-source workflow.

## Publication interpretation

The paper's strongest reproducible result is protocol-sensitive variability in retrospective ERA5 and ASOS agreement, not architectural superiority to large-scale Pangu-Weather, GraphCast or other global weather foundation models. Model comparison must retain its true data and verification population.

