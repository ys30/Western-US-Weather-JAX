# Frozen climatology-input sensitivity (engineering protocol)

This public note describes the **inference-time feature perturbation** in `scripts/frozen_spatial_eval.py`. It is distinct from a controlled matched-training ablation and is **not** a causal estimate of historical climatology.

## Design

The two evaluations load the **same source-trained checkpoint**, use **the same Southwest 2022 six-hourly ERA5 test fields**, source training normalization, test times and source-validation-selected persistence/climatology reference:

- `--climatology-mode target_history`: supply the target 2019–2021 month/hour climatology to both the CNN climatology channel and the fixed blended reference.
- `--climatology-mode source_global_feature_only`: replace **only the CNN climatology feature channel** with the scalar source-training mean; retain the original target-history blended reference, including the CNN's baseline input channel.

The feature-only mode uses the target pre-test historical climatology in the reference and its baseline feature. It therefore **does not remove all target historical information** and **does not test zero-history transfer**. The CNN encounters an input-channel distribution not seen in training. A loss of skill cannot establish that climate history is causally necessary for a suitably retrained alternative.

The older `source_global_constant` mode changes both the climatology feature and the blended reference and **must not be compared as a matched isolated channel perturbation**.

## Execution and acceptance

Run the [frozen spatial workflow](../.github/workflows/era5-frozen-spatial.yml). It exports `outputs/frozen-southwest/seed-N.json` and `outputs/frozen-feature-only/seed-N.json` as `frozen-southwest-results` artifacts for seeds 42, 7 and 123.

A valid comparison requires:

1. The workflow and source-trained checkpoints complete successfully.
2. Each seed's `source_weight_blend_rmse_kelvin` and `fixed_305k_blend_rmse_kelvin` are unchanged between modes (allowing only numerical serialization tolerance).
3. `target_test_start`, `target_history_last_timestamp`, and `fixed_hot_samples` agree.
4. Report per-seed CNN minus **the same** reference for whole-year and observed-truth-hot RMSE, with paired temporal-block intervals. The hot subset is defined by verifying ERA5 temperature >=305 K and does **not** measure warning detection.
5. No private manuscript text, publication figure or preliminary paper conclusion is committed to this public repository.

**Status:** Code and regression test are committed; numerical findings require successful real-ERA5 workflow artifacts and validation. Do not present this as a measured improvement or decline until those checks pass.

## Next scientific experiment

For a true matched training experiment, freeze the architecture, feature dimensionality, source 2019–2020 training split, 2021 model-selection rule, target 2022 evaluation, seeds, normalization and common reference. Train separately with a real climatology feature versus an explicitly defined replacement feature, rather than perturbing only at inference. Report equal-comparator differences, resource costs and sensitivity to target-history availability. Avoid interpreting two different trained models as an isolated causal effect without further assumptions.
