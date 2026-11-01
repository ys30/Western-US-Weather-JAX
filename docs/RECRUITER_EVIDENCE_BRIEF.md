# Recruiter evidence brief — ERA5 JAX GPU weather forecasting

**Project:** [Western-US-Weather-JAX](https://github.com/ys30/Western-US-Weather-JAX) · [Portfolio case study](https://ys30.github.io/projects/western-us-weather-jax.html)

## What I built
Developed a reproducible 24-hour-lead regional temperature forecasting workflow on six-hourly real ERA5 fields. Implemented JAX convolutional residual networks, training-only season/UTC-hour climatology, validation-selected persistence/climatology blending, fixed-305-K heat-emphasized loss, chronological holdouts, multi-seed tests, multiday truth-defined heat-event diagnostics and automated GitHub Actions. Verified execution on a real NVIDIA Tesla T4.

## Verified numerical evidence
- **GPU engineering:** 258.51 s CPU vs 33.29 s T4 GPU in the initial Colab matched-settings run (**7.76× descriptive elapsed-time ratio**). Three further audited GPU repetitions: 35.63, 32.72, 32.01 s (**33.45 ± 1.92 s**, sample standard deviation). Same CPU/GPU 2022 test RMSE **3.191155 K** and identical input SHA-256. Single prior CPU reference, so **not** a repeated paired speedup distribution.
- **Original region, 2022:** validation-selected blend annual/hot RMSE **2.99462/2.15930 K**. Heat-weighted residual CNN seeds 42/7/123 annual RMSE **2.98450/2.99741/2.99243 K**, fixed-305-K hot RMSE **2.05492/2.04506/2.03982 K**. All three improve hot RMSE, but only two improve annual RMSE.
- **Second, non-overlapping Southwest region, 2022:** locally retrained blend annual/hot **3.80252/2.40826 K**; CNN annual **3.77695/3.79943/3.78322 K**, hot **2.31885/2.31365/2.32763 K** (seeds 42/7/123). All three improve both metrics **after local retraining**.
- **Retrospective heat-event diagnostics:** truth-defined episodes with >=5% regional cells at >=305 K at one or more 6-hour valid times each day for >=2 consecutive days. Heat-weighted CNN improves hot-subset RMSE on original-region **2021: 7/7** and **2022: 8/8** events for each seed; second-region 2022 **8/10, 10/10, 10/10**. These are retrospective conditioned errors, **not** high-temperature event detection or operational forecast accuracy.
- **Uncertainty:** in original-region 2022, truth-hot time-block bootstrap (1, 3, 7, 14 days; 1000 draws each) yielded intervals below zero for each heat-weighted CNN seed. Results remain limited to the datasets, test window and event definition.

## Responsible interpretation
A compelling **JAX/HPC engineering and environmental machine-learning portfolio case**, with empirical evidence for heat-subset improvement over a stronger statistical baseline. It does **not** demonstrate uniform annual improvement, frozen-weight cross-region generalization, prospective forecast or warning-system validity, or a publication-ready independent external evaluation. The 2021 test was previously involved in model development.

## Evidence links
- [Full engineering and scientific verification](ENGINEERING_STATUS.md)
- [Tesla T4 GPU acceptance protocol](../docs/GPU_ACCEPTANCE.md)
- [2022 original region high-temperature CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41265640777)
- [2021 retrospective temporal sensitivity](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41267651677)
- [Southwest regional transfer-after-retraining](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41270894771)
