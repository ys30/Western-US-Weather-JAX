# Southwest 2023 matched-channel temporal replication

## Scope and comparison

This workflow repeats the 2022 matched-training architecture in a different **target test year**, Southwest 2023. Source 2019–2020 is used for fitting, source 2021 for model and blend selection, and the Southwest 2023 fields for verification. Target pre-test historical climatology uses Southwest 2019–2022, so this is **not** a strictly no-history transfer evaluation.

Historical and source-mean channel arms use the same network dimensionality, random seeds (42/7/123), optimization schedule and a common target-history persistence/climatology blend as their residual baseline. The source-mean arm replaces only its dedicated climatology input channel; it still receives target-history information indirectly via the residual baseline channel.

The 2023 experiment is **not an exact fixed-information-set replication** of 2022: target historical climatology is estimated from one extra year of pre-test data (2022). Consequently changes between 2022 and 2023 combine meteorological year shifts and changes in the historical climatology estimation window. Do not interpret them as an isolated temporal treatment effect.

## Evaluation and safeguards

- Test times must start 2023-01-01; training and validation end before that date.
- Conditional-hot RMSE is defined from **verifying** ERA5 temperature >=305 K. This is not forecast event detection or heat-warning skill.
- Export matched gridded error archives and compare source-mean minus historical CNN by resampling common calendar-day blocks (3, 7, 14 and 28 days; 2,000 draws per interval).
- Report both whole-year and hot-subset results with sample counts, uncertainty and per-seed signs.
- Different seeds use the same meteorological data; resampling intervals are correlated, descriptive, and not corrected for multiplicity.
- Historical climatology and ERA5 inputs do not establish operational issue-time availability.
- Do not report new numerical results until CI produces successfully validated actual ERA5 artifacts.

Workflow: [era5-matched-crossyear-2023.yml](../.github/workflows/era5-matched-crossyear-2023.yml).

Public code and aggregate engineering results are separate from the private manuscript, supplement and submission figures.
