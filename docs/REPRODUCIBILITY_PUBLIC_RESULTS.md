# Reproducing the published sensitivity tables (public engineering note)

This document covers reproducible **analysis artifacts**, not the private manuscript or submission figures.

## Verified CI and inputs

- Source ERA5 acquisition and training are implemented in the GitHub Actions workflows under `.github/workflows/`.
- The archived 2022 Southwest frozen-transfer evaluations distinguish **source-trained fixed weights with target pre-test historical climatology** from a separately trained no-climatology source-only protocol.
- The evaluated quantities are **retrospective ERA5 hindcast errors**, not operational forecast skill or heat-warning performance.
- Three initialization seeds: `42`, `7`, `123`. The same held-out observations underlie all three, so seed results are not independent replicates.
- For the climatology-adapted protocol, the comparator is a source-validation-selected persistence/climatology blend. Do not compare its errors directly as a causal ablation with source-only results, which use a different architecture, input set and persistence comparator.

## Public aggregate result

`results/bootstrap_block_length_sensitivity_v2.4.csv` contains 36 records:

- Three observed-temperature thresholds (303, 305, 307 K)
- Three seeds (42, 7, 123)
- Four calendar-time circular bootstrap blocks (3, 7, 14, 28 days)
- 2,000 draws per combination
- Paired CNN-minus-reference hot-subset RMSE differences (K), percentile 95% bounds and sample counts

To audit the committed aggregate:

```bash
python scripts/audit_published_bootstrap.py results/bootstrap_block_length_sensitivity_v2.4.csv
pytest -q tests/test_audit_published_bootstrap.py
```

Expected audit: 36 complete comparisons and 36 intervals with negative upper bounds. This describes the committed table; it does not establish familywise confidence or independent test-year generalization.

To recompute an analysis from **archived paired predictions**, use `scripts/bootstrap_block_sensitivity.py` for each seed's `seed-N-fields.npz` produced by the frozen-transfer workflow. The raw paired arrays can be larger than aggregate CSVs and should not be synthesized from overall RMSE summaries.

## Interpretation and caveats

- Negative CNN-minus-reference RMSE: CNN performed better on **observations actually at or above the threshold**.
- Thresholds are nested, post-hoc diagnostics, not independently held-out cohorts.
- Four bootstrap block lengths probe within-sample temporal dependence; no value was preregistered.
- Percentile intervals from different seeds and thresholds are correlated, not simultaneous.
- Do not label six-hourly regional 305 K occupancy runs as independent multi-day heatwave events.
- Publication manuscript text and submission-specific figure layouts are distributed separately, not stored in this public repository.
