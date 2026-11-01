# Reproducibility and submission readiness checklist

**Status:** working paper, not submission-ready. This ledger distinguishes verified CI results from pending data-availability and manuscript-quality work.

## Verified empirical provenance

| Evidence | CI run | Outcome |
| --- | --- | --- |
| Original-region 2022 held-out run | [41281229112](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41281229112) | Success; 21 JSON result files |
| Frozen source network, Southwest pre-test climatology | [41279255173](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41279255173) | Success |
| No-climatology source-trained frozen network | [41277714617](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41277714617) | Success |
| Manuscript plot generator | [41281229086](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41281229086) | Success; SVG and CSV in artifact |
| Forecast lead and temporal split unit tests | [41279255199](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41279255199) | Success |

## Results that must not be conflated

1. **Original-domain annual 2022 test:** 2/3 CNN seeds improve annual RMSE, all 3 improve observed >=305 K hot-subset RMSE against the original-region validation-selected blend.
2. **Southwest frozen-weight adaptation:** the source CNN and source normalization remain fixed, but the target pre-2022 historical temperature records are used to construct climatology feature maps; all three seeds show hot-subset improvements against the frozen protocol's source-selected blend, and bootstrap intervals exclude zero. Annual confidence intervals include zero.
3. **Southwest no-target-history source-only network:** the source model is trained without climatology features; all three seeds improve annual RMSE against 24-hour Persistence but worsen observed-hot RMSE, with paired bootstrap intervals entirely positive for hot RMSE differences.

Each protocol has a **different comparison baseline**. Results do not prove a causal contribution of climatology alone. Observed-hot is retrospective conditional RMSE, not deployable event detection.

## Reproducibility guardrails

- Code: https://github.com/ys30/Western-US-Weather-JAX
- The experimental run IDs above identify immutable historical code commits and archived CI results.
- ERA5 access scripts: `scripts/fetch_arco_era5.py`; raw ERA5 data are retrieved from an external provider, not included in this repo.
- Timeline audit: `weatherjax/time_audit.py` and `tests/test_time_audit.py`; 6-hour cadence, latest input t−4 for a 24-hour target at t, earlier frames t−7,t−6,t−5.
- Visualizations: `scripts/manuscript_figures.py` rebuilds from GitHub Actions JSON, not from typed-in manuscript values. Successful artifact: [paper-figures-artifact-derived](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41281229086).
- Before publication, record data retrieval timestamps, ERA5 metadata, geographic grid coordinates/spacing, dependency lockfile or environment manifest, file digests and external-data licence/acknowledgement requirements.

## Submission blockers

- [x] Main original-region 2022 CI run completed without cancellation.
- [x] Reproducible unit tests and source-only transfer workflow passed.
- [x] Manuscript result registry, main findings, confidence intervals and descriptive figure captions drafted.
- [x] Figure-generation workflow produces an artifact from archived results.
- [ ] Make permanent figure assets available in a stable release or committed manuscript folder. GitHub Actions artifacts expire.
- [ ] Review manuscript figure scales and labeling for journal requirements; include CI uncertainty graphically.
- [ ] Verify ERA5 spatial grid and time provenance from downloaded files (metadata and checksums).
- [ ] Audit parameter-development chronology, multiplicity and test reuse; strengthen independent external validation.
- [ ] Add correctly verified scientific references, ERA5 attribution and chosen journal formatting.
- [ ] Validate final manuscript claims and supplementary tables against pinned JSON archives.

## Recommended paper framing

**Supported:** In this specific 2022 Southwest test, frozen weights with pre-test target historical climatology reduced conditional extreme-heat RMSE against its source-selected blended reference, while climatology-free frozen source models improved domain-wide RMSE but worsened conditional hot RMSE against Persistence.

**Not supported:** A universally transportable model, causal identification of climatology's contribution, significance for future years or other geographies, or operational heat-warning skill.
