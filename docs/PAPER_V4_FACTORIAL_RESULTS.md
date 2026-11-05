# Paper v4.00 — factorial main-result evidence (2026-11-05)

## Provenance and reproducibility

- Source three-seed factorial CI: [41714440020](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41714440020)
- Source seven-seed factorial CI: [41721732092](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41721732092)
- Verified 160 original target JSON reports, reconstructed in [42123832170](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/42123832170).
- Script: `scripts/paper_v4_factorial_tables.py`; test: `tests/test_paper_v4_factorial_tables.py`.
- Seed list: 7, 11, 19, 29, 37, 42, 53, 71, 97, 123. Cases: Southwest and Southern Plains, 2022–2023.
- Detailed generated CSV: `results/paper-v4/factorial_10seed_4case_absolute_rmse.csv` (committed by replay workflow when available).

## Definitions

Let P mean persistence reference, B the validation-selected persistence–climatology blended reference,
S the source-mean replacement channel, and H the historical target-pretest climatology channel.

`Delta_P = RMSE(P,S) - RMSE(P,H)`
`Delta_B = RMSE(B,S) - RMSE(B,H)`
`I = Delta_B - Delta_P`

Positive Delta denotes lower RMSE for H; negative I means less incremental benefit with blend.

## Recovered ten-seed contrasts

| Case | Subset | Positive Delta_P | Positive Delta_B | Negative I | Median Delta_P (K) | Median Delta_B (K) | Median I (K) | Mean I (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Southwest 2022 | All | 9/10 | 7/10 | 10/10 | 0.081120 | 0.000845 | -0.079522 | -0.067794 |
| Southwest 2022 | Truth-hot >=305 K | 9/10 | 6/10 | 9/10 | 0.054385 | 0.001727 | -0.045036 | -0.046649 |
| Southwest 2023 | All | 9/10 | 9/10 | 9/10 | 0.063013 | 0.001905 | -0.060822 | -0.052861 |
| Southwest 2023 | Truth-hot >=305 K | 7/10 | 6/10 | 6/10 | 0.027355 | 0.006292 | -0.015455 | -0.023619 |
| Southern Plains 2022 | All | 10/10 | 6/10 | 10/10 | 0.110189 | 0.002122 | -0.110068 | -0.094101 |
| Southern Plains 2022 | Truth-hot >=305 K | 8/10 | 6/10 | 6/10 | 0.010309 | 0.006330 | -0.012275 | -0.012300 |
| Southern Plains 2023 | All | 10/10 | 8/10 | 10/10 | 0.074832 | 0.001173 | -0.075199 | -0.065735 |
| Southern Plains 2023 | Truth-hot >=305 K | 9/10 | 7/10 | 9/10 | 0.067351 | 0.009645 | -0.050017 | -0.047833 |

## Proposed Results 3.1 wording (draft)

Across four disjoint region–year evaluations with ten paired initializations per case,
the dedicated historical climatology channel showed consistently larger marginal benefit
when the residual reference was 24-hour persistence rather than a
persistence–climatology blend. For full-year verification, the median
reference-by-channel interaction ranged from −0.061 to −0.110 K among the four
cases; it was negative for 39 of the 40 case–seed comparisons. Restricting verification
to observed ERA5 gridcell temperatures of at least 305 K reduced the consistency
of the effect: median interactions ranged from −0.012 to −0.050 K, with
30 of 40 case–seed comparisons negative. The matched historical-channel benefit
under the blended reference was small (median 0.0008–0.0021 K for all times,
0.0017–0.0096 K for observed-hot cases). Thus the observed pattern supports
overlapping climatic information in the reference and the additional input
channel, though this factorial contrast by itself does not establish a unique
causal mechanism.

## Important interpretation limits

- Per-seed signs and medians are initialization summaries; four target cases are not four independent climate populations.
- Uncertainty intervals and any inferential p-values must be recomputed from paired temporal fields with seed dependence preserved.
- Truth-hot means verifying ERA5 >=305 K; the subset is outcome-conditioned, not an operational heat warning metric.
- A source-mean feature replacement holding a blend constant is NOT the same protocol as no target-region climatology at all.
- Source checkpoint selection used source validation overall RMSE, even when heat-weighted training was enabled.
- ASOS station results, spatial fields, and MOS benchmarking are not yet re-verified in this paper v4 replay.
