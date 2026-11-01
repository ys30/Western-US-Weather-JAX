# Figure 1 and Figure 2 source-to-repository audit

The successful [paper-figures workflow #41281229086](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41281229086) produced artifact `paper-figures-artifact-derived` (artifact ID `13073575917`), containing `adapted_hot.svg`, `source_only_hot.svg`, `comparison.csv` and a provenance README. The archived SVG payloads are the same length (1517 and 1501 characters) as the corresponding committed repository SVGs and the plot contents/labels were checked against the archive.

The numeric content of the archived six-row CSV was checked against `figures/manuscript/comparison.csv`. The archive CSV uses CRLF and its repository copy uses LF; this explains the difference in total byte/character counts (604 vs 597), **not** a scientific value discrepancy.

| Protocol | Seed | CNN annual K | Reference annual K | CNN hot K | Reference hot K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapted | 7 | 3.798109 | 3.799870 | 2.330832 | 2.478656 |
| Adapted | 42 | 3.790533 | 3.799870 | 2.347359 | 2.478656 |
| Adapted | 123 | 3.794606 | 3.799870 | 2.322750 | 2.478656 |
| Source-only | 7 | 4.098400 | 4.162715 | 2.563245 | 2.400407 |
| Source-only | 42 | 4.112426 | 4.162715 | 2.550601 | 2.400407 |
| Source-only | 123 | 4.099266 | 4.162715 | 2.538898 | 2.400407 |

**Result:** no mismatch was observed in the six archived numeric rows or the rounded SVG labels. Repeated reference bars are intentional. Source-only reference means **24-hour persistence**, whereas the adapted reference means **source-selected persistence/climatology blend**; never merge these baselines. Plotted values are rounded to three decimal places, whereas the CSV contains the underlying precision.

**Quality note:** this is an archive-versus-repository comparison. The underlying experiments and their CI provenance are documented separately in `docs/SUBMISSION_READINESS.md`. Figures 1–2 use truncated titles/legends rather than standalone journal-format axis labeling; final layout remains pending.
