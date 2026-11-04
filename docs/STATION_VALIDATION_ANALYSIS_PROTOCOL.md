# Station-validation and elevation QA: analysis protocol

Status: **protocol and code review; no station-skill result is inferred from this document**. Updated 2026-11-03. Public engineering/analysis record only; manuscript and submission figures are private.

## Existing implementations reviewed

- `scripts/fetch_asos_station_validation.py` downloads 2022–2023 ASOS/METAR temperatures from IEM for ABQ, ROW, ABI and OKC, deduplicates timestamps by median, and records source and CSV hashes.
- `scripts/verify_station_grid_elevations.py` retrieves ERA5 0.25-degree surface geopotential and IEM station metadata to compare station elevation with nearest-grid and bilinearly interpolated ERA5 elevation. The current implementation does **not** correct temperature.
- Associated Actions jobs have been committed, but commit creation or a submitted workflow is **not evidence of successful data retrieval or completed scientific validation**.

## Required scientific gates before reporting external skill

1. **Data success and provenance.** Verify successful Actions completion and inspect CSV row counts, missingness, station-year coverage, duplicates, source metadata and artifact digests. Retain observation units and timestamps. Avoid treating an IEM CSV as independently quality-controlled.
2. **Forecast-target alignment.** The CNN predicts the verifying ERA5 temperature at lead 24 h. Match station observations to the **target valid time** in UTC (not the input time) with a prespecified tolerance, e.g. nearest report within 30 minutes. Report match rate; run 15/30/60-minute sensitivity. Do not silently match a six-hourly 00/06/12/18 analysis timestamp to an unrelated hourly report.
3. **Space matching.** Sample each forecast and blended-baseline field at the station coordinate using a documented bilinear interpolator, with nearest-grid sensitivity and checks for longitude convention and latitude ordering. Use precisely the same interpolation for all comparison arms.
4. **Elevation audit.** Convert surface geopotential to height with 9.80665 m s^-2; check variable units and map resolution. Publish station-minus-grid elevation separately for nearest and bilinear sampling. Compare raw-temperature results first; elevation/lapse-rate adjustment is an explicitly labeled *sensitivity*, not a replacement for the original model outputs.
5. **Observation comparability.** METAR/ASOS observations are point measurements; ERA5 2-m temperature is a gridded analysis. Account for airport siting, sensor elevation, reports' QC, and representativeness error. Observations may have been assimilated into ERA5; this is **not an independent forecast benchmark**.
6. **Metrics and uncertainty.** Report matched station-year sample counts; bias, MAE, RMSE and CNN-minus-blend paired differences on identical station/times, including annual and >=305 K *observed-station* subsets. State subset counts. Cluster/moving-block resample by **day within station**, retaining the within-day dependence; report sensitivity to block length. Do not pool station reports as independent samples for nominal confidence intervals.
7. **Interpretation.** ERA5-input, ERA5-target retrospective model outputs are hindcast-style reconstructions, not evidence of deployable 24-h operational forecast skill. Distinguish agreement with stations from superiority against a true available-at-issue-time NWP forecast.
8. **Cross-seed robustness.** Preserve all individual seeds, station-years and experimental arms before reporting pooled effects. Report per-station-year distributions, not only aggregate wins. Keep uncorrected exploratory intervals and multiplicity caveats explicit.

## Code/CI audit issues to resolve

- The elevation script parses station height from the fifth HTML table cell. Confirm that column is elevation in **meters**, not a moved/renamed table field; a plausible 0–4000 numerical range check alone cannot establish units.
- The geopotential script assumes `geopotential_at_surface` exists in the cited ARCO data product and samples a 2022 timestamp. Verify actual variable availability, coordinate ordering, units and the source-data fetch in a completed run.
- `scipy` was added to the elevation workflow after initial creation; verify the latest workflow succeeds rather than assuming dependency repair suffices.
- The downloader's plausible temperature filter (-40 to 130 deg F) can drop genuine extreme cold; count and disclose all excluded observations. Timestamp matching and station QA remain separate tasks.
- GitHub Actions artifacts have finite retention. Mirror the analysis CSVs, summaries, plots and SHA256 manifests into durable, appropriate project storage; keep private manuscripts, supplementary DOCX and submission-only figure files **out** of the public repository.

## Public versus private policy

**Public GitHub:** analysis scripts, transparent processing protocols, CI, non-restricted input provenance, aggregate reproducible result CSV/JSON, statistical tables, and public-adaptation plots.

**Local/private only:** manuscript DOCX, journal submission supplements, cover letters, referee correspondence and submission-ready plates.

Do not overwrite earlier study protocols or conflate the regionally retrained Protocol A with frozen source-trained matched-channel Protocol B.
