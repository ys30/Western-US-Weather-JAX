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

## Implemented staged validation utilities (2026-11-03)

1. `scripts/sample_forecast_grid_at_stations.py`: accepts a **target-valid-time NetCDF** with `historical_K`, `source_mean_K`, and `baseline_K` on a shared `(time, latitude, longitude)` grid. Performs consistent bilinear/nearest spatial sampling at fixed ASOS coordinates, checks longitude convention and out-of-domain requests, and records SHA256 audit metadata.
2. `scripts/match_asos_prediction_times.py`: matches resulting station predictions to ASOS reports by nearest UTC timestamp within a prespecified tolerance, preserving the observation time and offset.
3. `scripts/evaluate_paired_station_predictions.py`: computes paired site-year annual/hot RMSE, MAE, bias and day-block intervals. Hot-subset bootstrap samples **full calendar days first**, *then* applies the observed ≥305 K condition. Empty bootstrap draws are tracked.
4. `.github/workflows/station-paired-evaluation-qa.yml`: runs synthetic unit/smoke checks for all three stages. **Workflow definition and push commits do not demonstrate CI success**; record completed run ID before claiming acceptance.

**Unresolved interface:** Existing experiment archives describe gridded NPZ prediction/error bundles, whereas the spatial sampling module consumes a three-arm NetCDF. Inspect the actual archive array keys, grid metadata, target-valid-time offsets and arms before implementing the NPZ-to-NetCDF adapter. Do not infer a time coordinate from file names or assume a source-trained model's target slice has the same timestamp as the corresponding input. Reuse identical station/time keys for all three prediction arms.

**No new scientific result is asserted by these commits.** Evaluation and optional station-elevation lapse-rate sensitivity require successful real-data staging, completed provenance/QA checks, and sufficient station report matches. Keep all manuscript versions and submission plates private.

## Archived NPZ interface verified from source and implemented

The producer `scripts/spatiotemporal_diagnostics.py` stores `time`, `truth_kelvin`, `cnn_error_kelvin`, `reference_error_kelvin`, `latitudes`, and `longitudes`. All error arrays are forecast minus ERA5 truth. Thus historical-channel forecast = truth + historical CNN error; source-mean forecast = truth + source-mean CNN error; shared reference = truth + reference error. The archival diagnostic producer stores target *verifying* timestamps, so no additional 24-hour time shift is applied.

`scripts/reconstruct_matched_archive_fields.py` now implements this transformation from **two NPZs for the same target and seed** into the common NetCDF grid consumed by the station sampler. It rejects mismatched timestamps, truth, reference errors, and spatial coordinates; does not retrain models; and provides no implied operational forecast skill. `scripts/test_reconstruct_matched_archive_fields.py` contains synthetic arithmetic and mismatch-rejection tests, added to the station QA workflow.

**Evidence boundary:** The archived NPZ field *schema* is verified from the producer code, but real archive pairing and the latest CI result still require successful runs and explicit artifact inspection. The public full-data archive for Southwest 2022 is Actions run 41400003722, artifact 13122165252, with expiration 2027-01-31. Do not insert real-world station-skill estimates or claim successful end-to-end validation until the archived datasets have been processed and the stage-by-stage audits reviewed.

## Southwest 2022 archived spatial acceptance gate

Workflow: `.github/workflows/archived-station-spatial-acceptance.yml`. Uses GitHub Actions run `41400003722`, artifact `13122165252`, rather than downloading ERA5 again or retraining. The runner downloads the previously archived files with read-only Actions permission; `scripts/accept_archived_station_sampling.py` checks every embedded SHA256/byte-count manifest entry, restores historical and source-mean prediction fields for seeds 42/7/123, and samples two Southwest station coordinates (ABQ and ROW) using the same bilinear scheme. It expects exactly 1,460 six-hourly valid times per station in calendar 2022. Output contains per-seed station-grid CSV, SHA256 digests, environment snapshot and acceptance summary. Intermediate NetCDF files are deleted after successful sampling to avoid redundant data uploads. Artifacts expire after 30 days and need further preservation.

This acceptance stage **does not fetch/match ASOS observed temperatures and does not compute station RMSE**; those are independent downstream gates. Workflow was configured and committed, but its success must be confirmed from the actual run before claiming that full archival stage passed. Source-mean and historical archived grid/ground-truth/reference mismatches fail loudly.

## First end-to-end observational evaluation configured (Southwest 2022)

The workflow `.github/workflows/southwest-asos-retrospective.yml` and runner `scripts/run_retrospective_station_case.py` now connect archived three-seed gridded predictions with fetched 2022 ABQ/ROW temperature reports. Workflow stages: original archived artifact download (run 41400003722), 53-record manifest check, paired NPZ reconstruction, shared bilinear station sampling, 2022 IEM ASOS acquisition, same-station nearest UTC valid-time matching, paired station-year scores, and exploratory seven-day calendar-block bootstrap. An additional 15/30/60 minute matched-row coverage audit is preserved; the primary comparison uses the predeclared ±30 minute tolerance.

**Status boundary:** these are committed workflow definitions, not yet a confirmed successful empirical run. Do not copy predicted sample totals into tables as observed matches. Airport observations are not necessarily independent of ERA5 assimilation, and height-adjusted sensitivity has not yet been integrated into this workflow. Existing four-case ten-seed ERA5-only summaries must not be replaced with new station skill until explicit run IDs, outputs and QA are verified. Artifacts are retained for 30 days, not permanent archiving.

## Acceptance hardening: station coverage and identical verification sample

The Southwest ASOS runner now requires at least **80% of expected forecast valid times** at each station (with at least 1,000 matched samples per site) before allowing station RMSE tables. It also asserts that all three fitted seeds use identical station, valid-time and observed-report keys. The purpose is to prevent numerical differences caused by missingness or sample-selection rather than model performance. A 15/30/60-minute matching-coverage audit is retained, with 30 minutes the primary tolerance.

The synthetic integrated regression `scripts/test_station_end_to_end.py` exercises archival NPZ reconstruction, spatial interpolation, nearest-report matching and paired evaluation with known exact errors. It does **not** validate a real observation, but its test is part of the CI QA workflow. Real-run conclusions remain withheld until the Actions job success, ASOS response metadata, station coverage counts and result artifact hashes are independently inspected.

## Elevation source QA and minimum hot-subset uncertainty gate

IEM's official network table labels elevation explicitly as **Elevation [m]**. The source parser now requires that field name and checks its unit before reading station metadata; a meter-versus-foot synthetic regression is run ahead of the online ERA5 geopotential audit. This is metadata QA, **not** evidence that the ERA5 acquisition or any lapse-rate-adjusted station comparison succeeded.

The paired station evaluator continues to report descriptive observed-hot counts and errors whenever observations exist, but only constructs its hot-subset calendar-day block interval for at least **30 hot observations spanning 10 distinct UTC calendar days**. Smaller or more temporally concentrated hot subsets have `ci_hot=null` and `hot_ci_eligible=false`. This conservative threshold is a display/inference guard, not a demonstration of adequate independent sample size, and cannot turn ERA5-assimilated airport observations into an independent operational forecast benchmark.

Current workflow and commit checks should be confirmed from GitHub Actions run/job logs before labeling these tests passed. Public workstream outputs can be posted only after archive SHA256, observation coverage, cross-seed matched-time and temperature-unit checks pass.

## ASOS source response preservation and offline preflight gate

The IEM downloader now emits a per-station-year row-exclusion audit distinguishing invalid timestamps, missing temperatures, conservative Fahrenheit bounds, out-of-year records, duplicate report timestamps and retained unique times. Each public-service response is saved unchanged as `*.iem-raw.csv`, with its original SHA256 stored in the station record; the normalized Kelvin CSV retains its separate digest. These files are placed in the public scientific Actions artifacts and must be preserved separately before artifact expiry.

The Southwest 2022 real-station workflow now executes the offline parser, matching, paired-statistics and NPZ-to-station synthetic regressions **before** downloading the earlier full ERA5 archive. A source or statistical script modification triggers re-evaluation. This makes parser/metadata regressions fail sooner, but a committed workflow still needs its actual GitHub Actions completion result before any scientific acceptance claim.

The 2022 ASOS source archive contains airport reports with limited observational QC and may overlap with observations assimilated into ERA5. The geographic-scale limitation and the mismatch between 0.25-degree gridded 2-m temperature and point airport temperature must remain explicit. The data do not establish deployable 24-hour operational forecast skill.

## Efficient and numerically-equivalent station bootstrap (November 2026)

The station-paired circular calendar-day bootstrap now operates on per-day **counts and squared-error sums**, rather than repeatedly concatenating per-day Pandas frames. For each resampled calendar-day sequence, the two arms' RMSEs are computed from the same number of selected reports. Missing calendar dates remain zero-count days; hot-subset filtering is applied to each day's sufficient statistics **without dropping non-hot calendar dates from the resampling universe**. This preserves the intended paired bootstrap distribution while avoiding many large DataFrame copies.

`scripts/test_calendar_bootstrap_equivalence.py` deterministically compares both 2.5%/97.5% quantiles and empty-resample counts against a naive row-level implementation for full-sample and observed-hot cases, including gaps in calendar coverage. The regression is now in both the fast station QA and the archive-backed ASOS real-data preflight workflows. **The equivalence and speed claims remain code-level until an actual passing CI job confirms execution; no new observational skill has been obtained from this change.**

## Station identifier ingestion hardening

The nearest-ASOS matcher now rejects null or blank station identifiers **before** converting the column to text. Previously, Pandas could turn missing station IDs into the literal string `NAN`, silently treating missingness as a station name. Empty-observation sites retain an explicit `match_rate=0` and `max_abs_offset_minutes=null` in the match audit, rather than omitting these fields. A deterministic missing-ID rejection regression has been added to the existing station time-alignment self-test.

This correction is a data-integrity fix, not a newly verified ASOS accuracy result. Latest push-triggered GitHub Actions success and real archived station-skill outputs remain unconfirmed from available integration APIs; the GitHub adapter exposes PR-associated commit run listing, not full push-run history.

## First user-reported successful Southwest ASOS workflow and results extraction

The Southwest retrospective ASOS workflow has been reported successful by the project operator. The repository now includes `scripts/summarize_asos_station_acceptance.py` and a dedicated workflow step that reads the **actual** `station-scored-seed-{42,7,123}.json` files plus `retrospective_station_results.json` and produces (1) a succinct Markdown GitHub Actions run summary and (2) `verified_station_metrics.csv` in the published artifact. It refuses to summarize missing seed/site combinations, low match coverage, score/match count mismatches, invalid primary tolerance or an absent cross-seed identical-observation attestation. A synthetic regression tests correct extraction and low-coverage rejection.

**Provenance boundary:** The original successful run ID, score JSON payloads and specific RMSE/CI numbers have not been independently fetched in this session; publication-ready site skill cannot be inferred solely from a green workflow. The latest reporter step will apply on new runs triggered by this change. Do not backfill numeric results from predictions or repeat the four-region ERA5-only results as airport verification.

## Manuscript-aligned ASOS bootstrap reporting (new workflow runs)

The earlier successful workflow reported by the project operator used **seven-day calendar blocks and 200 draws**. Those existing results should not be re-labelled as the manuscript's 14-day primary interval. A new workflow revision now computes **14-day paired calendar blocks with 2,000 draws** as primary and saves **7-day and 28-day** interval sensitivities from exactly the same matched station/seed records. The acceptance summary now rejects incompatible draws or missing sensitivity files and displays 14-day primary intervals next to the actual station RMSE. This is a future-run analysis configuration change: it does not alter model weights, observed data definitions or the archived successful run. A successful new run plus inspection of the resulting CSV and JSON is required before inserting new station estimates into the private manuscript or public portfolio.

## Artifact-only figure production (no fabricated station results)

The revised Southwest 2022 ASOS workflow now produces `verified_station_contrasts.svg` **only after** the actual 2022 station prediction/observation results satisfy the summary acceptance checks and generate `verified_station_metrics.csv`. The SVG plots six paired source-mean-minus-historical RMSE contrasts (ABQ/ROW × seeds 42/7/123) and their manuscript-aligned 14-day bootstrap intervals, with the zero-difference reference axis and per-row matched sample counts. It labels observations as retrospective station agreement, explicitly noting possible ERA5 assimilation overlap. The plotting script rejects incomplete stations/seeds, weak coverage, nonfinite metrics, inconsistent arithmetic, or inverted intervals.

`scripts/test_summarize_asos_station_acceptance.py` now additionally exercises the SVG generator using **synthetic** CSV rows, whereas the production workflow always reads accepted real ASOS outputs. No empirical figure has been inserted into the repository README, portfolio, or private manuscript until the successful run's data have been independently inspected. The GitHub Actions result artifact currently has 30-day retention; preserve the raw data and output CSV outside ephemeral Actions storage if needed for submission.
