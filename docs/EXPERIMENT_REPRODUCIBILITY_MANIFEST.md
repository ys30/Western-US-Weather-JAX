# ERA5 matched-model reproducibility manifest

Status: verified GitHub Actions run metadata (2026-11-02 UTC). This is an engineering manifest, not an archived manuscript or a proof of data-file checksum reproducibility.

| Evaluation | Source commit SHA | GitHub Actions run | Result artifact ID | Status |
| --- | --- | --- | --- | --- |
| Southwest 2022 | `d340194893bea498ee6928e29aeed58be49fdf45` | [41356348903](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41356348903) | `13104335134` | success |
| Southwest 2023 | `45bc4053ed5382348037bf5ed453c8f6095e069c` | [41362832891](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41362832891) | `13106656267` | success |
| Southern Plains 2022 | `37d61875b598491d4e255992128093d7b9650e97` | [41386676606](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41386676606) | `13115704221` | success |

## Configuration recovered from workflow files

- Source: ERA5 six-hourly 2 m temperature, source geographic box 31–43°N, 121–109°W; train 2019–2020, source validation 2021.
- Southwest target: 30–36°N, 108–102°W; target evaluation 2022 or 2023.
- Southern Plains target: 32–38°N, 101–95°W; target evaluation 2022.
- Architecture comparison: `historical` vs `source_mean` feature modes, three initialization seeds 42/7/123, 40 epochs, hot weight 2. Both arms retain the shared blend reference carrying target historical climatology.
- Six-hourly samples; forecast horizon 24 h (4 steps); truth-conditioned hot subset >=305 K.
- Direct paired CNN-minus-CNN comparison: calendar-day circular moving-block bootstrap, 2,000 draws, block lengths 3/7/14/28 days. Sign convention: source_mean minus historical RMSE.
- Target climatology: observations strictly before target test start; historical period differs between target 2022 and target 2023.
- Public experiment workflows: `.github/workflows/era5-matched-climatology.yml`, `era5-matched-crossyear-2023.yml`, `era5-matched-southern-plains.yml`. For exact historical settings, inspect each listed source commit rather than current main.

## Not yet verified or durably archived

1. SHA256 hashes of the original NetCDF input slices used by each completed run.
2. SHA256 hashes and external retention for fitted model checkpoint files.
3. SHA256 hashes of the exact Actions ZIP artifact bytes.
4. Environment lockfile / immutable dependency versions, ARCO upstream dataset snapshot or object version, and retrospective source-data availability.
5. Permanent long-term preservation of experiment artifacts beyond GitHub's configured retention.

**Artifact availability is temporary**: as checked on 2026-11-02, the three result artifacts have expiration timestamps around 2027-01-30 to 2027-01-31. The existing workflow uploads JSON summaries rather than checkpoints or full gridded prediction archives.

## Scientific interpretation

Observed-hot RMSE conditions on verifying ERA5 temperature and does not measure real-time heat warnings. ERA5 reanalysis is not proof of forecast-time information availability. The channel comparison is not a strict zero-history ablation, because both models see the historical-climatology blended baseline. The experiments and seed/block intervals are correlated and exploratory; avoid causal or multiplicity-adjusted significance claims.

Private manuscript, submission figures, supplementary information, and editorial reports are **not** included in this public repository.


## Verified SHA256-enabled re-runs (2026-11-02 UTC)

All three CI reruns completed successfully. Their downloadable result artifacts include `sha256-manifest.json` with **50 hashed files each**: 2 NetCDF inputs, 6 CNN checkpoint NPZ files, 6 gridded prediction/error NPZ files, and 36 result JSON/CSV files.

| Evaluation | Successful run | Result artifact ID | SHA256 of downloaded result ZIP |
| --- | --- | --- | --- |
| Southwest 2022 | [41394193646](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41394193646) | `13119675201` | `89affe98cc1c6dea47c8a333cfa39ee96aeb54800fa653aa87cb8661e83bd823` |
| Southwest 2023 | [41394199132](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41394199132) | `13119521152` | `9bedacc599750e36f25f43a95c9da71dfe1d25e763ff719d59f01e9a1769d885` |
| Southern Plains 2022 | [41394203685](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41394203685) | `13118593353` | `01a9e22fb0f278a81d8291226ed455543a6631c8e675de7523d9aacb5c4bddeb` |

The source NetCDF file SHA256 is identical across the three runs:
`ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6`.

**Retention caveat:** only JSON/CSV summaries and the SHA256 manifest were included in the uploaded result artifacts. The input NetCDF, trained checkpoint NPZ, and gridded-prediction NPZ **were hashed on the runner but not uploaded**; their recorded checksums do not make their bytes retrievable. External long-term preservation and dependency/data snapshot pinning remain outstanding.


## Full byte-preserving archive verification (Southwest 2022)

- Verified successful run: [41400003722](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41400003722); artifact ID `13122165252`.
- The 102,380,409-byte GitHub ZIP includes 54 files: 2 input NetCDF, 6 checkpoint NPZ, 6 gridded prediction NPZ, 12 diagnostics CSV, 24 experiment JSON results, 1 SHA256 manifest, and 3 environment/provenance text files.
- All 53 listed files exactly matched the SHA256 digest and byte count in `outputs/archive/sha256-manifest.json`; the manifest is not self-hashed.
- ZIP SHA256: `7476ae3978ed8d2a20385e7abad60f8a137e1034fe8124cfde419e1c4679197c`.
- This verifies the preserved bytes for this **new rerun**, not that they are byte-for-byte identical to the original 2022 study run. Artifact storage has finite retention and is not permanent archival infrastructure.


## Completed full-data archives for all three evaluations (verified 2026-11-02)

Each completed ZIP includes 54 files, of which 53 data/model/result/environment files were checked against the embedded SHA256 manifest; 0 missing entries, 0 digest or size mismatches. The manifest JSON is excluded from self-hashing.

| Evaluation | Successful Actions run | Full-data artifact ID | ZIP bytes | ZIP SHA256 |
| --- | --- | --- | ---: | --- |
| Southwest 2022 | [41400003722](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41400003722) | `13122165252` | 102380409 | `7476ae3978ed8d2a20385e7abad60f8a137e1034fe8124cfde419e1c4679197c` |
| Southwest 2023 | [41402185866](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41402185866) | `13122890221` | 105102847 | `ea3674409314c5659ddb0cd7861ced7b1115ad4b0631b9cc8e805532fd5c5d36` |
| Southern Plains 2022 | [41402189567](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41402189567) | `13121977420` | 101105883 | `8698e1514ec6e6ccc588dd44b53a83b141ed4e0b30914f983f0e98a41c217d4f` |

Each archive includes 2 ERA5 NetCDF inputs, 6 fitted checkpoint NPZ, 6 gridded prediction-error NPZ, 24 experiment JSON results, 12 diagnostic CSV, a self-excluded SHA256 manifest and 3 environment/provenance text files. This is an integrity check of the **new archival reruns**, not proof that they reproduce every historical byte from earlier successful runs. GitHub Actions artifact retention is finite; secure permanent off-platform storage remains outstanding.
