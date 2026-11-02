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
