# ERA5 reproducibility manifest — verified and pending

## Execution and source

| Item | Audited fact |
| --- | --- |
| Successful full original-domain test | [GitHub Actions #41285289285](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41285289285) |
| Python runtime configured | 3.11 (`actions/setup-python@v5`) |
| Install command | `pip install -e '.[test,zarr]' gcsfs` |
| Package constraints | `pyproject.toml`: JAX >=0.4.30; NumPy >=1.24; xarray >=2023.1; netCDF4 >=1.6; optional zarr >=2.16; pytest >=7 |
| Source data URL | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (Google ARCO ERA5) |
| Download selection | 2019-01-01 to 2022-12-31, 6-hour subsampling |
| Preflight checks | Kelvin units, time monotonicity and exact 6-hour cadence, grid dimensions, first eight frames finite and plausible temperature range |
| Experimental preflight output | `outputs/era5-crossyear/preflight.json`, included among workflow JSON artifacts |

## Missing information — do not infer

The `pyproject.toml` constraints do **not** establish exact installed package versions. The workflow does not pin all transitive dependencies. The file fetch routine does not write a SHA-256 hash or version of the underlying Zarr object. Neither the GitHub Actions run identifier nor a green test alone establishes the checksum of each NetCDF input.

For a publication-grade reproducible release, capture with an independent workflow or in a controlled rerun:

- exact package versions (`python -m pip freeze`), Python interpreter and platform details;
- input NetCDF SHA-256 hash and coordinate/time metadata;
- source object provenance and retrieved-at time;
- a stable archival identifier or release artifact for numeric outputs.

## Reproduction commands

The verified main workflow config executes:

```bash
python scripts/fetch_arco_era5.py --start 2019-01-01 --end 2022-12-31 --stride 6 --output input/era5_2019_2022.nc
python -m weatherjax.preflight input/era5_2019_2022.nc --timestep-hours 6 --lead-hours 24 --report outputs/era5-crossyear/preflight.json
```

This manifest is **code-derived** and does not claim independent examination of every historical NetCDF file.
