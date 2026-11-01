# ERA5 provider and temporal sampling provenance audit

## Verified from repository source code

- Fetch implementation: `scripts/fetch_arco_era5.py`
- External mirror: public Google Cloud ARCO ERA5 Zarr (ECMWF ERA5 reanalysis)
- Configured Zarr URI: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- Native collection naming indicates: hourly, 0.25-degree grid
- Script variable preference: `2m_temperature` (then `t2m`, `temperature_2m`)
- Processing: normalize longitude if necessary, spatial crop, `isel(time=slice(None,None,stride))` with `stride=6`
- Output: `t2m` dataset written to NetCDF, Kelvin units attribute, finite-value assertion
- Original region script defaults: latitude 31–43°N, longitude −121 to −109°
- Temporal no-lookahead tests: `weatherjax/time_audit.py`, checked on six-hourly model inputs

## Interpretation and key constraint

ERA5 is a **retrospective reanalysis**. The fact that input times precede target times does **not** guarantee operational real-time availability: input reanalysis may use observations assimilated later. The manuscript should use 'reanalysis-based hindcast' or 'retrospective prediction' and avoid claiming deployable day-ahead forecast readiness without an operational dataset evaluation.

## Not yet confirmed by code inspection alone

- Exact timestamp of each original download, the cloud object's immutable version ID, local NetCDF hashes
- Actual coordinates, resolution and length of every historical experiment's saved local file, beyond workflow configuration
- Provider-specific ERA5 citation, terms and acknowledgement corresponding to the cloud mirror
- Prospective validation on operationally available observational or analysis inputs

**Source-of-truth:** repository fetch implementation; these statements are not independent verification of historical raw ERA5 contents.
