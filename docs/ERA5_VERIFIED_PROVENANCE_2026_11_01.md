# Verified provenance-only ERA5 artifact — 2026-11-01

Source: [successful workflow #41288894933](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41288894933), artifact [era5-input-provenance](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41288894933), artifact ID `13076427083`.

Actual artifact contents inspected: `era5-file-manifest.json` and `pip-freeze.txt`. All workflow stages succeeded.

| Property | Actual manifest value |
| --- | --- |
| NetCDF filename | `era5_2019_2022.nc` |
| SHA-256 | `ed8bd0813282030e7f6b2d3447bdb2f4005d741d6daa8ec8c5435eb78387aed6` |
| File size | 56,182,643 bytes |
| Time steps | 5,844 |
| Grid | 49 × 49 |
| Latitudes | 31 to 43 degrees north |
| Longitudes | −121 to −109 degrees |
| First valid time | 2019-01-01T00:00:00 UTC |
| Last valid time | 2022-12-31T18:00:00 UTC |
| Data variable / units | `t2m` / K |
| Python interpreter | 3.11.17 |
| Captured at | 2026-11-01T08:28:18.421040+00:00 |

The raw artifact preserves the entire resolved `pip-freeze.txt` environment. The file checksum applies **only to this provenance-only retrieval**, not the data serialization used in prior successful training jobs; historical experiments have independent CI artifacts and commits. No operational-forecast availability or immutable source Zarr version is established by this check.
