# ERA5 small regional training guide

**Validated automatically:** regular time coordinate, Kelvin units, crop to 30–50°N and 125–102°W, and plausible temperature values in first eight frames. **Not yet demonstrated:** a real ERA5 GPU training run.

Obtain ERA5 *2m temperature* NetCDF from the Copernicus Climate Data Store (CDS) or prepare a small local Zarr subset. Ensure:
- variable name `t2m` (or specify `--variable`);
- spatial coordinates `latitude/longitude` or `lat/lon`;
- time dimension `time` or `valid_time`;
- hourly, consecutive timestamps (UTC), values in kelvin;
- enough frames for chronological split (at least 40; preferably years for real science);
- cut to the western-US window before training to control memory.

For a downloaded local file:
```bash
python -m weatherjax.preflight input/era5_t2m.nc --timestep-hours 1 --lead-hours 24 --report outputs/preflight.json
python -m weatherjax --data input/era5_t2m.nc --lead-steps 24 --epochs 4 --output outputs/era5-experiment
```

This MVP loads the entire cropped field into RAM and materializes lead-paired arrays. Keep files small and monitor memory. It does not yet stream from the cloud or automatically download data, and it must not be described as a complete cloud ERA5 training pipeline.

**Methodology caution:** this initial model uses one temperature map at time t to predict t+24h, without external future forcings; it is a deliberately weak baseline. Report test RMSE against persistence and do not extrapolate to deployed forecast skill.
