# Western-US-Weather-JAX

Phase 1 engineering MVP: a JAX CNN for fixed-lead 2m-temperature forecasting over the Western United States.

**Status:** the first verified smoke test uses synthetic data on CPU only. No real ERA5 skill or GPU training is claimed. See training scripts, CI checks, and the Colab walkthrough in this repository.

```bash
python -m pip install -e '.[test]'
pytest -q
python -m weatherjax --epochs 4
```

For hourly ERA5, set `--data /path/to/era5_t2m.nc --lead-steps 24`; verify input units and time intervals. Initial implementation is memory-resident and suited only to small input subsets.

Roadmap: (1) validate ERA5 ingestion and single-GPU training, (2) consider advanced JAX/GCP engineering depending on results, (3) consider a scientific paper only with a justified research contribution.
