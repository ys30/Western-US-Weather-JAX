# GPU acceptance protocol

Current GitHub-hosted CI uses an Ubuntu **CPU** runner; passing those jobs does NOT establish GPU execution.

The command below must run on a Colab, GCP, or self-hosted machine with an actual JAX GPU backend and real ERA5 NetCDF saved locally. The script rejects CPU-only execution and writes hardware-audited metrics.

Install a CUDA-compatible JAX build using the platform-specific instructions at https://docs.jax.dev/en/latest/installation.html before running:

```bash
pip install -e '.[test,zarr]'
python scripts/fetch_arco_era5.py --start 2020-01-01 --end 2021-12-31 --stride 6 --output input/real_era5.nc
python -m scripts.gpu_training --data input/real_era5.nc --epochs 20 --output outputs/gpu-era5
```

Acceptance requires (1) real GPU detected by `jax.devices()`; (2) valid ERA5 input; (3) training completion; (4) `gpu_metrics.json` with GPU device identification and held-out scores; (5) saved model weights. Do not label any work as **GPU verified** until these five checks have passed.

For fully automatic execution, attach an authorized GPU-equipped self-hosted runner or an approved managed GPU training environment. This repository deliberately does not spawn paid GPU instances automatically.
