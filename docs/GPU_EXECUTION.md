# GPU acceptance protocol

> **Current status note (November 2026):** Real ERA5 training on a Colab NVIDIA Tesla T4 has been completed and hardware-audited. The initial matched run measured 33.29 s GPU versus one 258.51 s CPU reference (a descriptive 7.76× ratio); three GPU repeats averaged 33.45 s (SD 1.92 s), with audited matched test RMSE 3.191155 K. This does **not** establish a distribution of paired CPU/GPU speedups. The instructions below describe acceptance procedures, and any statements that no GPU result exists or GPU work remains wholly unperformed are historical. The submitted *Atmospheric Research* manuscript is under review, not accepted or published. See [project README](../README.md) and [recruiter evidence](RECRUITER_EVIDENCE_BRIEF.md).


Current GitHub-hosted CI uses an Ubuntu **CPU** runner; passing those jobs does NOT establish GPU execution.

The command below must run on a Colab, GCP, or self-hosted machine with an actual JAX GPU backend and real ERA5 NetCDF saved locally. The script rejects CPU-only execution and writes hardware-audited metrics.

Install a CUDA-compatible JAX build using the platform-specific instructions at https://docs.jax.dev/en/latest/installation.html before running:

```bash
pip install -e '.[test,zarr]'
python scripts/fetch_arco_era5.py --start 2020-01-01 --end 2021-12-31 --stride 6 --output input/real_era5.nc
python -m scripts.gpu_training --data input/real_era5.nc --epochs 20 --output outputs/gpu-era5
```

Acceptance requires (1) real GPU detected by `jax.devices()`; (2) valid ERA5 input; (3) training completion; (4) `gpu_metrics.json` with GPU device identification and held-out scores; (5) saved model weights. These five checks have been met for the previously audited Colab T4 run; repeat the gate when using new hardware/configurations.

For fully automatic execution, attach an authorized GPU-equipped self-hosted runner or an approved managed GPU training environment. This repository deliberately does not spawn paid GPU instances automatically.
