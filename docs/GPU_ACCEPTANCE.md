# GPU acceptance protocol

The real ERA5 GPU entry point is `python scripts/gpu_training.py`; it intentionally refuses CPU-only JAX environments. CPU CI success **does not** establish that GPU training ran.

For comparable initial CPU/GPU training configurations, supply the same dataset, train/validation split, epoch count, history length and heat weight. The CPU run is performed via `python -m weatherjax`, the GPU via `scripts/gpu_training.py`. Both write elapsed training seconds into JSON metrics. GPU metrics additionally include audited GPU-resident JAX parameter devices. Compare elapsed times only with the caveat that platform-dependent JAX compilation and transfer costs are included, and record GPU type and software stack.

Example on a GPU-equipped machine after installing compatible JAX GPU wheels and preparing `input/era5_2019_2022.nc`:

```bash
python -m weatherjax --data input/era5_2019_2022.nc --epochs 40 --lead-steps 4 --history-steps 4 --zero-residual --time-features --train-until 2021-01-01 --validate-until 2022-01-01 --hot-weight 2 --hot-threshold-kelvin 305 --output outputs/cpu-parity
python scripts/gpu_training.py --data input/era5_2019_2022.nc --epochs 40 --history-steps 4 --train-until 2021-01-01 --validate-until 2022-01-01 --hot-weight 2 --hot-threshold-kelvin 305 --output outputs/gpu-parity
```

Gate passes only after (1) actual GPU hardware detected and training completed, (2) hardware audit saved with GPU-resident parameters, (3) CPU/GPU timings measured under stated matching settings, (4) numerical differences evaluated and documented. No GPU results are currently claimed.
