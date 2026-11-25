# GPU acceptance protocol

> **Current status note (November 2026):** Real ERA5 training on a Colab NVIDIA Tesla T4 has been completed and hardware-audited. The initial matched run measured 33.29 s GPU versus one 258.51 s CPU reference (a descriptive 7.76× ratio); three GPU repeats averaged 33.45 s (SD 1.92 s), with audited matched test RMSE 3.191155 K. This does **not** establish a distribution of paired CPU/GPU speedups. The instructions below describe acceptance procedures, and any statements that no GPU result exists or GPU work remains wholly unperformed are historical. The submitted *Atmospheric Research* manuscript is under review, not accepted or published. See [project README](../README.md) and [recruiter evidence](RECRUITER_EVIDENCE_BRIEF.md).


The real ERA5 GPU entry point is `python scripts/gpu_training.py`; it intentionally refuses CPU-only JAX environments. CPU CI success **does not** establish that GPU training ran.

For comparable initial CPU/GPU training configurations, supply the same dataset, train/validation split, epoch count, history length and heat weight. The CPU run is performed via `python -m weatherjax`, the GPU via `scripts/gpu_training.py`. Both write elapsed training seconds into JSON metrics. GPU metrics additionally include audited GPU-resident JAX parameter devices. Compare elapsed times only with the caveat that platform-dependent JAX compilation and transfer costs are included, and record GPU type and software stack.

Example on a GPU-equipped machine after installing compatible JAX GPU wheels and preparing `input/era5_2019_2022.nc`:

```bash
python -m weatherjax --data input/era5_2019_2022.nc --epochs 40 --lead-steps 4 --history-steps 4 --zero-residual --time-features --train-until 2021-01-01 --validate-until 2022-01-01 --hot-weight 2 --hot-threshold-kelvin 305 --output outputs/cpu-parity
python scripts/gpu_training.py --data input/era5_2019_2022.nc --epochs 40 --history-steps 4 --train-until 2021-01-01 --validate-until 2022-01-01 --hot-weight 2 --hot-threshold-kelvin 305 --output outputs/gpu-parity
```

Gate passes only after (1) actual GPU hardware detected and training completed, (2) hardware audit saved with GPU-resident parameters, (3) CPU/GPU timings measured under stated matching settings, (4) numerical differences evaluated and documented. Historical protocol statement (superseded by verified Colab T4 results noted above).

## Automated parity verifier

After obtaining actual measured CPU and GPU outputs with identical configurations, run:

```bash
python scripts/compare_gpu_cpu.py outputs/cpu-parity/metrics.json outputs/gpu-parity/gpu_metrics.json --output outputs/gpu-parity/acceptance.json
```

The verifier refuses a missing/incomplete GPU hardware audit, divergent data splits or model settings, nonfinite metrics, and CPU/GPU test RMSE deviations greater than 5% by default. Speedup is reported descriptively, not claimed without repeated measurements. Tests use synthetic audit fixtures and **do not** represent GPU execution.

## Opt-in GPU Actions workflow

[Real ERA5 GPU hardware acceptance](https://github.com/ys30/Western-US-Weather-JAX/actions/workflows/era5-gpu-acceptance.yml) is `workflow_dispatch` only. It does not run on ordinary pushes, and requires a configured self-hosted Linux x64 runner carrying the `gpu` label and a compatible NVIDIA CUDA driver. The runner owner is responsible for compute provisioning and costs. The workflow installs JAX CUDA support, fails if an actual GPU is not detected, trains matched CPU and GPU configurations on the same ERA5 years and uploads the parity audit. **This specific self-hosted GPU Actions workflow has not been used as evidence; separately audited Colab Tesla T4 executions have been completed.**
