# Repeated Colab GPU timing

> **Current status note (November 2026):** Real ERA5 training on a Colab NVIDIA Tesla T4 has been completed and hardware-audited. The initial matched run measured 33.29 s GPU versus one 258.51 s CPU reference (a descriptive 7.76× ratio); three GPU repeats averaged 33.45 s (SD 1.92 s), with audited matched test RMSE 3.191155 K. This does **not** establish a distribution of paired CPU/GPU speedups. The instructions below describe acceptance procedures, and any statements that no GPU result exists or GPU work remains wholly unperformed are historical. The submitted *Atmospheric Research* manuscript is under review, not accepted or published. See [project README](../README.md) and [recruiter evidence](RECRUITER_EVIDENCE_BRIEF.md).


After the first real Tesla T4 GPU acceptance, a **second** Colab GPU session can run at least three additional full training repetitions. This measures timing variability rather than treating the original **7.76×** CPU/GPU ratio as a stable hardware benchmark.

With a GPU runtime selected and real ERA5 saved as `input/era5_2019_2022.nc`:

```bash
python scripts/repeat_gpu_benchmark.py --data input/era5_2019_2022.nc --repeats 3 --epochs 40 --output outputs/gpu-repeats
```

Each repetition launches the hardware-audited GPU training script in a **fresh Python subprocess** (per-run compilation included), requires verified GPU parameter residency, checks the same model settings and input SHA-256, and writes `outputs/gpu-repeats/repeat-summary.json` with mean, sample standard deviation, range and per-run test RMSE. **Three audited GPU repetitions have now been completed as noted above.** Further CPU repetitions are needed to estimate a repeated speedup distribution. Three repetitions are a practical preliminary diagnostic rather than a statistically definitive benchmark.
