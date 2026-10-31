# Repeated Colab GPU timing

After the first real Tesla T4 GPU acceptance, a **second** Colab GPU session can run at least three additional full training repetitions. This measures timing variability rather than treating the original **7.76×** CPU/GPU ratio as a stable hardware benchmark.

With a GPU runtime selected and real ERA5 saved as `input/era5_2019_2022.nc`:

```bash
python scripts/repeat_gpu_benchmark.py --data input/era5_2019_2022.nc --repeats 3 --epochs 40 --output outputs/gpu-repeats
```

Each repetition launches the hardware-audited GPU training script in a **fresh Python subprocess** (per-run compilation included), requires verified GPU parameter residency, checks the same model settings and input SHA-256, and writes `outputs/gpu-repeats/repeat-summary.json` with mean, sample standard deviation, range and per-run test RMSE. **Do not claim repeated GPU performance until this is actually run.** Further CPU repetitions are needed to estimate a repeated speedup distribution. Three repetitions are a practical preliminary diagnostic rather than a statistically definitive benchmark.
