# Western-US-Weather-JAX

**Real ERA5 + JAX/CUDA regional 24-hour temperature forecasting and reproducible GPU engineering.** A four-frame residual CNN predicts six-hourly Western-US 2 m temperature fields at a **24-hour lead** on a **49 × 49** grid. This is a bounded research/engineering demonstrator, **not an operational weather forecast**.

## Verified results

| Evaluation | Persistence RMSE | Four-frame CNN RMSE | Relative overall improvement |
| --- | ---: | ---: | ---: |
| 2021 Jul–Dec holdout (train 2020) | 2.94224 K | 2.91864 K | 0.80% |
| Independent 2022 holdout (train 2019–2020; validate 2021) | 3.22027 K | 3.18849 K | 0.99% |
| 2022 holdout: validation-selected 305 K heat-weighted CNN | 3.22027 K | 3.19116 K | 0.90% |

**Known limitation:** the validation-selected, heat-weighted CNN has worse 2022 fixed-305-K heat-subset RMSE (**2.13841 K**) than persistence (**2.08005 K**). Do not interpret aggregate improvement as extreme-temperature forecasting skill.

## Actual GPU verification — Colab Tesla T4

A user-executed **NVIDIA Tesla T4** Google Colab run successfully trained the validated heat-weighted JAX CNN on real ERA5, with hardware-backed JAX GPU audit and matching input SHA-256. **40 epochs**, same seed and settings:

| Measurement | CPU | Tesla T4 GPU |
| --- | ---: | ---: |
| Training elapsed time | 258.51 s | 33.29 s |
| 2022 test RMSE | 3.191155 K | 3.191155 K |

**Observed elapsed-time ratio: 7.76×** in the initial Colab run. **Three additional real T4 GPU runs** completed in 35.63, 32.72 and 32.01 s (mean **33.45 s**, sample SD **1.92 s**), all with identical 2022 RMSE **3.191155 K**. Their mean relative to the earlier single CPU time suggests **~7.73×**, not a paired repeated speedup estimate. The test RMSE matched, GPU model parameters resided on GPU, and the automated CPU/GPU parity check passed. GPU timing now has three repeat measurements, but CPU timing is still single-run and hardware/thermal effects have not been controlled. The Colab session did not require a GitHub GPU runner.

## Reproduce and inspect

```bash
python -m pip install -e '.[test,zarr]' gcsfs
pytest -q
python -m weatherjax --epochs 4
```

Real ERA5 runs are defined in [GitHub Actions](.github/workflows/era5-crossyear.yml), with temporal split, validation-only heat weight selection and 2022 independent test reporting. The real GPU training path is in [scripts/gpu_training.py](scripts/gpu_training.py) and the [GPU acceptance protocol](docs/GPU_ACCEPTANCE.md). A manual Colab GPU session is an alternative to a configured self-hosted GPU runner.

For exact experiment tables, bootstrap sensitivity, evidence boundaries and the verified Colab GPU audit, see [Engineering Status](docs/ENGINEERING_STATUS.md).

## Remaining work

Repeat CPU timings under matched conditions and test additional GPU environments; improve fixed-threshold high-temperature skill; evaluate additional years, regions and model seeds. More complex models such as U-Net are **not yet verified**.
