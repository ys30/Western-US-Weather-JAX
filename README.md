# Western-US-Weather-JAX

**Real ERA5 + JAX/CUDA regional 24-hour temperature forecasting and reproducible GPU engineering.** A four-frame residual CNN predicts six-hourly Western-US 2 m temperature fields at a **24-hour lead** on a **49 × 49** grid. This is a bounded research/engineering demonstrator, **not an operational weather forecast**.

**For recruiters:** [One-page engineering and scientific evidence brief](docs/RECRUITER_EVIDENCE_BRIEF.md) · [Live portfolio case study](https://ys30.github.io/projects/western-us-weather-jax.html).

## Verified results

| Evaluation | Persistence RMSE | Four-frame CNN RMSE | Relative overall improvement |
| --- | ---: | ---: | ---: |
| 2021 Jul–Dec holdout (train 2020) | 2.94224 K | 2.91864 K | 0.80% |
| Independent 2022 holdout (train 2019–2020; validate 2021) | 3.22027 K | 3.18849 K | 0.99% |
| 2022 holdout: validation-selected 305 K heat-weighted CNN | 3.22027 K | 3.19116 K | 0.90% |

**Known limitation:** the validation-selected, heat-weighted CNN has worse 2022 fixed-305-K heat-subset RMSE (**2.13841 K**) than persistence (**2.08005 K**). Do not interpret aggregate improvement as extreme-temperature forecasting skill.

## Stronger independent-year baseline — important limitation

The 2022 evaluation also tested training-only month/hour climatology and a persistence/climatology mixture selected on **2021 validation data**. The blend achieved **2.99462 K RMSE** (75% persistence, 25% climatology), substantially better than the unweighted CNN (**3.18849 K**). **The original four-frame CNN does not beat this blend.** A newly tested **climatology-aware residual CNN** reached **2.98365 K** on the same 2022 holdout versus **2.99462 K** for the blend (~0.37% better), but had worse 305 K heat-tail RMSE (**2.17875 K** versus **2.15930 K**). This is one seed/year and is not yet statistically established. [Verified CI experiment](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41255803026).

## Three-seed robustness check (2022 independent test)

Three JAX CNN random seeds **42 / 7 / 123** were run in verified GitHub CI. Annual test RMSEs were **2.98365 / 2.99659 / 2.98882 K**, compared with the validation-selected blend at **2.99462 K**. Only **two of three seeds** beat the blend. On the fixed 305 K hot subset, RMSEs were **2.17875 / 2.09789 / 2.12385 K**, versus **2.15930 K** for the blend: again, two seeds beat the baseline but a different seed was worse. **All three 14-day paired moving-block bootstrap intervals for the annual CNN-minus-blend RMSE difference include zero.** Robust predictive superiority has **not** been demonstrated. [Verified multi-seed experiment](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41258489704).

## Heat-weight validation sweep — encouraging but qualified

Nine real ERA5 runs (three seeds × hot-loss weights 0, 1, 2) were completed. **2021 validation selected extra weight 2 for all seeds.** On the 2022 fixed-305-K test subset, the selected climatology-aware CNNs achieved **2.05492 / 2.04506 / 2.03982 K** for seeds 42 / 7 / 123, versus **2.15930 K** for the blended benchmark (about **4.8–5.5%** lower heat RMSE). Annual RMSEs were **2.98450 / 2.99741 / 2.99243 K**, versus **2.99462 K** for the blend; one seed was worse overall. **Do not claim statistical significance or cross-year robustness from this sweep.** [Verified CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41259276907).

**Further heat-tail robustness (2022):** The heat-weighted CNN passed corrected CI, and all three seeds have **negative 1-, 3-, 7-, and 14-day paired block-bootstrap intervals** for fixed-305-K hot RMSE versus the blend. In each seed, all **8 months with eligible heat observations (March–October)** improved; other months have no qualifying hot samples. This supports **within-year** robustness, **not** cross-year or other-region generalization. [Verified CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41265640777).

## 2021 retrospective temporal sensitivity: all three seeds improve

An additional real-ERA5 temporal backtest used 2018–2019 training, 2020 validation and 2021 testing. The validation-selected blend scored **2.90484 K annual** / **2.16951 K fixed-305-K hot** RMSE. Fixed extra hot-loss weight 2 climatology-aware CNNs (seeds 42 / 7 / 123) achieved **2.89004 / 2.90044 / 2.88990 K annual** and **2.06278 / 2.00450 / 2.04634 K hot**. All three seeds improved both metrics on this retrospective window. **2021 was previously used in model development**, so this is an informative *retrospective temporal sensitivity test*, **not an untouched independent external validation**. [Verified CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41267651677).

**Multiday high-temperature episode audit:** With a retrospective, truth-defined regional event rule (>=5% of grid cells at >=305 K during at least one 6-hour valid time on each of at least two consecutive days), the weight-2 climatology-aware CNN had lower within-event fixed-hot RMSE than the strong blend on **7/7 qualifying episodes in 2021** and **8/8 in 2022**, for **each of three seeds (42, 7, 123)**. These are correlated retrospective event-conditioned comparisons, **not forecast-event detection or independent prospective verification**. [2021 CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41268528184) · [2022 CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41268528080).

## Geographic sensitivity: second non-overlapping region

In a **new Southwest study area** (30–36°N, 108–102°W), three retrained heat-weighted climatology-aware CNNs all improved the 2022 annual RMSE (**3.77695 / 3.79943 / 3.78322 K**) versus the region-specific validation-selected blend (**3.80252 K**). Fixed-305-K hot RMSE was **2.31885 / 2.31365 / 2.32763 K**, versus **2.40826 K** for the blend (roughly **3.3–3.9%** improvement). Of 10 truth-defined multiday hot episodes, **8 / 10 / 10** improved across the three seeds. This supports **method transfer after retraining**, **not zero-shot transfer of a frozen network**. [Successful CI](https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41270894771).

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

Improve forecasting beyond the validation-selected climatology/persistence blend; repeat CPU timings under matched conditions and test additional GPU environments; improve fixed-threshold high-temperature skill; evaluate additional years, regions and model seeds. More complex models such as U-Net are **not yet verified**.
