# Independent 2021 backtest (new temporal window)

A second independent-year evaluation is configured with **2018–2019 training**, **2020 validation**, and **2021 untouched test**. It uses the same Western US region, 6-hour ERA5 2m temperature data and fixed 24-hour lead as the earlier 2022 evaluation.

The validation-selected persistence/climatology blend is recalculated only from the *earlier* training and validation periods, with no 2021 test leakage. The climatology-aware CNN is evaluated for seeds 42, 7, 123 with an extra 305 K heat loss weight of 2, **carried over as a preregistered candidate from the earlier 2022 exploratory work, not independently optimized for this 2021 window**. Because the 2021 data previously appeared in other model-development experiments, it is a **temporal sensitivity / retrospective backtest**, **not a genuinely untouched prospective generalization claim**. Do not treat the two periods as independent evidence of a pre-registered study or claim a 2021 result before CI finishes.

Workflow: [ERA5 2021 backtest](../.github/workflows/era5-2021-backtest.yml).
