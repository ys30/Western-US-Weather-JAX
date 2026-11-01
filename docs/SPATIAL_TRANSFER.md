# Southwest regional geographic-sensitivity experiment

The original ERA5 fetch defaults to latitudes 31–43°N and longitudes 121–109°W. This follow-up uses **30–36°N, 108–102°W**, a geographically distinct window with no shared grid-cell longitudes. It retains 2019–2020 training, 2021 validation and 2022 test, 6-hourly ERA5 input and 24-hour lead.

This is **not strict external zero-shot generalization** because the model is retrained within the new area. It instead tests whether the same method/hyperparameters and validation protocol transfer to another region. Stronger spatial external validation will freeze weights learned in the original region and evaluate them on a separately held-out region with compatible normalization. The 305 K high-temperature threshold and hot loss weight 2 are carried from previous analyses and are not retuned on the new 2022 test. If the new region lacks qualifying hot observations, report the heat metric as undefined, never zero or successful.

Workflow: `.github/workflows/era5-southwest-region.yml`.
