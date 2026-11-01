# Frozen-weights cross-region evaluation protocol

**Status:** Source model checkpoint export implemented; **a frozen-target evaluation result is NOT yet available**.

The original and Southwest ERA5 regions have different grid shapes, and the residual CNN's two extra channels are **local month/UTC-hour climatology** and a **locally weighted persistence/climatology blend**. A valid frozen network experiment must distinguish:

1. **Network weights**: train on original area 2019–2020 only, choose epoch on 2021 validation, export the exact JAX parameter arrays; freeze without any target optimization.
2. **Normalization**: use original-area training mean and standard deviation in target inference; never compute them using 2022 test truth. The checkpoint now records these values.
3. **Baseline weights**: use source validation's climatology blend coefficient, not target 2022 selection; checkpoint records the source coefficient.
4. **Spatially varying climatology**: cannot directly paste the source grid into the disjoint target region. Two distinct experiments are needed: (a) adapt local climatology from **target historical pre-2022 training dates**, with frozen CNN parameters, explicitly labeled **target-historical-feature adaptation**; (b) a stricter no-target-history experiment using a prespecified climatology-free compatible feature construction and separately trained source CNN.
5. **Truth masking**: select >=305 K hot test cells only for retrospective evaluation, not inference.
6. **Comparison**: report target local-retraining CNN versus region-specific blend separately; frozen network vs source-weighted blend must be computed with same inputs. Keep region split and timestamp/lead consistency.

Source-region checkpoint export is available via:
`python scripts/climatology_aware_cnn.py --data input/era5_west_2019_2022.nc --train-until 2021-01-01 --validate-until 2022-01-01 --seed 42 --hot-weight 2 --checkpoint-path outputs/source/model-42.npz --output outputs/source/result-42.json`

Note: this command uses the standard 2022 test to produce an output report, but **selection remains training/validation only**. A checkpoint file stores selected weights plus metadata, not the target region dataset.

**Do not label region-specific retraining as frozen spatial transfer.**
