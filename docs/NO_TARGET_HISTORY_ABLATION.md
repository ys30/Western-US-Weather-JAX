# Frozen source CNN: no-target-historical-climatology ablation

Evaluate original-region-trained CNN parameters, source-training mean/std, and source-validation blend coefficient on disjoint Southwest ERA5 2022, without retraining. Replace the target historical climatology feature by the **source global training mean** everywhere. This avoids target-region historical climatology/statistics but retains the necessary **four recent target temperature input frames** that are available at forecast issuance.

This is a **feature-ablation experiment**, not a full zero-target-observation or robust external-generalization demonstration. A constant climatology feature differs substantially from the source training feature distribution. Compare with frozen weights plus target pre-test historical climatology; no 2022 label selection is permitted. Results are unverified until CI finishes.
