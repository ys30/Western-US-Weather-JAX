# Working manuscript — research draft (not submitted)

## Title
Cross-Regional Transfer of Temperature Forecasting with JAX Convolutional Networks: Mean Accuracy, Heat-Tail Trade-offs, and Historical Climatology Adaptation

## Abstract (draft)
Regional transfer is an important challenge for data-driven temperature prediction, especially when extreme heat rather than average error determines performance. We evaluated 24-hour-ahead, six-hourly ERA5 2-m temperature prediction with residual JAX convolutional neural networks (CNNs) across geographically disjoint western United States domains. Source-region models were trained on 2019–2020, with 2021 reserved for validation, and evaluated in a separate Southwest domain in 2022. We distinguished frozen CNN weights with target pre-test historical climatology features from source-only models trained and transferred without regional climatology inputs. For frozen climatology-aware CNNs, heat-subset RMSE (observed temperature ≥305 K) was 2.347, 2.331 and 2.323 K across three seeds, compared with 2.479 K for the source-weighted blended reference. Paired 14-day time-block bootstrap 95% intervals for the heat RMSE difference were fully negative across seeds, while annual intervals included zero. Conversely, source-only climatology-free CNNs achieved annual RMSE 4.112–4.098 K versus 4.163 K persistence, but worsened heat-subset RMSE (2.539–2.563 versus 2.400 K), with 14-day bootstrap intervals for heat RMSE differences entirely positive. These contrasting protocols show that improvement in domain-wide mean error need not carry over to observed extreme-heat subsets. They also underscore that the reported climatology-aware transfer is not strict no-target-history zero-shot generalization.

## 1. Introduction — argument map
1. Temperature forecasting models are typically assessed by domain-wide skill; high-impact heat extremes warrant a separately reported metric.
2. Apparent transfer gains must distinguish a model trained again in the target region from a frozen model whose weights are reused.
3. A second distinction concerns local historical climatology as target-domain input information, which may be unavailable in some operational settings.
4. Contribution: a reproducible JAX/ERA5 evaluation that separates target-local retraining, frozen network transfer with historical feature adaptation, and source-only no-climatology transfer; evaluates paired time-block uncertainty; and reports both successful and unfavorable extremes results.

## 2. Methods

### 2.1 Data, geographical domains and temporal partitioning
ERA5 2-m temperature (Kelvin) was sampled every six hours across two nonoverlapping rectangular regions. The original domain was approximately latitude 31–43°N and longitude 121–109°W; the Southwest domain 30–36°N and 108–102°W. The principal cross-region experiments used original-region 2019–2020 data for training, 2021 for validation/epoch selection, and Southwest calendar 2022 for testing. Source training data determined normalization parameters, which were frozen during cross-region inference. Results pertain to these fixed domains and years; they are not random samples of all climates.

### 2.2 Forecast task and baseline
Four available six-hourly temperature frames and forecast-valid-time hour cyclical features were used in convolutional input tensors. The target was 24 hours ahead (four six-hourly steps). Persistence is the temperature field at the valid time minus four six-hourly steps. The climatology-aware variant also included spatial, monthly and UTC-hour historical climatology and a persistence/climatology blend; its blend coefficient was selected using source validation data. A local Southwest-selected blend in the independently retrained protocol is a different baseline and should not be equated with the frozen source-weight reference.

### 2.3 Models and transfer protocols
**A. Target-local retraining:** train Southwest CNNs using its own training period and validation to test model applicability after local adaptation; this is **not frozen transfer**.
**B. Frozen weights, historical feature adaptation:** train on original area; fix validation-selected weights, training normalization, and blend coefficient; construct Southwest spatial climatology from target data dated before 2022; infer 2022 without target model-gradient updates. This involves target historical data and is **not** strict no-target-history transfer.
**C. Source-only architecture:** train the source CNN from the outset without any climatology channels, using recent temperatures, cyclical hour and persistence. Freeze weights and normalization for Southwest 2022 and compare to persistence. Target recent observations are needed as forecast inputs but no target climatology model/statistics are fitted.
**D. Feature substitution sensitivity:** replace climatology features in the already-trained climatology-aware CNN with source training global mean. This induces a feature-distribution shift and is a sensitivity/ablation only, not an equivalent strict-transfer architecture.

Three initializations (seeds 42, 7, 123) were evaluated. Source model selection used original-region validation RMSE, not Southwest 2022 test performance.

### 2.4 Outcomes and uncertainty
Annual RMSE aggregates all test gridcells and valid times. The observed-hot subset restricts scored test points to **truth temperature ≥305 K** (about 31.85°C). Because truth identifies hot points retrospectively, this measures conditional error and **not** heat-event detection. Differences are calculated as CNN minus its **protocol-specific** baseline. Paired circular time-block bootstrap resamples valid times while preserving complete spatial fields, using 1000 draws at blocks of 1, 3, 7 and 14 days. Percentile intervals describe dependence-sensitive uncertainty **within this test domain/year**, not external-spatial replication or multiplicity-adjusted inference.

## 3. Results

### 3.1 Frozen network transfer with target historical climatology
In Southwest 2022, the frozen source-weight blended benchmark had annual RMSE 3.799870 K and truth-hot RMSE 2.478656 K. CNN annual RMSE was 3.790533, 3.798109 and 3.794606 K for seeds 42/7/123. CNN truth-hot RMSE was 2.347359, 2.330832 and 2.322750 K, approximately 5.30%, 5.96% and 6.29% below its own blended reference. The 14-day bootstrap 95% intervals for truth-hot CNN-minus-blend were [-0.170609, -0.095481], [-0.180439, -0.114352] and [-0.191826, -0.119412] K. Corresponding annual differences had intervals [-0.024538, 0.006258], [-0.008488, 0.005789] and [-0.019666, 0.009477] K, all spanning zero. All four tested block lengths had negative hot-subset intervals and annual intervals spanning zero.

### 3.2 Source-only no-target-historical-climatology transfer
The Southwest 2022 persistence baseline had annual RMSE 4.162715 K and truth-hot RMSE 2.400407 K. Climatology-free frozen CNNs had annual RMSE 4.112426, 4.098400 and 4.099266 K and truth-hot RMSE 2.550601, 2.563245 and 2.538898 K (seeds 42/7/123). At 14-day blocks, annual CNN-minus-persistence 95% intervals were [-0.067486, -0.031048], [-0.086444, -0.039725] and [-0.085941, -0.038732] K; hot-subset intervals were [0.114932, 0.189626], [0.132285, 0.197894] and [0.108560, 0.171113] K. All four block lengths gave the same qualitative signs. Thus mean accuracy gains coexisted with consistent hot-subset deterioration.

### 3.3 Feature-substitution sensitivity
Replacing the historical climatology channel with source-global-mean constants increased the source-weight blend hot RMSE from 2.478656 to 6.001585 K, and CNN hot RMSE from 2.347359/2.330832/2.322750 to 6.185947/5.924195/5.741222 K. Because both the baseline and learned-input distribution changed, this experiment is **not** a controlled estimate of the isolated causal benefit of climatology or definitive proof about strict spatial transfer.

## 4. Discussion — draft
The central finding is a **metric-dependent transfer trade-off**. Historical-feature-adapted frozen source networks improved hot-subset RMSE within the Southwest test year, although their small annual differences were statistically uncertain under the evaluated block resamples. A separately source-trained climatology-free model modestly improved annual prediction over persistence but worsened conditional hot-subset error. This distinction matters for evaluations in which extreme temperatures are of principal interest.

The adapted experiment retains a target historical-data requirement, whereas the no-climatology source-only protocol tests a narrower operational setting. Differences between the architectures, features and benchmarks prohibit a causal attribution of all skill differences solely to local climatology. Source-only results should remain visible even where they do not advance the high-heat objective.

## 5. Limitations and next analyses
- One main Southwest held-out year and two geographically disjoint rectangles do not demonstrate multi-region/multi-year external generalization.
- Prior exploration of historical time periods, feature choices and model weighting can influence apparent independence of an evaluation; record the development chronology explicitly.
- A threshold applied to observed truth does not validate heat-event detection or early-warning decision performance.
- Time-block bootstrap addresses temporal dependence but cannot solve broad spatial representativeness, multiple-comparison or model-selection issues.
- Comparing a climatology-aware CNN to the source-only CNN changes architecture/input/training distributions and comparator; further matched-design studies are needed before causal claims.
- Check exact time-index handling in all architectures: four available frames and a 24-hour target lead must not contain post-issuance target inputs.
- Consider an additional withheld Southwest year, independent domain and operational threshold-based detection metrics before a submission-ready manuscript.

## Data and code availability
Code and reproducible GitHub Actions: https://github.com/ys30/Western-US-Weather-JAX . ERA5 data access is automated in workflow scripts; the repository does not imply redistribution of ERA5 as owned data.

## Evidence ledger
- Frozen transfer numerics: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41273562903
- Frozen transfer paired bootstrap: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41274429714
- Source-only frozen spatial transfer: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41276916469
- Source-only paired bootstrap: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41277714617
- Feature-substitution sensitivity: https://github.com/ys30/Western-US-Weather-JAX/actions/runs/41275083685

**Editorial note:** This is a numerical-evidence draft, not a submission-ready article. Do not add literature citations, author lists or journal-specific claims until separately checked. The independent original-domain 2022 workflow was still in progress at the last CI check.
