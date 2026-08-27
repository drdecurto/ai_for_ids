# Final experiment protocol

## Axis 1: strict tabular panel

### Cells

- CIC-IDS2017 V1: 15 classes, no PCA.
- CIC-IDS2017 V3: grouped nine-class taxonomy, no PCA.
- CIC-UNSW-NB15 V1: ten classes, cleaned numeric features.
- CIC-UNSW-NB15 V2: ten classes, half-dimension PCA.

### Strict configurations

- Pruned Random Forest, strong 200-tree Random Forest, and 400-tree XGBoost under a large natural-distribution budget.
- The same three learners under the matched class-balanced budget.
- TabPFN 2.5 and TabICL at the matched budget.

TabSTAR is excluded from the strict panel because its CIC runs use a different held-out partition and a larger per-class budget.

### Split, sampling, and criterion

- Stratified 70/30 split with split seed 2026.
- Seeds 0-9 vary fitting-row sampling and model stochasticity.
- Large natural budget: at most 400,000 rows at observed class prevalence.
- Matched balanced budget: approximately 2,000 rows per class, bounded by availability.
- Preprocessing is fitted only on rows allowed to the corresponding budget.
- The weighted evaluation cap samples 5% per class with a 1,500-row floor; smaller classes remain complete and confusion rows are inverse-probability weighted.
- A non-benign class is rare when its weighted evaluation share is below 0.5%; the primary outcome is the unweighted mean of its classwise recall.

### Statistical inference

- Mean paired difference over ten seeds.
- Percentile paired bootstrap interval with fixed RNG seed.
- Exact two-sided sign-flip test.
- Holm adjustment within each six-comparator Axis-1 family.
- Win counts reported throughout.

## Factorial mechanism experiments

The same four-arm XGBoost design is run for ten paired seeds on CIC-IDS2017 V3 and CIC-UNSW-NB15 V1:

- small natural;
- small balanced;
- large natural;
- large balanced.

The design separates fitting-set size from class distribution. Oversampled arms report nominal and unique-row counts.

## Operating-point and cost analysis

Every rare model-class pair is classified as recovered, high-recall/low-precision, high-precision/low-recall, or missed using precision and recall thresholds of 0.5. Alert burden is predicted rare-label alerts per correctly classified rare record. Cost is reconstructed per seed from weighted confusion matrices, and sensitivity is reported across severity vectors and incumbents.

## Axis 2: controlled few-shot language models

The same frozen protocol is applied separately to two source variants.

| Property | Grouped CIC-IDS2017 | CIC-UNSW-NB15 V1 |
|---|---:|---:|
| Declared labels | 9 | 10 |
| Development rows | up to 400/class | up to 400/class |
| In-context rows | 5/class (45 total) | 5/class (50 total) |
| Evaluation rows | up to 30/class; 246/seed | 30/class; 300/seed |
| Selected features | 12, development-only mutual information | 12, development-only mutual information |
| Paired seeds | 10 | 10 |

- Six hosted models use one frozen configuration registry.
- The registry was selected on blind grouped-CIC probes and rechecked on five blind CIC-UNSW probes per model; all six passed 5/5 for parsing and clean termination.
- The matched Random Forest has 200 trees, unrestricted depth, Gini impurity, bootstrap sampling, and the campaign seed; it receives exactly the same in-context rows and selected columns as the hosted models.
- A single deterministic label-only retry is permitted after an unparsable response.
- Parse failures remain incorrect predictions for their true class and are not added as a macro-averaged class.
- Macro-F1 is averaged over the declared source-specific label set; parse rate is separate.
- Bonferroni correction is applied within each source-specific family of six model-anchor comparisons.
- The two panels permit source-specific replication and rank comparison, not cross-dataset transfer.

## Grouped-CIC data-rich references

Strong Random Forest and XGBoost references are fitted within each grouped-CIC seed on the in-context rows plus either the development partition or a disjoint 400,000-row natural sample. They use the same twelve features and evaluation rows as the hosted panel and are reported separately from the six-comparison family.
