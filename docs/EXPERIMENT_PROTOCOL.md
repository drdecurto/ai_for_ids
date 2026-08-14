# Final experiment protocol

## Axis 1: strict tabular panel

### Cells

- CIC-IDS2017 V1: 15 classes, no PCA.
- CIC-IDS2017 V3: grouped nine-class taxonomy, no PCA.
- CIC-UNSW-NB15 V1: ten classes, cleaned numeric features.
- CIC-UNSW-NB15 V2: ten classes, half-dimension PCA.

### Strict configurations

- Pruned Random Forest, large natural distribution.
- Strong 200-tree Random Forest, large natural distribution.
- 400-tree XGBoost, large natural distribution.
- The same three learners at the matched class-balanced budget.
- TabPFN 2.5 at the matched budget.
- TabICL at the matched budget.

TabSTAR is excluded from the strict panel because its CIC runs use a different held-out partition and up to 4,000 fitting rows per class.

### Split and sampling

- One stratified 70/30 split, split seed 2026.
- Seeds 0-9 vary fitting-row sampling and model stochasticity.
- Large natural budget: at most 400,000 rows at the observed class distribution.
- Matched balanced budget: approximately 2,000 rows per class, bounded by availability.
- Preprocessing is fitted only on the rows allowed to the corresponding budget.
- The weighted evaluation cap samples 5% per class with a 1,500-row floor; smaller classes are retained in full. Confusion rows are reweighted by inverse sampling fraction.

### Primary criterion

A non-benign class is rare when its weighted evaluation share is below 0.5%. The primary coverage outcome is the unweighted mean of classwise recall over that predeclared rare set.

### Statistical inference

- Mean paired difference over ten seeds.
- Percentile paired bootstrap interval with a fixed RNG seed.
- Exact two-sided sign-flip test.
- Holm adjustment within each six-comparator prior-fitted-model/cell family.
- Win counts are always reported.

## Factorial mechanism experiment

CIC-IDS2017 V3, XGBoost, ten paired seeds:

- small natural;
- small balanced;
- large natural;
- large balanced.

The experiment separates fitting-set size from class distribution. The large balanced arm uses oversampling and is reported with both nominal and unique-row counts.

## Operating-point and cost analysis

Every rare model-class pair is classified as:

- recovered: precision and recall >= 0.5;
- high recall / low precision;
- high precision / low recall;
- missed.

Alert burden is reported as predicted rare-label alerts per correctly classified rare record. Cost is reconstructed per seed from weighted confusion matrices. Positive crossover medians exclude dominance and negative-crossover cases.

## Axis 2: few-shot language models

- Grouped nine-class CIC-IDS2017 table.
- Up to 400 development rows per class.
- Five in-context rows per class.
- Up to 30 evaluation rows per class.
- Twelve features selected by mutual information on development rows only.
- Random Forest anchor receives the same in-context rows and selected columns.
- Model configuration is chosen on five development probes and must both parse and terminate cleanly.
- Ten paired seeds.
- Parse failures remain incorrect predictions for the true class.
- Macro-F1 is explicitly averaged over the declared closed label set.
- Parse rate is reported separately.
- Bonferroni correction covers the six language-model comparisons.
