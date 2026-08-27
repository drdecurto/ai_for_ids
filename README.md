# Tabular Foundation Models and Few-Shot LLMs for IoT Intrusion Detection

This release contains the complete persisted record and deterministic analysis code used by the final revision. The evidence is organised as in the self-contained manuscript:

1. **Preliminary screening:** five-seed results across ten Axis-1 cells and complementary few-shot protocol sweeps establish heterogeneity and boundary conditions.
2. **Confirmatory record:** four benign-dominated Axis-1 cells have complete paired ten-seed records. Axis 2 contains two complete source-specific, six-model, ten-seed campaigns: grouped CIC-IDS2017 and CIC-UNSW-NB15 V1.
3. **Targeted robustness evidence:** threshold sensitivity, two size-by-balance factorials, evaluation-cap checks, ten-seed ManyClass sensitivity, per-class operating points, cost vectors, and grouped-CIC data-rich tree references constrain the claims.

The second Axis-2 campaign is complete. Its calibration probes, run ledger, raw generations, paired tests, and cost record are released under `results/round4/raw/axis2_second_variant/`; no partial calls are used.

## Concise manuscript presentation

The anonymous manuscript presents the complete evidence in 28 pages. Cap validation is consolidated into one two-panel table, the preliminary Axis-2 sweeps into one compact table, and the two current-model Axis-2 campaigns into one cross-source table. Granular CSVs and generated tables remain available so that concision does not reduce auditability. The exact current manuscript fragments are mirrored in `results/manuscript_tables/` and `results/derived/`. See `docs/MANUSCRIPT_TABLE_MAP_ROUND4_CONCISE.md`.

## Principal evidence

- The strict Axis-1 record contains **320 seed-level observations**: four cells, eight model-budget configurations, and ten paired seeds.
- Accuracy and mean rare-class recall remain negatively rank-correlated at prevalence thresholds of 0.1%, 0.25%, 0.5%, and 1% in all 16 cell-threshold combinations.
- The same three accuracy leaders are the three rare-recall laggards in every cell-threshold combination.
- Two ten-seed $2\times2$ XGBoost experiments separate row budget from class distribution on CIC-IDS2017 V3 and CIC-UNSW-NB15 V1.
- The ManyClass sensitivity record contains four settings and ten seeds each.
- Axis 2 evaluates six hosted models and matched Random Forest anchors on grouped CIC-IDS2017 (45 fitting rows, 246 evaluation rows per seed) and CIC-UNSW-NB15 V1 (50 fitting rows, 300 evaluation rows per seed).
- Kimi-K3 leads both current-model panels, followed by GLM-5.2, but no positive model-anchor contrast survives six-comparison correction. Hosted-model ranks are identical across sources (Spearman $\rho_s=1.000$), while absolute performance is lower on CIC-UNSW-NB15.
- Hosted-model parse rates span 0.973--1.000 on grouped CIC-IDS2017 and 0.958--1.000 on CIC-UNSW-NB15 V1; unparsed responses remain in the denominator and are scored as errors.
- The two hosted campaigns cost USD 218.3418699 and USD 293.68404155 (USD 512.02591145 combined).
- Grouped-CIC development-rich Random Forest and XGBoost references distinguish controlled few-shot competitiveness from a practical labelled-data comparison.
- Macro-$F_1$, per-class precision/support, alerts per true positive, cap/full checks, and severity-weighted cost crossovers constrain interpretation of rare-class recall.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make all
```

The final-revision layer alone can be regenerated with:

```bash
make round4
```

No benchmark download, GPU, API key, or hosted-model call is required for analysis reproduction. Completed seed-level outputs and raw Axis-2 generations are included.

## Repository layout

```text
.
├── README.md
├── Makefile
├── data/
├── docs/
├── notebooks/
│   ├── 00_reproduce_manuscript_analysis.ipynb
│   ├── 01_reproduce_round4_revision.ipynb
│   └── experiments/
│       ├── revision4_experiments.ipynb
│       └── revision4_experiments_completed.ipynb
├── results/
│   ├── axis1/
│   ├── axis2/
│   ├── sensitivity/
│   ├── derived/
│   ├── round4/
│   │   ├── raw/axis2_second_variant/
│   │   ├── derived/
│   │   └── generated/
│   └── development_screening/
├── scripts/
│   ├── recompute_manuscript_results.py
│   ├── recompute_round4_revision.py
│   ├── recompute_axis2_second_variant.py
│   ├── verify_manuscript_claims.py
│   └── audit_manuscript_tables.py
└── tests/
```

## Reproducibility levels

1. **Exact numerical analysis:** fully reproducible from the included CSV/JSON record.
2. **Axis-2 rescoring:** raw generations from both current-model campaigns are included, allowing closed-label metrics, parse rates, model-anchor paired tests, and costs to be recomputed. All scheduled Qwen3.5 seeds in the CIC-UNSW-NB15 panel are complete; the preserved rerun audit documents the two initially interrupted jobs and their unchanged-configuration completion.
3. **Model re-execution:** source notebooks are retained but require the original benchmark sources, compatible package versions, GPUs and/or hosted endpoints. Provider versions and prices can change; the persisted outputs are the article's canonical record. Credentials are read only from environment variables or notebook secret stores and are not embedded in the release.

## Interpretation guardrails

- “Large natural” means at most 400,000 naturally distributed fitting rows, not the entire training partition.
- “Matched balanced” means approximately 2,000 fitting rows per class, bounded by availability.
- Rare-class recall is a class-unweighted mean over non-benign labels below the declared prevalence threshold.
- The fixed test split gives extremely small classes coarse recall resolution; seed intervals describe fitting/model variation, not resampling of the deployment population.
- The evaluation cap is reliable for the central recall comparison but is not treated as an exact substitute for full scoring on precision, alert burden, or cost.
- TabSTAR and preliminary cells are not inserted into the strict paired rank analysis.
- Failure to establish multiplicity-adjusted superiority is not interpreted as equivalence.
- The two Axis-2 panels are source-specific few-shot evaluations, not a cross-dataset-transfer experiment.
- Data-rich tree references are available on grouped CIC-IDS2017 only and answer a different information-budget question from the matched five-shot anchors.
