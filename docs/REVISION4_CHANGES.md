# Revision-4 changes

All final-revision tables are rebuilt by:

```bash
python scripts/recompute_round4_revision.py
```

The script validates the complete input record and invokes `scripts/recompute_axis2_second_variant.py` for independent reconstruction of the newly completed hosted-model panel.

## New and updated result records

| File | Status | Produced by |
|---|---|---|
| `results/sensitivity/exp4_manyclass.csv` | extended from three to ten seeds; the released three-seed rows are retained verbatim | Round-4 notebook, Experiment 4b |
| `results/sensitivity/exp5_size_by_balance_unsw.csv` | new ten-seed $2\times2$ XGBoost factorial on CIC-UNSW-NB15 V1 | Experiment 5 |
| `results/sensitivity/exp6_axis2_datarich.csv` and `_perclass.csv` | new strong-RF and XGBoost references on grouped-CIC Axis-2 rows | Experiment 6 |
| `results/round4/raw/axis2_second_variant/` | complete CIC-UNSW-NB15 V1 current-model panel: 70 run records, 70 raw-generation records, calibration probes, paired tests, and frozen registry | Experiment 7 |
| `results/round4/derived/` | all final-revision derived CSVs and `round4_summary.json` | analysis scripts |
| `results/round4/generated/table_axis2_two_variants_round4.tex` | compact cross-source Axis-2 manuscript table | `recompute_axis2_second_variant.py` |

## Source-map additions

| Manuscript component | Primary record | Derived artifact |
|---|---|---|
| Threshold sensitivity | `results/axis1/per_class.csv`, `metrics.csv` | `rare_threshold_sensitivity.csv`, generated threshold table |
| Cap validation and representative crossovers | Axis-1 metrics/confusions | `cap_operational_*`, `cap_crossover_*`, compact cap table |
| Core panel with capped/full macro-F1 | Axis-1 metrics, hash-matched full rows | `core_panel_round4.csv`, core table |
| Severity/incumbent sensitivity | Axis-1 confusions and per-class scores | `cost_crossover_all_vectors_*`, severity table |
| Rare-class precision/recall/support | Axis-1 per-class scores | `rare_per_class_operating_points.csv`, generated tables |
| ManyClass sensitivity | `exp4_manyclass.csv` | `manyclass_ten_seed_summary.csv`, paired tests |
| Two-source factorial | `exp1_size_by_balance.csv`, `exp5_size_by_balance_unsw.csv` | `factorial_two_source_*` |
| Grouped-CIC data-rich references | `exp6_axis2_datarich*.csv` | `axis2_datarich_*` |
| Grouped-CIC current-model Axis 2 | `results/axis2/raw_generations_*.jsonl`, campaign CSVs | `results/derived/axis2_closed_label_*` |
| CIC-UNSW current-model Axis 2 | `results/round4/raw/axis2_second_variant/*` | `axis2_second_variant_*`, `axis2_cross_variant_ordering.csv` |
| Preliminary Axis-1 screening | five-seed screening record | `legacy_omitted_cells.csv` |
| Preliminary Axis-2 screening | $k$-shot, scaling, feature-budget, cost records | `legacy_axis2_*.csv`, compact screening table |

## Second Axis-2 panel

The frozen six-model configuration registry was rechecked on five blind CIC-UNSW-NB15 probes per model; all six configurations parsed and terminated cleanly on all five probes. Ten paired seeds then scored 300 rows per seed for each hosted model and matched Random Forest. Two Qwen3.5 jobs interrupted before any completion was returned were rerun under the unchanged frozen configuration and both completed successfully. Independent rescoring reproduces the complete run ledger, paired tests, updated costs, and a hosted-model Spearman ordering of 1.000 relative to grouped CIC-IDS2017.

## Data provenance

The CIC-UNSW-NB15 raw table used by Experiments 5 and 7 was acquired through the notebook's pinned mirror and verified before cleaning against byte count 1,914,305,081 and SHA-256 `025a94239f29521e154be3a65d65268404d4c58600b66b94a4235eae87dbd8b5`. The cleaned table contains 3,540,241 rows and ten classes. The canonical CIC source remains cited in the manuscript.

## Concise manuscript displays

The 28-page article consolidates related evidence without deleting source data. The cap-validation table draws from the operational and crossover records; the preliminary Axis-2 table draws from the full screening grids; and Table 23 combines the two complete current-model panels. All granular tables remain under `results/round4/generated/`.

## Final-v4 consistency pass

The manuscript-facing release now uses **preliminary screening** consistently
for the five-seed evidence tier. Hosted-model parse accounting is stated
explicitly: rates span 0.973--1.000 on grouped CIC-IDS2017 and 0.958--1.000 on
CIC-UNSW-NB15 V1, with unparsed responses retained in the denominator as
errors. The 36 exact final manuscript table fragments are stored in
`results/manuscript_tables/` and synchronized byte-for-byte into
`results/derived/` by `scripts/sync_final_manuscript_tables.py`; the release
test suite verifies that mirror.
