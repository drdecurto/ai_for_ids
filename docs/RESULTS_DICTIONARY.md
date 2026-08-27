# Results dictionary

## Axis 1

### `results/axis1/metrics.csv`

One row per dataset cell, model, seed, and evaluation set. Key columns: `cell`, `model`, `seed`, `test_set`, `accuracy`, `macro_f1`, `n_train`, timing fields, and `test_hash`.

### `results/axis1/per_class.csv`

Per-class precision, recall, F1, and weighted support.

### `results/axis1/confusion.csv`

Persisted confusion entries used to reconstruct operating points and miss/false-alarm cost crossovers.

### `results/axis1/cap_validation.csv`

Classical full-versus-cap scoring. Final analyses retain only rows whose parent partition hashes match.

## Grouped-CIC Axis 2

### `results/axis2/results_*.csv`

Per-seed model records containing accuracy, stored macro-F1, parse rate, API cost, latency, and wall time. Closed-label manuscript scores are independently reconstructed from raw generations.

### `results/axis2/raw_generations_*.jsonl`

Raw predictions and metadata for the grouped nine-class campaign. These are the source of truth for closed-label scoring and per-class results.

## Sensitivity records

- `exp1_size_by_balance.csv`: CIC-IDS2017 V3 four-arm XGBoost factorial.
- `exp2_llm_leakage.csv` and `exp2_raw.jsonl`: Kimi-K3 selector audit.
- `exp3_calibration_probe.csv` and `exp3_chosen_config.json`: blind grouped-CIC configuration negotiation.
- `exp4_manyclass.csv`: TabPFN ManyClass sensitivity.
- `exp5_size_by_balance_unsw.csv`: CIC-UNSW-NB15 V1 factorial.
- `exp6_axis2_datarich.csv` and `_perclass.csv`: grouped-CIC data-rich references.
- `stage1_*`: Random Forest selector audit and overlap diagnostics.

## Final-revision second Axis-2 panel

`results/round4/raw/axis2_second_variant/` contains:

- `results.csv`: 70 source run rows (seven models including the anchor × ten seeds), metrics, parse rate, cost, and latency;
- `raw_generations.jsonl`: 70 raw prediction records, each with 300 labels and predictions;
- `paired_tests.csv`: supplied paired summaries for the six hosted models;
- `stage_a_probe.csv`: frozen-registry revalidation, all six configurations passing 5/5;
- `stage_a_chosen_config.json`: frozen configuration tags;
- `rerun_audit/`: superseded pre-rerun Qwen3.5 seed-2/seed-3 records, an exact comparison table, and the provenance note for the unchanged-configuration reruns.

`results/round4/derived/` contains the independently reconstructed outputs:

- `axis2_second_variant_per_seed.csv`;
- `axis2_second_variant_summary.csv`;
- `axis2_second_variant_paired_tests.csv`;
- `axis2_second_variant_cost.csv`;
- `axis2_cross_variant_ordering.csv`;
- `axis2_second_variant_validation.json`;
- `round4_summary.json`.

## Generated outputs

`results/derived/` and `results/round4/derived/` can be regenerated from the persisted record. They contain strict-panel summaries, rank correlations, factorial contrasts, cap checks, operating points, cost crossovers, both Axis-2 panels, verification JSON, figures, and LaTeX fragments. `results/round4/generated/table_axis2_two_variants_round4.tex` is the compact current-model table used by the manuscript.
