# Results dictionary

## `results/axis1/metrics.csv`

One row per cell, model, seed, and evaluation set.

Important columns include:

- `cell`: dataset-variant identifier.
- `model`: experiment configuration.
- `seed`: fitting/model seed.
- `test_set`: capped or full evaluation record.
- `accuracy`, `macro_f1`: aggregate metrics.
- `n_train`: nominal fitting instances.
- `fit_s`, `predict_s`: measured timings.
- `test_hash`: recorded partition identity.

## `results/axis1/per_class.csv`

Per-class precision, recall, F1, and support. The analysis script uses weighted per-class/confusion records where applicable.

## `results/axis1/confusion.csv`

Persisted confusion-matrix entries used to reconstruct rare-class operating points and miss/false-alarm cost crossovers.

## `results/axis1/cap_validation.csv`

Classical full-versus-cap scoring. The final analysis retains only rows whose parent partition hashes match.

## `results/axis2/results_*.csv`

Per-seed aggregate model records, including accuracy, the originally stored macro-F1, parse rate, API cost, latency, and wall time. The manuscript does **not** trust the stored macro-F1 blindly; it re-scores from raw generations.

## `results/axis2/raw_generations_*.jsonl`

Raw Axis-2 predictions and metadata. These files are the source of truth for corrected closed-label scoring and per-class results.

## `results/sensitivity/`

- `exp1_size_by_balance.csv`: four-arm XGBoost factorial.
- `exp2_llm_leakage.csv` and `exp2_raw.jsonl`: Kimi-K3 selector audit.
- `exp3_calibration_probe.csv`, `exp3_chosen_config.json`: blind configuration negotiation.
- `exp4_manyclass.csv`: TabPFN ManyClass sensitivity.
- `stage1_*`: Random Forest selector audit and overlap diagnostics.

## `results/derived/`

Regenerated outputs. These files can be deleted and recreated with the analysis script. The directory contains:

- strict-panel means and rank correlations;
- all 48 paired comparisons;
- factorial contrasts;
- cap-validation summaries;
- per-class operating points;
- cost crossovers;
- corrected Axis-2 metrics and tests;
- manuscript-ready figures and LaTeX fragments;
- `verification_summary.json`;
- `release_file_manifest_sha256.csv`.
