# Revision 5 — changes to the repository

This revision answers the second-round comments of two reviewers. It adds two layers of
evidence without changing any previously released record.

## 1. Analyses derived from the released record (no re-execution)

`scripts/round5_record_analysis.py` reads `results/axis1/confusion.csv`, `results/axis1/per_class.csv`,
the Axis-2 per-seed records and the factorial records, and writes `results/round5/record/`:

| File | Content |
|---|---|
| `metrics_extended_per_seed.csv` | balanced accuracy, MCC, Cohen's κ, G-mean, macro-F1, rare-class recall/precision/F1, attack recall, false-alarm rate, missed attacks and false alarms per 10^5 flows — per cell, configuration and seed, from the weighted confusion matrices |
| `metrics_extended_summary.csv` | means, SDs and within-cell ranks of the above |
| `baseline_and_inversion_magnitude.csv` | constant-benign accuracy, accuracy headroom retained by the panel, accuracy cost and rare-recall gain of the recall leader, operational counts |
| `metric_rank_correlations.csv` | Spearman rank correlation of each summary with accuracy and with rare-class recall; leaders |
| `effect_sizes_axis1.csv` | all 48 prior-fitted vs classical contrasts (rare recall, balanced accuracy, MCC, macro-F1, accuracy): mean Δ, bootstrap CI, SD of the paired differences, d_z, MDE80, wins, sign-flip p, Holm p |
| `effect_sizes_factorial.csv` | the eight factorial contrasts with the same quantities |
| `effect_sizes_axis2.csv` | the twelve Axis-2 model–anchor contrasts with the same quantities and Bonferroni p |
| `llm_applicability.csv` | cost and throughput of each hosted model extrapolated to one pass over the full held-out partitions |
| `recommended_operating_points.csv` | per cell and objective, the recommended configuration and its operational counts |
| `round5_record_summary.json` | headline numbers (permutation test of the inversion, MDE summary, baseline rows) |

`scripts/make_round5_tables.py` renders the manuscript tables (`results/round5/manuscript_tables/round5_*.tex`).

## 2. Re-execution on a second platform (`notebooks/experiments/round5_reexecution/`)

The strict Axis-1 panel was re-executed on an HPC cluster (NVIDIA H100) with the vendored cleaning
code (`common.py` transcribes `src/data` exactly; cleaned-table counts are asserted) and the
unchanged protocol, with four additions:

* `predict_proba` is stored for every run on the capped evaluation rows (`probs/*.npz`), from which
  per-class average precision, macro-AP over the rare set / attack classes / all classes, the binary
  attack-vs-benign AUPRC and ROC-AUC, and precision–recall curves are computed;
* five held-out designs: the reference row-level partition (`row2026`), two alternative stratified
  row-level partitions (`row2027`, `row2028`), a chronological within-class/within-capture-day split on
  parsed flow-start timestamps (`chrono`: earliest 70% of each class–day stratum fit, latest 30% held
  out; `temporal_check.py` verifies that no held-out flow starts before any fitting flow of its stratum,
  `temporal_boundary_check.csv`), and — on CIC-UNSW-NB15 only, where every class spans 39–40
  attacker–victim host pairs (11 for Analysis) — a host-pair-disjoint split (`hostpair`). A sixth
  design kept in the record but not in the manuscript, `temporal`, cuts each stratum at 70% of
  *capture-file order*; the timestamp audit showed that file order is not chronological for the
  benign class (the Monday–Wednesday files are only 60–76% time-ordered and the Thursday/Friday
  afternoon files sort before the morning files) nor for the grouped V3 classes, so its results are
  superseded by `chrono`;
* a leakage audit: exact feature-row duplicates across the partitions (with label-conflicting
  duplicates counted separately), nearest-fitting-row distances of every capped held-out row in the
  standardised feature space of each budget, and the recall of a one-nearest-neighbour memoriser
  (`leakage_audit.py`);
* session statistics of every class from the host/timestamp columns of the original flow releases
  (`prepare_data.py audit`; CIC-IDS2017 from GeneratedLabelledFlows, CIC-UNSW-NB15 from the
  CICFlowMeter output). The GeneratedLabelledFlows files align row by row with MachineLearningCVE
  for the Monday–Wednesday captures only; the CIC chronological split therefore uses capture-file
  order, which is the flow-termination order written by CICFlowMeter.

Outputs are released under `results/round5/mn5/` (`panel_per_seed.csv`, `panel_summary.csv`,
`inversion_stats.csv`, `contrast_by_split.csv`, `pr_curves.csv`, `leakage_all.csv`,
`session_structure.csv`, `split_manifest.csv`, `split_per_class.csv`, `temporal_boundary_check.csv`) together with the derived
`platform_replication.csv`, `design_shift.csv`, `partition_sd.csv` and `mn5_summary.json`
(`scripts/integrate_mn5.py`). The probability arrays (≈1 GB) are not redistributed; the code that
produced them and the SHA-256 of each file are listed in `results/round5/mn5/probs_manifest.csv`.

## 3. Manuscript

New or modified passages appear in colour in the marked manuscript. New sections: complementary summaries (3.3), held-out designs (3.5), effect sizes and
MDE (3.8), magnitude/metric families (4.1), robustness to the held-out design (4.4), few-shot
competitiveness vs applicability (5.4), practitioner guidance (6.1).

## 4. Corrections made during the revision

* `llm_applicability.csv` now takes the Nemotron-3 Ultra cost and latency from the corrected
  configuration rerun (`results/axis2/results_nemotron_corrected.csv`), the same record as the Axis-2
  summary table, and reports extrapolated hours both at the median endpoint latency and at the
  campaign wall rate (which includes queueing and retries).
* `recommended_operating_points.csv` separates "highest accuracy" from "fewest false alarms"; the
  pruned large forest emits the fewest false alarms in every cell, the accuracy leader does not.
* The vendored CIC-IDS2017 cleaning fills `Flow Bytes/s` and `Flow Packets/s` with whole-table medians
  before partitioning (1,564 of 2,522,362 cleaned rows, 0.06%); `temporal_check.py` counts them
  (`cic_imputation_counts.csv`).

