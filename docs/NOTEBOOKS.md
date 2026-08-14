# Notebook guide

## Canonical notebook

`00_reproduce_manuscript_analysis.ipynb`

- clean, restart-and-run;
- Google Colab compatible;
- installs analysis dependencies;
- requires no credentials;
- regenerates all derived tables and figures;
- runs release tests;
- ships without execution outputs or environment-specific paths; run all cells to regenerate them.

## Experiment-source notebooks

The notebooks under `notebooks/experiments/` are sanitised code records from the completed campaigns. Outputs, execution counters, credentials, and named repository URLs have been removed. They are retained to document model calls, loaders, fitting loops, and recovery logic.

They are **not** the canonical path for verifying manuscript numerics. Some were developed incrementally and contain recovery or continuation cells. Use the persisted results plus the canonical analysis notebook for exact reproduction. Independent model re-execution should consolidate the relevant cells, pin model/package versions, and record new raw-data hashes.

## Released experiment notebooks

- `axis1_ten_seed_campaign.ipynb`: strict Axis-1 and cap-validation campaign.
- `axis2_six_model_campaign.ipynb`: six-model Axis-2 campaign.
- `axis2_nemotron_configuration_rerun.ipynb`: calibrated Nemotron rerun.
- `axis2_feature_selection_audit.ipynb`: pooled/development/ICL selector audit.
- `sensitivity_experiments.ipynb`: size-by-balance and ManyClass sensitivity code.

Internal editorial-analysis notebooks and reviewer-specific decision cells are not part of
the public release because they are not required to reproduce any reported result.
