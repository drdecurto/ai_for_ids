# Notebook guide

## Analysis-only notebooks

- `notebooks/00_reproduce_manuscript_analysis.ipynb` rebuilds the original audited analysis layer from persisted records.
- `notebooks/01_reproduce_round4_revision.ipynb` rebuilds all final-revision derived CSVs and generated LaTeX tables, including the completed second Axis-2 panel. It requires no API calls or benchmark downloads.

## Experimental notebooks

- `notebooks/experiments/axis1_ten_seed_campaign.ipynb`: strict Axis-1 campaign.
- `notebooks/experiments/axis2_six_model_campaign.ipynb`: grouped-CIC current-model Axis-2 campaign.
- `notebooks/experiments/axis2_feature_selection_audit.ipynb`: Kimi-K3 selector audit.
- `notebooks/experiments/axis2_nemotron_configuration_rerun.ipynb`: corrected Nemotron configuration.
- `notebooks/experiments/sensitivity_experiments.ipynb`: earlier sensitivity experiments.
- `notebooks/experiments/revision4_experiments.ipynb`: complete Round-4 execution notebook, including the CIC-UNSW factorial, ten-seed ManyClass study, data-rich tree references, and the second current-model Axis-2 variant.
- `notebooks/experiments/revision4_experiments_completed.ipynb`: identical sanitized archival copy of the completed Round-4 notebook.
- `notebooks/experiments/revision4_experiments_completed_rerun.ipynb`: explicit archival alias for the final notebook after completion of the two initially interrupted Qwen3.5 jobs.
- `notebooks/experiments/round4_completed_extensions.ipynb`: compact record of the first three Round-4 add-on experiments; retained for continuity.

All released notebooks have outputs and execution counts cleared. API credentials are read from Colab Secrets or environment variables and are never stored in the notebook. Model re-execution is optional and environment-dependent; persisted CSV/JSON records are canonical for the manuscript.
