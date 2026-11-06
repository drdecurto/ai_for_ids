# Round-5 re-execution package

Scripts used to re-execute the strict Axis-1 panel on a second platform (HPC cluster, NVIDIA H100,
Slurm) with probabilistic outputs and alternative held-out designs. See `docs/REVISION5_CHANGES.md`.

* `common.py` — vendored loaders (exact transcription of `src/data`: cleaning, counts asserted),
  the released protocol (row split, cap, budgets, transform, model settings), the chronological
  within-class split, the host-pair-disjoint split, and the metric functions (weighted confusion
  matrix, balanced accuracy, MCC, weighted average precision, attack AUPRC/ROC-AUC).
* `prepare_data.py download|build|audit` — raw-table acquisition from the pinned mirrors, cleaned
  parquet tables with row metadata, session statistics, split manifests, exact-duplicate audit.
* `run_panel.py` — one (cell, split, seeds, models) run; resumable; writes metrics, per-class
  records, weighted confusion matrices and `probs/*.npz`.
* `leakage_audit.py` — nearest-fitting-row audit and 1-NN memoriser recall.
* `aggregate.py` — summaries released under `results/round5/mn5/`.
* `jobs/make_jobs.sh` — Slurm job generator (classical panel on CPU nodes, prior-fitted models on
  GPU nodes, one seed per GPU for the fifteen-class cell).

Environment (pinned): torch 2.9.1+cu126, transformers 4.57.1, tabpfn 6.0.6, tabpfn-extensions 0.2.1,
tabicl 0.1.3, xgboost 3.1.1, scikit-learn 1.6.1, pandas 2.3.3, huggingface-hub 0.36.0.
Set `IDS_SCRATCH` to the working directory; raw tables are fetched from the mirrors recorded in
`common.py` (CIC-UNSW-NB15 at the pinned revision with SHA-256 verification). No credentials are
embedded; TabPFN weights require a Hugging Face login with the licence accepted.
