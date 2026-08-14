# Tabular Foundation Models and Few-Shot LLMs for IoT Intrusion Detection

The release preserves the original two-axis benchmark architecture and is organised around the final audited evidential record. The canonical entry point is:

```bash
make all
```

This command rebuilds every manuscript table and figure from persisted CSV/JSON outputs, recomputes the corrected closed-label Axis-2 metrics, runs an independent manuscript-claim verifier, performs a second table-by-table statistical audit, and executes the release tests.

## Evidence supported by the release

- The strict Axis-1 core contains **eight configurations**, four dataset-variant cells, and ten paired seeds.
- Accuracy and mean recall on the predeclared rare classes are negatively rank-correlated on all four cells.
- In every cell, the three highest-accuracy configurations are also the three lowest-rare-recall configurations.
- A $2\times2$ size-by-balance experiment identifies class distribution as the principal measured lever on CIC-IDS2017 V3.
- TabPFN 2.5 and TabICL are compared with six classical configurations in 48 paired comparisons: 40 favour a prior-fitted model on all ten seeds, four favour a classical model on all ten seeds, and four are mixed.
- TabSTAR is a separate descriptive extension because its CIC campaign uses a different held-out partition and a larger fitted budget.
- The Axis-2 panel contains six language models plus a matched Random Forest anchor. Kimi-K3 has the strongest point estimates, but no multiplicity-adjusted advantage over the anchor is established.

## Repository layout

```text
.
├── README.md
├── Makefile
├── requirements.txt
├── requirements-experiments.txt
├── data/
│   └── README.md
├── docs/
│   ├── DATA_PROVENANCE.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── KNOWN_LIMITATIONS.md
│   ├── NOTEBOOKS.md
│   ├── NUMERICAL_VERIFICATION.md
│   ├── NUMERICAL_SOURCE_MAP.md
│   ├── TABLE_AND_STATISTICAL_AUDIT.md
│   ├── REPRODUCTION_SMOKE_TEST.md
│   ├── RESULTS_DICTIONARY.md
│   └── SECURITY_AND_ANONYMITY.md
├── notebooks/
│   ├── 00_reproduce_manuscript_analysis.ipynb
│   └── experiments/
├── results/
│   ├── axis1/
│   ├── axis2/
│   ├── sensitivity/
│   └── derived/
├── scripts/
│   ├── recompute_manuscript_results.py
│   ├── verify_manuscript_claims.py
│   └── audit_manuscript_tables.py
└── tests/
    └── test_release.py
```

## Quick reproduction

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make all
```

Expected final checks:

```text
claim verifier: PASS
statistical table audit: PASS (15 standalone checks; 16 with manuscript source)
7 passed
```

### Google Colab

Upload or clone the repository, open:

```text
notebooks/00_reproduce_manuscript_analysis.ipynb
```

and run all cells. The analysis-only path needs no API key, GPU, or benchmark download.

## Reproducibility levels

1. **Exact manuscript analysis.** Fully reproducible from the included result files.
2. **Metric re-scoring.** Axis-2 raw generations are included, so closed-label macro-$F_1$, parse rate, per-class scores, paired effects, and cost totals can be recomputed independently.
3. **Model re-execution.** Sanitised source notebooks are retained under `notebooks/experiments/`. Re-execution requires the benchmark data, compatible GPUs/model packages, and/or hosted endpoints. Endpoint versions and availability may change; the persisted outputs are the record used by the paper.

## Main generated artifacts

After `make all`, inspect:

- `results/derived/verification_summary.json`
- `results/derived/manuscript_claim_verification.json`
- `results/derived/table_statistical_audit.json`
- `results/derived/axis1_core_summary.csv`
- `results/derived/axis1_fm_vs_classical.csv`
- `results/derived/size_balance_contrasts.csv`
- `results/derived/operating_points_per_class.csv`
- `results/derived/cost_crossovers_summary.csv`
- `results/derived/axis2_closed_label_summary.csv`
- `results/derived/axis2_paired_tests.csv`
- `results/derived/release_file_manifest_sha256.csv`

## Interpretation guardrails

- “Large” means at most 400,000 naturally distributed fitting rows, not the complete training partition.
- “Matched balanced” means approximately 2,000 fitting rows per class, bounded by availability.
- Rare classes are non-benign labels below 0.5% of the weighted evaluation distribution.
- Recall is interpreted together with precision, alerts per true positive, stability, and cost crossovers.
- TabSTAR is not inserted into the strict paired rank analysis.
- Parse failures remain errors for their true classes but are not treated as an additional macro-averaged label.
- Failure to establish a multiplicity-adjusted advantage is not interpreted as equivalence.
