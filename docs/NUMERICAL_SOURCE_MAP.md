# Numerical source map

Headline claims are checked by `scripts/verify_manuscript_claims.py`; all statistical displays are independently recomputed by `scripts/audit_manuscript_tables.py`.

| Manuscript component | Primary source record | Derived artifact |
|---|---|---|
| Dataset cells and rare classes | `results/axis1/provenance.csv`, `per_class.csv` | dataset and rare-class LaTeX tables |
| Cap validation | Axis-1 metrics, cap validation, confusion rows | matching-hash and compact cap outputs |
| Axis-1 ranking and core panel | Axis-1 metrics and per-class scores | rank/core summaries and Figure 2 |
| Two-source size-by-balance mechanism | `exp1_size_by_balance.csv`, `exp5_size_by_balance_unsw.csv` | two-source factorial summaries/tables |
| ManyClass sensitivity | `results/round4/raw/exp4_manyclass.csv` | ten-seed summary and paired tests |
| Operating points and costs | Axis-1 per-class/confusion rows | precision-aware and cost-vector outputs |
| Preliminary screening | `results/development_screening/*.csv` | `legacy_*` summaries and compact screening table |
| Grouped-CIC current-model Axis 2 | `results/axis2/raw_generations_*.jsonl`, campaign CSVs | `results/derived/axis2_closed_label_*`, per-class figure |
| CIC-UNSW current-model Axis 2 | `results/round4/raw/axis2_second_variant/*` | `axis2_second_variant_*`, validation JSON |
| Cross-source Axis-2 table | both current-model source summaries | `axis2_cross_variant_ordering.csv`, `table_axis2_two_variants_round4.tex` |
| Grouped-CIC data-rich references | `exp6_axis2_datarich*.csv` | `axis2_datarich_*` and Table 24 |
| Selector audit | `exp2_llm_leakage.csv`, `exp2_raw.jsonl` | selector summary/effects and Table 26 |
| Independent audit | all records above | `results/derived/table_statistical_audit.json` |
| File integrity | every released result and source artifact | release manifests at derived and repository levels |

## Audit boundary

The release verifies all distributed result artifacts and records canonical dataset citations, cleaning rules, row/class counts, and available raw-source hashes. Hosted endpoint identities are represented by recorded model strings and frozen configurations; future provider identity, behaviour, availability, and pricing are not guaranteed.
