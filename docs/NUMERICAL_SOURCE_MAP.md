# Numerical source map

All manuscript tables and figures are rebuilt by `scripts/recompute_manuscript_results.py`. Headline claims are checked by `scripts/verify_manuscript_claims.py`, and every statistical table is independently recomputed by `scripts/audit_manuscript_tables.py`.

| Manuscript component | Primary source record | Derived artifact |
|---|---|---|
| Table 2: dataset rows, classes, held-out support | `results/axis1/provenance.csv`, `per_class.csv` | `table_dataset_cells.tex` |
| Table 3: rare-class sets | `results/axis1/per_class.csv` | `table_rare_classes.tex` |
| Table 5: full-versus-cap validation | `metrics.csv`, `cap_validation.csv` | `cap_validation_matching_hash_rows.csv`, `table_cap_validation.tex` |
| Table 6: blind configuration negotiation | `results/sensitivity/exp3_calibration_probe.csv`, `exp3_chosen_config.json` | `axis2_calibration_probe.csv`, `table_axis2_calibration.tex` |
| Table 7 and Figure 2: accuracy versus rare recall | `metrics.csv`, `per_class.csv` | `axis1_rank_inversion.csv`, `table_rank_summary.tex`, `rank_inversion_core8.pdf` |
| Table 8: size-by-balance experiment | `results/sensitivity/exp1_size_by_balance.csv` | `size_balance_summary.csv`, `size_balance_contrasts.csv`, `table_size_balance_combined.tex` |
| Table 9: selected prior-fitted comparisons | `metrics.csv`, `per_class.csv` | `axis1_fm_vs_classical.csv`, `table_tabpfn_selected_pairs.tex` |
| Table 10: ManyClass sensitivity | `results/sensitivity/exp4_manyclass.csv` | `manyclass_summary.csv`, `table_manyclass.tex` |
| Table 11: stability and efficiency | `metrics.csv` | `axis1_core_summary.csv`, `table_stability_efficiency.tex` |
| Table 12: operating points | `per_class.csv`, `confusion.csv` | `operating_points_per_class.csv`, `operating_points_selected_summary.csv`, `table_operating_selected.tex` |
| Table 13: TabSTAR named-column control | TabSTAR and generic-name Axis-1 rows | `tabstar_named_vs_generic.csv`, `table_tabstar_control.tex` |
| Table 14: alarm-cost crossovers | `results/axis1/confusion.csv` | `cost_crossovers_per_seed.csv`, `cost_crossovers_summary.csv`, `table_cost_selected.tex` |
| Table 15 and Figure 3: Axis-2 closed-label panel | `results/axis2/raw_generations_*.jsonl`, campaign CSVs | `axis2_closed_label_summary.csv`, `axis2_paired_tests.csv`, `axis2_perclass_closed_label.pdf` |
| Table 16: feature-selection audit | `results/sensitivity/exp2_llm_leakage.csv`, `exp2_raw.jsonl` | `axis2_mi_llm_summary.csv`, `axis2_mi_llm_effects.csv`, `table_axis2_mi_combined.tex` |
| Independent table audit | all records above | `table_statistical_audit.json` |
| Released-artifact integrity | every file under `results/axis1`, `axis2`, `sensitivity`, and `derived` | `release_file_manifest_sha256.csv` |

## Audit boundary

The supplied runs did not retain complete SHA-256 values for every originally downloaded benchmark table. The release verifies all distributed result artifacts, records canonical dataset citations and final row/class counts, and recomputes the actual held-out support used by the reported runs. Hosted endpoint identities are represented by recorded model strings and frozen configurations; future endpoint identity is not guaranteed.
