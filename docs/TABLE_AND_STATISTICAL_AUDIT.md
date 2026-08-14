# Table and statistical audit

## Scope

This audit covers every numerical/statistical table in the final 21-page manuscript **Tabular Foundation Models and Few-Shot LLMs for IoT Intrusion Detection**. The manuscript contains no appendices. Complete records that would otherwise require long auxiliary tables are distributed as CSV/JSON files in the repository.

The audit is independent of the table-generation script:

- table generation: `repository/scripts/recompute_manuscript_results.py`;
- independent recomputation: `repository/scripts/audit_manuscript_tables.py`;
- machine-readable result: `repository/results/derived/table_statistical_audit.json`.

Final standalone audit status: **PASS - 15 numerical checks** (Tables 2--16).
When the manuscript source is present beside the repository, the script performs a
16th check of headline manuscript literals and structural constraints.

## Table-by-table record

| Table | Content | Primary persisted source | Independent checks | Status |
|---:|---|---|---|:---:|
| 2 | Dataset-variant cells | `results/axis1/provenance.csv`, `per_class.csv` | Cleaned rows, class counts, held-out rows, benign shares, smallest attack supports | PASS |
| 3 | Predeclared rare classes | `results/axis1/per_class.csv` | Non-benign classes below 0.5% of weighted held-out distribution | PASS |
| 4 | Axis-1 model panel and budgets | Protocol and run metadata | Eight strict configurations; TabSTAR identified as separate extension | PASS |
| 5 | Evaluation-cap validation | `metrics.csv`, `cap_validation.csv` | Matching parent-partition hashes; mean and maximum cap-minus-full errors | PASS |
| 6 | Axis-2 frozen configurations | `results/sensitivity/exp3_calibration_probe.csv`, `exp3_chosen_config.json` | Six configurations; 5/5 clean development probes; clean stop reason | PASS |
| 7 | Accuracy and rare-recall ranking | `metrics.csv`, `per_class.csv` | Four Spearman values; same top-three accuracy and bottom-three recall sets | PASS |
| 8 | Size-by-balance factorial | `results/sensitivity/exp1_size_by_balance.csv` | Four arm means, paired effects, 50,000-sample bootstrap intervals, exact tests, Holm adjustment | PASS |
| 9 | Selected prior-fitted pairwise effects | `metrics.csv`, `per_class.csv` | Complete 48-comparison record; 40/4/4 direction count; family-level Holm values | PASS |
| 10 | ManyClass sensitivity | `results/sensitivity/exp4_manyclass.csv` | All 12 runs; means and standard deviations; rare-recall range 0.00610 | PASS |
| 11 | Stability and computational placement | `metrics.csv`, `axis1_core_summary.csv` | Rare-recall means/SD, fitted rows, fit time, prediction time | PASS |
| 12 | Selected rare-class operating points | `per_class.csv`, `confusion.csv` | Recovered/HR-LP/HP-LR/missed counts and seed-averaged alerts per true positive | PASS |
| 13 | TabSTAR named-column control | TabSTAR and generic-name rows | Paired effects, intervals, wins, exact tests for accuracy, macro-$F_1$, rare recall | PASS |
| 14 | Alarm-cost crossovers | `results/axis1/confusion.csv` | 560 seed-level comparisons; uniform and severity-weighted scenarios; selected medians | PASS |
| 15 | Axis-2 six-model panel | Raw generation JSONL and campaign CSVs | 60 LLM seed records; closed-label scores; parse rates; costs; six adjusted paired tests | PASS |
| 16 | Axis-2 selector audit | `results/sensitivity/exp2_llm_leakage.csv`, `exp2_raw.jsonl` | 30 Kimi-K3 runs; 387 identical rows/seed; paired selector effects | PASS |

## Headline consistency checks

The audit additionally confirms:

- strict Axis-1 record: **320** records;
- rank correlations: **-0.452, -0.452, -0.429, -0.476**;
- prior-fitted comparison directions: **40 all-seed wins, 4 all-seed losses, 4 mixed**;
- factorial effects: **+0.509, +0.260, +0.242, -0.007**;
- Axis-2 total API cost: **USD 218.3418699**;
- Kimi-K3 versus RF closed-label macro-$F_1$: **+0.02803**, adjusted **p=0.11719**;
- CIC-IDS2017 held-out benign share: **83.1158%**;
- when the manuscript source is present, it contains no `\appendix` command and no unsupported harsh disclaimer wording.

## Corrected inherited values

The final audit identified and corrected two values inherited from earlier drafts:

1. The CIC-IDS2017 held-out benign share is **83.1%**, computed from 628,945 benign rows among 756,709 held-out rows. The earlier 80.2% figure described a different source-level distribution and did not correspond to the actual fixed partition used by the reported runs.
2. Axis-2 macro-$F_1$ is computed over the declared closed label set. Unparsed generations remain incorrect predictions for their true classes and are summarized separately through parse rate; they are not inserted as an additional macro-averaged class.

## Reproduction command

```bash
cd repository
make all
```

Expected final output:

```text
claim verifier: PASS
statistical table audit: PASS (15 standalone checks; 16 with manuscript source)
7 passed
```
