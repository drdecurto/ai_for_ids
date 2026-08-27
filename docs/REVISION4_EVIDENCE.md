# Final-revision evidence audit

## Included confirmatory and robustness evidence

- Four strict Axis-1 cells, eight configurations, ten seeds: 320 observations.
- Rare-class prevalence thresholds of 0.1%, 0.25%, 0.5%, and 1%.
- Ten-seed size-by-balance factorials on CIC-IDS2017 V3 and CIC-UNSW-NB15 V1.
- Ten-seed ManyClass sensitivity for four alphabet/redundancy settings.
- Hash-matched capped/full comparisons for recall, precision, alerts per true positive, false alarms, macro-F1, and selected cost crossovers.
- Complete CIC-IDS2017 V1 rare-class precision, recall, and support profiles.
- Alternative severity vectors and strong-incumbent cost crossovers.
- Grouped-CIC development-rich and large-natural Random Forest/XGBoost Axis-2 references on the same features and evaluation rows as the hosted campaign.
- Two complete current-model Axis-2 panels under one frozen protocol: grouped CIC-IDS2017 and CIC-UNSW-NB15 V1, each with six hosted models, a matched Random Forest, ten paired seeds, calibration probes, raw generations, reliability accounting, cost, and corrected paired tests.

## Included as preliminary screening

The six Axis-1 cells outside the strict ten-seed panel and the five-seed $k$-shot, scaling, feature-budget, and endpoint-cost records are retained under `results/development_screening/`. Their seed depth, model panels, or partitions differ from the confirmatory record. They are reported descriptively and excluded from confirmatory confidence intervals and multiplicity families.

## Axis-2 scope

The two current-model panels are source-specific controlled few-shot evaluations. They support cross-source rank comparison, but they do not test training on one source and evaluation on another, temporal drift, zero-day behaviour, or open-set detection. The final CIC-UNSW-NB15 ledger is complete for every scheduled model-seed job. Qwen3.5 reaches parse rate 1.000 after the two initially interrupted jobs are rerun under the unchanged frozen configuration; the audit copies of the superseded records remain available for provenance.

## Canonical outputs

The completed add-on records under `results/round4/raw/`, the two Axis-2 raw-generation records, and the pre-existing audited result record are the canonical numerical inputs. `scripts/recompute_round4_revision.py` validates the grids and invokes `scripts/recompute_axis2_second_variant.py` to reconstruct the second Axis-2 panel independently from raw generations.
