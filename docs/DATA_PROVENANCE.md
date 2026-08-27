# Data provenance and audit boundary

## Authoritative citations

- CIC-IDS2017: Sharafaldin, Lashkari, and Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, 2018.
- UNSW-NB15: Moustafa and Slay, *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*, 2015.

The manuscript uses a CICFlowMeter re-derivation of UNSW-NB15 and names it CIC-UNSW-NB15 to avoid conflating the derived feature table with the original 49-feature release.

## Preserved in this release

- cleaned row and class counts;
- final metrics, per-class records, confusion matrices, and cap-validation records;
- raw hosted-model generations for both current-model Axis-2 sources;
- the complete second-source run ledger, calibration probes, frozen configuration registry, paired tests, parse failures, latency, and costs;
- calibration choices and sensitivity results;
- recorded split/test hashes where an experiment emitted them;
- SHA-256 hashes for every released result artifact.

The completed CIC-UNSW-NB15 Axis-2 record contains 70 source-specific run records: six hosted models and one matched Random Forest across ten paired seeds. Each record contains 300 scored predictions. Qwen3.5 completes all ten seeds with parse rate 1.000. Two jobs that were interrupted before any completion was returned were rerun under the unchanged frozen configuration; the superseded pre-rerun ledger and generations are retained under `axis2_second_variant/rerun_audit/`, while the parent directory contains the canonical completed record.

## Raw-table acquisition record

The final-revision notebook acquired `CICFlowMeter_out.csv` from a pinned mirror and checked the file before cleaning against byte count 1,914,305,081 and SHA-256 `025a94239f29521e154be3a65d65268404d4c58600b66b94a4235eae87dbd8b5`. The cleaned CIC-UNSW-NB15 table contains 3,540,241 rows and ten classes. The benchmark file itself is not redistributed.

## Audit boundary

The supplied historical runs do not preserve a complete raw-source SHA-256 record for every benchmark table, redistribution rights for the benchmark captures, or immutable identities for future hosted-model endpoints. The paper and repository make no stronger provenance claim than the persisted record supports.
