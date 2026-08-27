# Benchmark data

The benchmark captures are **not redistributed** in this release.

The final manuscript uses two source families:

1. **CIC-IDS2017**, cited to Sharafaldin, Lashkari, and Ghorbani (2018). The release uses a cleaned flow table with 2,522,362 rows. Variant V1 retains the 15-class taxonomy; V3 groups related DoS, Patator, and Web-Attack labels into nine classes.
2. **CIC-UNSW-NB15**, the CICFlowMeter re-derivation of the UNSW-NB15 captures. The original dataset is cited to Moustafa and Slay (2015). The cleaned table contains 3,540,241 rows, nine attack classes, and benign. V1 uses the cleaned numeric feature space; V2 uses half-dimension PCA fitted within the declared fitting budget.

Public mirrors used during execution were transport sources rather than authoritative citations. For the CIC-UNSW-NB15 final-revision experiments, the notebook pinned and verified `CICFlowMeter_out.csv` at 1,914,305,081 bytes with SHA-256 `025a94239f29521e154be3a65d65268404d4c58600b66b94a4235eae87dbd8b5` before cleaning. Historical source tables do not all have an equivalent archived content hash, so the repository does not claim byte-level identity of every future redownload.

The release preserves:

- final row and class counts;
- experiment loaders and sanitised source notebooks;
- fitting/evaluation hashes recorded by the runs;
- released predictions, confusion matrices, calibration probes, raw generations, paired tests, and derived-artifact hashes for both current-model Axis-2 panels.

To rerun model fitting, place locally acquired tables where the selected experiment notebook expects them, or set the documented local-path variables. Never commit benchmark archives, credentials, or private download tokens.
