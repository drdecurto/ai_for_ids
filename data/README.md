# Benchmark data

The benchmark captures are **not redistributed** in this release.

The final manuscript uses two source families:

1. **CIC-IDS2017**, cited to Sharafaldin, Lashkari, and Ghorbani (2018). The release uses a cleaned flow table with 2,522,362 rows. Variant V1 retains the 15-class taxonomy; V3 groups related DoS, Patator, and Web-Attack labels into nine classes.
2. **CIC-UNSW-NB15**, the CICFlowMeter re-derivation of the UNSW-NB15 captures. The original dataset is cited to Moustafa and Slay (2015). The cleaned table contains 3,540,241 rows, nine attack classes, and benign. V1 uses the cleaned numeric feature space; V2 uses half-dimension PCA fitted within the declared fitting budget.

Public mirrors used during execution were transport sources rather than authoritative citations. Because the original uploaded run bundles did not preserve raw-source content hashes for every cell, this repository does **not** claim byte-level identity of a future redownload. It does preserve:

- final row and class counts;
- the experiment loaders and source notebooks;
- fitting/evaluation hashes recorded by the runs;
- all released predictions, confusion matrices, raw generations, and derived-artifact hashes.

To rerun model fitting, place locally acquired tables where the selected experiment notebook expects them, or adapt the loader cell to a local path. Never commit benchmark archives, credentials, or private download tokens.
