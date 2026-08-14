# Data provenance and audit boundary

## Authoritative citations

- CIC-IDS2017: Sharafaldin, Lashkari, and Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, 2018.
- UNSW-NB15: Moustafa and Slay, *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*, 2015.

The manuscript uses a CICFlowMeter re-derivation of UNSW-NB15 and names it CIC-UNSW-NB15 to avoid conflating the derived feature table with the original 49-feature release.

## Preserved in this release

- cleaned row and class counts;
- final metrics, per-class records, confusion matrices, and cap-validation records;
- raw Axis-2 generations;
- calibration choices and sensitivity results;
- recorded split/test hashes where the experiment emitted them;
- SHA-256 hashes for every released result artifact.

## Not preserved in the supplied runs

- a complete raw-source SHA-256 record for every downloaded benchmark table;
- redistribution rights for the benchmark captures;
- immutable identities for future hosted model endpoints.

The paper and repository make no stronger provenance claim than this record supports.
