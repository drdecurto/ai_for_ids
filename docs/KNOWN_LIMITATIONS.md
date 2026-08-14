# Known limitations

1. **Static, closed-label captures.** The release does not establish temporal drift robustness, open-set recognition, zero-day detection, or cross-dataset transfer.
2. **Raw-source hashing.** The original run bundles did not preserve content checksums for every downloaded benchmark table. Result-artifact hashing is complete; future raw redownload identity is not guaranteed.
3. **Evaluation cap.** Recall is class-conditional and is preserved under within-class sampling, while macro-F1 can be more sensitive to rare-label false positives. The manuscript therefore centres the confirmatory claim on paired rare recall.
4. **CIC cap validation.** Later full-scoring rows with non-matching partition hashes are excluded rather than treated as paired.
5. **TabSTAR.** Its CIC results use a different held-out partition and up to 4,000 rows per class; they are descriptive and never merged with strict-panel inference.
6. **ManyClass.** The sensitivity uses three seeds and four wrapper settings; it bounds but does not exhaust implementation dependence.
7. **Large balanced arm.** The factorial arm oversamples minority rows. Nominal and unique counts are both reported.
8. **Hosted endpoints.** Axis-2 latency, price, and output can change with serving configuration, model updates, or availability.
9. **Calibration size.** Configuration negotiation uses five development probes per candidate configuration.
10. **Statistical power.** Ten seeds improve exact-test resolution but do not establish equivalence when a comparison is non-significant.
11. **Cost vectors.** Uniform and severity-weighted costs are illustrative sensitivity assumptions, not calibrated organisational losses.
