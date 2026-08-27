# Known limitations

1. **Static benchmark sources.** The confirmatory evidence covers four Axis-1 cells from two cleaned sources and two Axis-2 variants; temporal drift, zero-day attacks, and open-set recognition are not evaluated.
2. **Source-specific Axis-2 panels.** Applying the same few-shot protocol to two sources is not cross-dataset transfer. No model is trained or prompted on one source and tested on the other.
3. **Small-class resolution.** The fixed split leaves very small held-out supports for classes such as Heartbleed, producing coarse recall steps.
4. **Fixed held-out partition.** Seed intervals vary fitted rows, model stochasticity, and capped evaluation subsamples, not the deployment population or split seed.
5. **Evaluation cap.** Rare recall is preserved closely, but precision-, alert-, and cost-dependent quantities can be more sensitive to the cap.
6. **Oversampling.** Large balanced factorial arms reuse minority rows; nominal row count is reported separately from unique-row count.
7. **ManyClass wrapper.** The ten-seed sensitivity supports limited configuration robustness on the evaluated fifteen-class cell, not invariance to every decomposition.
8. **Hosted endpoints.** Accuracy, parse rate, latency, availability, and price can change with provider updates. The final ledger is complete, but two Qwen3.5 jobs required an unchanged-configuration rerun after interruption before any completion was returned; the pre-rerun and completed records are preserved for audit.
9. **Multiplicity and equivalence.** Failure to establish corrected superiority does not establish equivalence.
10. **Data-rich references.** Development-rich tree references are available on grouped CIC-IDS2017 only; they are included to distinguish matched few-shot comparison from practical labelled-data competitiveness.
11. **Cost assumptions.** Miss severities are illustrative sensitivity vectors rather than measured organisational losses.
