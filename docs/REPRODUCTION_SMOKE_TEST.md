# Reproduction smoke test

The packaged analysis was executed from the repository root with:

```bash
make all
```

The run:

- regenerated the base and final-revision derived tables and figures;
- independently reconstructed the completed CIC-UNSW-NB15 Axis-2 panel from 70 raw-generation records;
- passed the manuscript-claim verifier;
- passed the independent 16-check numerical audit, with a 17th manuscript-literal check when the manuscript source is supplied beside the repository;
- passed all release integrity tests (`14 passed`).

The analysis-only path uses the persisted run records and requires no API key, GPU, hosted-model call, or benchmark download.
