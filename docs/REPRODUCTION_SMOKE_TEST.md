# Reproduction smoke test

The packaged analysis was executed from the repository root with:

```bash
make all
```

The run:

- regenerated all derived tables and figures;
- passed the manuscript-claim verifier;
- passed the independent 15-check numerical audit (and the optional 16th manuscript check when the manuscript source is present);
- passed all release integrity tests (`7 passed`).

The analysis-only path uses the persisted run records and requires no API key, GPU, or benchmark download.
