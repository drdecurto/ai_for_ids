# Numerical verification summary

All values below are independently recomputed from persisted seed-level or raw-generation records by `scripts/verify_manuscript_claims.py` and `scripts/audit_manuscript_tables.py`.

## Axis 1

- Strict record: 320 observations.
- Pairwise directions: 40 foundation-model wins on all ten seeds, 4 classical wins on all ten, 4 mixed.
- Accuracy/rare-recall Spearman correlations: -0.45238, -0.45238, -0.42857, and -0.47619.
- Balance effects on rare recall: +0.50913 and +0.25993 on CIC-IDS2017 V3; +0.57919 and +0.31953 on CIC-UNSW-NB15 V1.
- ManyClass mean rare-recall spread: 0.00302995; no setting differs from the reference after Holm correction.

## Axis 2: grouped CIC-IDS2017

- Matched Random Forest: accuracy 0.804065, macro-F1 0.805513.
- Kimi-K3: accuracy 0.818293, macro-F1 0.833546, delta +0.028033, 9/10 wins, adjusted p = 0.117188.
- GLM-5.2: delta +0.013225, 8/10 wins, adjusted p = 0.128906.
- Hosted-model cost: USD 218.3418699.

## Axis 2: CIC-UNSW-NB15 V1

- Matched Random Forest: accuracy 0.453000, macro-F1 0.433203.
- Kimi-K3: accuracy 0.489000, macro-F1 0.473185, delta +0.039982, 9/10 wins, adjusted p = 0.105469.
- GLM-5.2: accuracy 0.485333, macro-F1 0.466030, delta +0.032827, 8/10 wins, adjusted p = 0.234375.
- Qwen3.5: accuracy 0.419, macro-F1 0.396, parse rate 1.000, mean difference versus the matched forest -0.037, two of ten seed-level wins, and Bonferroni-adjusted p=0.2227. The two initially interrupted jobs completed successfully when rerun under the unchanged frozen configuration.
- Hosted-model cost: USD 293.68404155.

## Cross-source Axis 2

- Hosted-model macro-F1 rank correlation: Spearman 1.000.
- No positive model-anchor contrast survives six-comparison correction on either source.
- Combined hosted-model cost: USD 512.02591145.
- Grouped-CIC development-rich tree references reach macro-F1 0.91117-0.91898.
