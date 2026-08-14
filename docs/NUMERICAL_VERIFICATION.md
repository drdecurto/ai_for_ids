# Numerical verification record

The analysis script checks and regenerates the following manuscript-defining quantities.

## Axis 1

- 320 strict core records: 4 cells x 8 configurations x 10 seeds.
- Spearman accuracy-versus-rare-recall correlations:
  - CIC-IDS2017 V1: -0.45238
  - CIC-IDS2017 V3: -0.45238
  - CIC-UNSW-NB15 V1: -0.42857
  - CIC-UNSW-NB15 V2: -0.47619
- The top three accuracy configurations equal the bottom three rare-recall configurations in every cell.
- 48 prior-fitted-versus-classical comparisons:
  - prior-fitted model wins all ten seeds: 40;
  - classical model wins all ten seeds: 4;
  - mixed: 4.

## Factorial experiment

Paired rare-recall contrasts on CIC-IDS2017 V3:

- balance at the small budget: +0.50913;
- balance at the large budget: +0.25993;
- size at the natural distribution: +0.24214;
- size at the balanced distribution: -0.00706.

## Axis 2

Corrected closed-label means:

- Random Forest: accuracy 0.80407, macro-F1 0.80551.
- Kimi-K3: accuracy 0.81829, macro-F1 0.83355, adjusted p = 0.11719.
- GLM-5.2: accuracy 0.80081, macro-F1 0.81874, adjusted p = 0.12891.
- Total recorded language-model cost: USD 218.3418699.

The conclusion is “no multiplicity-adjusted advantage detected,” not equivalence.

Run:

```bash
make all
```

and compare against `results/derived/verification_summary.json`.
