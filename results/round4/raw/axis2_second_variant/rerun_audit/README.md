# Qwen3.5 endpoint-rerun audit

Two Qwen3.5 model-seed jobs (seeds 2 and 3) were interrupted by a provider-side endpoint migration that returned HTTP 404 transport errors before any completion was produced during the first execution of the CIC-UNSW-NB15 V1 campaign. They were rerun under the unchanged frozen `nothink_template` endpoint configuration. Both reruns completed successfully.

The parent directory contains the canonical completed record used by every analysis and manuscript table. This audit directory retains the superseded pre-rerun ledger and raw generations solely to document the transition. No completed model-seed job or row-level prediction was selectively replaced.

Final Qwen3.5 record: 10/10 completed seeds, 3,000 scored rows, parse rate 1.000, accuracy 0.419, macro-F1 0.396, and no unparsed outputs.
