"""
Multi-seed validation of the operational foundation-model panel (Axis 1)
========================================================================

What this script does
---------------------
For one (dataset, variant) cell it runs the operational foundation-model panel
across N random splits, saving each model's results to disk **as soon as that
model finishes**, then prints seed-mean accuracy with bootstrap 95% confidence
intervals and paired Wilcoxon p-values (every model vs Random Forest, plus the
TabPFN vs TabPFN 2.5 comparison).

Why incremental saving
----------------------
Models differ wildly in cost. TabPFN / TabPFN 2.5 / TabICL / Random Forest are
cheap (minutes per cell); TabSTAR fine-tunes a LoRA each seed and is ~5x more
expensive. Saving after every model means an interrupted or crashed run keeps
the cheap results you already paid for, and you can re-run just the slow model
(`--models tabstar`) without redoing the rest.

Recommended budget
------------------
Five seeds is the standard. Three is a faster floor with wider CIs. Two is
enough to spot bugs but gives no meaningful CI.

Usage
-----
Run every variant of one dataset (default dataset cic_ids2017):
    python multiseed_evaluation.py
    python multiseed_evaluation.py --dataset n_baiot

Run a single cell (pin the variant):
    python multiseed_evaluation.py --dataset n_baiot --variant V4

Re-run only the expensive model on a cell:
    python multiseed_evaluation.py --dataset cic_ids2017 --variant V4 --models tabstar

Sweep every cell of Table 8 (each model still saved as it finishes):
    python multiseed_evaluation.py --all-cells

Output
------
Per model, under <out-dir> (default multiseed_runs/<dataset>_<variant>):
    <model>_per_seed.csv   per-seed metrics (audit trail)
    <model>_summary.json   aggregate metrics (seed mean + bootstrap CI)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.data import CICIDS2017, N_BaIoT, CIC_UNSW
from src.pipelines import MultiSeedRunner, compare_runners


# ---------------------------------------------------------------------------
# Defaults pulled from the notebook
# ---------------------------------------------------------------------------

SEEDS: List[int] = [0, 1, 2, 3, 4]   # paper's Axis-2 protocol
TEST_SIZE: float = 0.3               # paper's Axis-1 protocol

# Mirrors Table 1 of the paper (operational foundation-model panel).
# memory_bound_per_class caps rows/class for the foundation models.
MODEL_SPECS: List[Dict] = [
    {"name": "random_forest"},
    {"name": "tabpfn",     "model_kwargs": {"version": "2"}, "memory_bound_per_class": 2000},
    {"name": "tabpfn_2.5",                                   "memory_bound_per_class": 2000},
    {"name": "tabicl",                                       "memory_bound_per_class": 2000},
    {"name": "tabstar",                                      "memory_bound_per_class": 4000},
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Reset root handlers and install a clean formatter (matches the notebook)."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

# Variant -> (pca, classes_mapping). V1 = no PCA, no class mapping; V4 = both.
VARIANT_FLAGS = {
    "V1": (False, False),
    "V2": (True,  False),
    "V3": (False, True),
    "V4": (True,  True),
}

DATASET_BUILDERS = {
    "cic_ids2017": CICIDS2017,
    "n_baiot":     N_BaIoT,
    "cic_unsw":    CIC_UNSW,
}

# CIC-UNSW only reports V1, V2 in the paper; other cells are skipped.
TABLE8_CELLS = [
    ("cic_ids2017", "V1"), ("cic_ids2017", "V2"),
    ("cic_ids2017", "V3"), ("cic_ids2017", "V4"),
    ("n_baiot",     "V1"), ("n_baiot",     "V2"),
    ("n_baiot",     "V3"), ("n_baiot",     "V4"),
    ("cic_unsw",    "V1"), ("cic_unsw",    "V2"),
]

def load_dataset(dataset_name: str, variant: str):
    """Build, load, and report dimensions of a (dataset, variant) cell."""
    if dataset_name not in DATASET_BUILDERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_BUILDERS)}."
        )
    if variant not in VARIANT_FLAGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {list(VARIANT_FLAGS)}."
        )

    pca, classes_mapping = VARIANT_FLAGS[variant]
    dataset = DATASET_BUILDERS[dataset_name](pca=pca, classes_mapping=classes_mapping)
    dataset.load()
    print(
        f"Loaded {len(dataset.data):,} rows, "
        f"{dataset.data.shape[1] - 1} features, "
        f"{dataset.data['Label'].nunique()} classes."
    )
    return dataset


# ---------------------------------------------------------------------------
# The one and only run path: panel for a single cell, saved as it goes
# ---------------------------------------------------------------------------

def run_cell(
    dataset_name: str,
    variant: str,
    out_dir: str,
    seeds: Iterable[int] = SEEDS,
    test_size: float = TEST_SIZE,
    models: Optional[List[str]] = None,
) -> Dict[str, MultiSeedRunner]:
    """Run the panel for one (dataset, variant) cell.

    Each model is run and **saved to disk immediately after it finishes**, so an
    interrupted run keeps everything completed so far. After all requested
    models finish, prints the panel summary and the standard comparisons.

    Parameters
    ----------
    models : optional subset of model names to run (default: all of MODEL_SPECS).
        e.g. ["tabstar"] to re-run only the expensive model on this cell.
    """
    seeds = list(seeds)

    specs = MODEL_SPECS
    if models is not None:
        wanted = set(models)
        specs = [spec for spec in MODEL_SPECS if spec["name"] in wanted]
        if not specs:
            raise ValueError(
                f"No model specs left after filtering by {sorted(wanted)}. "
                f"Known models: {[s['name'] for s in MODEL_SPECS]}"
            )

    print(f"\n{'=' * 72}\nCell: {dataset_name} {variant}  ->  {out_dir}\n{'=' * 72}")
    dataset = load_dataset(dataset_name, variant)

    runners: Dict[str, MultiSeedRunner] = {}
    rows = []
    for spec in specs:
        name = spec["name"]
        print(f"\n--- {name} ({len(seeds)} seeds) ---")

        # Only pass optional kwargs that the spec actually sets, so model
        # defaults (e.g. TabPFN 2.5's version) are preserved.
        runner_kwargs = dict(
            dataset=dataset,
            model_name=name,
            seeds=seeds,
            test_size=test_size,
        )
        if "model_kwargs" in spec:
            runner_kwargs["model_kwargs"] = spec["model_kwargs"]
        if "memory_bound_per_class" in spec:
            runner_kwargs["memory_bound_per_class"] = spec["memory_bound_per_class"]

        runner = MultiSeedRunner(**runner_kwargs)
        runner.run_all()
        runner.save(out_dir)            # <-- persist this model before moving on
        runners[name] = runner

        s = runner.summary()
        print(
            f"  acc = {s['accuracy_mean']:.4f}  "
            f"95% CI [{s['accuracy_lo95']:.4f}, {s['accuracy_hi95']:.4f}]  "
            f"saved -> {out_dir}"
        )
        rows.append({
            "model":        name,
            "acc_mean":     s["accuracy_mean"],
            "acc_lo95":     s["accuracy_lo95"],
            "acc_hi95":     s["accuracy_hi95"],
            "macroF1_mean": s["macro_f1_mean"],
            "n_seeds":      s["n_seeds"],
        })

    _report(runners, rows)
    return runners


def _report(runners: Dict[str, MultiSeedRunner], rows: List[Dict]) -> None:
    """Print the panel summary and the standard paired comparisons."""
    panel_df = (
        pd.DataFrame(rows)
        .sort_values("acc_mean", ascending=False)
        .reset_index(drop=True)
    )
    print("\nPanel summary:")
    print(panel_df.round(4).to_string(index=False))

    # Comparison vs Random Forest, the natural anchor.
    if "random_forest" in runners:
        rf = runners["random_forest"]
        cmp_rows = [
            {
                "model":           name,
                "mean_diff_vs_RF": (c := compare_runners(r, rf, metric="accuracy"))["mean_diff"],
                "wilcoxon_p":      c["wilcoxon_p"],
                "n_seeds":         c["n_seeds"],
            }
            for name, r in runners.items() if name != "random_forest"
        ]
        if cmp_rows:
            cmp_df = (
                pd.DataFrame(cmp_rows)
                .sort_values("mean_diff_vs_RF", ascending=False)
                .reset_index(drop=True)
            )
            print("\nPairwise comparisons vs Random Forest:")
            print(cmp_df.round(4).to_string(index=False))

    # The specific TabPFN vs TabPFN 2.5 question from the paper.
    if "tabpfn" in runners and "tabpfn_2.5" in runners:
        c = compare_runners(runners["tabpfn_2.5"], runners["tabpfn"], metric="accuracy")
        print("\nPaired Wilcoxon (TabPFN 2.5 vs TabPFN):")
        print(f"  mean diff         = {c['mean_diff']:+.4f}")
        print(f"  p-value (2-sided) = {c['wilcoxon_p']:.4f}  (n_seeds = {c['n_seeds']})")
        print(
            "  p < 0.05 => the per-seed distributions are distinguishable; "
            "otherwise the single-seed gain is not statistically reliable. "
            "If 0.05 <= p <= 0.15, increase to ten seeds before concluding."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed validation of the operational foundation-model panel (Axis 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=list(DATASET_BUILDERS), default="cic_ids2017",
                        help="Dataset name (default: cic_ids2017). Ignored with --all-cells.")
    parser.add_argument("--variant", choices=list(VARIANT_FLAGS), default=None,
                        help="Dataset variant (default: all variants of the dataset). "
                             "Ignored with --all-cells.")
    parser.add_argument("--all-cells", action="store_true",
                        help="Sweep every (dataset, variant) cell of Table 8 instead of one cell.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Seed list (default: {SEEDS}).")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help=f"Test split fraction (default: {TEST_SIZE}).")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=[spec["name"] for spec in MODEL_SPECS],
                        help="Subset of models to run (default: all). e.g. --models tabstar.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for a single cell (default: "
                             "multiseed_runs/<dataset>_<variant>). Ignored when more "
                             "than one variant runs.")
    parser.add_argument("--out-root", type=str, default="multiseed_runs",
                        help="Output root used for --all-cells (default: multiseed_runs).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO).")
    return parser.parse_args(argv)


def variants_for(dataset_name: str) -> List[str]:
    """Variants the paper reports for a dataset, in order (from TABLE8_CELLS)."""
    return [v for (d, v) in TABLE8_CELLS if d == dataset_name]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(getattr(logging, args.log_level))

    # Decide which (dataset, variant) cells to run.
    if args.all_cells:
        cells = list(TABLE8_CELLS)
    elif args.variant is not None:
        cells = [(args.dataset, args.variant)]
    else:
        # Only --dataset given: run every variant of that dataset.
        cells = [(args.dataset, v) for v in variants_for(args.dataset)]

    # --out-dir only makes sense for a single cell; otherwise keep cells separate.
    single_cell = len(cells) == 1
    for dataset_name, variant in cells:
        out_dir = (
            args.out_dir if (single_cell and args.out_dir)
            else f"{args.out_root}/{dataset_name}_{variant}"
        )
        run_cell(
            dataset_name=dataset_name,
            variant=variant,
            out_dir=out_dir,
            seeds=args.seeds,
            test_size=args.test_size,
            models=args.models,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
