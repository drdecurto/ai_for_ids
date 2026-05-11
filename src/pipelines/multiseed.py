"""
src/pipelines/multiseed.py

Multi-seed validation orchestration for the operational foundation-model
benchmark (Axis 1 in the paper).

The single-seed protocol used in
``models_use_example.ipynb`` answers "how does this model behave on a
single 70/30 split?" — sufficient for descriptive reporting, but not for
statistical testing of close cells (e.g., the 0.97-percentage-point
TabPFN -> TabPFN 2.5 gain on CIC-IDS2017 V_4). The protocol implemented
here resamples the split itself across N seeds, yielding the seed-level
distributions on which the bootstrap CIs and paired Wilcoxon tests are
computed.

Quick start
-----------
.. code-block:: python

    from src.data import CICIDS2017
    from src.pipelines.multiseed import MultiSeedRunner, compare_runners

    dataset = CICIDS2017(pca=False, classes_mapping=False)
    dataset.load()

    rA = MultiSeedRunner(
        dataset=dataset,
        model_name="tabpfn",         # TabPFN release 2 (legacy)
        model_kwargs={"version": "2"},
        seeds=[0, 1, 2, 3, 4],
        test_size=0.3,
        memory_bound_per_class=2000,  # FM memory budget; matches the paper
    )
    rA.run_all()

    rB = MultiSeedRunner(
        dataset=dataset,
        model_name="tabpfn_2.5",     # TabPFN 2.5 (new entrant)
        seeds=[0, 1, 2, 3, 4],
        test_size=0.3,
        memory_bound_per_class=2000,
    )
    rB.run_all()

    summary_a = rA.summary()        # mean +/- 95% CI for accuracy and
                                    # macro-F1
    summary_b = rB.summary()
    cmp_table = compare_runners(rA, rB)   # paired Wilcoxon p-value, etc.

The class is self-contained: it does its own seeding, splitting, fitting,
prediction, and metric computation. It does not modify any module-level
config; per-seed configuration overrides go through
``src.models.seedutils.build_seeded_model``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data._base import TabularDataset
from src.data.utils import balanced_sample
from src.models.seedutils import build_seeded_model, set_global_seed


# ---------------------------------------------------------------------------
# Per-seed result container
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    """One per-seed evaluation. Stored verbatim so the student can audit
    individual seeds before aggregating."""

    seed: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]   # class label -> F1
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    n_train: int
    n_test: int


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def _bootstrap_ci(
    samples: Sequence[float],
    n_resamples: int = 5_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean of a small sample.

    Returns (mean, lo, hi). For very small N (<2) we return the lone
    value with a degenerate CI; for N=1 the bootstrap is uninformative
    by design and we surface that fact rather than fabricate a CI.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))

    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(arr.mean()), float(lo), float(hi))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
class MultiSeedRunner:
    """
    Run one foundation model on N seeds against the operational test
    partition, with optional class-balanced fitting subsample to
    respect each foundation model's memory bound.
    """

    def __init__(
        self,
        dataset: TabularDataset,
        model_name: str,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        test_size: float = 0.3,
        label_col: str = "Label",
        memory_bound_per_class: int | None = None,
        stratify: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        bootstrap_resamples: int = 5_000,
    ) -> None:
        """
        Args:
            dataset: A loaded ``TabularDataset`` instance (i.e. the
                student has already called ``dataset.load()``). The
                runner does not mutate the dataset.
            model_name: Name forwarded to
                ``src.models.seedutils.build_seeded_model``.
                Examples: ``"tabpfn"``, ``"tabpfn_2.5"``, ``"tabicl"``,
                ``"tabstar"``, ``"random_forest"``.
            seeds: Iterable of integer seeds to evaluate. The paper's
                Axis-2 protocol uses 5 seeds. For a smaller compute
                budget pass e.g. ``seeds=(0, 1, 2)``; the bootstrap CIs
                widen accordingly.
            test_size: Fraction of the cleaned dataset reserved for
                test. The paper's Axis-1 protocol uses 0.3.
            label_col: Column with the multi-class label. Used for
                stratification and per-class metric reporting.
            memory_bound_per_class: If set, the train partition is
                class-balanced to at most this many rows per class
                before fitting. Recommended values: 2000 for TabPFN /
                TabPFN 2.5 / TabICL; 4000 for TabSTAR. Set to ``None``
                for classical baselines.
            stratify: If True, the train/test split is stratified on
                the label column. Highly recommended for the imbalanced
                IDS datasets (otherwise rare classes can vanish from
                the test split entirely on certain seeds).
            model_kwargs: Extra keyword arguments forwarded to the
                model constructor (e.g., ``{"version": "2"}`` to force
                the legacy TabPFN release).
            bootstrap_resamples: Number of bootstrap resamples for the
                mean-accuracy CI. Default 5,000 matches the paper.
        """
        self.dataset = dataset
        self.model_name = model_name
        self.seeds = list(seeds)
        self.test_size = float(test_size)
        self.label_col = label_col
        self.memory_bound_per_class = memory_bound_per_class
        self.stratify = bool(stratify)
        self.model_kwargs = dict(model_kwargs or {})
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.results: list[SeedResult] = []

    # ---------------------------------------------------------------- run
    def run_all(self, verbose: bool = True) -> list[SeedResult]:
        """Run the model on every seed and return per-seed results."""
        self.results = []
        for s in self.seeds:
            if verbose:
                logging.info(
                    f"[MultiSeedRunner] {self.model_name} seed={s}"
                )
            self.results.append(self._run_one(seed=s))
        return self.results

    def _run_one(self, seed: int) -> SeedResult:
        """One seed: split, fit, predict, score."""
        # 1. Set global RNG state (numpy / random / torch)
        set_global_seed(seed)

        # 2. Stratified split with this seed
        stratify_col = self.label_col if self.stratify else None
        train_df, test_df = self.dataset.train_test_split(
            test_size=self.test_size,
            random_state=seed,
            stratify_col=stratify_col,
        )

        # 3. Optional memory-bounded class-balanced subsample of TRAIN
        #    only. The TEST partition is never subsampled — it remains
        #    the operational stratified imbalanced reference.
        if self.memory_bound_per_class is not None:
            train_df = balanced_sample(
                train_df,
                category_col=self.label_col,
                n_per_class=self.memory_bound_per_class,
            )

        x_train = train_df.drop(columns=[self.label_col]).values
        y_train = train_df[self.label_col].values
        x_test = test_df.drop(columns=[self.label_col]).values
        y_test = test_df[self.label_col].values

        # 4. Build a fresh, seed-specific model
        model = build_seeded_model(
            self.model_name,
            seed=seed,
            **self.model_kwargs,
        )

        # 5. Fit
        t0 = time.time()
        model.fit(x_train, y_train)
        fit_secs = time.time() - t0

        # 6. Predict
        t0 = time.time()
        y_pred = model.predict(x_test)
        pred_secs = time.time() - t0

        # 7. Metrics
        acc = float(accuracy_score(y_test, y_pred))
        macro = float(f1_score(y_test, y_pred, average="macro"))

        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            output_dict=True,
            zero_division=0,
        )
        # Drop the average rows that classification_report adds
        non_class_keys = {"accuracy", "macro avg", "weighted avg"}
        per_class_f1: dict[str, float] = {}
        per_class_p: dict[str, float] = {}
        per_class_r: dict[str, float] = {}
        for cls, metrics in report.items():
            if cls in non_class_keys or not isinstance(metrics, dict):
                continue
            per_class_f1[str(cls)] = float(metrics.get("f1-score", 0.0))
            per_class_p[str(cls)] = float(metrics.get("precision", 0.0))
            per_class_r[str(cls)] = float(metrics.get("recall", 0.0))

        return SeedResult(
            seed=seed,
            accuracy=acc,
            macro_f1=macro,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_p,
            per_class_recall=per_class_r,
            fit_seconds=fit_secs,
            predict_seconds=pred_secs,
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
        )

    # ------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        """
        Aggregate per-seed results.

        Returns a dict with the following keys:
            seeds:                   list of seeds actually run
            n_seeds:                 len(seeds)
            accuracy_mean:           seed-mean overall accuracy
            accuracy_lo95, _hi95:    bootstrap 95 % CI of the mean
            macro_f1_mean / lo / hi: same for macro-F1
            per_class_f1:            dict[class_label, dict] with mean / CI
            per_seed_table:          pandas DataFrame, one row per seed
        """
        if not self.results:
            raise RuntimeError(
                "MultiSeedRunner.summary() called before run_all()."
            )

        accs = [r.accuracy for r in self.results]
        macros = [r.macro_f1 for r in self.results]

        rng = np.random.default_rng(0)
        m_acc, lo_acc, hi_acc = _bootstrap_ci(
            accs, self.bootstrap_resamples, rng=rng
        )
        m_mf, lo_mf, hi_mf = _bootstrap_ci(
            macros, self.bootstrap_resamples, rng=rng
        )

        # per-class aggregation
        all_classes = sorted(
            {
                cls
                for r in self.results
                for cls in r.per_class_f1
            }
        )
        per_class: dict[str, dict[str, float]] = {}
        for cls in all_classes:
            f1s = [r.per_class_f1.get(cls, 0.0) for r in self.results]
            mu, lo, hi = _bootstrap_ci(
                f1s, self.bootstrap_resamples, rng=rng
            )
            per_class[cls] = {
                "f1_mean": mu,
                "f1_lo95": lo,
                "f1_hi95": hi,
                "n_seeds_with_class": int(
                    sum(1 for r in self.results if cls in r.per_class_f1)
                ),
            }

        rows = []
        for r in self.results:
            rows.append(
                {
                    "seed": r.seed,
                    "accuracy": r.accuracy,
                    "macro_f1": r.macro_f1,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "fit_seconds": r.fit_seconds,
                    "predict_seconds": r.predict_seconds,
                }
            )
        per_seed_df = pd.DataFrame(rows)

        return {
            "model_name": self.model_name,
            "seeds": [r.seed for r in self.results],
            "n_seeds": len(self.results),
            "accuracy_mean": m_acc,
            "accuracy_lo95": lo_acc,
            "accuracy_hi95": hi_acc,
            "macro_f1_mean": m_mf,
            "macro_f1_lo95": lo_mf,
            "macro_f1_hi95": hi_mf,
            "per_class_f1": per_class,
            "per_seed_table": per_seed_df,
        }

    # -------------------------------------------------------------- save
    def save(self, out_dir: str) -> None:
        """
        Persist per-seed metrics and the aggregate summary.

        Two files are written:
          - ``{out_dir}/{model_name}_per_seed.csv``  one row per seed.
          - ``{out_dir}/{model_name}_summary.json``  aggregate metrics.

        ``out_dir`` is created if it does not already exist.
        """
        os.makedirs(out_dir, exist_ok=True)
        summary = self.summary()
        per_seed_df = summary.pop("per_seed_table")

        per_seed_path = os.path.join(
            out_dir, f"{self.model_name}_per_seed.csv"
        )
        summary_path = os.path.join(
            out_dir, f"{self.model_name}_summary.json"
        )
        per_seed_df.to_csv(per_seed_path, index=False)

        # Make summary JSON-serialisable (drop the DataFrame, cast types)
        clean: dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, np.generic):
                clean[k] = v.item()
            else:
                clean[k] = v
        with open(summary_path, "w") as f:
            json.dump(clean, f, indent=2)

        logging.info(
            f"[MultiSeedRunner] saved {per_seed_path} and {summary_path}"
        )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------
def compare_runners(
    runner_a: MultiSeedRunner,
    runner_b: MultiSeedRunner,
    metric: str = "accuracy",
) -> dict[str, Any]:
    """
    Paired non-parametric comparison between two MultiSeedRunner runs
    on the same set of seeds.

    The Wilcoxon signed-rank test is the right tool here because seeds
    are matched: for every seed we have one accuracy value from each
    runner, and the paired difference is the natural unit of variability.

    Args:
        runner_a: First runner. Must have completed ``run_all()``.
        runner_b: Second runner, same seeds as ``runner_a``.
        metric: ``"accuracy"`` (default) or ``"macro_f1"``.

    Returns:
        A dict with:
            seeds:           list of paired seeds.
            metric:          the metric compared.
            mean_a / mean_b: seed-mean of the metric for each runner.
            mean_diff:       runner_a mean minus runner_b mean.
            wilcoxon_stat:   the W statistic.
            wilcoxon_p:      two-sided p-value.
                             Reject H0 ("the two methods are
                             distributionally equivalent on this metric")
                             if p < 0.05.
            n_seeds:         len(seeds).
        If the seed sets differ, only the intersection is used and a
        warning is logged.
    """
    from scipy.stats import wilcoxon

    seeds_a = {r.seed: r for r in runner_a.results}
    seeds_b = {r.seed: r for r in runner_b.results}
    common = sorted(set(seeds_a).intersection(seeds_b))
    if len(common) < len(seeds_a) or len(common) < len(seeds_b):
        logging.warning(
            "compare_runners: runner_a has seeds %s but runner_b has %s; "
            "using intersection %s",
            sorted(seeds_a), sorted(seeds_b), common,
        )

    if metric == "accuracy":
        a_vals = np.array([seeds_a[s].accuracy for s in common])
        b_vals = np.array([seeds_b[s].accuracy for s in common])
    elif metric == "macro_f1":
        a_vals = np.array([seeds_a[s].macro_f1 for s in common])
        b_vals = np.array([seeds_b[s].macro_f1 for s in common])
    else:
        raise ValueError(
            f"metric must be 'accuracy' or 'macro_f1', got {metric!r}"
        )

    # Wilcoxon needs non-zero differences. If every paired diff is 0 we
    # surface that explicitly rather than crashing scipy.
    diffs = a_vals - b_vals
    if np.all(diffs == 0):
        return {
            "seeds": common,
            "metric": metric,
            "mean_a": float(a_vals.mean()),
            "mean_b": float(b_vals.mean()),
            "mean_diff": 0.0,
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": 1.0,
            "n_seeds": len(common),
            "note": "All paired differences are zero; runners are identical on this metric.",
        }

    stat, p = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
    return {
        "seeds": common,
        "metric": metric,
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "mean_diff": float(a_vals.mean() - b_vals.mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "n_seeds": len(common),
    }


# ---------------------------------------------------------------------------
# Convenience: cross-model panel sweep
# ---------------------------------------------------------------------------
def run_panel(
    dataset: TabularDataset,
    model_specs: list[dict[str, Any]],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    out_dir: str | None = None,
    **runner_kwargs: Any,
) -> dict[str, MultiSeedRunner]:
    """
    Run a panel of models on the same dataset, all under the same
    seeds. Useful for reproducing one row of Table 8 from the paper.

    Args:
        dataset: A loaded TabularDataset.
        model_specs: List of dicts, each with at least ``"name"``
            (forwarded to ``build_seeded_model``) and optionally
            ``"model_kwargs"`` and ``"memory_bound_per_class"``.
            Example::

                model_specs = [
                    {"name": "random_forest"},
                    {"name": "tabpfn",     "model_kwargs": {"version": "2"},
                     "memory_bound_per_class": 2000},
                    {"name": "tabpfn_2.5", "memory_bound_per_class": 2000},
                    {"name": "tabicl",     "memory_bound_per_class": 2000},
                    {"name": "tabstar",    "memory_bound_per_class": 4000},
                ]

        seeds: Iterable of seeds used for every model in the panel.
        out_dir: If provided, every runner saves its per-seed CSV and
            summary JSON in this directory.
        **runner_kwargs: Extra keyword arguments forwarded to every
            ``MultiSeedRunner`` (e.g., ``test_size=0.3``).

    Returns:
        Dict mapping ``model_name -> MultiSeedRunner``. Each runner has
        already completed ``run_all()``.
    """
    runners: dict[str, MultiSeedRunner] = {}
    for spec in model_specs:
        name = spec["name"]
        runner = MultiSeedRunner(
            dataset=dataset,
            model_name=name,
            seeds=seeds,
            model_kwargs=spec.get("model_kwargs", {}),
            memory_bound_per_class=spec.get("memory_bound_per_class"),
            **runner_kwargs,
        )
        runner.run_all()
        if out_dir is not None:
            runner.save(out_dir)
        runners[name] = runner

    return runners
