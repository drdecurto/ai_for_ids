#!/usr/bin/env python3
"""Recompute the final revision analysis from persisted result files.

No model is fitted here. The script validates seed completeness, recomputes the
strict-panel and robustness summaries, and regenerates the final revision tables.
The completed second hosted-model variant is independently rescored from raw generations.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from recompute_axis2_second_variant import recompute as recompute_axis2_second_variant

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results"
R4 = ROOT / "results" / "round4" / "raw"
HIST = ROOT / "results" / "development_screening"
OUT = ROOT / "results" / "round4" / "derived"
GEN = ROOT / "results" / "round4" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

# Reconstruct the completed CIC-UNSW-NB15 V1 Axis-2 panel from raw generations.
A2_SECOND_INFO = recompute_axis2_second_variant(ROOT)

CELLS = ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1", "cic_unsw_V2"]
CELL_LABEL = {
    "cic_ids2017_V1": "CIC-IDS2017 V1",
    "cic_ids2017_V3": "CIC-IDS2017 V3",
    "cic_unsw_V1": "CIC-UNSW-NB15 V1",
    "cic_unsw_V2": "CIC-UNSW-NB15 V2",
}
CORE = [
    "rf_pruned_full", "rf_strong_full", "xgb_full",
    "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
    "tabpfn_2.5", "tabicl",
]
CLASSICAL = CORE[:6]
FM = CORE[6:]
MODEL_LABEL = {
    "rf_pruned_full": "Pruned RF, large",
    "rf_strong_full": "Strong RF, large",
    "xgb_full": "XGBoost, large",
    "rf_pruned_balanced": "Pruned RF, balanced",
    "rf_strong_balanced": "Strong RF, balanced",
    "xgb_balanced": "XGBoost, balanced",
    "tabpfn_2.5": "TabPFN 2.5",
    "tabicl": "TabICL",
    "tabstar": "TabSTAR",
}
ABBR = {
    "rf_pruned_full": "RF-P/L", "rf_strong_full": "RF-S/L", "xgb_full": "XGB/L",
    "rf_pruned_balanced": "RF-P/B", "rf_strong_balanced": "RF-S/B", "xgb_balanced": "XGB/B",
    "tabpfn_2.5": "TabPFN", "tabicl": "TabICL", "tabstar": "TabSTAR",
}
SEVERITY = {
    "Heartbleed": 8, "Infiltration": 8, "Web Attack": 4,
    "Web Attack  Brute Force": 4, "Web Attack  XSS": 4,
    "Web Attack  Sql Injection": 4, "Bot": 3, "Brute Force": 2,
    "FTP-Patator": 2, "SSH-Patator": 2, "Port Scan": 1,
    "PortScan": 1, "DDoS": 1, "DoS Hulk": 1,
    "DoS GoldenEye": 1, "DoS slowloris": 1,
    "DoS Slowhttptest": 1, "Backdoor": 8, "Shellcode": 6,
    "Worms": 6, "Exploits": 4, "Analysis": 2,
    "Reconnaissance": 1, "Generic": 1, "Fuzzers": 1, "DoS": 1,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def signflip(values: Sequence[float]) -> float:
    d = np.asarray(values, dtype=float)
    if d.size == 0:
        return float("nan")
    observed = abs(float(d.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        null.append(abs(float(np.mean(d * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap_ci(values: Sequence[float], n_boot: int = 50000, seed: int = 2026) -> tuple[float, float]:
    d = np.asarray(values, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for position, idx in enumerate(order):
        running = max(running, (len(p) - position) * p[idx])
        out[idx] = min(1.0, running)
    return out


def tex(value: object) -> str:
    return (str(value).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def assert_grid(df: pd.DataFrame, group_cols: list[str], expected_groups: int, seeds: Iterable[int] = range(10)) -> None:
    expected_seeds = set(seeds)
    groups = list(df.groupby(group_cols, dropna=False))
    assert len(groups) == expected_groups, (group_cols, len(groups), expected_groups)
    for key, g in groups:
        assert set(g["seed"].astype(int)) == expected_seeds, (key, sorted(set(g.seed)), expected_seeds)
        assert len(g) == len(expected_seeds), (key, len(g))


# ------------------------------------------------------------------
# Load and validate the record
# ------------------------------------------------------------------
M = pd.read_csv(R3 / "axis1" / "metrics.csv").drop_duplicates(
    ["cell", "seed", "model", "test_set"], keep="last"
)
PC = pd.read_csv(R3 / "axis1" / "per_class.csv")
CM = pd.read_csv(R3 / "axis1" / "confusion.csv")
A2_SEED = pd.read_csv(R3 / "derived" / "axis2_closed_label_per_seed.csv")
A2_SUM = pd.read_csv(R3 / "derived" / "axis2_closed_label_summary.csv")
A2_RAW_SUM = pd.read_csv(R3 / "axis2" / "results_original_six_model_campaign.csv")
A2_CORRECTED = pd.read_csv(R3 / "axis2" / "results_nemotron_corrected.csv")
CIC_FACT = pd.read_csv(R3 / "sensitivity" / "exp1_size_by_balance.csv")
MC = pd.read_csv(R4 / "exp4_manyclass.csv")
UNSW_FACT = pd.read_csv(R4 / "exp5_size_by_balance_unsw.csv")
A2_RICH = pd.read_csv(R4 / "exp6_axis2_datarich.csv")
A2_RICH_PC = pd.read_csv(R4 / "exp6_axis2_datarich_perclass.csv")

assert_grid(CIC_FACT, ["arm"], 4)
assert_grid(UNSW_FACT, ["arm"], 4)
assert_grid(MC, ["alphabet_size", "n_estimators_redundancy"], 4)
assert_grid(A2_RICH, ["reference", "learner"], 4)
assert len(A2_RICH_PC) > 0
assert MC["wrapper_engaged"].astype(bool).all()
assert set(MC["n_classes"].astype(int)) == {15}

INPUT_MANIFEST = []
for path in [
    R3 / "axis1" / "metrics.csv", R3 / "axis1" / "per_class.csv", R3 / "axis1" / "confusion.csv",
    R3 / "derived" / "axis2_closed_label_per_seed.csv", R3 / "derived" / "axis2_closed_label_summary.csv",
    R3 / "axis2" / "results_original_six_model_campaign.csv", R3 / "axis2" / "results_nemotron_corrected.csv",
    R3 / "sensitivity" / "exp1_size_by_balance.csv",
    R4 / "exp4_manyclass.csv", R4 / "exp5_size_by_balance_unsw.csv",
    R4 / "exp6_axis2_datarich.csv", R4 / "exp6_axis2_datarich_perclass.csv",
    R4 / "axis2_second_variant" / "results.csv",
    R4 / "axis2_second_variant" / "raw_generations.jsonl",
    R4 / "axis2_second_variant" / "paired_tests.csv",
    R4 / "axis2_second_variant" / "stage_a_probe.csv",
    R4 / "axis2_second_variant" / "stage_a_chosen_config.json",
]:
    INPUT_MANIFEST.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
pd.DataFrame(INPUT_MANIFEST).to_csv(OUT / "input_manifest.csv", index=False)

# Determine benign labels and prevalence from the persisted weighted supports.
BENIGN: dict[str, str] = {}
SUPPORT: dict[str, pd.Series] = {}
for cell in CELLS:
    base = PC[(PC.cell == cell) & (PC.seed == 0) & (PC.model == "rf_pruned_full") & (PC.test_set == "capped")]
    benign = [c for c in base["class"].astype(str).unique() if c.lower() == "benign"]
    assert len(benign) == 1
    BENIGN[cell] = benign[0]
    SUPPORT[cell] = base.groupby("class").support.mean().sort_index()


def rare_set(cell: str, threshold: float = 0.005) -> list[str]:
    shares = SUPPORT[cell] / float(SUPPORT[cell].sum())
    return sorted(c for c, share in shares.items() if c != BENIGN[cell] and float(share) < threshold)


RARE = {cell: rare_set(cell) for cell in CELLS}

# ------------------------------------------------------------------
# Strict panel with capped and hash-matched full macro-F1
# ------------------------------------------------------------------
Mc = M[(M.test_set == "capped") & M.model.isin(CORE)].copy()
PCc = PC[(PC.test_set == "capped") & PC.model.isin(CORE + ["tabstar"])].copy()
CMc = CM[(CM.test_set == "capped") & CM.model.isin(CORE + ["tabstar"])].copy()

rare_rows = []
for cell in CELLS:
    g = PCc[(PCc.cell == cell) & PCc["class"].isin(RARE[cell])]
    s = g.groupby(["cell", "seed", "model"], as_index=False).agg(
        rare_recall=("recall", "mean"), rare_precision=("precision", "mean"), rare_f1=("f1", "mean")
    )
    rare_rows.append(s)
rare_seed = pd.concat(rare_rows, ignore_index=True)
core_seed = Mc.merge(rare_seed, on=["cell", "seed", "model"], how="left")
core_summary = core_seed.groupby(["cell", "model"], as_index=False).agg(
    accuracy=("accuracy", "mean"), macro_f1_cap=("macro_f1", "mean"),
    rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
    n_train=("n_train", "mean"), fit_s=("fit_s", "mean"), predict_s=("predict_s", "mean")
)
core_summary["accuracy_rank"] = core_summary.groupby("cell")["accuracy"].rank(method="min", ascending=False).astype(int)
core_summary["macro_f1_rank"] = core_summary.groupby("cell")["macro_f1_cap"].rank(method="min", ascending=False).astype(int)
core_summary["rare_recall_rank"] = core_summary.groupby("cell")["rare_recall"].rank(method="min", ascending=False).astype(int)

full_rows = []
for cell in CELLS:
    for model in CLASSICAL:
        cap = M[(M.cell == cell) & (M.model == model) & (M.test_set == "capped")].set_index("seed")
        full = M[(M.cell == cell) & (M.model == model) & (M.test_set == "full")].set_index("seed")
        for seed in sorted(set(cap.index) & set(full.index)):
            if str(cap.loc[seed, "test_hash"]) != str(full.loc[seed, "test_hash"]):
                continue
            full_rows.append({
                "cell": cell, "model": model, "seed": int(seed),
                "accuracy_full": float(full.loc[seed, "accuracy"]),
                "macro_f1_full": float(full.loc[seed, "macro_f1"]),
                "macro_f1_cap": float(cap.loc[seed, "macro_f1"]),
            })
full_match = pd.DataFrame(full_rows)
full_match.to_csv(OUT / "full_partition_matching_rows.csv", index=False)
full_summary = full_match.groupby(["cell", "model"], as_index=False).agg(
    n_full=("seed", "size"), accuracy_full=("accuracy_full", "mean"),
    macro_f1_full=("macro_f1_full", "mean"), mean_cap_minus_full=("macro_f1_cap", "mean")
)
# Convert the last field from mean cap value to mean difference.
cap_means = full_match.groupby(["cell", "model"])["macro_f1_cap"].mean()
full_means = full_match.groupby(["cell", "model"])["macro_f1_full"].mean()
diff_map = (cap_means - full_means).to_dict()
full_summary["mean_cap_minus_full"] = [diff_map[(r.cell, r.model)] for r in full_summary.itertuples()]
core_summary = core_summary.merge(full_summary, on=["cell", "model"], how="left")
core_summary["cell_label"] = core_summary.cell.map(CELL_LABEL)
core_summary["model_label"] = core_summary.model.map(MODEL_LABEL)
core_summary.to_csv(OUT / "core_panel_round4.csv", index=False)

# ------------------------------------------------------------------
# Rare-class threshold sensitivity
# ------------------------------------------------------------------
threshold_rows = []
threshold_comparison_rows = []
for threshold in [0.001, 0.0025, 0.005, 0.01]:
    for cell in CELLS:
        R = rare_set(cell, threshold)
        rr = (PCc[(PCc.cell == cell) & PCc.model.isin(CORE) & PCc["class"].isin(R)]
              .groupby(["seed", "model"], as_index=False).recall.mean()
              .rename(columns={"recall": "rare_recall"}))
        merged = Mc[Mc.cell == cell].merge(rr, on=["seed", "model"])
        means = merged.groupby("model").agg(accuracy=("accuracy", "mean"), rare_recall=("rare_recall", "mean"))
        rho = float(means.accuracy.corr(means.rare_recall, method="spearman"))
        top3 = set(means.accuracy.nlargest(3).index)
        bottom3 = set(means.rare_recall.nsmallest(3).index)
        threshold_rows.append({
            "threshold": threshold, "cell": cell, "n_rare": len(R), "spearman": rho,
            "same_three": top3 == bottom3, "accuracy_leader": means.accuracy.idxmax(),
            "rare_recall_leader": means.rare_recall.idxmax(), "rare_classes": "; ".join(R),
            **{f"rr_{model}": float(means.loc[model, "rare_recall"]) for model in CORE},
        })
        pvt = merged.pivot(index="seed", columns="model", values="rare_recall")
        for fm in FM:
            for classical in CLASSICAL:
                d = (pvt[fm] - pvt[classical]).dropna().to_numpy()
                threshold_comparison_rows.append({
                    "threshold": threshold, "cell": cell, "fm": fm, "classical": classical,
                    "mean_delta": float(d.mean()), "wins": int((d > 0).sum()),
                    "p_raw": signflip(d),
                })
threshold_df = pd.DataFrame(threshold_rows)
threshold_cmp = pd.DataFrame(threshold_comparison_rows)
threshold_df.to_csv(OUT / "rare_threshold_sensitivity.csv", index=False)
threshold_cmp.to_csv(OUT / "rare_threshold_pairwise.csv", index=False)
threshold_counts = (threshold_cmp.assign(
    outcome=np.select([threshold_cmp.wins.eq(10), threshold_cmp.wins.eq(0)], ["FM 10/10", "classical 10/10"], default="mixed")
).groupby(["threshold", "outcome"]).size().unstack(fill_value=0).reset_index())
threshold_counts.to_csv(OUT / "rare_threshold_pairwise_counts.csv", index=False)

# ------------------------------------------------------------------
# Factorial size x balance in two sources
# ------------------------------------------------------------------
def factorial_summary(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = df.groupby(["arm", "size", "balance"], as_index=False).agg(
        n_train=("n_train", "mean"), n_unique=("n_unique_rows", "mean"),
        oversample_factor=("oversample_factor", "mean"), accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), rare_recall=("rare_recall", "mean"),
        rare_recall_sd=("rare_recall", "std"), fit_s=("fit_s", "mean")
    )
    summary.insert(0, "dataset", dataset)
    contrasts = [
        ("balance at small budget", "small_balanced", "small_natural"),
        ("balance at large budget", "large_balanced", "large_natural"),
        ("size under natural sampling", "large_natural", "small_natural"),
        ("size under balanced sampling", "large_balanced", "small_balanced"),
    ]
    rows = []
    for label, left, right in contrasts:
        for metric in ["accuracy", "macro_f1", "rare_recall"]:
            pvt = df.pivot(index="seed", columns="arm", values=metric)
            d = (pvt[left] - pvt[right]).to_numpy()
            lo, hi = bootstrap_ci(d)
            rows.append({
                "dataset": dataset, "contrast": label, "metric": metric,
                "left": left, "right": right, "mean_delta": float(d.mean()),
                "ci_lo": lo, "ci_hi": hi, "wins": int((d > 0).sum()),
                "p_raw": signflip(d),
            })
    contrasts_df = pd.DataFrame(rows)
    return summary, contrasts_df

cic_fact_summary, cic_fact_contrasts = factorial_summary(CIC_FACT, "CIC-IDS2017 V3")
unsw_fact_summary, unsw_fact_contrasts = factorial_summary(UNSW_FACT, "CIC-UNSW-NB15 V1")
fact_summary = pd.concat([cic_fact_summary, unsw_fact_summary], ignore_index=True)
fact_contrasts = pd.concat([cic_fact_contrasts, unsw_fact_contrasts], ignore_index=True)
# Holm within each dataset and metric over the four planned contrasts.
fact_contrasts["p_holm_4"] = np.nan
for (dataset, metric), idx in fact_contrasts.groupby(["dataset", "metric"]).groups.items():
    fact_contrasts.loc[list(idx), "p_holm_4"] = holm_adjust(fact_contrasts.loc[list(idx), "p_raw"])
fact_summary.to_csv(OUT / "factorial_two_source_summary.csv", index=False)
fact_contrasts.to_csv(OUT / "factorial_two_source_contrasts.csv", index=False)

# ------------------------------------------------------------------
# ManyClass 10-seed sensitivity
# ------------------------------------------------------------------
mc_summary_rows = []
for (a, r), g in MC.groupby(["alphabet_size", "n_estimators_redundancy"]):
    lo, hi = bootstrap_ci(g.rare_recall.to_numpy())
    mc_summary_rows.append({
        "alphabet_size": int(a), "n_estimators_redundancy": int(r), "n": len(g),
        "accuracy": float(g.accuracy.mean()), "macro_f1": float(g.macro_f1.mean()),
        "rare_recall": float(g.rare_recall.mean()), "rare_recall_sd": float(g.rare_recall.std(ddof=1)),
        "ci_lo": lo, "ci_hi": hi, "fit_s": float(g.fit_s.mean()),
    })
mc_summary = pd.DataFrame(mc_summary_rows).sort_values(["alphabet_size", "n_estimators_redundancy"])
base = MC[(MC.alphabet_size == 5) & (MC.n_estimators_redundancy == 1)].set_index("seed")
mc_pairs = []
for (a, r), g in MC.groupby(["alphabet_size", "n_estimators_redundancy"]):
    if (a, r) == (5, 1):
        continue
    other = g.set_index("seed")
    d = other.loc[base.index, "rare_recall"].to_numpy() - base.rare_recall.to_numpy()
    lo, hi = bootstrap_ci(d)
    mc_pairs.append({
        "alphabet_size": int(a), "n_estimators_redundancy": int(r),
        "mean_delta_vs_a5r1": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
        "wins": int((d > 0).sum()), "p_raw": signflip(d),
    })
mc_pairs = pd.DataFrame(mc_pairs)
mc_pairs["p_holm_3"] = holm_adjust(mc_pairs.p_raw)
mc_summary["spread_from_min"] = mc_summary.rare_recall - mc_summary.rare_recall.min()
mc_summary.to_csv(OUT / "manyclass_ten_seed_summary.csv", index=False)
mc_pairs.to_csv(OUT / "manyclass_paired_tests.csv", index=False)

# ------------------------------------------------------------------
# Axis 2 data-rich references and current campaign costs
# ------------------------------------------------------------------
a2_rich_summary = A2_RICH.groupby(["reference", "learner"], as_index=False).agg(
    n_fit=("n_fit", "mean"), n_fit_unique=("n_fit_unique", "mean"), n_eval=("n_eval", "mean"),
    accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), fit_s=("fit_s", "mean")
)
anchor = (A2_RAW_SUM[A2_RAW_SUM.model == "RandomForest"]
          .drop_duplicates("seed").set_index("seed")
          .rename(columns={"macro_f1": "macro_f1_closed"}))
kimi = A2_SEED[A2_SEED.model == "Kimi-K3"].set_index("seed")
a2_comparisons = []
for (reference, learner), g in A2_RICH.groupby(["reference", "learner"]):
    candidate = g.set_index("seed")
    for comparator_name, comparator, metric_name in [
        ("matched 45-row RF", anchor, "macro_f1_closed"),
        ("Kimi-K3", kimi, "macro_f1_closed"),
    ]:
        common = sorted(set(candidate.index) & set(comparator.index))
        d = candidate.loc[common, "macro_f1"].to_numpy() - comparator.loc[common, metric_name].to_numpy()
        lo, hi = bootstrap_ci(d)
        a2_comparisons.append({
            "reference": reference, "learner": learner, "comparator": comparator_name,
            "mean_delta_macro_f1": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "wins": int((d > 0).sum()), "p_raw": signflip(d),
        })
a2_comparisons = pd.DataFrame(a2_comparisons)
a2_comparisons["p_holm_8"] = holm_adjust(a2_comparisons.p_raw)
a2_rich_summary.to_csv(OUT / "axis2_datarich_summary.csv", index=False)
a2_comparisons.to_csv(OUT / "axis2_datarich_comparisons.csv", index=False)

# Cost per 1,000 scored rows from the current six-model campaign.  The
# corrected Nemotron rerun supersedes the original Nemotron endpoint record.
a2_cost_source = pd.concat([
    A2_RAW_SUM[(A2_RAW_SUM.kind == "llm") & ~A2_RAW_SUM.model.str.contains("Nemotron", case=False, na=False)],
    A2_CORRECTED[(A2_CORRECTED.kind == "llm") & A2_CORRECTED.model.str.contains("Nemotron", case=False, na=False)],
], ignore_index=True)
a2_cost = a2_cost_source.groupby("model", as_index=False).agg(
    total_usd=("usd", "sum"), n_seed_rows=("seed", "size"), median_latency_s=("latency_median_s", "median")
)
# All current seeds use the same evaluation count, visible in the closed-label record.
n_eval_total = A2_SEED[A2_SEED.model != "RandomForest"].groupby("model").n_eval.sum()
a2_cost["n_scored"] = a2_cost.model.map(n_eval_total).astype(int)
a2_cost["usd_per_1000"] = 1000 * a2_cost.total_usd / a2_cost.n_scored
a2_cost.to_csv(OUT / "axis2_current_cost_per_1000.csv", index=False)

# Preliminary screening tables are read from the archived experiment CSVs.
# The grouped v7_5 directory is the five-shot primary configuration cited in Round 2.
legacy_kshot = pd.read_csv(HIST / "e2_summary.csv")
legacy_cost = pd.read_csv(HIST / "e5_cost.csv")
legacy_scaling = pd.read_csv(HIST / "e9_pairs.csv")
legacy_kshot.to_csv(OUT / "legacy_axis2_kshot.csv", index=False)
legacy_cost.to_csv(OUT / "legacy_axis2_cost.csv", index=False)
legacy_scaling.to_csv(OUT / "legacy_axis2_scaling.csv", index=False)

# ------------------------------------------------------------------
# Cap validation for operational quantities and cost crossovers
# ------------------------------------------------------------------
def alarm_quantities(matrix: pd.DataFrame, cell: str, rare: list[str], weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = weights or {}
    benign = BENIGN[cell]
    false_alarm = float(matrix[(matrix.true_label == benign) & (matrix.pred_label != benign)]["count"].sum())
    miss_rows = matrix[(matrix.true_label != benign) & (matrix.pred_label == benign)]
    miss = float((miss_rows["count"] * miss_rows.true_label.map(lambda x: weights.get(x, 1.0))).sum())
    rare_alerts = float(matrix[matrix.pred_label.isin(rare)]["count"].sum())
    rare_tp = float(matrix[(matrix.true_label == matrix.pred_label) & matrix.true_label.isin(rare)]["count"].sum())
    return {
        "false_alarm": false_alarm, "miss": miss, "rare_alerts": rare_alerts, "rare_tp": rare_tp,
        "alerts_per_tp": rare_alerts / rare_tp if rare_tp > 0 else float("nan"),
    }

cap_op_rows = []
for cell in CELLS:
    for model in CLASSICAL:
        cap_meta = M[(M.cell == cell) & (M.model == model) & (M.test_set == "capped")].set_index("seed")
        full_meta = M[(M.cell == cell) & (M.model == model) & (M.test_set == "full")].set_index("seed")
        for seed in sorted(set(cap_meta.index) & set(full_meta.index)):
            if str(cap_meta.loc[seed, "test_hash"]) != str(full_meta.loc[seed, "test_hash"]):
                continue
            cap_cm = CM[(CM.cell == cell) & (CM.model == model) & (CM.seed == seed) & (CM.test_set == "capped")]
            full_cm = CM[(CM.cell == cell) & (CM.model == model) & (CM.seed == seed) & (CM.test_set == "full")]
            if cap_cm.empty or full_cm.empty:
                continue
            qc = alarm_quantities(cap_cm, cell, RARE[cell])
            qf = alarm_quantities(full_cm, cell, RARE[cell])
            pc_cap = PC[(PC.cell == cell) & (PC.model == model) & (PC.seed == seed) & (PC.test_set == "capped") & PC["class"].isin(RARE[cell])]
            pc_full = PC[(PC.cell == cell) & (PC.model == model) & (PC.seed == seed) & (PC.test_set == "full") & PC["class"].isin(RARE[cell])]
            row = {"cell": cell, "model": model, "seed": int(seed), "regime": "large" if model.endswith("_full") else "balanced"}
            for k, v in qc.items(): row[f"{k}_cap"] = v
            for k, v in qf.items(): row[f"{k}_full"] = v
            row["rare_recall_cap"] = float(pc_cap.recall.mean())
            row["rare_recall_full"] = float(pc_full.recall.mean())
            row["rare_precision_cap"] = float(pc_cap.precision.mean())
            row["rare_precision_full"] = float(pc_full.precision.mean())
            cap_op_rows.append(row)
cap_op = pd.DataFrame(cap_op_rows)
for metric in ["false_alarm", "miss", "alerts_per_tp", "rare_recall", "rare_precision"]:
    cap_op[f"d_{metric}"] = cap_op[f"{metric}_cap"] - cap_op[f"{metric}_full"]
cap_op["rel_false_alarm_error"] = cap_op.d_false_alarm / cap_op.false_alarm_full.replace(0, np.nan)
cap_op["rel_miss_error"] = cap_op.d_miss / cap_op.miss_full.replace(0, np.nan)
cap_op.to_csv(OUT / "cap_operational_matching_rows.csv", index=False)
cap_summary = cap_op.groupby(["cell", "regime"], as_index=False).agg(
    n=("seed", "size"), mean_d_rr=("d_rare_recall", "mean"), max_abs_d_rr=("d_rare_recall", lambda s: float(np.nanmax(np.abs(s)))),
    mean_d_precision=("d_rare_precision", "mean"), max_abs_d_precision=("d_rare_precision", lambda s: float(np.nanmax(np.abs(s)))),
    mean_d_aptp=("d_alerts_per_tp", "mean"), max_abs_d_aptp=("d_alerts_per_tp", lambda s: float(np.nanmax(np.abs(s)))),
    mean_rel_fa=("rel_false_alarm_error", "mean"), max_abs_rel_fa=("rel_false_alarm_error", lambda s: float(np.nanmax(np.abs(s)))),
)
cap_summary.to_csv(OUT / "cap_operational_summary.csv", index=False)

# Capped vs full crossover against pruned RF for classical configurations.
cap_crossover_rows = []
for cell in CELLS:
    for scoring in ["cap", "full"]:
        g = cap_op[cap_op.cell == cell]
        base = g[g.model == "rf_pruned_full"].set_index("seed")
        for model in [m for m in CLASSICAL if m != "rf_pruned_full"]:
            cand = g[g.model == model].set_index("seed")
            for seed in sorted(set(base.index) & set(cand.index)):
                miss_gain = base.loc[seed, f"miss_{scoring}"] - cand.loc[seed, f"miss_{scoring}"]
                extra_fa = cand.loc[seed, f"false_alarm_{scoring}"] - base.loc[seed, f"false_alarm_{scoring}"]
                rho = extra_fa / miss_gain if abs(miss_gain) > 1e-12 else float("nan")
                cap_crossover_rows.append({"cell": cell, "model": model, "seed": seed, "scoring": scoring, "rho": rho})
cap_crossover = pd.DataFrame(cap_crossover_rows)
cap_cross_wide = cap_crossover.pivot_table(index=["cell", "model", "seed"], columns="scoring", values="rho").reset_index()
cap_cross_wide = cap_cross_wide[(cap_cross_wide.cap > 0) & (cap_cross_wide.full > 0)]
cap_cross_wide["relative_error"] = (cap_cross_wide.cap - cap_cross_wide.full) / cap_cross_wide.full
cap_cross_summary = cap_cross_wide.groupby(["cell", "model"], as_index=False).agg(
    n=("seed", "size"), median_cap=("cap", "median"), median_full=("full", "median"),
    mean_relative_error=("relative_error", "mean"), max_abs_relative_error=("relative_error", lambda s: float(np.max(np.abs(s))))
)
cap_cross_wide.to_csv(OUT / "cap_crossover_matching_rows.csv", index=False)
cap_cross_summary.to_csv(OUT / "cap_crossover_summary.csv", index=False)

# ------------------------------------------------------------------
# Severity vector and miss/false-alarm crossover sensitivity
# ------------------------------------------------------------------
severity_rows = []
for cell in CELLS:
    shares = SUPPORT[cell] / SUPPORT[cell].sum()
    for cls, share in shares.items():
        if cls == BENIGN[cell]:
            continue
        w = float(SEVERITY.get(cls, 1.0))
        severity_rows.append({
            "cell": cell, "class": cls, "prevalence_pct": 100 * float(share), "rare": cls in RARE[cell],
            "uniform": 1.0, "attenuated": 1.0 + 0.5 * (w - 1.0),
            "declared": w, "amplified": 1.0 + 2.0 * (w - 1.0),
        })
severity_df = pd.DataFrame(severity_rows)
severity_df.to_csv(OUT / "severity_weight_vectors.csv", index=False)

weight_vectors: dict[str, dict[str, dict[str, float]]] = {}
for cell in CELLS:
    g = severity_df[severity_df.cell == cell]
    weight_vectors[cell] = {
        name: dict(zip(g["class"], g[name])) for name in ["uniform", "attenuated", "declared", "amplified"]
    }

cost_rows = []
for cell in CELLS:
    for vector_name, weights in weight_vectors[cell].items():
        quantities = {}
        for model in CORE:
            rows = []
            for seed, matrix in CMc[(CMc.cell == cell) & (CMc.model == model)].groupby("seed"):
                q = alarm_quantities(matrix, cell, RARE[cell], weights)
                rows.append({"seed": int(seed), **q})
            quantities[model] = pd.DataFrame(rows).set_index("seed")
        for incumbent in ["rf_pruned_full", "rf_strong_full", "xgb_full"]:
            base = quantities[incumbent]
            for candidate in [m for m in CORE if m != incumbent]:
                cand = quantities[candidate]
                for seed in sorted(set(base.index) & set(cand.index)):
                    miss_gain = base.loc[seed, "miss"] - cand.loc[seed, "miss"]
                    extra_fa = cand.loc[seed, "false_alarm"] - base.loc[seed, "false_alarm"]
                    rho = extra_fa / miss_gain if abs(miss_gain) > 1e-12 else float("nan")
                    status = "positive" if np.isfinite(rho) and rho > 0 else "non-positive-or-undefined"
                    cost_rows.append({
                        "cell": cell, "vector": vector_name, "incumbent": incumbent, "candidate": candidate,
                        "seed": seed, "miss_gain": miss_gain, "extra_false_alarms": extra_fa,
                        "rho": rho, "status": status,
                    })
cost_seed = pd.DataFrame(cost_rows)
cost_seed.to_csv(OUT / "cost_crossover_all_vectors_per_seed.csv", index=False)
cost_summary_rows = []
for keys, g in cost_seed.groupby(["cell", "vector", "incumbent", "candidate"]):
    positive = g[(g.status == "positive") & np.isfinite(g.rho)].rho.to_numpy()
    cost_summary_rows.append({
        "cell": keys[0], "vector": keys[1], "incumbent": keys[2], "candidate": keys[3],
        "positive_seeds": len(positive), "rho_median": float(np.median(positive)) if len(positive) else np.nan,
        "rho_q1": float(np.quantile(positive, .25)) if len(positive) else np.nan,
        "rho_q3": float(np.quantile(positive, .75)) if len(positive) else np.nan,
        "one_over_rho": float(1 / np.median(positive)) if len(positive) and np.median(positive) > 0 else np.nan,
    })
cost_summary = pd.DataFrame(cost_summary_rows)
cost_summary.to_csv(OUT / "cost_crossover_all_vectors_summary.csv", index=False)

# ------------------------------------------------------------------
# Precision-aware rare-class evidence, support/resolution, and TabSTAR
# ------------------------------------------------------------------
op = PCc.groupby(["cell", "model", "class"], as_index=False).agg(
    precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), support=("support", "mean")
)
op = op[[cls in RARE[cell] for cell, cls in zip(op.cell, op["class"])]]
op["operating_point"] = np.select(
    [(op.precision >= .5) & (op.recall >= .5),
     (op.precision < .5) & (op.recall >= .5),
     (op.precision >= .5) & (op.recall < .5)],
    ["recovered", "high recall / low precision", "high precision / low recall"], default="missed"
)
op["alerts_per_tp_class"] = np.where(op.precision > 0, 1 / op.precision, np.inf)
op.to_csv(OUT / "rare_per_class_operating_points.csv", index=False)

selected_models = {
    "cic_ids2017_V1": ["xgb_full", "tabpfn_2.5", "rf_pruned_full", "tabstar"],
    "cic_unsw_V2": ["xgb_full", "tabpfn_2.5", "rf_pruned_full"],
}
op_summary_rows = []
for cell, models in selected_models.items():
    for model in models:
        g = op[(op.cell == cell) & (op.model == model)]
        counts = g.operating_point.value_counts()
        ratios = []
        for seed, matrix in CMc[(CMc.cell == cell) & (CMc.model == model)].groupby("seed"):
            q = alarm_quantities(matrix, cell, RARE[cell])
            ratios.append(q["alerts_per_tp"])
        op_summary_rows.append({
            "cell": cell, "model": model, "recovered": int(counts.get("recovered", 0)),
            "high_recall_low_precision": int(counts.get("high recall / low precision", 0)),
            "high_precision_low_recall": int(counts.get("high precision / low recall", 0)),
            "missed": int(counts.get("missed", 0)), "alerts_per_tp": float(np.mean(ratios)),
        })
op_summary = pd.DataFrame(op_summary_rows)
op_summary.to_csv(OUT / "precision_aware_selected_summary.csv", index=False)

resolution_rows = []
for cell in CELLS:
    for cls in RARE[cell]:
        support = float(SUPPORT[cell][cls])
        # Persisted support is the reweighted full-partition support; the discrete
        # cap resolution uses the unweighted cap count where available. Infer it
        # from the number of nonzero weighted instances only when exact count is
        # stored in the notebook/manuscript; otherwise report full support.
        resolution_rows.append({"cell": cell, "class": cls, "weighted_full_support": support, "full_recall_step": 1.0 / support})
resolution_df = pd.DataFrame(resolution_rows)
resolution_df.to_csv(OUT / "rare_class_support_resolution.csv", index=False)

# ------------------------------------------------------------------
# Descriptive continuity tables from Round 2, transcribed from the persisted
# five-seed experiment record and cross-checked against its archived source.
# ------------------------------------------------------------------
omitted_rows = [
    # CIC-IDS2017 V2 and V4: values in the archived five-seed screening tables.
    ("CIC-IDS2017 V2", "Pruned RF, large", .9881, .3880, .192),
    ("CIC-IDS2017 V2", "Pruned RF, balanced", .9321, .4980, .732),
    ("CIC-IDS2017 V2", "Strong RF, balanced", .9652, .5640, .791),
    ("CIC-IDS2017 V2", "XGBoost, large", .9981, .7920, .706),
    ("CIC-IDS2017 V2", "TabPFN", .9531, .5880, np.nan),
    ("CIC-IDS2017 V2", "TabPFN 2.5", .9773, .6060, np.nan),
    ("CIC-IDS2017 V2", "TabICL", .9710, .6250, np.nan),
    ("CIC-IDS2017 V2", "TabSTAR", .9150, .4880, np.nan),
    ("CIC-IDS2017 V4", "Pruned RF, large", .9903, .4930, .212),
    ("CIC-IDS2017 V4", "Pruned RF, balanced", .9520, .5420, .831),
    ("CIC-IDS2017 V4", "Strong RF, balanced", .9689, .5810, .857),
    ("CIC-IDS2017 V4", "XGBoost, large", .9984, .8510, .705),
    ("CIC-IDS2017 V4", "TabPFN", .9590, .6000, np.nan),
    ("CIC-IDS2017 V4", "TabPFN 2.5", .9808, .6300, np.nan),
    ("CIC-IDS2017 V4", "TabICL", .9721, .6530, np.nan),
    ("CIC-IDS2017 V4", "TabSTAR", .9387, .5573, np.nan),
    ("N-BaIoT V1", "Random Forest", .9953, .9834, np.nan),
    ("N-BaIoT V1", "TabPFN", .9568, .8749, np.nan),
    ("N-BaIoT V1", "TabPFN 2.5", .9497, .8659, np.nan),
    ("N-BaIoT V1", "TabICL", .9572, .8764, np.nan),
    ("N-BaIoT V1", "TabSTAR", .9907, .9872, np.nan),
    ("N-BaIoT V2", "Random Forest", .9212, .9190, np.nan),
    ("N-BaIoT V2", "TabPFN", .9537, .8706, np.nan),
    ("N-BaIoT V2", "TabPFN 2.5", .9575, .8883, np.nan),
    ("N-BaIoT V2", "TabICL", .9545, .8730, np.nan),
    ("N-BaIoT V2", "TabSTAR", .9633, .9641, np.nan),
    ("N-BaIoT V3", "Random Forest", .9998, .9997, np.nan),
    ("N-BaIoT V3", "TabPFN", .9997, .9995, np.nan),
    ("N-BaIoT V3", "TabPFN 2.5", .9997, .9994, np.nan),
    ("N-BaIoT V3", "TabICL", .9996, .9993, np.nan),
    ("N-BaIoT V3", "TabSTAR", .9991, .9985, np.nan),
    ("N-BaIoT V4", "Random Forest", .9947, .9918, np.nan),
    ("N-BaIoT V4", "TabPFN", .9992, .9987, np.nan),
    ("N-BaIoT V4", "TabPFN 2.5", .9993, .9988, np.nan),
    ("N-BaIoT V4", "TabICL", .9989, .9982, np.nan),
    ("N-BaIoT V4", "TabSTAR", .9979, .9966, np.nan),
]
omitted = pd.DataFrame(omitted_rows, columns=["cell", "model", "accuracy", "macro_f1", "rare_recall_if_available"])
omitted["evidence"] = "Five-seed preliminary descriptive record"
omitted.to_csv(OUT / "legacy_omitted_cells.csv", index=False)

# ------------------------------------------------------------------
# LaTeX table fragments
# ------------------------------------------------------------------
ROW = r" \\"

def write_tex_table(path: Path, rows: list[str]) -> None:
    fixed = []
    for line in rows:
        if line.endswith("\\") and not line.endswith("\\\\"):
            line += "\\"
        fixed.append(line)
    path.write_text("\n".join(fixed) + "\n", encoding="utf-8")

# Core panel table
lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Cell & Configuration & Acc. & Macro-$F_1$ (cap) & Macro-$F_1$ (full) & $n_{full}$ & Rare recall & SD & ranks A/F/R \\", r"\midrule"]
for cell in CELLS:
    g = core_summary[core_summary.cell == cell].sort_values("accuracy_rank")
    for j, row in enumerate(g.itertuples()):
        mf_full = "--" if pd.isna(row.macro_f1_full) else f"{row.macro_f1_full:.3f}"
        nfull = "--" if pd.isna(row.n_full) else str(int(row.n_full))
        label = CELL_LABEL[cell] if j == 0 else ""
        lines.append(f"{label} & {tex(MODEL_LABEL[row.model])} & {row.accuracy:.4f} & {row.macro_f1_cap:.3f} & {mf_full} & {nfull} & {row.rare_recall:.3f} & {row.rare_recall_sd:.4f} & {row.accuracy_rank}/{row.macro_f1_rank}/{row.rare_recall_rank} \\")
    if cell != CELLS[-1]: lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_core_panel_round4.tex", lines)

# Threshold table
lines = [r"\begin{tabular}{@{}llrrcllrrrr@{}}", r"\toprule",
         r"Threshold & Cell & $|\mathcal R|$ & $\rho_s$ & Same three & Accuracy leader & Recall leader & XGB/L & XGB/B & TabPFN & TabICL \\", r"\midrule"]
for threshold in [0.001, 0.0025, 0.005, 0.01]:
    g = threshold_df[threshold_df.threshold == threshold]
    for j, (_, row) in enumerate(g.iterrows()):
        label = f"{100*threshold:g}\\%" if j == 0 else ""
        lines.append(
            f"{label} & {CELL_LABEL[row['cell']]} & {int(row['n_rare'])} & "
            f"{row['spearman']:+.3f} & {'yes' if bool(row['same_three']) else 'no'} & "
            f"{ABBR[row['accuracy_leader']]} & {ABBR[row['rare_recall_leader']]} & "
            f"{row['rr_xgb_full']:.3f} & {row['rr_xgb_balanced']:.3f} & "
            f"{row['rr_tabpfn_2.5']:.3f} & {row['rr_tabicl']:.3f}" + ROW
        )
    if threshold != .01:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_threshold_sensitivity_round4.tex", lines)

# Factorial means and effects
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Dataset & Arm & Presented & Unique & Oversampling & Accuracy & Macro-$F_1$ & Rare recall \\", r"\midrule"]
for dataset in ["CIC-IDS2017 V3", "CIC-UNSW-NB15 V1"]:
    g = fact_summary[fact_summary.dataset == dataset]
    order = ["small_natural", "small_balanced", "large_natural", "large_balanced"]
    for j, arm in enumerate(order):
        row = g[g.arm == arm].iloc[0]
        label = dataset if j == 0 else ""
        lines.append(f"{label} & {arm.replace('_',' ')} & {row.n_train:,.0f} & {row.n_unique:,.0f} & {row.oversample_factor:.2f}$\times$ & {row.accuracy:.4f} & {row.macro_f1:.3f} & {row.rare_recall:.3f} \\")
    if dataset != "CIC-UNSW-NB15 V1": lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_factorial_means_round4.tex", lines)

rrc = fact_contrasts[fact_contrasts.metric == "rare_recall"]
lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Dataset & Contrast & Mean $\Delta$ & 95\% CI & Positive seeds & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for dataset in ["CIC-IDS2017 V3", "CIC-UNSW-NB15 V1"]:
    g = rrc[rrc.dataset == dataset]
    for j, row in enumerate(g.itertuples()):
        label = dataset if j == 0 else ""
        lines.append(f"{label} & {row.contrast} & {row.mean_delta:+.3f} & [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] & {row.wins}/10 & {row.p_holm_4:.4f} \\")
    if dataset != "CIC-UNSW-NB15 V1": lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_factorial_effects_round4.tex", lines)

# ManyClass
lines = [r"\begin{tabular}{@{}rrrrrrrr@{}}", r"\toprule",
         r"Alphabet & Redundancy & $n$ & Accuracy & Macro-$F_1$ & Rare recall & 95\% CI & Fit (s) \\", r"\midrule"]
for row in mc_summary.itertuples():
    lines.append(f"{row.alphabet_size} & {row.n_estimators_redundancy} & {row.n} & {row.accuracy:.4f} & {row.macro_f1:.3f} & {row.rare_recall:.3f} & [{row.ci_lo:.3f}, {row.ci_hi:.3f}] & {row.fit_s:.2f} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_manyclass_round4.tex", lines)

# Axis2 data-rich
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Fitting reference & Learner & Rows & Unique & Accuracy & Macro-$F_1$ & $\Delta$ vs 45-row RF & $\Delta$ vs Kimi \\", r"\midrule"]
for row in a2_rich_summary.itertuples():
    d_rf = a2_comparisons[(a2_comparisons.reference == row.reference) & (a2_comparisons.learner == row.learner) & (a2_comparisons.comparator == "matched 45-row RF")].iloc[0]
    d_k = a2_comparisons[(a2_comparisons.reference == row.reference) & (a2_comparisons.learner == row.learner) & (a2_comparisons.comparator == "Kimi-K3")].iloc[0]
    lines.append(f"{row.reference} & {row.learner} & {row.n_fit:,.0f} & {row.n_fit_unique:,.0f} & {row.accuracy:.3f} & {row.macro_f1:.3f} & {d_rf.mean_delta_macro_f1:+.3f} & {d_k.mean_delta_macro_f1:+.3f} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_axis2_datarich_round4.tex", lines)

# Current Axis2 per-1000 cost
lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Model & Scored rows & Campaign USD & USD / 1{,}000 & Median latency (s) \\", r"\midrule"]
for row in a2_cost.sort_values("total_usd", ascending=False).itertuples():
    lines.append(f"{tex(row.model)} & {row.n_scored:,} & {row.total_usd:.2f} & {row.usd_per_1000:.2f} & {row.median_latency_s:.2f} \\")
lines += [r"\midrule", f"Total & {a2_cost.n_scored.sum():,} & {a2_cost.total_usd.sum():.2f} & -- & -- \\", r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_axis2_cost_round4.tex", lines)

# Cap operational summary
lines = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
         r"Cell & Regime & $n$ & mean $\Delta R$ & max $|\Delta R|$ & mean $\Delta P$ & max $|\Delta P|$ & mean $\Delta$ alerts/TP & max $|\Delta|$ & mean rel. FA err. & max $|$rel. FA err.$|$ \\", r"\midrule"]
for row in cap_summary.itertuples():
    lines.append(f"{CELL_LABEL[row.cell]} & {row.regime} & {row.n} & {row.mean_d_rr:+.4f} & {row.max_abs_d_rr:.4f} & {row.mean_d_precision:+.4f} & {row.max_abs_d_precision:.4f} & {row.mean_d_aptp:+.3f} & {row.max_abs_d_aptp:.3f} & {row.mean_rel_fa:+.3f} & {row.max_abs_rel_fa:.3f} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_cap_operational_round4.tex", lines)

# Cost selected: candidates against XGB/L under four vectors
sel = cost_summary[(cost_summary.incumbent == "xgb_full") & cost_summary.candidate.isin(["xgb_balanced", "tabpfn_2.5"])].copy()
lines = [r"\begin{tabular}{@{}llllrr@{}}", r"\toprule",
         r"Cell & Candidate vs XGB/L & Weight vector & Positive seeds & Median $\rho^*$ & $1/\rho^*$ \\", r"\midrule"]
for cell in CELLS:
    g = sel[sel.cell == cell]
    for j, row in enumerate(g.sort_values(["candidate", "vector"]).itertuples()):
        label = CELL_LABEL[cell] if j == 0 else ""
        inv = "--" if pd.isna(row.one_over_rho) else f"{row.one_over_rho:.3f}"
        rho = "--" if pd.isna(row.rho_median) else f"{row.rho_median:.3f}"
        lines.append(f"{label} & {ABBR[row.candidate]} & {row.vector} & {row.positive_seeds}/10 & {rho} & {inv} \\")
    if cell != CELLS[-1]: lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_cost_sensitivity_round4.tex", lines)

# Severity weights (unique taxonomies only)
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Taxonomy & Class & Prev. (\%) & Rare & Uniform & Attenuated & Declared & Amplified \\", r"\midrule"]
for cell, label in [("cic_ids2017_V1", "CIC-IDS2017 V1"), ("cic_ids2017_V3", "CIC-IDS2017 V3"), ("cic_unsw_V1", "CIC-UNSW-NB15")]:
    g = severity_df[severity_df.cell == cell].sort_values("prevalence_pct")
    for j, (_, row) in enumerate(g.iterrows()):
        lab = label if j == 0 else ""
        lines.append(
            f"{lab} & {tex(row['class'])} & {row['prevalence_pct']:.4f} & "
            f"{'yes' if bool(row['rare']) else 'no'} & {row['uniform']:g} & "
            f"{row['attenuated']:g} & {row['declared']:g} & {row['amplified']:g}" + ROW
        )
    if cell != "cic_unsw_V1":
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_severity_vectors_round4.tex", lines)

# Precision-aware selected summary
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Cell & Model & Recovered & High-R/low-P & High-P/low-R & Missed & Alerts/TP \\", r"\midrule"]
for cell in selected_models:
    g = op_summary[op_summary.cell == cell]
    for j, row in enumerate(g.itertuples()):
        lab = CELL_LABEL[cell] if j == 0 else ""
        lines.append(f"{lab} & {MODEL_LABEL[row.model]} & {row.recovered} & {row.high_recall_low_precision} & {row.high_precision_low_recall} & {row.missed} & {row.alerts_per_tp:.2f} \\")
    if cell != list(selected_models)[-1]: lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_precision_aware_round4.tex", lines)

# CIC V1 per-class precision/recall/support for selected models.
lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
         r"Class (support) & \multicolumn{2}{c}{XGB/L} & \multicolumn{2}{c}{TabPFN} & \multicolumn{2}{c}{TabSTAR} & \multicolumn{2}{c}{RF-P/L} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r" & P & R & P & R & P & R & P & R \\", r"\midrule"]
for cls in RARE["cic_ids2017_V1"]:
    values = []
    for model in ["xgb_full", "tabpfn_2.5", "tabstar", "rf_pruned_full"]:
        r = op[(op.cell == "cic_ids2017_V1") & (op.model == model) & (op["class"] == cls)].iloc[0]
        values.extend([r.precision, r.recall])
    support = SUPPORT["cic_ids2017_V1"][cls]
    lines.append(f"{tex(cls)} ({support:,.0f}) & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_rare_perclass_cic_v1_round4.tex", lines)

# Omitted/continuity cells condensed table.
lines = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule", r"Cell & Model & Accuracy & Macro-$F_1$ \\", r"\midrule"]
for cell in omitted.cell.unique():
    g = omitted[omitted.cell == cell]
    for j, row in enumerate(g.itertuples()):
        lab = cell if j == 0 else ""
        lines.append(f"{lab} & {tex(row.model)} & {row.accuracy:.4f} & {row.macro_f1:.4f} \\")
    if cell != omitted.cell.unique()[-1]: lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
write_tex_table(GEN / "table_legacy_omitted_cells_round4.tex", lines)

# Legacy k-shot table (preserve the actual CSV schema rather than hard-code values).
# Inspect and normalize likely column names.
legacy_kshot.to_latex(GEN / "table_legacy_kshot_raw.tex", index=False, escape=True, float_format=lambda x: f"{x:.3f}")
legacy_cost.to_latex(GEN / "table_legacy_cost_raw.tex", index=False, escape=True, float_format=lambda x: f"{x:.3f}")
legacy_scaling.to_latex(GEN / "table_legacy_scaling_raw.tex", index=False, escape=True, float_format=lambda x: f"{x:.3f}")

# Compact source summary for prose and response drafting.
summary = {
    "input_checks": {
        "manyclass_rows": len(MC), "unsw_factorial_rows": len(UNSW_FACT),
        "axis2_datarich_rows": len(A2_RICH), "axis2_datarich_perclass_rows": len(A2_RICH_PC),
        "axis2_second_variant_raw_records": A2_SECOND_INFO["n_raw_records"],
        "axis2_second_variant_complete": A2_SECOND_INFO["status"] == "PASS",
    },
    "threshold": {
        "n_cell_thresholds": len(threshold_df),
        "all_negative": bool((threshold_df.spearman < 0).all()),
        "all_same_three": bool(threshold_df.same_three.all()),
        "rho_min": float(threshold_df.spearman.min()), "rho_max": float(threshold_df.spearman.max()),
    },
    "factorial_rare_recall": rrc.to_dict("records"),
    "manyclass": {
        "spread": float(mc_summary.rare_recall.max() - mc_summary.rare_recall.min()),
        "summary": mc_summary.to_dict("records"), "paired": mc_pairs.to_dict("records"),
    },
    "axis2_datarich": {
        "summary": a2_rich_summary.to_dict("records"), "comparisons": a2_comparisons.to_dict("records"),
        "matched_anchor_macro_f1": float(A2_SUM[A2_SUM.model == "RandomForest"].macro_f1_closed.iloc[0]),
        "kimi_macro_f1": float(A2_SUM[A2_SUM.model == "Kimi-K3"].macro_f1_closed.iloc[0]),
    },
    "current_axis2_total_cost": float(a2_cost.total_usd.sum()),
    "axis2_second_variant": A2_SECOND_INFO,
    "combined_axis2_total_cost": float(a2_cost.total_usd.sum()) + float(A2_SECOND_INFO["total_cost_usd"]),
    "core_count": {"configurations": len(CORE), "cells": len(CELLS), "seeds": 10, "seed_level_fits": len(CORE)*len(CELLS)*10},
}
(OUT / "round4_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=2, default=str))
