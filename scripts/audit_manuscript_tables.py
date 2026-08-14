#!/usr/bin/env python3
"""Independent numerical audit of every statistical table in the manuscript.

The audit starts from the persisted run records rather than importing the table
builder. It recomputes the quantities used by the manuscript and compares them
with the released derived CSV and LaTeX fragments. A machine-readable report is
written to ``results/derived/table_statistical_audit.json``.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
AX1 = ROOT / "results" / "axis1"
AX2 = ROOT / "results" / "axis2"
SENS = ROOT / "results" / "sensitivity"
OUT = ROOT / "results" / "derived"
MANUSCRIPT_CANDIDATES = [
    ROOT.parent / "manuscript" / "manuscript_content.tex",
    ROOT / "manuscript_snapshot" / "manuscript_content.tex",
]
MANUSCRIPT = next((path for path in MANUSCRIPT_CANDIDATES if path.exists()), MANUSCRIPT_CANDIDATES[0])

CORE = [
    "rf_pruned_full", "rf_strong_full", "xgb_full",
    "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
    "tabpfn_2.5", "tabicl",
]
CLASSICAL = CORE[:6]
FM = ["tabpfn_2.5", "tabicl"]
CELLS = ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1", "cic_unsw_V2"]
CELL_LABEL = {
    "cic_ids2017_V1": "CIC-IDS2017 V1",
    "cic_ids2017_V3": "CIC-IDS2017 V3",
    "cic_unsw_V1": "CIC-UNSW-NB15 V1",
    "cic_unsw_V2": "CIC-UNSW-NB15 V2",
}
SEV = {
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

checks: list[dict[str, object]] = []


def record(name: str, details: str = "") -> None:
    checks.append({"name": name, "status": "PASS", "details": details})


def close(a: float, b: float, tol: float = 1e-10) -> None:
    if not np.isclose(a, b, rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"{a!r} != {b!r}")


def signflip(d: Sequence[float]) -> float:
    d = np.asarray(d, dtype=float)
    observed = abs(float(d.mean()))
    values = [
        abs(float(np.mean(d * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(d))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def bootstrap_ci(d: Sequence[float], n_boot: int = 50000, seed: int = 2026) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_tex_contains(filename: str, snippets: Iterable[str]) -> None:
    text = (OUT / filename).read_text()
    missing = [s for s in snippets if s not in text]
    if missing:
        raise AssertionError(f"{filename} missing {missing[:4]}")


# -------------------------------------------------------------------------
# Inputs and strict-panel completeness
# -------------------------------------------------------------------------
metrics = pd.read_csv(AX1 / "metrics.csv").drop_duplicates(
    ["cell", "seed", "model", "test_set"], keep="last"
)
per_class = pd.read_csv(AX1 / "per_class.csv")
confusion = pd.read_csv(AX1 / "confusion.csv")
core_metrics = metrics[(metrics.test_set == "capped") & metrics.model.isin(CORE)].copy()
assert len(core_metrics) == 320
assert set(core_metrics.cell) == set(CELLS)
assert core_metrics.groupby(["cell", "model"]).seed.nunique().eq(10).all()
assert core_metrics.groupby("cell").test_hash.nunique().eq(1).all()
record("Strict Axis-1 panel", "320 records; 4 cells x 8 configurations x 10 seeds; one parent test hash per cell")

# Rare classes and support record.
rare: dict[str, list[str]] = {}
benign: dict[str, str] = {}
support_by_cell: dict[str, pd.Series] = {}
for cell in CELLS:
    base = per_class[
        (per_class.cell == cell) & (per_class.seed == 0) &
        (per_class.model == "rf_pruned_full") & (per_class.test_set == "capped")
    ]
    support = base.groupby("class").support.mean()
    b = [label for label in support.index if str(label).lower() == "benign"]
    assert len(b) == 1
    benign[cell] = b[0]
    support_by_cell[cell] = support
    total = float(support.sum())
    rare[cell] = sorted(
        label for label, n in support.items()
        if label != b[0] and float(n) / total < 0.005
    )

expected_rare = {
    "cic_ids2017_V1": [
        "Bot", "DoS GoldenEye", "DoS Slowhttptest", "DoS slowloris",
        "FTP-Patator", "Heartbleed", "Infiltration", "SSH-Patator",
        "Web Attack  Brute Force", "Web Attack  Sql Injection", "Web Attack  XSS",
    ],
    "cic_ids2017_V3": ["Bot", "Brute Force", "Heartbleed", "Infiltration", "Web Attack"],
    "cic_unsw_V1": ["Analysis", "Backdoor", "DoS", "Generic", "Reconnaissance", "Shellcode", "Worms"],
    "cic_unsw_V2": ["Analysis", "Backdoor", "DoS", "Generic", "Reconnaissance", "Shellcode", "Worms"],
}
assert rare == expected_rare
record("Table 3 rare-class definitions", "11, 5, 7 and 7 classes under the predeclared 0.5% threshold")

# Dataset table, including the corrected held-out CIC benign share.
provenance = pd.read_csv(AX1 / "provenance.csv").set_index("cell")
assert int(provenance.loc["cic_ids2017_V1", "rows"]) == 2_522_362
assert int(provenance.loc["cic_ids2017_V3", "classes"]) == 9
assert int(provenance.loc["cic_unsw_V1", "rows"]) == 3_540_241
assert int(provenance.loc["cic_unsw_V2", "classes"]) == 10
for cell, heldout, share, smallest, smallest_n in [
    ("cic_ids2017_V1", 756_709, 0.831158344885558, "Heartbleed", 3),
    ("cic_ids2017_V3", 756_709, 0.831158344885554, "Heartbleed", 3),
    ("cic_unsw_V1", 1_062_073, 0.974695713006546, "Worms", 74),
    ("cic_unsw_V2", 1_062_073, 0.974695713006546, "Worms", 74),
]:
    support = support_by_cell[cell]
    close(float(support.sum()), heldout, 1e-9)
    close(float(support[benign[cell]] / support.sum()), share, 1e-12)
    assert support.drop(index=benign[cell]).idxmin() == smallest
    close(float(support[smallest]), smallest_n, 1e-9)
assert_tex_contains("table_dataset_cells.tex", ["83.1\\%", "97.5\\%", "Heartbleed (3)", "Worms (74)"])
record("Table 2 dataset cells", "Corrected CIC held-out benign share = 83.1%; all row, class and held-out counts verified")

# -------------------------------------------------------------------------
# Recompute Axis-1 summaries and ranks from per-seed metrics/per-class data.
# -------------------------------------------------------------------------
pc_core = per_class[(per_class.test_set == "capped") & per_class.model.isin(CORE)].copy()
pc_core["is_rare"] = [label in rare[cell] for cell, label in zip(pc_core.cell, pc_core["class"])]
rare_seed = (pc_core[pc_core.is_rare]
             .groupby(["cell", "seed", "model"], as_index=False)
             .agg(rare_recall=("recall", "mean")))
core = core_metrics.merge(rare_seed, on=["cell", "seed", "model"], how="left")
summary = (core.groupby(["cell", "model"], as_index=False)
           .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
                n_train=("n_train", "mean"), fit_s=("fit_s", "mean"), predict_s=("predict_s", "mean")))
summary["accuracy_rank"] = summary.groupby("cell").accuracy.rank(method="min", ascending=False).astype(int)
summary["recall_rank"] = summary.groupby("cell").rare_recall.rank(method="min", ascending=False).astype(int)

released_summary = pd.read_csv(OUT / "axis1_core_summary.csv")[summary.columns]
assert_frame_equal(
    summary.sort_values(["cell", "model"]).reset_index(drop=True),
    released_summary.sort_values(["cell", "model"]).reset_index(drop=True),
    check_exact=False, rtol=1e-12, atol=1e-12,
)
rank_expected = {
    "cic_ids2017_V1": -0.4523809523809524,
    "cic_ids2017_V3": -0.4523809523809524,
    "cic_unsw_V1": -0.4285714285714286,
    "cic_unsw_V2": -0.4761904761904762,
}
for cell in CELLS:
    g = summary[summary.cell == cell]
    rho = float(g.accuracy.corr(g.rare_recall, method="spearman"))
    close(rho, rank_expected[cell])
    top = set(g.nlargest(3, "accuracy").model)
    bottom = set(g.nsmallest(3, "rare_recall").model)
    assert top == bottom == {"rf_pruned_full", "rf_strong_full", "xgb_full"}
assert_tex_contains("table_rank_summary.tex", ["-0.452", "-0.429", "-0.476", "0.926", "0.695"])
record("Table 7 rank inversion", "All four Spearman coefficients and top-three/bottom-three memberships recomputed")

# -------------------------------------------------------------------------
# Cap validation table.
# -------------------------------------------------------------------------
cap_rows = []
for cell in CELLS:
    for model in CLASSICAL:
        g = metrics[(metrics.cell == cell) & (metrics.model == model)].pivot(index="seed", columns="test_set")
        for seed in sorted(set(metrics[(metrics.cell == cell) & (metrics.model == model)].seed)):
            try:
                if str(g.loc[seed, ("test_hash", "capped")]) != str(g.loc[seed, ("test_hash", "full")]):
                    continue
                cap_rows.append({
                    "cell": cell,
                    "model": model,
                    "seed": seed,
                    "regime": "large" if model.endswith("_full") else "balanced",
                    "d_accuracy": float(g.loc[seed, ("accuracy", "capped")] - g.loc[seed, ("accuracy", "full")]),
                    "d_macro_f1": float(g.loc[seed, ("macro_f1", "capped")] - g.loc[seed, ("macro_f1", "full")]),
                })
            except KeyError:
                pass
cap = pd.DataFrame(cap_rows)
cap_summary = (cap.groupby(["cell", "regime"], as_index=False)
               .agg(n=("seed", "size"), mean_d_accuracy=("d_accuracy", "mean"),
                    max_abs_d_accuracy=("d_accuracy", lambda x: np.abs(x).max()),
                    mean_d_macro_f1=("d_macro_f1", "mean"),
                    max_abs_d_macro_f1=("d_macro_f1", lambda x: np.abs(x).max())))
released_cap = pd.read_csv(OUT / "cap_validation_matching_hash_summary.csv")
assert_frame_equal(
    cap_summary.sort_values(["cell", "regime"]).reset_index(drop=True),
    released_cap.sort_values(["cell", "regime"]).reset_index(drop=True),
    check_exact=False, rtol=1e-12, atol=1e-12,
)
assert set(cap_summary[cap_summary.cell.str.startswith("cic_ids")].n) == {9}
assert set(cap_summary[cap_summary.cell.str.startswith("cic_unsw")].n) == {30}
assert_tex_contains("table_cap_validation.tex", ["0.093466", "-0.000084", "-0.000262", "0.006010"])
record("Table 5 cap validation", "156 matching-hash model-seed rows; all means and maxima verified")

# -------------------------------------------------------------------------
# Factorial size-by-balance table and paired tests.
# -------------------------------------------------------------------------
sb = pd.read_csv(SENS / "exp1_size_by_balance.csv")
sb_summary = (sb.groupby(["arm", "size", "balance"], as_index=False)
              .agg(n_train=("n_train", "mean"), n_unique=("n_unique_rows", "mean"),
                   accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                   rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std")))
released_sb = pd.read_csv(OUT / "size_balance_summary.csv")
for arm in sb_summary.arm:
    a = sb_summary[sb_summary.arm == arm].iloc[0]
    b = released_sb[released_sb.arm == arm].iloc[0]
    for col in ["n_train", "n_unique", "accuracy", "macro_f1", "rare_recall", "rare_recall_sd"]:
        close(float(a[col]), float(b[col]))
pivot = sb.pivot(index="seed", columns="arm", values="rare_recall")
contrast_defs = [
    ("balance at small", "small_balanced", "small_natural"),
    ("balance at large", "large_balanced", "large_natural"),
    ("size at natural", "large_natural", "small_natural"),
    ("size at balanced", "large_balanced", "small_balanced"),
]
computed = []
for label, a, b in contrast_defs:
    d = (pivot[a] - pivot[b]).to_numpy()
    lo, hi = bootstrap_ci(d)
    computed.append({"contrast": label, "mean_delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "p_raw": signflip(d), "wins": int((d > 0).sum())})
adj = holm([r["p_raw"] for r in computed])
for r, p in zip(computed, adj):
    r["p_holm_4"] = p
released_contrasts = pd.read_csv(OUT / "size_balance_contrasts.csv").set_index("contrast")
for r in computed:
    rel = released_contrasts.loc[r["contrast"]]
    for col in ["mean_delta", "ci_lo", "ci_hi", "p_raw", "p_holm_4", "wins"]:
        close(float(r[col]), float(rel[col]))
assert_tex_contains("table_size_balance_combined.tex", ["+0.509", "+0.260", "+0.242", "-0.007", "0.0078"])
record("Table 8 size-distribution factorial", "Four arm means, bootstrap intervals, exact sign-flip tests and Holm values verified")

# -------------------------------------------------------------------------
# 48 paired prior-fitted versus classical comparisons.
# -------------------------------------------------------------------------
pair_rows = []
for cell in CELLS:
    pvt = core[core.cell == cell].pivot(index="seed", columns="model", values="rare_recall")
    for fm in FM:
        family = []
        for comparator in CLASSICAL:
            d = (pvt[fm] - pvt[comparator]).to_numpy()
            lo, hi = bootstrap_ci(d)
            family.append({"cell": cell, "fm": fm, "versus": comparator,
                           "mean_delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
                           "p_raw": signflip(d), "wins": int((d > 0).sum())})
        adjusted = holm([r["p_raw"] for r in family])
        for row, p in zip(family, adjusted):
            row["p_holm_6"] = p
            row["direction"] = "FM all 10" if row["wins"] == 10 else (
                "classical all 10" if row["wins"] == 0 else "mixed")
            pair_rows.append(row)
pairs = pd.DataFrame(pair_rows)
assert len(pairs) == 48
assert pairs.direction.value_counts().to_dict() == {"FM all 10": 40, "classical all 10": 4, "mixed": 4}
assert (pairs[pairs.versus.str.endswith("_full")].wins == 10).all()
released_pairs = pd.read_csv(OUT / "axis1_fm_vs_classical.csv")
for keys, row in pairs.set_index(["cell", "fm", "versus"]).iterrows():
    rel = released_pairs.set_index(["cell", "fm", "versus"]).loc[keys]
    for col in ["mean_delta", "ci_lo", "ci_hi", "p_raw", "p_holm_6", "wins"]:
        close(float(row[col]), float(rel[col]))
assert_tex_contains("table_tabpfn_selected_pairs.tex", ["-0.065", "+0.628", "+0.573", "+0.550", "0.0117"])
record("Table 9 selected paired effects", "Complete 48-comparison record independently recomputed: 40/4/4")

# -------------------------------------------------------------------------
# ManyClass and stability tables.
# -------------------------------------------------------------------------
many = pd.read_csv(SENS / "exp4_manyclass.csv")
many_summary = (many.groupby(["alphabet_size", "n_estimators_redundancy"], as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                     rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std")))
released_many = pd.read_csv(OUT / "manyclass_summary.csv")
for keys, row in many_summary.set_index(["alphabet_size", "n_estimators_redundancy"]).iterrows():
    rel = released_many.set_index(["alphabet_size", "n_estimators_redundancy"]).loc[keys]
    for col in ["accuracy", "macro_f1", "rare_recall", "rare_recall_sd"]:
        close(float(row[col]), float(rel[col]))
close(float(many_summary.rare_recall.max() - many_summary.rare_recall.min()), 0.0060951996339612124)
assert_tex_contains("table_manyclass.tex", ["0.8641", "0.8660", "0.8630", "0.8599"])
record("Table 10 ManyClass sensitivity", "All 12 runs and the 0.00610 rare-recall range verified")

released_stability = pd.read_csv(OUT / "axis1_core_summary.csv")
for cell in CELLS:
    for model in ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl"]:
        a = summary[(summary.cell == cell) & (summary.model == model)].iloc[0]
        b = released_stability[(released_stability.cell == cell) & (released_stability.model == model)].iloc[0]
        for col in ["rare_recall", "rare_recall_sd", "n_train", "fit_s", "predict_s"]:
            close(float(a[col]), float(b[col]))
assert_tex_contains("table_stability_efficiency.tex", ["0.1115", "0.0004", "1782.1", "400,000"])
record("Table 11 stability and efficiency", "All displayed means, standard deviations, fitting rows and timings verified")

# -------------------------------------------------------------------------
# Operating-point and TabSTAR-control tables.
# -------------------------------------------------------------------------
pc_ext = per_class[(per_class.test_set == "capped") & per_class.model.isin(CORE + ["tabstar", "tabstar__generic_names"])].copy()
op = (pc_ext.groupby(["cell", "model", "class"], as_index=False)
      .agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), support=("support", "mean")))
op = op[[label in rare[cell] for cell, label in zip(op.cell, op["class"])]]
op["category"] = np.select(
    [(op.precision >= .5) & (op.recall >= .5),
     (op.precision < .5) & (op.recall >= .5),
     (op.precision >= .5) & (op.recall < .5)],
    ["recovered", "high recall / low precision", "high precision / low recall"],
    default="missed",
)
selected = {
    "cic_ids2017_V1": ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl", "tabstar", "rf_pruned_full"],
    "cic_unsw_V2": ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl", "rf_pruned_full"],
}
op_rows = []
cm_ext = confusion[(confusion.test_set == "capped") & confusion.model.isin(CORE + ["tabstar", "tabstar__generic_names"])]
for cell, models in selected.items():
    for model in models:
        g = op[(op.cell == cell) & (op.model == model)]
        vc = g.category.value_counts()
        ratios = []
        for seed, seed_cm in cm_ext[(cm_ext.cell == cell) & (cm_ext.model == model)].groupby("seed"):
            tp = seed_cm[(seed_cm.true_label == seed_cm.pred_label) & seed_cm.true_label.isin(rare[cell])]["count"].sum()
            alerts = seed_cm[seed_cm.pred_label.isin(rare[cell])]["count"].sum()
            ratios.append(float(alerts / tp) if tp > 0 else np.inf)
        op_rows.append({
            "cell": cell, "model": model,
            "recovered": int(vc.get("recovered", 0)),
            "high_recall_low_precision": int(vc.get("high recall / low precision", 0)),
            "high_precision_low_recall": int(vc.get("high precision / low recall", 0)),
            "missed": int(vc.get("missed", 0)),
            "alerts_per_tp_pooled": float(np.mean(ratios)),
        })
op_calc = pd.DataFrame(op_rows).set_index(["cell", "model"])
op_rel = pd.read_csv(OUT / "operating_points_selected_summary.csv").set_index(["cell", "model"])
for keys, row in op_calc.iterrows():
    rel = op_rel.loc[keys]
    for col in ["recovered", "high_recall_low_precision", "high_precision_low_recall", "missed", "alerts_per_tp_pooled"]:
        close(float(row[col]), float(rel[col]))
assert_tex_contains("table_operating_selected.tex", ["8 & 2 & 0 & 1 & 1.35", "6 & 4 & 0 & 1 & 1.65", "6 & 3 & 0 & 2 & 2.71"])
record("Table 12 operating points", "All category counts and seed-averaged alerts-per-true-positive values verified")

# TabSTAR named-column negative control.
tab_rows = []
for metric in ["accuracy", "macro_f1", "rare_recall"]:
    if metric == "rare_recall":
        src = (pc_ext[(pc_ext.cell == "cic_ids2017_V3") & pc_ext["class"].isin(rare["cic_ids2017_V3"])]
               .groupby(["seed", "model"], as_index=False).recall.mean()
               .pivot(index="seed", columns="model", values="recall"))
    else:
        src = metrics[(metrics.cell == "cic_ids2017_V3") & (metrics.test_set == "capped")].pivot(
            index="seed", columns="model", values=metric)
    d = (src["tabstar"] - src["tabstar__generic_names"]).dropna().to_numpy()
    lo, hi = bootstrap_ci(d)
    tab_rows.append({"metric": metric, "mean_delta_named_minus_generic": d.mean(),
                     "ci_lo": lo, "ci_hi": hi, "p_raw": signflip(d), "wins": int((d > 0).sum())})
tab_calc = pd.DataFrame(tab_rows).set_index("metric")
tab_rel = pd.read_csv(OUT / "tabstar_named_vs_generic.csv").set_index("metric")
for metric, row in tab_calc.iterrows():
    rel = tab_rel.loc[metric]
    for col in ["mean_delta_named_minus_generic", "ci_lo", "ci_hi", "p_raw", "wins"]:
        close(float(row[col]), float(rel[col]))
assert_tex_contains("table_tabstar_control.tex", ["-0.0021", "+0.0001", "+0.0105", "0.7539"])
record("Table 13 TabSTAR name control", "All three paired effects, intervals and exact tests verified")

# -------------------------------------------------------------------------
# Cost crossovers from confusion matrices.
# -------------------------------------------------------------------------
def alarm_table(cell: str, severity: dict[str, float] | None) -> pd.DataFrame:
    rows = []
    b = benign[cell]
    g = confusion[(confusion.cell == cell) & (confusion.test_set == "capped") & confusion.model.isin(CORE)]
    for (seed, model), p in g.groupby(["seed", "model"]):
        miss = p[(p.true_label != b) & (p.pred_label == b)]
        if severity:
            miss_count = float((miss["count"] * miss.true_label.map(lambda x: severity.get(x, 1.0))).sum())
        else:
            miss_count = float(miss["count"].sum())
        false_alarm = float(p[(p.true_label == b) & (p.pred_label != b)]["count"].sum())
        rows.append({"seed": seed, "model": model, "miss": miss_count, "false_alarm": false_alarm})
    return pd.DataFrame(rows)

cost_rows = []
for scenario, weights in [("uniform", None), ("severity weighted", SEV)]:
    for cell in CELLS:
        a = alarm_table(cell, weights)
        base = a[a.model == "rf_pruned_full"].set_index("seed")
        for model in [m for m in CORE if m != "rf_pruned_full"]:
            other = a[a.model == model].set_index("seed")
            ratios = []
            for seed in sorted(base.index):
                dm = float(base.loc[seed, "miss"] - other.loc[seed, "miss"])
                df = float(other.loc[seed, "false_alarm"] - base.loc[seed, "false_alarm"])
                rho = np.nan if abs(dm) <= 1e-12 else df / dm
                if np.isfinite(rho) and rho > 0:
                    ratios.append(rho)
            cost_rows.append({
                "scenario": scenario, "cell": cell, "model": model,
                "positive_seeds": len(ratios),
                "rho_median_positive": float(np.median(ratios)) if ratios else np.nan,
                "one_over_rho": float(1 / np.median(ratios)) if ratios else np.nan,
            })
cost_calc = pd.DataFrame(cost_rows).set_index(["scenario", "cell", "model"])
cost_rel = pd.read_csv(OUT / "cost_crossovers_summary.csv").set_index(["scenario", "cell", "model"])
for keys, row in cost_calc.iterrows():
    rel = cost_rel.loc[keys]
    for col in ["positive_seeds", "rho_median_positive", "one_over_rho"]:
        close(float(row[col]), float(rel[col]))
assert_tex_contains("table_cost_selected.tex", ["0.041 & 24.6", "0.386 & 2.6", "2.029 & 0.5", "1.465 & 0.7"])
record("Table 14 cost crossovers", "560 seed-level cost comparisons and all selected medians independently reconstructed")

# -------------------------------------------------------------------------
# Axis 2 closed-label scoring from raw generations.
# -------------------------------------------------------------------------
old_raw = read_jsonl(AX2 / "raw_generations_original_six_model_campaign.jsonl")
nem_raw = read_jsonl(AX2 / "raw_generations_nemotron_corrected.jsonl")
panel_raw = [r for r in old_raw if not r["model"].startswith("Nemotron")] + nem_raw
old_results = pd.read_csv(AX2 / "results_original_six_model_campaign.csv")
nem_results = pd.read_csv(AX2 / "results_nemotron_corrected.csv")
panel_results = pd.concat([
    old_results[~old_results.model.str.startswith("Nemotron", na=False)],
    nem_results[nem_results.model.str.startswith("Nemotron", na=False)],
], ignore_index=True)
rf = old_results[old_results.model == "RandomForest"].drop_duplicates("seed").set_index("seed")
seed_rows = []
for r in panel_raw:
    y = np.asarray(r["y_true"], dtype=str)
    pred = np.asarray([v if v is not None else "__unparsed__" for v in r["y_pred"]], dtype=str)
    labels = sorted(set(y))
    seed_rows.append({
        "seed": int(r["seed"]), "model": r["model"], "config": r["config"],
        "accuracy": accuracy_score(y, pred),
        "macro_f1_closed": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "parse_rate": np.mean([v is not None for v in r["y_pred"]]),
        "n_eval": len(y), "unparsed": sum(v is None for v in r["y_pred"]),
    })
seed_calc = pd.DataFrame(seed_rows)
assert set(seed_calc.n_eval) == {246}
seed_rel = pd.read_csv(OUT / "axis2_closed_label_per_seed.csv")
assert_frame_equal(
    seed_calc.sort_values(["model", "seed"]).reset_index(drop=True),
    seed_rel.sort_values(["model", "seed"]).reset_index(drop=True),
    check_exact=False, rtol=1e-12, atol=1e-12,
)
summary_rows = []
for model, g in seed_calc.groupby("model"):
    gs = g.set_index("seed")
    d = gs.macro_f1_closed.to_numpy() - rf.loc[gs.index, "macro_f1"].to_numpy()
    lo, hi = bootstrap_ci(d)
    source_cost = panel_results[panel_results.model == model]
    summary_rows.append({
        "model": model, "accuracy": g.accuracy.mean(), "macro_f1_closed": g.macro_f1_closed.mean(),
        "parse_rate": g.parse_rate.mean(), "usd": source_cost.usd.sum(),
        "mean_delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
        "p_raw": signflip(d), "p_bonferroni_6": min(1.0, 6 * signflip(d)),
        "wins": int((d > 0).sum()),
    })
a2_calc = pd.DataFrame(summary_rows).set_index("model")
a2_rel = pd.read_csv(OUT / "axis2_closed_label_summary.csv").set_index("model")
for model, row in a2_calc.iterrows():
    rel = a2_rel.loc[model]
    for col in ["accuracy", "macro_f1_closed", "parse_rate", "usd", "mean_delta", "ci_lo", "ci_hi", "p_raw", "p_bonferroni_6", "wins"]:
        close(float(row[col]), float(rel[col]))
close(float(rf.accuracy.mean()), float(a2_rel.loc["RandomForest", "accuracy"]))
close(float(rf.macro_f1.mean()), float(a2_rel.loc["RandomForest", "macro_f1_closed"]))
close(float(a2_calc.usd.sum()), 218.3418699)
assert_tex_contains("table_axis2_summary.tex", ["0.818 & 0.834", "+0.028 & 0.1172", "93.52"])
record("Table 15 Axis-2 panel", "All 60 raw LLM seed records rescored over the closed nine-label set; six adjusted paired tests verified")

# -------------------------------------------------------------------------
# Axis-2 calibration and feature-selection audit.
# -------------------------------------------------------------------------
chosen = json.loads((SENS / "exp3_chosen_config.json").read_text())
cal = pd.read_csv(SENS / "exp3_calibration_probe.csv")
for model, config in chosen.items():
    row = cal[(cal.model == model) & (cal.config == config)].iloc[0]
    assert int(row.clean) == int(row["of"]) == 5
    assert str(row.finish_reason) == "stop"
assert_tex_contains("table_axis2_calibration.tex", ["plain\\_4096", "plain\\_512", "nothink\\_template"])
record("Table 6 Axis-2 calibration", "All six frozen configurations pass 5/5 clean development probes")

mi = pd.read_csv(SENS / "exp2_llm_leakage.csv")
raw_mi = read_jsonl(SENS / "exp2_raw.jsonl")
assert len(raw_mi) == 30 and {len(r["y_true"]) for r in raw_mi} == {387}
mi_summary = (mi.groupby("regime", as_index=False)
              .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                   parse_rate=("parse_rate", "mean"), wall_min=("wall_min", "sum")))
mi_rel = pd.read_csv(OUT / "axis2_mi_llm_summary.csv")
for regime, row in mi_summary.set_index("regime").iterrows():
    rel = mi_rel.set_index("regime").loc[regime]
    for col in ["accuracy", "macro_f1", "parse_rate", "wall_min"]:
        close(float(row[col]), float(rel[col]))
mi_pvt = mi.pivot(index="seed", columns="regime")
mi_effects = []
for metric in ["accuracy", "macro_f1", "parse_rate"]:
    for regime in ["development_only", "icl_only"]:
        d = (mi_pvt[metric][regime] - mi_pvt[metric]["legacy_pooled"]).to_numpy()
        lo, hi = bootstrap_ci(d)
        mi_effects.append({"metric": metric, "regime": regime, "mean_delta": d.mean(),
                           "ci_lo": lo, "ci_hi": hi, "p_raw": signflip(d),
                           "wins": int((d > 0).sum())})
mi_effect_rel = pd.read_csv(OUT / "axis2_mi_llm_effects.csv").set_index(["metric", "regime"])
for row in mi_effects:
    rel = mi_effect_rel.loc[(row["metric"], row["regime"])]
    for col in ["mean_delta", "ci_lo", "ci_hi", "p_raw", "wins"]:
        close(float(row[col]), float(rel[col]))
assert_tex_contains("table_axis2_mi_combined.tex", ["0.821", "0.831", "+0.012", "+0.022", "0.0059"])
record("Table 16 Axis-2 selector audit", "30 Kimi-K3 runs; 387 identical evaluation rows per seed across selector regimes")

# -------------------------------------------------------------------------
# Manuscript-level consistency checks.
# -------------------------------------------------------------------------
if MANUSCRIPT.exists():
    text = MANUSCRIPT.read_text()
    assert "\\appendix" not in text.lower()
    assert "measurement study" not in text.lower()
    assert "introduces no" not in text.lower()
    for literal in [
        "$83.1\\%$", "$0.831$", "$4\\times8\\times10=320$", "$+0.509$",
        "$+0.260$", "$+0.242$", "$-0.007$", "2\\times6\\times4=48",
        "\\$218.34", "$p_{\\mathrm{adj}}=0.117$",
    ]:
        assert literal in text, literal
    record("Manuscript numerical literals", "Headline numbers agree with the independently recomputed tables; no appendices or harsh disclaimer phrasing")

report = {
    "status": "PASS",
    "checks": checks,
    "headline": {
        "axis1_records": 320,
        "rank_correlations": rank_expected,
        "pairwise_counts": {"FM all 10": 40, "classical all 10": 4, "mixed": 4},
        "factorial_effects": {r["contrast"]: float(r["mean_delta"]) for r in computed},
        "axis2_total_cost_usd": float(a2_calc.usd.sum()),
        "cic_ids2017_heldout_benign_share": float(support_by_cell["cic_ids2017_V1"][benign["cic_ids2017_V1"]] / support_by_cell["cic_ids2017_V1"].sum()),
    },
}
audit_path = OUT / "table_statistical_audit.json"
audit_path.write_text(json.dumps(report, indent=2))

# Refresh the release manifest so the independent audit itself is covered.
import hashlib
manifest_path = OUT / "release_file_manifest_sha256.csv"
manifest_rows = []
for path in sorted((ROOT / "results").rglob("*")):
    if path.is_file() and path != manifest_path:
        manifest_rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

print(json.dumps({"status": report["status"], "checks": len(checks), "names": [c["name"] for c in checks]}, indent=2))
