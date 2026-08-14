#!/usr/bin/env python3
"""Regenerate all manuscript tables, figures and verification summaries.

This script is intentionally analysis-only: it uses the persisted predictions,
confusion matrices and experiment records released with the repository. It does
not download the underlying benchmark corpora or call hosted models.

Usage
-----
    python scripts/recompute_manuscript_results.py

Outputs are written to results/derived and manuscript-ready LaTeX/figures can be
copied from there.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
AX1 = ROOT / "results" / "axis1"
AX2 = ROOT / "results" / "axis2"
SENS = ROOT / "results" / "sensitivity"
OUT = ROOT / "results" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

CORE = [
    "rf_pruned_full", "rf_strong_full", "xgb_full",
    "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
    "tabpfn_2.5", "tabicl",
]
CLASSICAL = [
    "rf_pruned_full", "rf_strong_full", "xgb_full",
    "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
]
FM = ["tabpfn_2.5", "tabicl"]
CELL_ORDER = ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1", "cic_unsw_V2"]
CELL_LABEL = {
    "cic_ids2017_V1": "CIC-IDS2017 V1",
    "cic_ids2017_V3": "CIC-IDS2017 V3",
    "cic_unsw_V1": "CIC-UNSW-NB15 V1",
    "cic_unsw_V2": "CIC-UNSW-NB15 V2",
}
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
    "tabstar__generic_names": "TabSTAR, generic names",
}
ABBR = {
    "rf_pruned_full": "RF-P/L",
    "rf_strong_full": "RF-S/L",
    "xgb_full": "XGB/L",
    "rf_pruned_balanced": "RF-P/B",
    "rf_strong_balanced": "RF-S/B",
    "xgb_balanced": "XGB/B",
    "tabpfn_2.5": "TabPFN",
    "tabicl": "TabICL",
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


def signflip(d: Sequence[float]) -> float:
    d = np.asarray(d, dtype=float)
    obs = abs(float(d.mean()))
    vals = [
        abs(float(np.mean(d * np.asarray(s, dtype=float))))
        for s in itertools.product((-1.0, 1.0), repeat=len(d))
    ]
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))


def bootstrap_ci(d: Sequence[float], n_boot: int = 50000, seed: int = 2026) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        candidate = (m - i) * p[idx]
        running = max(running, candidate)
        out[idx] = min(1.0, running)
    return out


def tex_escape(value: object) -> str:
    s = str(value)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Axis 1: strict shared-partition panel
# ---------------------------------------------------------------------------
M = pd.read_csv(AX1 / "metrics.csv").drop_duplicates(
    ["cell", "seed", "model", "test_set"], keep="last"
)
PC = pd.read_csv(AX1 / "per_class.csv")
CM = pd.read_csv(AX1 / "confusion.csv")
Mc = M[(M.test_set == "capped") & M.model.isin(CORE)].copy()
PCc = PC[(PC.test_set == "capped") & PC.model.isin(CORE + ["tabstar", "tabstar__generic_names"])].copy()
CMc = CM[(CM.test_set == "capped") & CM.model.isin(CORE + ["tabstar", "tabstar__generic_names"])].copy()

# Identify rare labels from the persisted weighted support (<0.5% of the cell).
RARE: dict[str, list[str]] = {}
BENIGN: dict[str, str] = {}
for cell in CELL_ORDER:
    base = PCc[(PCc.cell == cell) & (PCc.seed == 0) & (PCc.model == "rf_pruned_full")]
    benign = [x for x in base["class"].astype(str).unique() if x.lower() == "benign"]
    if len(benign) != 1:
        raise RuntimeError(f"Could not determine benign label for {cell}")
    BENIGN[cell] = benign[0]
    supports = base.groupby("class", as_index=True).support.mean()
    total = float(supports.sum())
    RARE[cell] = sorted(
        c for c, v in supports.items()
        if c != BENIGN[cell] and float(v) / total < 0.005
    )

rare_seed = (
    PCc[PCc["class"].isin(set(sum(RARE.values(), [])))]
    .assign(is_rare=lambda d: [cl in RARE[cell] for cell, cl in zip(d.cell, d["class"])])
)
rare_seed = rare_seed[rare_seed.is_rare]
rare_seed = (rare_seed.groupby(["cell", "seed", "model"], as_index=False)
             .agg(rare_recall=("recall", "mean"), rare_precision=("precision", "mean"), rare_f1=("f1", "mean")))

core = Mc.merge(rare_seed, on=["cell", "seed", "model"], how="left")
core_summary = (core.groupby(["cell", "model"], as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                     rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
                     n_train=("n_train", "mean"), fit_s=("fit_s", "mean"), predict_s=("predict_s", "mean")))
core_summary["accuracy_rank"] = core_summary.groupby("cell")["accuracy"].rank(method="min", ascending=False).astype(int)
core_summary["recall_rank"] = core_summary.groupby("cell")["rare_recall"].rank(method="min", ascending=False).astype(int)
core_summary["rank_shift"] = core_summary.recall_rank - core_summary.accuracy_rank
core_summary["cell_label"] = core_summary.cell.map(CELL_LABEL)
core_summary["model_label"] = core_summary.model.map(MODEL_LABEL)
core_summary.to_csv(OUT / "axis1_core_summary.csv", index=False)

rank_rows = []
for cell in CELL_ORDER:
    g = core_summary[core_summary.cell == cell]
    rho = float(g.accuracy.corr(g.rare_recall, method="spearman"))
    top_acc = g.nsmallest(3, "accuracy_rank").sort_values("accuracy_rank").model.tolist()
    bottom_rec = g.nlargest(3, "recall_rank").sort_values("recall_rank", ascending=False).model.tolist()
    rank_rows.append({
        "cell": cell, "cell_label": CELL_LABEL[cell], "spearman": rho,
        "top3_accuracy": "; ".join(top_acc),
        "bottom3_recall": "; ".join(bottom_rec),
        "same_three": set(top_acc) == set(bottom_rec),
    })
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT / "axis1_rank_inversion.csv", index=False)

# Full paired FM-vs-classical record with Holm correction within each cell/model family.
pair_rows = []
for cell in CELL_ORDER:
    pvt = core[core.cell == cell].pivot(index="seed", columns="model", values="rare_recall")
    for fm in FM:
        family = []
        for cl in CLASSICAL:
            d = (pvt[fm] - pvt[cl]).dropna().to_numpy()
            lo, hi = bootstrap_ci(d)
            family.append({
                "cell": cell, "fm": fm, "versus": cl, "n": len(d),
                "mean_delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                "p_raw": signflip(d), "wins": int((d > 0).sum()),
            })
        adj = holm_adjust([r["p_raw"] for r in family])
        for r, a in zip(family, adj):
            r["p_holm_6"] = float(a)
            pair_rows.append(r)
pairs = pd.DataFrame(pair_rows)
pairs["direction"] = np.where(pairs.wins == 10, "FM all 10", np.where(pairs.wins == 0, "classical all 10", "mixed"))
pairs.to_csv(OUT / "axis1_fm_vs_classical.csv", index=False)

# Cap validation restricted to matching partition hashes.
cap_rows = []
for cell in CELL_ORDER:
    for model in CLASSICAL:
        g = M[(M.cell == cell) & (M.model == model)].pivot(index="seed", columns="test_set")
        for seed in sorted(set(M[(M.cell == cell) & (M.model == model)].seed)):
            try:
                hcap = g.loc[seed, ("test_hash", "capped")]
                hfull = g.loc[seed, ("test_hash", "full")]
            except KeyError:
                continue
            if str(hcap) != str(hfull):
                continue
            cap_rows.append({
                "cell": cell, "model": model, "seed": seed,
                "regime": "large" if model.endswith("_full") else "balanced",
                "d_accuracy": float(g.loc[seed, ("accuracy", "capped")] - g.loc[seed, ("accuracy", "full")]),
                "d_macro_f1": float(g.loc[seed, ("macro_f1", "capped")] - g.loc[seed, ("macro_f1", "full")]),
            })
cap = pd.DataFrame(cap_rows)
cap_summary = (cap.groupby(["cell", "regime"], as_index=False)
               .agg(n=("seed", "size"), mean_d_accuracy=("d_accuracy", "mean"),
                    max_abs_d_accuracy=("d_accuracy", lambda s: float(np.max(np.abs(s)))),
                    mean_d_macro_f1=("d_macro_f1", "mean"),
                    max_abs_d_macro_f1=("d_macro_f1", lambda s: float(np.max(np.abs(s))))))
cap.to_csv(OUT / "cap_validation_matching_hash_rows.csv", index=False)
cap_summary.to_csv(OUT / "cap_validation_matching_hash_summary.csv", index=False)

# Operating-point classification and alerts/TP, generated from per-class metrics.
op = (PCc.groupby(["cell", "model", "class"], as_index=False)
      .agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), support=("support", "mean")))
op = op[[cl in RARE[cell] for cell, cl in zip(op.cell, op["class"])]]
op["operating_point"] = np.select(
    [
        (op.precision >= 0.5) & (op.recall >= 0.5),
        (op.precision < 0.5) & (op.recall >= 0.5),
        (op.precision >= 0.5) & (op.recall < 0.5),
    ],
    ["recovered", "high recall / low precision", "high precision / low recall"],
    default="missed",
)
op["alerts_per_tp"] = np.where(op.precision > 0, 1.0 / op.precision, np.inf)
op.to_csv(OUT / "operating_points_per_class.csv", index=False)

# Selected aggregate operating points used in the main manuscript.
selected = {
    "cic_ids2017_V1": ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl", "tabstar", "rf_pruned_full"],
    "cic_unsw_V2": ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl", "rf_pruned_full"],
}
op_sum_rows = []
for cell, models in selected.items():
    for model in models:
        g = op[(op.cell == cell) & (op.model == model)]
        counts = g.operating_point.value_counts()
        # Compute alerts per true positive directly from the persisted, reweighted
        # confusion matrix, then average the seed-level ratios. An alert is any
        # prediction into a rare label; a true positive is a correct rare-label
        # prediction.
        ratios = []
        for seed, cm_seed in CMc[(CMc.cell == cell) & (CMc.model == model)].groupby("seed"):
            tp = float(cm_seed[(cm_seed.true_label == cm_seed.pred_label) & cm_seed.true_label.isin(RARE[cell])]["count"].sum())
            alerts = float(cm_seed[cm_seed.pred_label.isin(RARE[cell])]["count"].sum())
            ratios.append(alerts / tp if tp > 0 else np.inf)
        pooled = float(np.mean(ratios)) if ratios else np.nan
        op_sum_rows.append({
            "cell": cell, "model": model,
            "recovered": int(counts.get("recovered", 0)),
            "high_recall_low_precision": int(counts.get("high recall / low precision", 0)),
            "high_precision_low_recall": int(counts.get("high precision / low recall", 0)),
            "missed": int(counts.get("missed", 0)),
            "alerts_per_tp_pooled": pooled,
            "mean_precision": float(g.precision.mean()),
            "mean_recall": float(g.recall.mean()),
        })
op_summary = pd.DataFrame(op_sum_rows)
op_summary.to_csv(OUT / "operating_points_selected_summary.csv", index=False)

# Cost crossover against the pruned large-budget RF, core panel only.
def alarms(cell: str, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or {}
    benign = BENIGN[cell]
    g = CMc[(CMc.cell == cell) & CMc.model.isin(CORE)]
    rows = []
    for (seed, model), p in g.groupby(["seed", "model"]):
        miss = p[(p.true_label != benign) & (p.pred_label == benign)]
        rows.append({
            "seed": seed, "model": model,
            "miss": float((miss["count"] * miss.true_label.map(lambda x: weights.get(x, 1.0))).sum()),
            "false_alarm": float(p[(p.true_label == benign) & (p.pred_label != benign)]["count"].sum()),
        })
    return pd.DataFrame(rows)

cost_rows = []
for scenario, weights in [("uniform", None), ("severity weighted", SEV)]:
    for cell in CELL_ORDER:
        A = alarms(cell, weights)
        base = A[A.model == "rf_pruned_full"].set_index("seed")
        for model in [m for m in CORE if m != "rf_pruned_full"]:
            other = A[A.model == model].set_index("seed")
            for seed in sorted(set(base.index) & set(other.index)):
                dm = float(base.loc[seed, "miss"] - other.loc[seed, "miss"])
                df = float(other.loc[seed, "false_alarm"] - base.loc[seed, "false_alarm"])
                if abs(dm) <= 1e-12:
                    status = "no finite crossover"
                    rho = np.nan
                else:
                    rho = df / dm
                    status = "positive crossover" if rho > 0 else "dominance / negative crossover"
                cost_rows.append({
                    "scenario": scenario, "cell": cell, "model": model, "seed": seed,
                    "base_miss": float(base.loc[seed, "miss"]),
                    "model_miss": float(other.loc[seed, "miss"]),
                    "base_false_alarm": float(base.loc[seed, "false_alarm"]),
                    "model_false_alarm": float(other.loc[seed, "false_alarm"]),
                    "rho": rho, "status": status,
                })
cost_seed = pd.DataFrame(cost_rows)
cost_seed.to_csv(OUT / "cost_crossovers_per_seed.csv", index=False)
cost_summary_rows = []
for (scenario, cell, model), g in cost_seed.groupby(["scenario", "cell", "model"]):
    pos = g[g.status == "positive crossover"].rho.dropna().to_numpy()
    cost_summary_rows.append({
        "scenario": scenario, "cell": cell, "model": model,
        "positive_seeds": len(pos), "dominance_or_negative": int((g.status != "positive crossover").sum()),
        "rho_median_positive": float(np.median(pos)) if len(pos) else np.nan,
        "rho_q1_positive": float(np.quantile(pos, 0.25)) if len(pos) else np.nan,
        "rho_q3_positive": float(np.quantile(pos, 0.75)) if len(pos) else np.nan,
        "one_over_rho": float(1.0 / np.median(pos)) if len(pos) and np.median(pos) > 0 else np.nan,
    })
cost_summary = pd.DataFrame(cost_summary_rows)
cost_summary.to_csv(OUT / "cost_crossovers_summary.csv", index=False)

# TabSTAR named-name negative control, kept outside the confirmatory panel.
tabstar_control_rows = []
for metric in ["accuracy", "macro_f1", "rare_recall"]:
    if metric == "rare_recall":
        src = rare_seed[rare_seed.cell == "cic_ids2017_V3"].pivot(index="seed", columns="model", values=metric)
    else:
        src = M[(M.cell == "cic_ids2017_V3") & (M.test_set == "capped")].pivot(index="seed", columns="model", values=metric)
    d = (src["tabstar"] - src["tabstar__generic_names"]).dropna().to_numpy()
    lo, hi = bootstrap_ci(d)
    tabstar_control_rows.append({
        "metric": metric, "n": len(d), "mean_delta_named_minus_generic": float(d.mean()),
        "ci_lo": lo, "ci_hi": hi, "p_raw": signflip(d), "wins": int((d > 0).sum()),
    })
tabstar_control = pd.DataFrame(tabstar_control_rows)
tabstar_control.to_csv(OUT / "tabstar_named_vs_generic.csv", index=False)

# ---------------------------------------------------------------------------
# Additional experiments
# ---------------------------------------------------------------------------
size_balance = pd.read_csv(SENS / "exp1_size_by_balance.csv")
size_balance_summary = (size_balance.groupby(["arm", "size", "balance"], as_index=False)
                        .agg(n_train=("n_train", "mean"), n_unique=("n_unique_rows", "mean"),
                             oversample=("oversample_factor", "mean"), accuracy=("accuracy", "mean"),
                             macro_f1=("macro_f1", "mean"), rare_recall=("rare_recall", "mean"),
                             rare_recall_sd=("rare_recall", "std"), fit_s=("fit_s", "mean")))
size_balance_summary.to_csv(OUT / "size_balance_summary.csv", index=False)

sb_pvt = size_balance.pivot(index="seed", columns="arm", values="rare_recall")
contrasts = [
    ("balance at small", "small_balanced", "small_natural"),
    ("balance at large", "large_balanced", "large_natural"),
    ("size at natural", "large_natural", "small_natural"),
    ("size at balanced", "large_balanced", "small_balanced"),
]
sb_rows = []
for label, a, b in contrasts:
    d = (sb_pvt[a] - sb_pvt[b]).dropna().to_numpy()
    lo, hi = bootstrap_ci(d)
    sb_rows.append({
        "contrast": label, "a": a, "b": b, "n": len(d), "mean_delta": float(d.mean()),
        "ci_lo": lo, "ci_hi": hi, "p_raw": signflip(d), "wins": int((d > 0).sum()),
    })
sb_adj = holm_adjust([r["p_raw"] for r in sb_rows])
for r, p in zip(sb_rows, sb_adj):
    r["p_holm_4"] = float(p)
size_balance_contrasts = pd.DataFrame(sb_rows)
size_balance_contrasts.to_csv(OUT / "size_balance_contrasts.csv", index=False)

many = pd.read_csv(SENS / "exp4_manyclass.csv")
many_summary = (many.groupby(["alphabet_size", "n_estimators_redundancy"], as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                     rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
                     fit_s=("fit_s", "mean")))
many_summary.to_csv(OUT / "manyclass_summary.csv", index=False)

mi = pd.read_csv(SENS / "exp2_llm_leakage.csv")
mi_summary = (mi.groupby(["regime", "model"], as_index=False)
              .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                   parse_rate=("parse_rate", "mean"), wall_min=("wall_min", "sum")))
mi_summary.to_csv(OUT / "axis2_mi_llm_summary.csv", index=False)
mi_pvt = mi.pivot(index="seed", columns="regime")
mi_rows = []
for metric in ["accuracy", "macro_f1", "parse_rate"]:
    for regime in ["development_only", "icl_only"]:
        d = (mi_pvt[metric][regime] - mi_pvt[metric]["legacy_pooled"]).dropna().to_numpy()
        lo, hi = bootstrap_ci(d)
        mi_rows.append({
            "metric": metric, "regime": regime, "reference": "legacy_pooled",
            "mean_delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_raw": signflip(d), "wins": int((d > 0).sum()),
        })
mi_effects = pd.DataFrame(mi_rows)
mi_effects.to_csv(OUT / "axis2_mi_llm_effects.csv", index=False)

calibration = pd.read_csv(SENS / "exp3_calibration_probe.csv")
calibration.to_csv(OUT / "axis2_calibration_probe.csv", index=False)
chosen_config = json.loads((SENS / "exp3_chosen_config.json").read_text())
(OUT / "axis2_chosen_config.json").write_text(json.dumps(chosen_config, indent=2))

# ---------------------------------------------------------------------------
# Axis 2: corrected closed-label panel
# ---------------------------------------------------------------------------
raw_old = [json.loads(x) for x in (AX2 / "raw_generations_original_six_model_campaign.jsonl").read_text().splitlines() if x.strip()]
raw_nem = [json.loads(x) for x in (AX2 / "raw_generations_nemotron_corrected.jsonl").read_text().splitlines() if x.strip()]
# Five retained models from the original campaign; Nemotron is replaced by the blind-calibrated run.
raw_panel = [r for r in raw_old if not r["model"].startswith("Nemotron")] + raw_nem

res_old = pd.read_csv(AX2 / "results_original_six_model_campaign.csv")
res_nem = pd.read_csv(AX2 / "results_nemotron_corrected.csv")
res_panel = pd.concat([
    res_old[~res_old.model.str.startswith("Nemotron", na=False)],
    res_nem[res_nem.model.str.startswith("Nemotron", na=False)],
], ignore_index=True)
rf = res_old[res_old.model == "RandomForest"].drop_duplicates("seed").copy()

axis2_seed_rows = []
perclass_rows = []
for r in raw_panel:
    y = np.asarray(r["y_true"], dtype=str)
    pred = np.asarray([p if p is not None else "__unparsed__" for p in r["y_pred"]], dtype=str)
    labels = sorted(set(y))
    p, rc, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    axis2_seed_rows.append({
        "seed": int(r["seed"]), "model": r["model"], "config": r["config"],
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_closed": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "parse_rate": float(np.mean([v is not None for v in r["y_pred"]])),
        "n_eval": len(y), "unparsed": int(sum(v is None for v in r["y_pred"])),
    })
    for label, pv, rv, fv, sv in zip(labels, p, rc, f, sup):
        perclass_rows.append({
            "seed": int(r["seed"]), "model": r["model"], "class": label,
            "precision": float(pv), "recall": float(rv), "f1": float(fv), "support": int(sv),
        })
axis2_seed = pd.DataFrame(axis2_seed_rows)
axis2_seed.to_csv(OUT / "axis2_closed_label_per_seed.csv", index=False)
axis2_pc = pd.DataFrame(perclass_rows)
axis2_pc.to_csv(OUT / "axis2_closed_label_per_class_per_seed.csv", index=False)

# Add costs and latency from matching persisted result rows.
axis2_seed = axis2_seed.merge(
    res_panel[["seed", "model", "usd", "latency_median_s", "wall_min"]],
    on=["seed", "model"], how="left"
)
axis2_summary = (axis2_seed.groupby(["model", "config"], as_index=False)
                 .agg(accuracy=("accuracy", "mean"), macro_f1_closed=("macro_f1_closed", "mean"),
                      parse_rate=("parse_rate", "mean"), unparsed=("unparsed", "sum"),
                      usd=("usd", "sum"), latency_median_s=("latency_median_s", "median"),
                      wall_min=("wall_min", "sum")))
rf_seed = rf.set_index("seed")
axis2_test_rows = []
for model, g in axis2_seed.groupby("model"):
    gs = g.set_index("seed").sort_index()
    common = sorted(set(gs.index) & set(rf_seed.index))
    for metric, rf_metric in [("accuracy", "accuracy"), ("macro_f1_closed", "macro_f1")]:
        d = gs.loc[common, metric].to_numpy() - rf_seed.loc[common, rf_metric].to_numpy()
        lo, hi = bootstrap_ci(d)
        axis2_test_rows.append({
            "model": model, "metric": metric, "n": len(d), "mean_delta": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi, "p_raw": signflip(d), "wins": int((d > 0).sum()),
        })
axis2_tests = pd.DataFrame(axis2_test_rows)
for metric in axis2_tests.metric.unique():
    idx = axis2_tests.metric == metric
    axis2_tests.loc[idx, "p_bonferroni_6"] = np.minimum(1.0, 6 * axis2_tests.loc[idx, "p_raw"])
axis2_tests.to_csv(OUT / "axis2_paired_tests.csv", index=False)
axis2_summary = axis2_summary.merge(
    axis2_tests[axis2_tests.metric == "macro_f1_closed"][["model", "mean_delta", "ci_lo", "ci_hi", "p_raw", "p_bonferroni_6", "wins"]],
    on="model", how="left"
)
rf_row = pd.DataFrame([{
    "model": "RandomForest", "config": "matched anchor", "accuracy": rf.accuracy.mean(),
    "macro_f1_closed": rf.macro_f1.mean(), "parse_rate": 1.0, "unparsed": 0,
    "usd": 0.0, "latency_median_s": 0.0, "wall_min": 0.0,
    "mean_delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
    "p_raw": np.nan, "p_bonferroni_6": np.nan, "wins": np.nan,
}])
axis2_summary = pd.concat([rf_row, axis2_summary], ignore_index=True)
axis2_summary.to_csv(OUT / "axis2_closed_label_summary.csv", index=False)
axis2_pc_mean = (axis2_pc.groupby(["model", "class"], as_index=False)
                 .agg(precision=("precision", "mean"), recall=("recall", "mean"),
                      f1=("f1", "mean"), support=("support", "mean")))
axis2_pc_mean.to_csv(OUT / "axis2_closed_label_per_class_mean.csv", index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# 1. Corrected methodology flowchart.
fig, ax = plt.subplots(figsize=(13.6, 6.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=1.1)
ax.text(0.5, 0.93, "Shared construction: canonical capture -> cleaning -> label taxonomy -> stratified split", ha="center", va="center", fontsize=12, bbox=box)
ax.annotate("", xy=(0.27, 0.78), xytext=(0.46, 0.89), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.73, 0.78), xytext=(0.54, 0.89), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.74, "Axis 1: tabular detector measurement", ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
ax.text(0.75, 0.74, "Axis 2: few-shot language-model control", ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
ax.text(0.25, 0.57, "Strict core panel (8 configurations)\nShared capped rows within each cell and seed\nSix classical arms + TabPFN 2.5 + TabICL", ha="center", va="center", fontsize=10.5, bbox=box)
ax.text(0.75, 0.57, "Disjoint development, ICL and evaluation rows\n12 development-selected features\n5 shots per class; RF anchor on identical inputs", ha="center", va="center", fontsize=10.5, bbox=box)
ax.annotate("", xy=(0.25, 0.46), xytext=(0.25, 0.52), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.75, 0.46), xytext=(0.75, 0.52), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.36, "Primary outcome: rare-class recall\nReport accuracy, macro-F1, precision, alerts/TP\n10 paired seeds; bootstrap CI; exact sign flip", ha="center", va="center", fontsize=10.5, bbox=box)
ax.text(0.75, 0.36, "Closed-label accuracy and macro-F1\nParse failures retained as errors; parse rate separate\n10 paired seeds; Bonferroni across six models", ha="center", va="center", fontsize=10.5, bbox=box)
ax.annotate("", xy=(0.25, 0.20), xytext=(0.25, 0.30), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.75, 0.20), xytext=(0.75, 0.30), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.12, "Separate extensions\nTabSTAR (different CIC partition)\n2x2 balance-size, cap, names and ManyClass checks", ha="center", va="center", fontsize=9.8, bbox=box)
ax.text(0.75, 0.12, "Protocol checks\nBlind configuration negotiation\nPooled vs development-only vs ICL-only feature selection", ha="center", va="center", fontsize=9.8, bbox=box)
fig.tight_layout()
fig.savefig(OUT / "methodology_flowchart.pdf", bbox_inches="tight")
fig.savefig(OUT / "methodology_flowchart.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 2. Rank inversion: mean-rank slope chart with cell-to-cell ranges.
rank_models = [
    "xgb_full", "rf_strong_full", "rf_pruned_full", "xgb_balanced",
    "tabpfn_2.5", "tabicl", "rf_strong_balanced", "rf_pruned_balanced",
]
rank_ranges = (core_summary.groupby("model")
               .agg(acc_mean=("accuracy_rank", "mean"),
                    acc_min=("accuracy_rank", "min"),
                    acc_max=("accuracy_rank", "max"),
                    rec_mean=("recall_rank", "mean"),
                    rec_min=("recall_rank", "min"),
                    rec_max=("recall_rank", "max"))
               .loc[rank_models])
rank_colors = plt.get_cmap("cividis")(np.linspace(0.08, 0.92, len(rank_models)))
fig, ax = plt.subplots(figsize=(8.3, 7.2))
for (model, r), color in zip(rank_ranges.iterrows(), rank_colors):
    ax.plot([0, 1], [r.acc_mean, r.rec_mean], lw=2.2, color=color, alpha=0.95, zorder=2)
    ax.scatter([0, 1], [r.acc_mean, r.rec_mean], s=65, color=color,
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.vlines(0, r.acc_min, r.acc_max, color=color, lw=5, alpha=0.35, zorder=1)
    ax.vlines(1, r.rec_min, r.rec_max, color=color, lw=5, alpha=0.35, zorder=1)
    ax.text(-0.045, r.acc_mean, ABBR[model], ha="right", va="center",
            fontsize=9, fontweight="bold", color=color)
ax.set_xlim(-0.28, 1.17); ax.set_ylim(8.55, 0.45)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Mean accuracy rank", "Mean rare-recall rank"], fontsize=11)
ax.set_yticks(range(1, 9))
ax.set_ylabel("Rank across configurations (1 = highest)")
ax.grid(axis="y", linewidth=0.5, alpha=0.35)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.text(0.5, 8.35, "Lines rising toward rank 1 indicate improved position under the recall criterion.",
        ha="center", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "rank_inversion_core8.pdf", bbox_inches="tight")
fig.savefig(OUT / "rank_inversion_core8.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 3. Axis-2 single accessible heatmap: F1 and recall rows for each LLM.
models_heat = ["Kimi-K3", "GLM-5.2", "MiniMax-M3", "Qwen3.5-397B-A17B", "DeepSeek-V4-Pro", "Nemotron-3-Ultra-550b-a55b"]
classes = sorted(axis2_pc_mean["class"].unique())
rows_heat, row_labels = [], []
for model in models_heat:
    g = axis2_pc_mean[axis2_pc_mean.model == model].set_index("class").reindex(classes)
    rows_heat.append(g.f1.to_numpy()); row_labels.append(f"{model} - F1")
    rows_heat.append(g.recall.to_numpy()); row_labels.append(f"{model} - recall")
heat = np.asarray(rows_heat)
fig, ax = plt.subplots(figsize=(13.4, 7.8))
im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="cividis")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=42, ha="right", fontsize=8)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("Score")
ax.set_xlabel("Closed label set")
fig.tight_layout()
fig.savefig(OUT / "axis2_perclass_closed_label.pdf", bbox_inches="tight")
fig.savefig(OUT / "axis2_perclass_closed_label.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# LaTeX fragments
# ---------------------------------------------------------------------------
# Dataset-cell table, grounded in the persisted split and provenance record.
provenance = pd.read_csv(AX1 / "provenance.csv").set_index("cell")
dataset_lines = [
    r"\begin{tabular}{@{}llrrrrl@{}}",
    r"\toprule",
    r"Cell & Source & Cleaned rows & Classes & Held-out rows & Benign & Smallest held-out class \\",
    r"\midrule",
]
for cell in CELL_ORDER:
    base = PCc[(PCc.cell == cell) & (PCc.seed == 0) &
               (PCc.model == "rf_pruned_full") & (PCc.test_set == "capped")]
    support = base.groupby("class").support.mean()
    benign = BENIGN[cell]
    total = float(support.sum())
    non_benign = support.drop(index=benign)
    smallest_label = str(non_benign.idxmin())
    smallest_n = float(non_benign.min())
    source = "CIC-IDS2017" if cell.startswith("cic_ids") else "CIC-UNSW-NB15"
    variant = cell.rsplit("_", 1)[-1].replace("V", "V")
    dataset_lines.append(
        f"{variant} & {source} & {int(provenance.loc[cell, 'rows']):,} & "
        f"{int(provenance.loc[cell, 'classes'])} & {total:,.0f} & "
        f"{100.0 * float(support[benign]) / total:.1f}\\% & "
        f"{tex_escape(' '.join(smallest_label.split()))} ({smallest_n:,.0f}) \\\\"
    )
dataset_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_dataset_cells.tex").write_text("\n".join(dataset_lines))

# Evaluation-cap validation, generated from matching parent-partition hashes.
cap_lines = [
    r"\begin{tabular}{@{}llrrrrr@{}}",
    r"\toprule",
    r"Cell & Regime & $n$ & Mean $\Delta$ accuracy & Max $|\Delta|$ accuracy & Mean $\Delta$ macro-$F_1$ & Max $|\Delta|$ macro-$F_1$ \\",
    r"\midrule",
]
for cell in CELL_ORDER:
    for regime in ["balanced", "large"]:
        r = cap_summary[(cap_summary.cell == cell) & (cap_summary.regime == regime)].iloc[0]
        cap_lines.append(
            f"{CELL_LABEL[cell]} & {regime} & {int(r.n)} & {r.mean_d_accuracy:.6f} & "
            f"{r.max_abs_d_accuracy:.6f} & {r.mean_d_macro_f1:.6f} & "
            f"{r.max_abs_d_macro_f1:.6f} \\\\"
        )
cap_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_cap_validation.tex").write_text("\n".join(cap_lines))

# Compact rank summary.
rank_lines = [r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule", r"Cell & $\rho_s$ & Acc. leader & Recall & Recall leader & Recall \\", r"\midrule"]
for cell in CELL_ORDER:
    g = core_summary[core_summary.cell == cell]
    a = g.sort_values("accuracy", ascending=False).iloc[0]
    rr = g.sort_values("rare_recall", ascending=False).iloc[0]
    rho = rank_df.loc[rank_df.cell == cell, "spearman"].iloc[0]
    rank_lines.append(f"{CELL_LABEL[cell]} & {rho:.3f} & {tex_escape(ABBR[a.model])} & {a.rare_recall:.3f} & {tex_escape(ABBR[rr.model])} & {rr.rare_recall:.3f} \\\\")
rank_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_rank_summary.tex").write_text("\n".join(rank_lines))

# 2x2 arm means.
sb_lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Budget & Distribution & Rows & Unique rows & Macro-$F_1$ & Rare recall \\" , r"\midrule"]
for arm in ["small_natural", "small_balanced", "large_natural", "large_balanced"]:
    r = size_balance_summary[size_balance_summary.arm == arm].iloc[0]
    sb_lines.append(f"{r['size'].title()} & {r['balance'].title()} & {r.n_train:,.0f} & {r.n_unique:,.0f} & {r.macro_f1:.3f} & {r.rare_recall:.3f} \\\\")
sb_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_size_balance_means.tex").write_text("\n".join(sb_lines))

# Selected pairwise comparisons, TabPFN only in main text.
sel_pairs = pairs[pairs.fm == "tabpfn_2.5"].copy()
pair_lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Cell & Comparator & $\Delta$ recall & 95\% CI & Wins & $p_{\mathrm{Holm}}$ \\" , r"\midrule"]
for cell in CELL_ORDER:
    for cl in ["xgb_balanced", "rf_strong_balanced", "xgb_full", "rf_pruned_full"]:
        r = sel_pairs[(sel_pairs.cell == cell) & (sel_pairs.versus == cl)].iloc[0]
        pair_lines.append(f"{CELL_LABEL[cell]} & {tex_escape(ABBR[cl])} & {r.mean_delta:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {int(r.wins)}/10 & {r.p_holm_6:.4f} \\\\")
    pair_lines.append(r"\addlinespace")
pair_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_tabpfn_selected_pairs.tex").write_text("\n".join(pair_lines))

# Stability/efficiency table.
stab_lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule", r"Cell & Model & Rare recall & SD & Fit rows & Fit (s) & Predict (s) \\" , r"\midrule"]
for cell in CELL_ORDER:
    for model in ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl"]:
        r = core_summary[(core_summary.cell == cell) & (core_summary.model == model)].iloc[0]
        stab_lines.append(f"{CELL_LABEL[cell]} & {tex_escape(ABBR[model])} & {r.rare_recall:.3f} & {r.rare_recall_sd:.4f} & {r.n_train:,.0f} & {r.fit_s:.2f} & {r.predict_s:.1f} \\\\")
    stab_lines.append(r"\addlinespace")
stab_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_stability_efficiency.tex").write_text("\n".join(stab_lines))

# Selected operating-point summary.
op_lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule", r"Cell & Model & Rec. & HR/LP & HP/LR & Missed & Alerts/TP \\" , r"\midrule"]
for cell in selected:
    for model in selected[cell]:
        r = op_summary[(op_summary.cell == cell) & (op_summary.model == model)].iloc[0]
        alert = "$\\infty$" if not np.isfinite(r.alerts_per_tp_pooled) else f"{r.alerts_per_tp_pooled:.2f}"
        op_lines.append(f"{CELL_LABEL[cell]} & {tex_escape(MODEL_LABEL[model])} & {int(r.recovered)} & {int(r.high_recall_low_precision)} & {int(r.high_precision_low_recall)} & {int(r.missed)} & {alert} \\\\")
    op_lines.append(r"\addlinespace")
op_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_operating_selected.tex").write_text("\n".join(op_lines))

# Selected cost table: XGB large and TabPFN against pruned large.
cost_lines = [r"\begin{tabular}{@{}lllrrr@{}}", r"\toprule", r"Cell & Model & Costs & Positive seeds & Median $\rho^*$ & $1/\rho^*$ \\" , r"\midrule"]
for cell in CELL_ORDER:
    for model in ["xgb_full", "tabpfn_2.5"]:
        for scen in ["uniform", "severity weighted"]:
            r = cost_summary[(cost_summary.cell == cell) & (cost_summary.model == model) & (cost_summary.scenario == scen)].iloc[0]
            cost_lines.append(f"{CELL_LABEL[cell]} & {tex_escape(ABBR[model])} & {scen.title()} & {int(r.positive_seeds)}/10 & {r.rho_median_positive:.3f} & {r.one_over_rho:.1f} \\\\")
    cost_lines.append(r"\addlinespace")
cost_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_cost_selected.tex").write_text("\n".join(cost_lines))

# Axis-2 table.
axis2_order = ["RandomForest", "Kimi-K3", "GLM-5.2", "MiniMax-M3", "Qwen3.5-397B-A17B", "DeepSeek-V4-Pro", "Nemotron-3-Ultra-550b-a55b"]
a2_lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule", r"Model & Accuracy & Macro-$F_1$ & Parse & $\Delta F_1$ & $p_{\mathrm{Bonf}}$ & Cost (USD) \\" , r"\midrule"]
for model in axis2_order:
    r = axis2_summary[axis2_summary.model == model].iloc[0]
    delta = "--" if pd.isna(r.mean_delta) else f"{r.mean_delta:+.3f}"
    p = "--" if pd.isna(r.p_bonferroni_6) else f"{r.p_bonferroni_6:.4f}"
    name = (model.replace("RandomForest", "Random Forest")
                 .replace("Qwen3.5-397B-A17B", "Qwen3.5")
                 .replace("Nemotron-3-Ultra-550b-a55b", "Nemotron-3 Ultra"))
    a2_lines.append(f"{tex_escape(name)} & {r.accuracy:.3f} & {r.macro_f1_closed:.3f} & {r.parse_rate:.3f} & {delta} & {p} & {r.usd:.2f} \\\\")
a2_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_axis2_summary.tex").write_text("\n".join(a2_lines))

# Main-text rare-label definitions.
rare_lines = [
    r"\begin{tabular}{@{}lp{0.72\linewidth}@{}}",
    r"\toprule",
    r"Cell & Classes below the 0.5\% evaluation-share threshold \\",
    r"\midrule",
]
for cell in CELL_ORDER:
    rare_text = ", ".join(" ".join(str(label).split()) for label in RARE[cell])
    rare_lines.append(f"{CELL_LABEL[cell]} & {tex_escape(rare_text)} " + r"\\")
rare_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_rare_classes.tex").write_text("\n".join(rare_lines))

# Complete records are distributed as CSV rather than manuscript appendices.

# Factorial paired effects.
sb_effect_labels = {
    "balance at small": "Balance, small budget",
    "balance at large": "Balance, large budget",
    "size at natural": "Size, natural distribution",
    "size at balanced": "Size, balanced distribution",
}
sb_effect = [
    r"\begin{tabular}{@{}lrrrr@{}}",
    r"\toprule",
    r"Contrast & $\Delta$ rare recall & 95\% CI & Wins & $p_{\mathrm{Holm}}$ \\",
    r"\midrule",
]
for contrast in ["balance at small", "balance at large", "size at natural", "size at balanced"]:
    r = size_balance_contrasts[size_balance_contrasts.contrast == contrast].iloc[0]
    sb_effect.append(
        f"{sb_effect_labels[contrast]} & {r.mean_delta:+.3f} & "
        f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {int(r.wins)}/10 & {r.p_holm_4:.4f} \\\\"
    )
sb_effect += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_size_balance_effects.tex").write_text("\n".join(sb_effect))

# ManyClass sensitivity.
many_lines = [
    r"\begin{tabular}{@{}rrrrrr@{}}",
    r"\toprule",
    r"Alphabet & Redundancy & Accuracy & Macro-$F_1$ & Rare recall & SD \\",
    r"\midrule",
]
for _, r in many_summary.sort_values(["alphabet_size", "n_estimators_redundancy"]).iterrows():
    many_lines.append(
        f"{int(r.alphabet_size)} & {int(r.n_estimators_redundancy)} & {r.accuracy:.4f} & "
        f"{r.macro_f1:.4f} & {r.rare_recall:.4f} & {r.rare_recall_sd:.4f} \\\\"
    )
many_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_manyclass.tex").write_text("\n".join(many_lines))

# Axis-2 feature-selection audit summaries.
mi_regime_label = {
    "development_only": "Development only",
    "icl_only": "ICL only",
    "legacy_pooled": "Pooled labelled",
}
mi_lines = [
    r"\begin{tabular}{@{}lrrrr@{}}",
    r"\toprule",
    r"Selector & Accuracy & Macro-$F_1$ & Parse rate & Wall time (min) \\",
    r"\midrule",
]
for regime in ["development_only", "icl_only", "legacy_pooled"]:
    r = mi_summary[mi_summary.regime == regime].iloc[0]
    mi_lines.append(
        f"{mi_regime_label[regime]} & {r.accuracy:.3f} & {r.macro_f1:.3f} & "
        f"{r.parse_rate:.3f} & {r.wall_min:.1f} \\\\"
    )
mi_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_axis2_mi_summary.tex").write_text("\n".join(mi_lines))

mi_effect_lines = [
    r"\begin{tabular}{@{}llrrrr@{}}",
    r"\toprule",
    r"Selector vs pooled & Metric & $\Delta$ & 95\% CI & Wins & $p$ \\",
    r"\midrule",
]
for metric in ["macro_f1", "parse_rate"]:
    for regime in ["development_only", "icl_only"]:
        r = mi_effects[(mi_effects.metric == metric) & (mi_effects.regime == regime)].iloc[0]
        metric_label = "Macro-$F_1$" if metric == "macro_f1" else "Parse rate"
        mi_effect_lines.append(
            f"{mi_regime_label[regime]} & {metric_label} & {r.mean_delta:+.3f} & "
            f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {int(r.wins)}/10 & {r.p_raw:.4f} \\\\"
        )
mi_effect_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_axis2_mi_effects.tex").write_text("\n".join(mi_effect_lines))

# TabSTAR semantic-name negative control.
metric_label = {"accuracy": "Accuracy", "macro_f1": "Macro-$F_1$", "rare_recall": "Rare recall"}
tabstar_lines = [
    r"\begin{tabular}{@{}lrrrr@{}}",
    r"\toprule",
    r"Metric & Named $-$ generic & 95\% CI & Wins & $p$ \\",
    r"\midrule",
]
for metric in ["accuracy", "macro_f1", "rare_recall"]:
    r = tabstar_control[tabstar_control.metric == metric].iloc[0]
    tabstar_lines.append(
        f"{metric_label[metric]} & {r.mean_delta_named_minus_generic:+.4f} & "
        f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] & {int(r.wins)}/{int(r.n)} & {r.p_raw:.4f} \\\\"
    )
tabstar_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_tabstar_control.tex").write_text("\n".join(tabstar_lines))

# Blind configuration-negotiation table.
cal_name = {
    "Qwen3.5-397B-A17B": "Qwen3.5",
    "Nemotron-3-Ultra-550b-a55b": "Nemotron-3 Ultra",
}
cal_lines = [
    r"\begin{tabular}{@{}llrrl@{}}",
    r"\toprule",
    r"Model & Frozen configuration & Clean & Probes & Finish \\",
    r"\midrule",
]
for model in ["GLM-5.2", "DeepSeek-V4-Pro", "Kimi-K3", "MiniMax-M3", "Qwen3.5-397B-A17B", "Nemotron-3-Ultra-550b-a55b"]:
    config = chosen_config[model]
    r = calibration[(calibration.model == model) & (calibration.config == config)].iloc[0]
    cal_lines.append(
        f"{tex_escape(cal_name.get(model, model))} & {tex_escape(config)} & {int(r.clean)} & "
        f"{int(r['of'])} & {tex_escape(str(r.finish_reason))} \\\\"
    )
cal_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_axis2_calibration.tex").write_text("\n".join(cal_lines))


# Main-text combined sensitivity tables. These compact panels replace the
# former appendices while preserving every result required by the revision.
size_balance_combined = (
    r"\begin{minipage}{\linewidth}" + "\n"
    + r"\centering" + "\n"
    + r"\textbf{Panel A: arm means.}\\[3pt]" + "\n"
    + (OUT / "table_size_balance_means.tex").read_text() + "\n"
    + r"\par\medskip" + "\n"
    + r"\textbf{Panel B: planned paired contrasts.}\\[3pt]" + "\n"
    + (OUT / "table_size_balance_effects.tex").read_text() + "\n"
    + r"\end{minipage}"
)
(OUT / "table_size_balance_combined.tex").write_text(size_balance_combined)

axis2_mi_combined = (
    r"\begin{minipage}{\linewidth}" + "\n"
    + r"\centering" + "\n"
    + r"\textbf{Panel A: selector-regime means.}\\[3pt]" + "\n"
    + (OUT / "table_axis2_mi_summary.tex").read_text() + "\n"
    + r"\par\medskip" + "\n"
    + r"\textbf{Panel B: paired changes relative to the pooled-labelled selector.}\\[3pt]" + "\n"
    + (OUT / "table_axis2_mi_effects.tex").read_text() + "\n"
    + r"\end{minipage}"
)
(OUT / "table_axis2_mi_combined.tex").write_text(axis2_mi_combined)

# Complete machine-readable records remain in CSV. Obsolete manuscript
# appendix fragments are intentionally not part of the release.
for obsolete in [
    "appendix_core_panel.tex",
    "appendix_pairwise.tex",
    "appendix_operating_points.tex",
]:
    path = OUT / obsolete
    if path.exists():
        path.unlink()

# Machine-readable verification summary.
summary = {
    "axis1_core_records": int(len(core)),
    "axis1_core_cells": CELL_ORDER,
    "rank_correlations": {r["cell"]: r["spearman"] for _, r in rank_df.iterrows()},
    "pairwise_counts": pairs.direction.value_counts().to_dict(),
    "axis2_total_cost_usd": float(axis2_summary[axis2_summary.model != "RandomForest"].usd.sum()),
    "axis2_models": axis2_order,
    "rare_classes": RARE,
}
(OUT / "verification_summary.json").write_text(json.dumps(summary, indent=2))

# Hash every distributed result artifact except the manifest itself. This is
# deliberately executed after all derived outputs above have been written.
manifest_path = OUT / "release_file_manifest_sha256.csv"
manifest_rows = []
for path in sorted((ROOT / "results").rglob("*")):
    if path.is_file() and path != manifest_path:
        manifest_rows.append({
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

print(json.dumps(summary, indent=2))
