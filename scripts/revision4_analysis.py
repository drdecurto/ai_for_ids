#!/usr/bin/env python3
"""Revision-4 analyses derived from the persisted round-3 record.

Everything here is analysis-only (no model execution). It answers:

  R1-3  rare-class threshold sensitivity (0.1 / 0.25 / 0.5 / 1.0 %)
  R1-4  evaluation-cap validation for operational metrics
        (false alarms, misses, rare-alert precision, alerts/TP, crossovers)
  R1-7  severity-weight vector, alternative vectors, crossovers vs strong incumbents
  R1-8  confidence intervals for the ManyClass sensitivity (n available seeds)
  R4-2  per-configuration macro-F1 (capped, and full-partition where hashes match)
  R4-4  per-class precision / recall / support for the rare set
  R4-m  per-1,000-row Axis-2 cost; component-sum check; audit-row explanation

Outputs go to results/derived/rev4/ (CSV) and to the manuscript generated/ folder
(LaTeX fragments).  Run from the repository root:

    python scripts/revision4_analysis.py --manuscript ../manuscript
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AX1 = ROOT / "results" / "axis1"
AX2 = ROOT / "results" / "axis2"
SENS = ROOT / "results" / "sensitivity"
OUT = ROOT / "results" / "derived" / "rev4"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--manuscript", default=str(ROOT.parent / "manuscript"))
ARGS = ap.parse_args()
GEN = Path(ARGS.manuscript) / "generated"
GEN.mkdir(parents=True, exist_ok=True)

CORE = ["rf_pruned_full", "rf_strong_full", "xgb_full",
        "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
        "tabpfn_2.5", "tabicl"]
CLASSICAL = CORE[:6]
FM = ["tabpfn_2.5", "tabicl"]
CELLS = ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1", "cic_unsw_V2"]
CELL_LABEL = {"cic_ids2017_V1": "CIC-IDS2017 V1", "cic_ids2017_V3": "CIC-IDS2017 V3",
              "cic_unsw_V1": "CIC-UNSW-NB15 V1", "cic_unsw_V2": "CIC-UNSW-NB15 V2"}
ABBR = {"rf_pruned_full": "RF-P/L", "rf_strong_full": "RF-S/L", "xgb_full": "XGB/L",
        "rf_pruned_balanced": "RF-P/B", "rf_strong_balanced": "RF-S/B", "xgb_balanced": "XGB/B",
        "tabpfn_2.5": "TabPFN", "tabicl": "TabICL", "tabstar": "TabSTAR"}
LONG = {"rf_pruned_full": "Pruned RF, large", "rf_strong_full": "Strong RF, large",
        "xgb_full": "XGBoost, large", "rf_pruned_balanced": "Pruned RF, balanced",
        "rf_strong_balanced": "Strong RF, balanced", "xgb_balanced": "XGBoost, balanced",
        "tabpfn_2.5": "TabPFN 2.5", "tabicl": "TabICL", "tabstar": "TabSTAR"}

# Predeclared illustrative severity weights (identical to recompute_manuscript_results.py).
SEV = {
    "Heartbleed": 8, "Infiltration": 8, "Web Attack": 4,
    "Web Attack  Brute Force": 4, "Web Attack  XSS": 4, "Web Attack  Sql Injection": 4,
    "Bot": 3, "Brute Force": 2, "FTP-Patator": 2, "SSH-Patator": 2,
    "Port Scan": 1, "PortScan": 1, "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1,
    "DoS slowloris": 1, "DoS Slowhttptest": 1,
    "Backdoor": 8, "Shellcode": 6, "Worms": 6, "Exploits": 4, "Analysis": 2,
    "Reconnaissance": 1, "Generic": 1, "Fuzzers": 1, "DoS": 1,
}


# ---------------------------------------------------------------- helpers
def signflip(d: Sequence[float]) -> float:
    d = np.asarray(d, dtype=float)
    obs = abs(float(d.mean()))
    vals = [abs(float(np.mean(d * np.asarray(s, dtype=float))))
            for s in itertools.product((-1.0, 1.0), repeat=len(d))]
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))


def bootstrap_ci(d, n_boot=50000, seed=2026):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))


def t_ci(x, conf=0.95):
    """Student-t interval for a mean (used where n is too small for a bootstrap)."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return (float(x.mean() - h), float(x.mean() + h))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        out[idx] = min(1.0, running)
    return out


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_"))


def fmt(v, nd=3, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


# ---------------------------------------------------------------- load
M = pd.read_csv(AX1 / "metrics.csv").drop_duplicates(["cell", "seed", "model", "test_set"], keep="last")
PC = pd.read_csv(AX1 / "per_class.csv")
CM = pd.read_csv(AX1 / "confusion.csv")
Mc = M[(M.test_set == "capped") & M.model.isin(CORE)].copy()
PCc = PC[(PC.test_set == "capped")].copy()
CMc = CM[(CM.test_set == "capped")].copy()

BENIGN = {}
SHARE = {}
for cell in CELLS:
    base = PCc[(PCc.cell == cell) & (PCc.seed == 0) & (PCc.model == "rf_pruned_full")]
    ben = [x for x in base["class"].astype(str).unique() if x.lower() == "benign"]
    BENIGN[cell] = ben[0]
    sup = base.groupby("class").support.mean()
    SHARE[cell] = sup / float(sup.sum())


def rare_set(cell, thr):
    return sorted(c for c, v in SHARE[cell].items() if c != BENIGN[cell] and v < thr)


RARE = {c: rare_set(c, 0.005) for c in CELLS}


def rare_recall_by_seed(cell, R, models=None):
    g = PCc[(PCc.cell == cell) & PCc["class"].isin(R)]
    if models is not None:
        g = g[g.model.isin(models)]
    return g.groupby(["seed", "model"]).recall.mean().rename("rare_recall").reset_index()


# ========================================================================
# R1-3  Threshold sensitivity
# ========================================================================
THRS = [0.001, 0.0025, 0.005, 0.01]
rows = []
for thr in THRS:
    n_fm10 = n_cl10 = n_mixed = 0
    for cell in CELLS:
        R = rare_set(cell, thr)
        rs = rare_recall_by_seed(cell, R, CORE)
        core = Mc[Mc.cell == cell].merge(rs, on=["seed", "model"])
        s = core.groupby("model").agg(acc=("accuracy", "mean"), rr=("rare_recall", "mean"))
        rho = float(s.acc.corr(s.rr, method="spearman"))
        top = set(s.acc.nlargest(3).index)
        bot = set(s.rr.nsmallest(3).index)
        pvt = core.pivot(index="seed", columns="model", values="rare_recall")
        for fm in FM:
            for cl in CLASSICAL:
                d = (pvt[fm] - pvt[cl]).to_numpy()
                w = int((d > 0).sum())
                n_fm10 += w == 10
                n_cl10 += w == 0
                n_mixed += 0 < w < 10
        rows.append({
            "threshold": thr, "cell": cell, "n_rare": len(R), "spearman": rho,
            "same_three": top == bot, "acc_leader": s.acc.idxmax(), "recall_leader": s.rr.idxmax(),
            **{f"rr_{m}": float(s.rr[m]) for m in CORE},
            "rare_classes": "; ".join(R),
        })
    for r in rows:
        if r["threshold"] == thr and "fm_all10" not in r:
            r["fm_all10"], r["classical_all10"], r["mixed"] = n_fm10, n_cl10, n_mixed
thr_df = pd.DataFrame(rows)
thr_df.to_csv(OUT / "threshold_sensitivity.csv", index=False)

with open(GEN / "table_threshold_sensitivity.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrcllrrrr@{}}\n\\toprule\n")
    f.write("Threshold & Cell & $|\\mathcal{R}|$ & $\\rho_s$ & Same three & Acc.\\ leader & Recall leader "
            "& XGB/L & XGB/B & TabPFN & TabICL \\\\\n\\midrule\n")
    for thr in THRS:
        g = thr_df[thr_df.threshold == thr]
        for i, (_, r) in enumerate(g.iterrows()):
            lab = f"{thr*100:g}\\%" if i == 0 else ""
            f.write(f"{lab} & {CELL_LABEL[r.cell]} & {r.n_rare} & {r.spearman:+.3f} & "
                    f"{'yes' if r.same_three else 'no'} & {ABBR[r.acc_leader]} & {ABBR[r.recall_leader]} & "
                    f"{r.rr_xgb_full:.3f} & {r.rr_xgb_balanced:.3f} & {r['rr_tabpfn_2.5']:.3f} & {r.rr_tabicl:.3f} \\\\\n")
        r0 = g.iloc[0]
        f.write(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{48 paired comparisons at {thr*100:g}\\%: "
                f"prior-fitted model on all ten seeds {r0.fm_all10}; classical on all ten seeds {r0.classical_all10}; "
                f"mixed {r0.mixed}.}}}} \\\\\n")
        if thr != THRS[-1]:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ========================================================================
# R4-2  Core panel: accuracy, macro-F1 (cap), macro-F1 (full, matching hash), rare recall
# ========================================================================
rs05 = pd.concat([rare_recall_by_seed(c, RARE[c], CORE).assign(cell=c) for c in CELLS])
core = Mc.merge(rs05, on=["cell", "seed", "model"], how="left")

# Full-partition macro-F1 restricted to seed rows whose partition hash matches the capped row.
full_rows = []
for cell in CELLS:
    for model in CLASSICAL:
        g = M[(M.cell == cell) & (M.model == model)]
        gc_ = g[g.test_set == "capped"].set_index("seed")
        gf = g[g.test_set == "full"].set_index("seed")
        for seed in sorted(set(gc_.index) & set(gf.index)):
            if str(gc_.loc[seed, "test_hash"]) != str(gf.loc[seed, "test_hash"]):
                continue
            full_rows.append({"cell": cell, "model": model, "seed": seed,
                              "macro_f1_full": float(gf.loc[seed, "macro_f1"]),
                              "accuracy_full": float(gf.loc[seed, "accuracy"]),
                              "macro_f1_cap": float(gc_.loc[seed, "macro_f1"])})
full_df = pd.DataFrame(full_rows)
full_sum = full_df.groupby(["cell", "model"], as_index=False).agg(
    n_full=("seed", "size"), macro_f1_full=("macro_f1_full", "mean"),
    macro_f1_cap_matched=("macro_f1_cap", "mean"))

panel = (core.groupby(["cell", "model"], as_index=False)
         .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
              rare_recall=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
              rare_recall_min=("rare_recall", "min"), rare_recall_max=("rare_recall", "max")))
panel = panel.merge(full_sum, on=["cell", "model"], how="left")
panel["accuracy_rank"] = panel.groupby("cell")["accuracy"].rank(method="min", ascending=False).astype(int)
panel["macro_f1_rank"] = panel.groupby("cell")["macro_f1"].rank(method="min", ascending=False).astype(int)
panel["recall_rank"] = panel.groupby("cell")["rare_recall"].rank(method="min", ascending=False).astype(int)
panel.to_csv(OUT / "core_panel_full.csv", index=False)

# Spearman between macro-F1 and rare recall, and between accuracy and macro-F1, per cell
rk = []
for cell in CELLS:
    g = panel[panel.cell == cell]
    rk.append({"cell": cell,
               "rho_acc_rare": float(g.accuracy.corr(g.rare_recall, method="spearman")),
               "rho_f1_rare": float(g.macro_f1.corr(g.rare_recall, method="spearman")),
               "rho_acc_f1": float(g.accuracy.corr(g.macro_f1, method="spearman")),
               "f1_leader": g.loc[g.macro_f1.idxmax(), "model"]})
rk_df = pd.DataFrame(rk)
rk_df.to_csv(OUT / "core_panel_rank_correlations.csv", index=False)

with open(GEN / "table_core_panel.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n")
    f.write("Cell & Configuration & Accuracy & Macro-$F_1$ (cap) & Macro-$F_1$ (full) & $n_{\\mathrm{full}}$ "
            "& Rare-class recall & SD & Ranks (A/F/R) \\\\\n\\midrule\n")
    for cell in CELLS:
        g = panel[panel.cell == cell].set_index("model").loc[CORE].reset_index()
        for i, r in g.iterrows():
            lab = CELL_LABEL[cell] if i == 0 else ""
            mf = fmt(r.macro_f1_full) if not np.isnan(r.macro_f1_full) else "--"
            nf = f"{int(r.n_full)}" if not np.isnan(r.n_full) else "--"
            f.write(f"{lab} & {LONG[r.model]} & {r.accuracy:.4f} & {r.macro_f1:.3f} & {mf} & {nf} & "
                    f"{r.rare_recall:.3f} & {r.rare_recall_sd:.4f} & "
                    f"{r.accuracy_rank}/{r.macro_f1_rank}/{r.recall_rank} \\\\\n")
        if cell != CELLS[-1]:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ========================================================================
# R1-4  Cap validation for operational metrics (classical configs, matching hashes)
# ========================================================================
def alarm_quantities(cm_df, cell, R):
    """Return benign false alarms, attack misses, rare alerts, rare TPs, from one confusion matrix."""
    b = BENIGN[cell]
    fa = float(cm_df[(cm_df.true_label == b) & (cm_df.pred_label != b)]["count"].sum())
    miss = float(cm_df[(cm_df.true_label != b) & (cm_df.pred_label == b)]["count"].sum())
    alerts = float(cm_df[cm_df.pred_label.isin(R)]["count"].sum())
    tp = float(cm_df[(cm_df.true_label == cm_df.pred_label) & cm_df.true_label.isin(R)]["count"].sum())
    return fa, miss, alerts, tp


op_rows = []
for cell in CELLS:
    R = RARE[cell]
    for model in CLASSICAL:
        g = M[(M.cell == cell) & (M.model == model)]
        gc_ = g[g.test_set == "capped"].set_index("seed")
        gf = g[g.test_set == "full"].set_index("seed")
        for seed in sorted(set(gc_.index) & set(gf.index)):
            if str(gc_.loc[seed, "test_hash"]) != str(gf.loc[seed, "test_hash"]):
                continue
            cmc = CM[(CM.cell == cell) & (CM.model == model) & (CM.seed == seed) & (CM.test_set == "capped")]
            cmf = CM[(CM.cell == cell) & (CM.model == model) & (CM.seed == seed) & (CM.test_set == "full")]
            if cmc.empty or cmf.empty:
                continue
            fa_c, m_c, al_c, tp_c = alarm_quantities(cmc, cell, R)
            fa_f, m_f, al_f, tp_f = alarm_quantities(cmf, cell, R)
            # per-class rare recall difference (cap - full), mean over rare set
            pc_c = PC[(PC.cell == cell) & (PC.model == model) & (PC.seed == seed) & (PC.test_set == "capped") & PC["class"].isin(R)]
            pc_f = PC[(PC.cell == cell) & (PC.model == model) & (PC.seed == seed) & (PC.test_set == "full") & PC["class"].isin(R)]
            rr_c = float(pc_c.recall.mean())
            rr_f = float(pc_f.recall.mean())
            rp_c = float(pc_c.precision.mean())
            rp_f = float(pc_f.precision.mean())
            op_rows.append({
                "cell": cell, "model": model, "seed": seed,
                "regime": "large" if model.endswith("_full") else "balanced",
                "fa_cap": fa_c, "fa_full": fa_f, "miss_cap": m_c, "miss_full": m_f,
                "alerts_cap": al_c, "alerts_full": al_f, "tp_cap": tp_c, "tp_full": tp_f,
                "aptp_cap": al_c / tp_c if tp_c > 0 else np.nan,
                "aptp_full": al_f / tp_f if tp_f > 0 else np.nan,
                "rare_recall_cap": rr_c, "rare_recall_full": rr_f,
                "rare_precision_cap": rp_c, "rare_precision_full": rp_f,
            })
opv = pd.DataFrame(op_rows)
opv["d_fa_rel"] = (opv.fa_cap - opv.fa_full) / opv.fa_full
opv["d_miss_rel"] = (opv.miss_cap - opv.miss_full) / opv.miss_full
opv["d_aptp"] = opv.aptp_cap - opv.aptp_full
opv["d_rare_recall"] = opv.rare_recall_cap - opv.rare_recall_full
opv["d_rare_precision"] = opv.rare_precision_cap - opv.rare_precision_full
opv.to_csv(OUT / "cap_validation_operational_rows.csv", index=False)

opv_sum = (opv.groupby(["cell", "regime"], as_index=False)
           .agg(n=("seed", "size"),
                mean_d_fa_rel=("d_fa_rel", "mean"), max_abs_d_fa_rel=("d_fa_rel", lambda s: float(np.nanmax(np.abs(s)))),
                mean_d_miss_rel=("d_miss_rel", "mean"), max_abs_d_miss_rel=("d_miss_rel", lambda s: float(np.nanmax(np.abs(s)))),
                mean_d_aptp=("d_aptp", "mean"), max_abs_d_aptp=("d_aptp", lambda s: float(np.nanmax(np.abs(s)))),
                mean_d_rr=("d_rare_recall", "mean"), max_abs_d_rr=("d_rare_recall", lambda s: float(np.nanmax(np.abs(s)))),
                mean_d_rp=("d_rare_precision", "mean"), max_abs_d_rp=("d_rare_precision", lambda s: float(np.nanmax(np.abs(s))))))
opv_sum.to_csv(OUT / "cap_validation_operational_summary.csv", index=False)

# Crossover validation: rho* against RF-P/L computed from capped and from full matrices, same seeds
cx_rows = []
for cell in CELLS:
    g = opv[opv.cell == cell]
    for scoring in ["cap", "full"]:
        base = g[g.model == "rf_pruned_full"].set_index("seed")
        for model in [m for m in CLASSICAL if m != "rf_pruned_full"]:
            other = g[g.model == model].set_index("seed")
            for seed in sorted(set(base.index) & set(other.index)):
                dm = float(base.loc[seed, f"miss_{scoring}"] - other.loc[seed, f"miss_{scoring}"])
                dfa = float(other.loc[seed, f"fa_{scoring}"] - base.loc[seed, f"fa_{scoring}"])
                rho = np.nan if abs(dm) < 1e-12 else dfa / dm
                cx_rows.append({"cell": cell, "model": model, "seed": seed, "scoring": scoring, "rho": rho,
                                "positive": bool(rho > 0) if not np.isnan(rho) else False})
cx = pd.DataFrame(cx_rows)
cx.to_csv(OUT / "cap_validation_crossover_rows.csv", index=False)
cx_pvt = cx.pivot_table(index=["cell", "model", "seed"], columns="scoring", values="rho").reset_index()
cx_pvt = cx_pvt[(cx_pvt["cap"] > 0) & (cx_pvt["full"] > 0)]
cx_pvt["rel_err"] = (cx_pvt["cap"] - cx_pvt["full"]) / cx_pvt["full"]
cx_sum = (cx_pvt.groupby(["cell", "model"], as_index=False)
          .agg(n=("seed", "size"), med_cap=("cap", "median"), med_full=("full", "median"),
               mean_rel_err=("rel_err", "mean"), max_abs_rel_err=("rel_err", lambda s: float(np.max(np.abs(s))))))
cx_sum.to_csv(OUT / "cap_validation_crossover_summary.csv", index=False)

with open(GEN / "table_cap_validation_operational.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n")
    f.write(" & & & \\multicolumn{2}{c}{Rare-class recall} & \\multicolumn{2}{c}{Rare-class precision} & "
            "\\multicolumn{2}{c}{False alarms (rel.)} & \\multicolumn{2}{c}{Alerts/TP} \\\\\n")
    f.write("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n")
    f.write("Cell & Regime & $n$ & mean $\\Delta$ & max $|\\Delta|$ & mean $\\Delta$ & max $|\\Delta|$ & "
            "mean $\\Delta$ & max $|\\Delta|$ & mean $\\Delta$ & max $|\\Delta|$ \\\\\n\\midrule\n")
    for _, r in opv_sum.iterrows():
        f.write(f"{CELL_LABEL[r.cell]} & {r.regime} & {int(r.n)} & {r.mean_d_rr:+.4f} & {r.max_abs_d_rr:.4f} & "
                f"{r.mean_d_rp:+.4f} & {r.max_abs_d_rp:.4f} & {r.mean_d_fa_rel:+.3f} & {r.max_abs_d_fa_rel:.3f} & "
                f"{r.mean_d_aptp:+.3f} & {r.max_abs_d_aptp:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(GEN / "table_cap_validation_crossover.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n")
    f.write("Cell & Model & $n$ & Median $\\rho^*$ (cap) & Median $\\rho^*$ (full) & Mean rel.\\ error & Max $|$rel.\\ error$|$ \\\\\n\\midrule\n")
    for cell in CELLS:
        g = cx_sum[cx_sum.cell == cell]
        for i, (_, r) in enumerate(g.iterrows()):
            lab = CELL_LABEL[cell] if i == 0 else ""
            f.write(f"{lab} & {ABBR[r.model]} & {int(r.n)} & {r.med_cap:.3f} & {r.med_full:.3f} & "
                    f"{r.mean_rel_err:+.3f} & {r.max_abs_rel_err:.3f} \\\\\n")
        if cell != CELLS[-1]:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ========================================================================
# R1-7  Severity weights, alternative vectors, crossovers vs strong incumbents
# ========================================================================
def cost_vectors(cell):
    R = RARE[cell]
    classes = [c for c in SHARE[cell].index if c != BENIGN[cell]]
    uniform = {c: 1.0 for c in classes}
    severity = {c: float(SEV.get(c, 1.0)) for c in classes}
    rare_flat = {c: (4.0 if c in R else 1.0) for c in classes}
    steep = {c: (float(SEV.get(c, 1.0)) ** 2 if c in R else float(SEV.get(c, 1.0))) for c in classes}
    share = SHARE[cell]
    inv = {c: float(min(8.0, max(1.0, 0.001 / max(share[c], 1e-9)))) for c in classes}
    return {"uniform": uniform, "severity": severity, "rare-flat": rare_flat, "steep": steep, "inverse-prevalence": inv}


vec_rows = []
for cell in CELLS:
    vs = cost_vectors(cell)
    for c in vs["uniform"]:
        vec_rows.append({"cell": cell, "class": c, "share_pct": 100 * SHARE[cell][c], "rare": c in RARE[cell],
                         **{k: v[c] for k, v in vs.items()}})
vec_df = pd.DataFrame(vec_rows)
vec_df.to_csv(OUT / "severity_weight_vectors.csv", index=False)

with open(GEN / "table_severity_weights.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n")
    f.write("Source & Class & Share (\\%) & Rare & Severity & Rare-flat & Steep & Inv.-prev. \\\\\n\\midrule\n")
    for cell, src in [("cic_ids2017_V1", "CIC-IDS2017 (V1 labels)"), ("cic_ids2017_V3", "CIC-IDS2017 (V3 labels)"), ("cic_unsw_V1", "CIC-UNSW-NB15")]:
        g = vec_df[vec_df.cell == cell].sort_values("share_pct")
        for i, (_, r) in enumerate(g.iterrows()):
            lab = src if i == 0 else ""
            f.write(f"{lab} & {esc(r['class'])} & {r.share_pct:.3f} & {'yes' if r.rare else 'no'} & "
                    f"{r.severity:g} & {r['rare-flat']:g} & {r.steep:g} & {r['inverse-prevalence']:.1f} \\\\\n")
        if cell != "cic_unsw_V1":
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")


def alarms(cell, weights):
    b = BENIGN[cell]
    g = CMc[(CMc.cell == cell) & CMc.model.isin(CORE)]
    rows_ = []
    for (seed, model), p in g.groupby(["seed", "model"]):
        miss = p[(p.true_label != b) & (p.pred_label == b)]
        rows_.append({"seed": seed, "model": model,
                      "miss": float((miss["count"] * miss.true_label.map(lambda x: weights.get(x, 1.0))).sum()),
                      "fa": float(p[(p.true_label == b) & (p.pred_label != b)]["count"].sum())})
    return pd.DataFrame(rows_)


INCUMBENTS = ["rf_pruned_full", "rf_strong_full", "xgb_full"]
cost_rows = []
for cell in CELLS:
    vs = cost_vectors(cell)
    for vname, w in vs.items():
        A = alarms(cell, w)
        for inc in INCUMBENTS:
            base = A[A.model == inc].set_index("seed")
            for model in [m for m in CORE if m != inc]:
                other = A[A.model == model].set_index("seed")
                rhos, status = [], []
                for seed in sorted(set(base.index) & set(other.index)):
                    dm = float(base.loc[seed, "miss"] - other.loc[seed, "miss"])
                    dfa = float(other.loc[seed, "fa"] - base.loc[seed, "fa"])
                    if abs(dm) < 1e-12:
                        status.append("none"); continue
                    rho = dfa / dm
                    if rho > 0:
                        rhos.append(rho); status.append("positive")
                    elif dm > 0 and dfa <= 0:
                        status.append("dominates")   # fewer misses and no more false alarms
                    else:
                        status.append("dominated")   # more misses and more false alarms, or negative
                pos = np.asarray(rhos)
                cost_rows.append({"cell": cell, "vector": vname, "incumbent": inc, "model": model,
                                  "positive_seeds": len(pos), "dominates_seeds": status.count("dominates"),
                                  "dominated_seeds": status.count("dominated"),
                                  "rho_median": float(np.median(pos)) if len(pos) else np.nan,
                                  "rho_q1": float(np.quantile(pos, .25)) if len(pos) else np.nan,
                                  "rho_q3": float(np.quantile(pos, .75)) if len(pos) else np.nan})
cost_df = pd.DataFrame(cost_rows)
cost_df.to_csv(OUT / "cost_crossovers_all_vectors.csv", index=False)


def cost_cell(r):
    if r.positive_seeds == 0:
        if r.dominates_seeds == 10:
            return "dom."
        return "--"
    s = f"{r.rho_median:.3f} ({int(r.positive_seeds)})"
    return s


VEC_ORDER = ["uniform", "severity", "rare-flat", "steep", "inverse-prevalence"]
VEC_LAB = {"uniform": "Uniform", "severity": "Severity", "rare-flat": "Rare-flat", "steep": "Steep", "inverse-prevalence": "Inv.-prev."}
SEL = [("rf_pruned_full", "xgb_full"), ("rf_pruned_full", "tabpfn_2.5"),
       ("rf_strong_full", "xgb_full"), ("rf_strong_full", "xgb_balanced"), ("rf_strong_full", "tabpfn_2.5"),
       ("xgb_full", "xgb_balanced"), ("xgb_full", "tabpfn_2.5"), ("xgb_full", "tabicl")]
with open(GEN / "table_cost_sensitivity.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lll" + "r" * len(VEC_ORDER) + "@{}}\n\\toprule\n")
    f.write("Cell & Incumbent & Model & " + " & ".join(VEC_LAB[v] for v in VEC_ORDER) + " \\\\\n\\midrule\n")
    for cell in CELLS:
        first = True
        for inc, model in SEL:
            vals = []
            for v in VEC_ORDER:
                r = cost_df[(cost_df.cell == cell) & (cost_df.vector == v) & (cost_df.incumbent == inc) & (cost_df.model == model)].iloc[0]
                vals.append(cost_cell(r))
            f.write(f"{CELL_LABEL[cell] if first else ''} & {ABBR[inc]} & {ABBR[model]} & " + " & ".join(vals) + " \\\\\n")
            first = False
        if cell != CELLS[-1]:
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ========================================================================
# R4-4  Per-class precision / recall / support for the rare set
# ========================================================================
PCall = PC[(PC.test_set == "capped")]
pc_mean = (PCall.groupby(["cell", "model", "class"], as_index=False)
           .agg(precision=("precision", "mean"), recall=("recall", "mean"), support=("support", "mean")))
pc_mean.to_csv(OUT / "per_class_mean_all.csv", index=False)


def write_rare_table(path, cell, models, caption_models):
    R = RARE[cell]
    g = pc_mean[(pc_mean.cell == cell) & pc_mean["class"].isin(R)]
    sup = g.groupby("class").support.first().sort_values()
    with open(path, "w") as f:
        f.write("\\begin{tabular}{@{}lr" + "rr" * len(models) + "@{}}\n\\toprule\n")
        f.write(" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{caption_models[m]}}}" for m in models) + " \\\\\n")
        f.write("".join(f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(models))) + "\n")
        f.write("Class & Support & " + " & ".join("P & R" for _ in models) + " \\\\\n\\midrule\n")
        for cl, s in sup.items():
            cells = []
            for m in models:
                r = g[(g.model == m) & (g["class"] == cl)]
                if r.empty:
                    cells.append("-- & --")
                else:
                    cells.append(f"{float(r.precision.iloc[0]):.3f} & {float(r.recall.iloc[0]):.3f}")
            f.write(f"{esc(cl)} & {int(round(s)):,} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n")
        cells = []
        for m in models:
            r = g[g.model == m]
            cells.append(f"{r.precision.mean():.3f} & {r.recall.mean():.3f}")
        f.write("Unweighted mean & & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


write_rare_table(GEN / "table_rare_perclass_cic_v1.tex", "cic_ids2017_V1",
                 ["xgb_full", "xgb_balanced", "tabpfn_2.5", "tabicl", "tabstar", "rf_pruned_full"],
                 {"xgb_full": "XGB/L", "xgb_balanced": "XGB/B", "tabpfn_2.5": "TabPFN", "tabicl": "TabICL",
                  "tabstar": "TabSTAR$^{\\dagger}$", "rf_pruned_full": "RF-P/L"})
write_rare_table(GEN / "table_rare_perclass_unsw_v2.tex", "cic_unsw_V2",
                 ["xgb_full", "xgb_balanced", "rf_strong_balanced", "tabpfn_2.5", "tabicl", "rf_pruned_full"],
                 {"xgb_full": "XGB/L", "xgb_balanced": "XGB/B", "rf_strong_balanced": "RF-S/B",
                  "tabpfn_2.5": "TabPFN", "tabicl": "TabICL", "rf_pruned_full": "RF-P/L"})

# Resolution of the headline measure: attainable recall grid per rare class (support retained whole)
res_rows = []
for cell in CELLS:
    for cl in RARE[cell]:
        s = float(pc_mean[(pc_mean.cell == cell) & (pc_mean["class"] == cl)].support.iloc[0])
        res_rows.append({"cell": cell, "class": cl, "support": s, "retained_whole": s <= 1500,
                         "recall_step": 1.0 / s if s <= 1500 else np.nan})
pd.DataFrame(res_rows).to_csv(OUT / "rare_class_resolution.csv", index=False)

# ========================================================================
# R1-8  ManyClass: intervals on the available seeds
# ========================================================================
many = pd.read_csv(SENS / "exp4_manyclass.csv")
mc_rows = []
for (a, rdd), g in many.groupby(["alphabet_size", "n_estimators_redundancy"]):
    x = g.rare_recall.to_numpy()
    lo, hi = bootstrap_ci(x) if len(x) >= 10 else t_ci(x)
    mc_rows.append({"alphabet": a, "redundancy": rdd, "n": len(x), "accuracy": g.accuracy.mean(),
                    "macro_f1": g.macro_f1.mean(), "rare_recall": x.mean(), "sd": x.std(ddof=1),
                    "ci_lo": lo, "ci_hi": hi, "min": x.min(), "max": x.max()})
mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(OUT / "manyclass_intervals.csv", index=False)
# Paired differences between settings (all share the same seeds)
mp = many.pivot(index="seed", columns=["alphabet_size", "n_estimators_redundancy"], values="rare_recall")
ref = mp[(5, 1)]
mcd = []
for col in mp.columns:
    if col == (5, 1):
        continue
    d = (mp[col] - ref).to_numpy()
    lo, hi = bootstrap_ci(d) if len(d) >= 10 else t_ci(d)
    mcd.append({"setting": f"a{col[0]}_r{col[1]}", "mean_delta_vs_a5_r1": d.mean(), "ci_lo": lo, "ci_hi": hi,
                "p_signflip": signflip(d), "n": len(d)})
pd.DataFrame(mcd).to_csv(OUT / "manyclass_paired_vs_a5r1.csv", index=False)
n_seeds = int(many.seed.nunique())
with open(GEN / "table_manyclass.tex", "w") as f:
    f.write("\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n")
    f.write("Alphabet & Redundancy & $n$ & Accuracy & Macro-$F_1$ & Rare-class recall & 95\\% CI \\\\\n\\midrule\n")
    for _, r in mc_df.iterrows():
        f.write(f"{int(r.alphabet)} & {int(r.redundancy)} & {int(r.n)} & {r.accuracy:.4f} & {r.macro_f1:.4f} & "
                f"{r.rare_recall:.4f} & [{r.ci_lo:.4f}, {r.ci_hi:.4f}] \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
grid = mc_df.rare_recall
print(f"ManyClass: n_seeds={n_seeds}; spread across settings={grid.max()-grid.min():.4f}")

# ========================================================================
# R4 minor  Axis-2: per-1,000-row cost, component sums, audit-row explanation
# ========================================================================
res_old = pd.read_csv(AX2 / "results_original_six_model_campaign.csv")
res_nem = pd.read_csv(AX2 / "results_nemotron_corrected.csv")
res_panel = pd.concat([res_old[~res_old.model.str.startswith("Nemotron", na=False)],
                       res_nem[res_nem.model.str.startswith("Nemotron", na=False)]], ignore_index=True)
llm = res_panel[res_panel.kind == "llm"].copy()
raw_old = [json.loads(x) for x in (AX2 / "raw_generations_original_six_model_campaign.jsonl").read_text().splitlines() if x.strip()]
n_eval = {}
for r in raw_old:
    n_eval[(r["model"], int(r["seed"]))] = len(r["y_true"])
llm["n_eval"] = [n_eval.get((m, int(s)), 246) for m, s in zip(llm.model, llm.seed)]
cost = (llm.groupby("model", as_index=False)
        .agg(usd=("usd", "sum"), rows=("n_eval", "sum"), latency=("latency_median_s", "median"), wall=("wall_min", "sum")))
cost["usd_per_1k_rows"] = cost.usd / cost.rows * 1000
cost["usd_per_seed"] = cost.usd / 10
total = float(cost.usd.sum())
cost.to_csv(OUT / "axis2_cost_per_1k.csv", index=False)
print("Axis-2 total USD (unrounded):", total, "sum of rounded-to-cents components:", round(sum(round(v, 2) for v in cost.usd), 2))

# Closed-label summary (recomputed exactly as in the main script) with the added cost column
ax2 = pd.read_csv(ROOT / "results" / "derived" / "axis2_closed_label_summary.csv")
MODEL_NAMES = {"RandomForest": "Random Forest", "Kimi-K3": "Kimi-K3", "GLM-5.2": "GLM-5.2", "MiniMax-M3": "MiniMax-M3",
               "Qwen3.5-397B-A17B": "Qwen3.5", "DeepSeek-V4-Pro": "DeepSeek-V4-Pro", "Nemotron-3-Ultra-550b-a55b": "Nemotron-3 Ultra"}
order = ["RandomForest", "Kimi-K3", "GLM-5.2", "MiniMax-M3", "Qwen3.5-397B-A17B", "DeepSeek-V4-Pro", "Nemotron-3-Ultra-550b-a55b"]
with open(GEN / "table_axis2_summary.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n")
    f.write("Model & Accuracy & Macro-$F_1$ & Parse & $\\Delta F_1$ & $p_{\\mathrm{Bonf}}$ & Cost (USD) & USD / 1{,}000 rows & Latency (s) \\\\\n\\midrule\n")
    for m in order:
        r = ax2[ax2.model == m].iloc[0]
        c = cost[cost.model == m]
        if m == "RandomForest":
            f.write(f"{MODEL_NAMES[m]} & {r.accuracy:.3f} & {r.macro_f1_closed:.3f} & 1.000 & -- & -- & 0.00 & 0.00 & -- \\\\\n")
        else:
            f.write(f"{MODEL_NAMES[m]} & {r.accuracy:.3f} & {r.macro_f1_closed:.3f} & {r.parse_rate:.3f} & "
                    f"{r.mean_delta:+.3f} & {r.p_bonferroni_6:.4f} & {float(c.usd.iloc[0]):.2f} & "
                    f"{float(c.usd_per_1k_rows.iloc[0]):.2f} & {float(c.latency.iloc[0]):.2f} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"Total & & & & & & {total:.2f} & & \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# Axis-2 per-class precision/recall mean table (closed-label) for the appendix
ax2pc = pd.read_csv(ROOT / "results" / "derived" / "axis2_closed_label_per_class_mean.csv")
with open(GEN / "table_axis2_perclass.tex", "w") as f:
    models = [m for m in order if m != "RandomForest"]
    classes = sorted(ax2pc["class"].unique())
    f.write("\\begin{tabular}{@{}lr" + "rr" * len(models) + "@{}}\n\\toprule\n")
    f.write(" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{MODEL_NAMES[m]}}}" for m in models) + " \\\\\n")
    f.write("".join(f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(models))) + "\n")
    f.write("Class & Rows/seed & " + " & ".join("P & R" for _ in models) + " \\\\\n\\midrule\n")
    for cl in classes:
        s = ax2pc[(ax2pc["class"] == cl)].support.iloc[0]
        cells = []
        for m in models:
            r = ax2pc[(ax2pc.model == m) & (ax2pc["class"] == cl)]
            cells.append(f"{float(r.precision.iloc[0]):.2f} & {float(r.recall.iloc[0]):.2f}" if not r.empty else "-- & --")
        f.write(f"{esc(cl)} & {int(round(s))} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# Audit-row explanation numbers
mi = pd.read_csv(SENS / "exp2_llm_leakage.csv")
print("Selector audit rows per seed: see notebook protocol (50/class, 500 dev) ->", "387 = 7x50 + 31 + 6")

# ========================================================================
# Summary of headline numbers for the response letter
# ========================================================================
summary = {
    "threshold_same_three_all": bool(thr_df.same_three.all()),
    "threshold_rho_range": [float(thr_df.spearman.min()), float(thr_df.spearman.max())],
    "threshold_direction_counts": {str(t): thr_df[thr_df.threshold == t].iloc[0][["fm_all10", "classical_all10", "mixed"]].to_dict() for t in THRS},
    "n_rare_by_threshold": {str(t): thr_df[thr_df.threshold == t].set_index("cell").n_rare.to_dict() for t in THRS},
    "core_panel_rank_correlations": rk_df.to_dict("records"),
    "cap_operational_summary": opv_sum.to_dict("records"),
    "cap_crossover_summary": cx_sum.to_dict("records"),
    "manyclass": {"n_seeds": n_seeds, "spread": float(grid.max() - grid.min()), "rows": mc_df.to_dict("records"), "paired": mcd},
    "axis2_cost": cost.to_dict("records"), "axis2_total_usd": total,
}
(OUT / "rev4_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("done ->", OUT)

# ========================================================================
# R1-2  Factorial replication on CIC-UNSW-NB15 V1 (notebook Experiment 5)
# ========================================================================
p5 = SENS / "exp5_size_by_balance_unsw.csv"
if p5.is_file():
    e5 = pd.read_csv(p5)
    arms5 = e5.groupby(["size", "balance"]).agg(
        n_train=("n_train", "mean"), n_unique=("n_unique_rows", "mean"),
        oversample=("oversample_factor", "mean"), accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), rare_recall=("rare_recall", "mean"),
        rr_sd=("rare_recall", "std"))
    pv5 = e5.pivot(index="seed", columns="arm", values="rare_recall")
    plan5 = [("Balance, small budget", "small_balanced", "small_natural"),
             ("Balance, large budget", "large_balanced", "large_natural"),
             ("Size, natural distribution", "large_natural", "small_natural"),
             ("Size, balanced distribution", "large_balanced", "small_balanced")]
    c5 = []
    for lab, a, b in plan5:
        d = (pv5[a] - pv5[b]).to_numpy()
        lo, hi = bootstrap_ci(d)
        c5.append({"contrast": lab, "delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
                   "wins": int((d > 0).sum()), "n": len(d), "p_signflip": signflip(d)})
    ph = np.asarray([r["p_signflip"] for r in c5], dtype=float)
    order = np.argsort(ph); run = 0.0; adj = np.empty_like(ph)
    for i, idx in enumerate(order):
        run = max(run, (len(ph) - i) * ph[idx]); adj[idx] = min(1.0, run)
    for r, q in zip(c5, adj):
        r["p_holm"] = float(q)
    c5df = pd.DataFrame(c5)
    arms5.reset_index().to_csv(OUT / "exp5_unsw_factorial_arms.csv", index=False)
    c5df.to_csv(OUT / "exp5_unsw_factorial_contrasts.csv", index=False)
    with open(GEN / "table_size_balance_unsw.tex", "w") as f:
        f.write("\\begin{minipage}{\\linewidth}\n\\centering\n")
        f.write("\\textbf{Panel A: arm means.}\\\\[3pt]\n")
        f.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n")
        f.write("Budget & Distribution & Rows & Unique rows & Macro-$F_1$ & Rare-class recall \\\\\n\\midrule\n")
        for sz, bl in [("small", "natural"), ("small", "balanced"),
                       ("large", "natural"), ("large", "balanced")]:
            r = arms5.loc[(sz, bl)]
            f.write(f"{sz.capitalize()} & {bl.capitalize()} & {int(round(r.n_train)):,} & "
                    f"{int(round(r.n_unique)):,} & {r.macro_f1:.3f} & {r.rare_recall:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\par\\medskip\n")
        f.write("\\textbf{Panel B: planned paired contrasts.}\\\\[3pt]\n")
        f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
        f.write("Contrast & $\\Delta$ rare-class recall & 95\\% CI & Wins & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
        for _, r in c5df.iterrows():
            f.write(f"{r.contrast} & {r.delta:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                    f"{r.wins}/{r.n} & {r.p_holm:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{minipage}\n")
    print("exp5 factorial ->", GEN / "table_size_balance_unsw.tex")
else:
    print("exp5 outputs not found; skipped")

# ========================================================================
# R1-6  Data-rich tree references on the Axis-2 evaluation rows (Experiment 6)
# ========================================================================
p6 = SENS / "exp6_axis2_datarich.csv"
if p6.is_file():
    e6 = pd.read_csv(p6)
    rf_anchor = (pd.read_csv(AX2 / "results_original_six_model_campaign.csv")
                 .query("model=='RandomForest'").drop_duplicates("seed").set_index("seed"))
    per_seed = pd.read_csv(ROOT / "results" / "derived" / "axis2_closed_label_per_seed.csv")
    best_name = per_seed.groupby("model").macro_f1_closed.mean().idxmax()
    best = per_seed[per_seed.model == best_name].set_index("seed")
    rep = e6.drop_duplicates("seed").set_index("seed").anchor_accuracy_refit
    assert float((rep - rf_anchor.loc[rep.index, "accuracy"]).abs().max()) < 1e-12, \
        "anchor reproduction failed; Experiment 6 rows are not on the campaign split"
    d6 = []
    for ref in ["development-rich", "large natural"]:
        for lr in ["strong RF", "XGBoost"]:
            g = e6[(e6.reference == ref) & (e6.learner == lr)].set_index("seed").sort_index()
            d = g.macro_f1.to_numpy() - rf_anchor.loc[g.index, "macro_f1"].to_numpy()
            db = g.macro_f1.to_numpy() - best.loc[g.index, "macro_f1_closed"].to_numpy()
            lo, hi = bootstrap_ci(d)
            d6.append({"reference": ref, "learner": lr, "n_fit": g.n_fit.mean(),
                       "accuracy": g.accuracy.mean(), "macro_f1": g.macro_f1.mean(),
                       "delta_vs_anchor": d.mean(), "ci_lo": lo, "ci_hi": hi,
                       "wins": int((d > 0).sum()), "p_signflip": signflip(d),
                       "best_llm": best_name, "delta_vs_best_llm": db.mean(),
                       "wins_vs_best_llm": int((db > 0).sum()), "p_vs_best_llm": signflip(db)})
    d6df = pd.DataFrame(d6)
    d6df.to_csv(OUT / "exp6_axis2_datarich_summary.csv", index=False)
    LAB6 = {("development-rich", "strong RF"): "Strong RF, in-context $\\cup$ development",
            ("development-rich", "XGBoost"): "XGBoost, in-context $\\cup$ development",
            ("large natural", "strong RF"): "Strong RF, in-context $\\cup$ 400k natural",
            ("large natural", "XGBoost"): "XGBoost, in-context $\\cup$ 400k natural"}
    with open(GEN / "table_axis2_datarich.tex", "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n")
        f.write("Reference & Fitting rows & Accuracy & Macro-$F_1$ & "
                "$\\Delta F_1$ vs.\\ anchor & Wins & $p$ \\\\\n\\midrule\n")
        f.write(f"Random Forest anchor (in-context rows) & 45 & {rf_anchor.accuracy.mean():.3f} & "
                f"{rf_anchor.macro_f1.mean():.3f} & -- & -- & -- \\\\\n")
        for _, r in d6df.iterrows():
            f.write(f"{LAB6[(r.reference, r.learner)]} & {int(round(r.n_fit)):,} & "
                    f"{r.accuracy:.3f} & {r.macro_f1:.3f} & {r.delta_vs_anchor:+.3f} & "
                    f"{r.wins}/10 & {r.p_signflip:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("exp6 data-rich ->", GEN / "table_axis2_datarich.tex")
else:
    print("exp6 outputs not found; skipped")
