#!/usr/bin/env python3
"""Round-5 analysis-only extensions computed from the persisted ai_for_ids record.

Everything here is derived from the released result files (weighted confusion
matrices, per-class records, Axis-2 per-seed records, cost ledgers). No model is
re-executed. Outputs go to <out>/ as CSV plus a JSON summary used by the
manuscript text.

Usage: python round5_record_analysis.py --results <ai_for_ids/results> --out <dir>
"""
import argparse, itertools, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

CELL_LABEL = {"cic_ids2017_V1": "CIC-IDS2017 V1", "cic_ids2017_V3": "CIC-IDS2017 V3",
              "cic_unsw_V1": "CIC-UNSW-NB15 V1", "cic_unsw_V2": "CIC-UNSW-NB15 V2"}
MODEL_LABEL = {"rf_pruned_full": "RF-P/L", "rf_strong_full": "RF-S/L", "xgb_full": "XGB/L",
               "rf_pruned_balanced": "RF-P/B", "rf_strong_balanced": "RF-S/B", "xgb_balanced": "XGB/B",
               "tabpfn_2.5": "TabPFN 2.5", "tabicl": "TabICL"}
STRICT = list(MODEL_LABEL)
CLASSICAL = STRICT[:6]
PRIOR = ["tabpfn_2.5", "tabicl"]
RARE_SHARE = 0.005
FULL_TEST_ROWS = {"cic_ids2017_V1": 756_709, "cic_ids2017_V3": 756_709,
                  "cic_unsw_V1": 1_062_073, "cic_unsw_V2": 1_062_073}


# ----------------------------------------------------------------------------- stats
def sign_flip_p(d):
    d = np.asarray(d, float); o = abs(d.mean())
    return float(np.mean([abs(np.mean(d * np.array(x))) >= o - 1e-15
                          for x in itertools.product((-1, 1), repeat=len(d))]))

def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    b = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (len(p) - i) * p[idx]); out[idx] = min(1.0, run)
    return out

def paired_summary(d):
    """Effect size with 95% CI, standardised effect d_z, and minimum detectable effects.

    MDE80 is the smallest mean paired difference that a two-sided paired t test at
    alpha=0.05 would detect with 80% power given the observed SD of the differences
    and n pairs; MDE_ci is the half-width of the 95% t interval (the smallest
    difference whose interval excludes zero)."""
    d = np.asarray(d, float); n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1); t_pow = stats.t.ppf(0.80, n - 1)
    lo, hi = boot_ci(d)
    return dict(n=n, mean_delta=d.mean(), ci_lo=lo, ci_hi=hi, sd_diff=sd,
                d_z=(d.mean() / sd if sd > 0 else np.inf), wins=int((d > 0).sum()),
                p_signflip=sign_flip_p(d), mde80=(t_crit + t_pow) * se, mde_ci=t_crit * se)


# ----------------------------------------------------------------------------- metrics from cm
def cm_metrics(cm: pd.DataFrame):
    """Metrics from a (weighted) confusion matrix with true labels in rows."""
    labs = list(cm.index); M = cm.to_numpy(float)
    tp = np.diag(M); row = M.sum(1); col = M.sum(0); N = M.sum()
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    acc = tp.sum() / N
    # multi-class MCC (Gorodkin 2004)
    cov_tp = N * tp.sum() - (row * col).sum()
    den = math.sqrt(max(N**2 - (col**2).sum(), 0) * max(N**2 - (row**2).sum(), 0))
    mcc = cov_tp / den if den > 0 else 0.0
    pe = (row * col).sum() / N**2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    present = row > 0
    bal_acc = rec[present].mean()
    gmean = float(np.exp(np.log(np.clip(rec[present], 1e-12, 1)).mean()))
    ben = [i for i, c in enumerate(labs) if str(c).lower() == "benign"]
    share = row / N
    rare_idx = [i for i in range(len(labs)) if i not in ben and share[i] < RARE_SHARE and row[i] > 0]
    rare_recall = rec[rare_idx].mean() if rare_idx else np.nan
    rare_f1 = f1[rare_idx].mean() if rare_idx else np.nan
    rare_prec = prec[rare_idx].mean() if rare_idx else np.nan
    # binary attack view
    att = [i for i in range(len(labs)) if i not in ben]
    attack_rows = row[att].sum()
    missed_attacks = M[np.ix_(att, ben)].sum() if ben else 0.0   # attack rows predicted benign
    benign_rows = row[ben].sum() if ben else 0.0
    false_alarms = M[np.ix_(ben, att)].sum() if ben else 0.0
    rare_rows = row[rare_idx].sum()
    rare_missed = rare_rows - tp[rare_idx].sum()
    return dict(accuracy=acc, balanced_accuracy=bal_acc, mcc=mcc, kappa=kappa,
                macro_f1=f1[present].mean(), gmean=gmean, rare_recall=rare_recall,
                rare_precision=rare_prec, rare_f1=rare_f1, n_rare=len(rare_idx),
                attack_recall=(1 - missed_attacks / attack_rows) if attack_rows else np.nan,
                false_alarm_rate=(false_alarms / benign_rows) if benign_rows else np.nan,
                missed_attacks_per_100k=1e5 * missed_attacks / N,
                missed_rare_per_100k=1e5 * rare_missed / N,
                false_alarms_per_100k=1e5 * false_alarms / N,
                benign_share=benign_rows / N, n_rows=N)


def load_cms(conf):
    out = {}
    for (cell, seed, model, ts), g in conf.groupby(["cell", "seed", "model", "test_set"]):
        cm = g.pivot_table(index="true_label", columns="pred_label", values="count",
                           aggfunc="sum", fill_value=0.0)
        labs = sorted(set(cm.index) | set(cm.columns))
        out[(cell, seed, model, ts)] = cm.reindex(index=labs, columns=labs, fill_value=0.0)
    return out


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); R = Path(a.results); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    summary = {}

    conf = pd.read_csv(R / "axis1/confusion.csv")
    conf = conf[conf.model.isin(STRICT)]
    cms = load_cms(conf)

    # 1. extended metrics per seed (capped evaluation) --------------------------------
    rows = []
    for (cell, seed, model, ts), cm in cms.items():
        if ts != "capped": continue
        m = cm_metrics(cm); m.update(cell=cell, seed=seed, model=model); rows.append(m)
    per_seed = pd.DataFrame(rows).sort_values(["cell", "model", "seed"])
    per_seed.to_csv(O / "metrics_extended_per_seed.csv", index=False)

    metrics = ["accuracy", "balanced_accuracy", "mcc", "kappa", "macro_f1", "gmean", "rare_recall",
               "rare_precision", "rare_f1", "attack_recall", "false_alarm_rate",
               "missed_attacks_per_100k", "missed_rare_per_100k", "false_alarms_per_100k"]
    summ = per_seed.groupby(["cell", "model"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    for m in ["accuracy", "balanced_accuracy", "mcc", "macro_f1", "gmean", "rare_recall"]:
        summ[f"{m}_rank"] = summ.groupby("cell")[f"{m}_mean"].rank(ascending=False, method="min").astype(int)
    summ["model_label"] = summ.model.map(MODEL_LABEL); summ["cell_label"] = summ.cell.map(CELL_LABEL)
    summ.to_csv(O / "metrics_extended_summary.csv", index=False)

    # 2. constant-benign baseline and inversion magnitude -----------------------------
    base_rows = []
    for cell, g in summ.groupby("cell"):
        bshare = per_seed[per_seed.cell == cell].benign_share.mean()
        acc_lead = g.loc[g.accuracy_mean.idxmax()]; rec_lead = g.loc[g.rare_recall_mean.idxmax()]
        worst_acc = g.accuracy_mean.min()
        base_rows.append(dict(
            cell=cell, cell_label=CELL_LABEL[cell], benign_share=bshare, constant_benign_accuracy=bshare,
            panel_accuracy_min=worst_acc, panel_accuracy_max=g.accuracy_mean.max(),
            panel_headroom_min=(worst_acc - bshare) / (1 - bshare),
            panel_headroom_max=(g.accuracy_mean.max() - bshare) / (1 - bshare),
            accuracy_leader=acc_lead.model, accuracy_leader_rare_recall=acc_lead.rare_recall_mean,
            recall_leader=rec_lead.model, recall_leader_accuracy=rec_lead.accuracy_mean,
            accuracy_cost_of_recall_leader=acc_lead.accuracy_mean - rec_lead.accuracy_mean,
            rare_recall_gain_of_recall_leader=rec_lead.rare_recall_mean - acc_lead.rare_recall_mean,
            missed_rare_per_100k_accuracy_leader=acc_lead.missed_rare_per_100k_mean,
            missed_rare_per_100k_recall_leader=rec_lead.missed_rare_per_100k_mean,
            false_alarms_per_100k_accuracy_leader=acc_lead.false_alarms_per_100k_mean,
            false_alarms_per_100k_recall_leader=rec_lead.false_alarms_per_100k_mean,
            balanced_accuracy_leader=g.loc[g.balanced_accuracy_mean.idxmax()].model,
            mcc_leader=g.loc[g.mcc_mean.idxmax()].model,
            macro_f1_leader=g.loc[g.macro_f1_mean.idxmax()].model))
    baseline = pd.DataFrame(base_rows); baseline.to_csv(O / "baseline_and_inversion_magnitude.csv", index=False)

    # 3. rank correlations between summaries, per cell --------------------------------
    corr_rows = []
    for cell, g in summ.groupby("cell"):
        acc_top3 = set(g.nlargest(3, "accuracy_mean").model)
        rr_bottom3 = set(g.nsmallest(3, "rare_recall_mean").model)
        for m in ["balanced_accuracy", "mcc", "kappa", "macro_f1", "gmean", "rare_f1", "attack_recall"]:
            corr_rows.append(dict(cell=cell, cell_label=CELL_LABEL[cell], metric=m,
                                  rho_with_accuracy=spearman(g[f"{m}_mean"], g.accuracy_mean),
                                  rho_with_rare_recall=spearman(g[f"{m}_mean"], g.rare_recall_mean),
                                  leader=g.loc[g[f"{m}_mean"].idxmax()].model,
                                  leader_is_accuracy_leader=g.loc[g[f"{m}_mean"].idxmax()].model == g.loc[g.accuracy_mean.idxmax()].model,
                                  same_three=acc_top3 == rr_bottom3))
        corr_rows.append(dict(cell=cell, cell_label=CELL_LABEL[cell], metric="rare_recall",
                              rho_with_accuracy=spearman(g.rare_recall_mean, g.accuracy_mean),
                              rho_with_rare_recall=1.0, leader=g.loc[g.rare_recall_mean.idxmax()].model,
                              leader_is_accuracy_leader=False, same_three=acc_top3 == rr_bottom3))
    corr = pd.DataFrame(corr_rows); corr.to_csv(O / "metric_rank_correlations.csv", index=False)

    # 3b. permutation null for the pooled inversion: 4 cells x 8 configurations ---------
    rng = np.random.default_rng(2026); obs = []
    for cell, g in summ.groupby("cell"): obs.append(spearman(g.accuracy_mean, g.rare_recall_mean))
    obs_mean = float(np.mean(obs)); null = []
    for _ in range(20000):
        vals = []
        for cell, g in summ.groupby("cell"):
            vals.append(spearman(g.accuracy_mean, rng.permutation(g.rare_recall_mean.to_numpy())))
        null.append(np.mean(vals))
    null = np.asarray(null)
    summary["inversion_permutation"] = dict(observed_mean_rho=obs_mean, per_cell=obs,
                                            p_two_sided=float(np.mean(np.abs(null) >= abs(obs_mean))),
                                            p_one_sided=float(np.mean(null <= obs_mean)))

    # 4. effect sizes and MDE: all Axis-1 prior-fitted vs classical contrasts -----------
    eff = []
    for cell in CELL_LABEL:
        for pf in PRIOR:
            for cl in CLASSICAL:
                for metric in ["rare_recall", "balanced_accuracy", "mcc", "macro_f1", "accuracy"]:
                    a_ = per_seed[(per_seed.cell == cell) & (per_seed.model == pf)].set_index("seed")[metric]
                    b_ = per_seed[(per_seed.cell == cell) & (per_seed.model == cl)].set_index("seed")[metric]
                    d = (a_ - b_.reindex(a_.index)).to_numpy()
                    s = paired_summary(d); s.update(cell=cell, model=pf, comparator=cl, metric=metric,
                                                    family="axis1_prior_vs_classical"); eff.append(s)
    eff = pd.DataFrame(eff)
    for (cell, pf, metric), g in eff.groupby(["cell", "model", "metric"]):
        eff.loc[g.index, "p_holm_6"] = holm(g.p_signflip.to_numpy())
    eff.to_csv(O / "effect_sizes_axis1.csv", index=False)

    # 4b. factorial contrasts: add d_z and MDE from the raw arms ------------------------
    fac = []
    for tag, f in [("CIC-IDS2017 V3", R / "sensitivity/exp1_size_by_balance.csv"),
                   ("CIC-UNSW-NB15 V1", R / "round4/raw/exp5_size_by_balance_unsw.csv")]:
        df = pd.read_csv(f)
        armcol = "arm" if "arm" in df.columns else [c for c in df.columns if "arm" in c][0]
        rcol = [c for c in df.columns if "rare" in c and "recall" in c][0]
        piv = df.pivot_table(index="seed", columns=armcol, values=rcol)
        for name, l, r in [("balance at small budget", "small_balanced", "small_natural"),
                           ("balance at large budget", "large_balanced", "large_natural"),
                           ("size under natural sampling", "large_natural", "small_natural"),
                           ("size under balanced sampling", "large_balanced", "small_balanced")]:
            if l in piv.columns and r in piv.columns:
                s = paired_summary((piv[l] - piv[r]).to_numpy()); s.update(dataset=tag, contrast=name,
                                                                          metric="rare_recall"); fac.append(s)
    pd.DataFrame(fac).to_csv(O / "effect_sizes_factorial.csv", index=False)

    # 4c. Axis-2 contrasts with d_z and MDE ------------------------------------------
    ax2 = []
    for tag, fl, fa in [("grouped CIC-IDS2017", R / "derived/axis2_closed_label_per_seed.csv", R / "axis2/results_original_six_model_campaign.csv"),
                        ("CIC-UNSW-NB15 V1", R / "round4/derived/axis2_second_variant_per_seed.csv", R / "round4/raw/axis2_second_variant/results.csv")]:
        llm = pd.read_csv(fl).drop_duplicates(["seed", "model"], keep="last")
        piv = llm.pivot_table(index="seed", columns="model", values="macro_f1_closed")
        anc = pd.read_csv(fa).drop_duplicates(["seed", "model"], keep="last")
        anchor = anc[anc.model == "RandomForest"].set_index("seed").macro_f1.reindex(piv.index)
        for m in [c for c in piv.columns if c != "RandomForest"]:
            d = (piv[m] - anchor).dropna().to_numpy()
            s = paired_summary(d); s.update(source=tag, model=m, metric="macro_f1_closed"); ax2.append(s)
    ax2 = pd.DataFrame(ax2)
    for src, g in ax2.groupby("source"):
        ax2.loc[g.index, "p_bonferroni_6"] = np.minimum(1.0, 6 * g.p_signflip)
    ax2.to_csv(O / "effect_sizes_axis2.csv", index=False)

    # 4d. MDE summary ------------------------------------------------------------------
    rr = eff[eff.metric == "rare_recall"]
    summary["mde"] = dict(
        axis1_rare_recall_sd_diff_median=float(rr.sd_diff.median()),
        axis1_rare_recall_mde80_median=float(rr.mde80.median()),
        axis1_rare_recall_mde80_min=float(rr.mde80.min()), axis1_rare_recall_mde80_max=float(rr.mde80.max()),
        axis1_n_contrasts_with_abs_delta_below_mde80=int((rr.mean_delta.abs() < rr.mde80).sum()),
        axis2_mde80_cic=float(ax2[ax2.source == "grouped CIC-IDS2017"].mde80.median()),
        axis2_mde80_unsw=float(ax2[ax2.source == "CIC-UNSW-NB15 V1"].mde80.median()),
        axis2_mde80_rows=ax2[["source", "model", "mean_delta", "ci_lo", "ci_hi", "sd_diff", "d_z", "mde80"]].to_dict("records"),
        signflip_floor=2 / 2**10)

    # 5. LLM applicability: throughput and cost extrapolation ---------------------------
    appl = []
    for tag, f, n_rows_seed, cell in [("grouped CIC-IDS2017", R / "axis2/results_original_six_model_campaign.csv", 246, "cic_ids2017_V3"),
                                      ("CIC-UNSW-NB15 V1", R / "round4/raw/axis2_second_variant/results.csv", 300, "cic_unsw_V1")]:
        df = pd.read_csv(f).drop_duplicates(["seed", "model"], keep="last")
        if cell == "cic_ids2017_V3":   # the corrected Nemotron rerun supersedes the original endpoint record (as in the Axis-2 summary table)
            corr = pd.read_csv(R / "axis2/results_nemotron_corrected.csv")
            df = pd.concat([df[~df.model.str.contains("Nemotron", case=False, na=False)],
                            corr[corr.model.str.contains("Nemotron", case=False, na=False)]], ignore_index=True)
        df = df[df.kind == "llm"]
        g = df.groupby("model").agg(usd=("usd", "sum"), lat=("latency_median_s", "median"),
                                    wall_min=("wall_min", "sum"), seeds=("seed", "nunique"))
        g["rows_scored"] = g.seeds * n_rows_seed
        g["usd_per_1000"] = 1000 * g.usd / g.rows_scored
        g["rows_per_hour_sequential"] = 3600 / g.lat
        g["seconds_per_row_wall"] = 60 * g.wall_min / g.rows_scored
        n_full = FULL_TEST_ROWS[cell]
        g["usd_full_partition"] = g.usd_per_1000 * n_full / 1000
        g["hours_full_partition_at_campaign_rate"] = g.seconds_per_row_wall * n_full / 3600   # campaign wall time incl. queueing and retries
        g["hours_full_partition_at_endpoint_latency"] = g.lat * n_full / 3600                     # sequential calls at the median endpoint latency
        g["source"] = tag; g["full_partition_rows"] = n_full
        appl.append(g.reset_index())
    appl = pd.concat(appl); appl.to_csv(O / "llm_applicability.csv", index=False)
    met = pd.read_csv(R / "axis1/metrics.csv")
    xgb = met[(met.model == "xgb_full") & (met.test_set == "full")].groupby("cell").agg(
        predict_s=("predict_s", "mean"), n=("n_test_rows", "first"))
    summary["tree_full_partition_predict_s"] = xgb.to_dict("index")

    # 6. decision guidance: recommended operating points per cell -------------------------
    ops = []
    for cell, g in summ.groupby("cell"):
        def pick(col, asc=False):
            r = g.sort_values(col, ascending=asc).iloc[0]; return r
        for objective, col, asc in [("maximise accuracy", "accuracy_mean", False),
                                    ("fewest false alarms", "false_alarms_per_100k_mean", True),
                                    ("maximise rare-class coverage", "rare_recall_mean", False),
                                    ("balanced aggregate (macro-F1)", "macro_f1_mean", False),
                                    ("balanced accuracy", "balanced_accuracy_mean", False),
                                    ("MCC", "mcc_mean", False)]:
            r = pick(col, asc)
            ops.append(dict(cell=cell, cell_label=CELL_LABEL[cell], objective=objective, model=r.model,
                            model_label=MODEL_LABEL[r.model], accuracy=r.accuracy_mean,
                            rare_recall=r.rare_recall_mean, macro_f1=r.macro_f1_mean,
                            balanced_accuracy=r.balanced_accuracy_mean, mcc=r.mcc_mean,
                            false_alarms_per_100k=r.false_alarms_per_100k_mean,
                            missed_rare_per_100k=r.missed_rare_per_100k_mean))
    pd.DataFrame(ops).to_csv(O / "recommended_operating_points.csv", index=False)

    summary["baseline"] = baseline.to_dict("records")
    summary["rank_correlations"] = corr.to_dict("records")
    (O / "round5_record_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote", sorted(p.name for p in O.iterdir()))


if __name__ == "__main__":
    main()
