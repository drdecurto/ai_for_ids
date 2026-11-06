#!/usr/bin/env python3
"""Integrate the second-platform summaries (pulled to back/summary) with the released record.

python integrate_mn5.py --summary ../back/summary --out mn5

Produces in <out>/: copies of the summaries, plus
  platform_replication.csv  per cell/model: record mean vs re-executed row2026 mean (accuracy, rare recall)
  design_shift.csv          per cell/model: temporal/hostpair minus row2026 (re-executed) for the main metrics
  partition_sd.csv          per cell/contrast: SD of the rare-recall contrast across row2026/2027/2028
  mn5_summary.json          headline numbers used in the manuscript text
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ORDER = ["rf_pruned_full", "rf_strong_full", "xgb_full", "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced", "tabpfn_2.5", "tabicl"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--summary", required=True); ap.add_argument("--out", default=str(HERE / "mn5"))
    ap.add_argument("--record", default=str(HERE / "out"))
    a = ap.parse_args(); S = Path(a.summary); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for f in S.glob("*.csv"):
        if f.resolve() != (O / f.name).resolve(): shutil.copy2(f, O / f.name)   # --summary and --out may be the same directory
    ps = pd.read_csv(O / "panel_summary.csv"); per = pd.read_csv(O / "panel_per_seed.csv")
    rec = pd.read_csv(Path(a.record) / "metrics_extended_summary.csv")
    summ = {}
    # platform replication
    rows = []
    for _, r in rec.iterrows():
        q = ps[(ps.cell == r.cell) & (ps.model == r.model) & (ps.split == "row2026")]
        if len(q):
            q = q.iloc[0]
            rows.append(dict(cell=r.cell, model=r.model, n_seeds=int(q.accuracy_count), record_accuracy=r.accuracy_mean, mn5_accuracy=q.accuracy_mean,
                             record_rare_recall=r.rare_recall_mean, mn5_rare_recall=q.rare_recall_mean,
                             d_accuracy=q.accuracy_mean - r.accuracy_mean, d_rare_recall=q.rare_recall_mean - r.rare_recall_mean,
                             record_rare_sd=r.rare_recall_std, mn5_rare_sd=q.rare_recall_std))
    rep = pd.DataFrame(rows); rep.to_csv(O / "platform_replication.csv", index=False)
    if len(rep):
        summ["platform"] = dict(max_abs_d_accuracy=float(rep.d_accuracy.abs().max()), max_abs_d_rare=float(rep.d_rare_recall.abs().max()),
                                median_abs_d_rare=float(rep.d_rare_recall.abs().median()),
                                rank_corr_rare=float(stats.spearmanr(rep.record_rare_recall, rep.mn5_rare_recall).statistic),
                                rank_corr_acc=float(stats.spearmanr(rep.record_accuracy, rep.mn5_accuracy).statistic))
    # design shift
    rows = []
    for (cell, model), g in ps.groupby(["cell", "model"]):
        base = g[g.split == "row2026"]
        if not len(base): continue
        b = base.iloc[0]
        for d in ["temporal", "chrono", "hostpair", "row2027", "row2028"]:
            q = g[g.split == d]
            if not len(q): continue
            q = q.iloc[0]
            rows.append(dict(cell=cell, model=model, design=d, accuracy=q.accuracy_mean, rare_recall=q.rare_recall_mean,
                             macro_ap_rare=q.get("macro_ap_rare_mean", np.nan), attack_auprc=q.get("attack_auprc_mean", np.nan),
                             d_accuracy=q.accuracy_mean - b.accuracy_mean, d_rare_recall=q.rare_recall_mean - b.rare_recall_mean,
                             d_macro_ap_rare=q.get("macro_ap_rare_mean", np.nan) - b.get("macro_ap_rare_mean", np.nan)))
    ds = pd.DataFrame(rows); ds.to_csv(O / "design_shift.csv", index=False)
    # partition SD of contrasts
    con = pd.read_csv(O / "contrast_by_split.csv")
    rows = []
    for (cell, metric, m, c), g in con[con.split.str.startswith("row")].groupby(["cell", "metric", "model", "comparator"]):
        if len(g) < 2: continue
        rows.append(dict(cell=cell, metric=metric, model=m, comparator=c, n_partitions=len(g), mean_over_partitions=g.mean_delta.mean(),
                         sd_over_partitions=g.mean_delta.std(ddof=1), min_delta=g.mean_delta.min(), max_delta=g.mean_delta.max(),
                         sign_agree=bool((np.sign(g.mean_delta) == np.sign(g.mean_delta.iloc[0])).all()),
                         median_within_sd=g.sd_diff.median()))
    psd = pd.DataFrame(rows); psd.to_csv(O / "partition_sd.csv", index=False)
    if len(psd):
        rr = psd[psd.metric == "rare_recall"]
        summ["partition"] = dict(n_contrasts=len(rr), median_sd_over_partitions=float(rr.sd_over_partitions.median()),
                                 max_sd_over_partitions=float(rr.sd_over_partitions.max()), median_within_sd=float(rr.median_within_sd.median()),
                                 n_sign_disagree=int((~rr.sign_agree).sum()))
    inv = pd.read_csv(O / "inversion_stats.csv") if (O / "inversion_stats.csv").is_file() else None
    if inv is not None: summ["inversion"] = inv.to_dict("records")
    if len(ds):
        for dname in ("temporal", "chrono"):
          t = ds[ds.design == dname]
          if not len(t): continue
          summ[dname] = dict(d_rare_recall_min=float(t.d_rare_recall.min()), d_rare_recall_max=float(t.d_rare_recall.max()),
                                d_accuracy_min=float(t.d_accuracy.min()), d_accuracy_max=float(t.d_accuracy.max()),
                                rows=t[["cell", "model", "accuracy", "rare_recall", "d_accuracy", "d_rare_recall", "macro_ap_rare"]].to_dict("records"))
        h = ds[ds.design == "hostpair"]
        if len(h): summ["hostpair"] = h[["cell", "model", "accuracy", "rare_recall", "d_accuracy", "d_rare_recall", "macro_ap_rare"]].to_dict("records")
    (O / "mn5_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("temporal", "chrono", "inversion", "hostpair")}, indent=1))


if __name__ == "__main__":
    main()
