#!/usr/bin/env python3
"""Aggregate the run_panel outputs of all splits into compact summaries (run on the second platform).

python aggregate.py --runs $S/runs/r5 --audit $S/audit --out $S/summary

Writes: panel_per_seed.csv, panel_summary.csv (per split/cell/model means+sd+ranks),
inversion_stats.csv (per split/cell: rho accuracy vs rare recall, same-three, leaders),
contrast_by_split.csv (prior-fitted minus classical rare-recall contrasts per split),
pr_curves.csv (weighted PR curves of rare classes on a 101-point recall grid, per model,
seed-averaged), leakage_all.csv, and copies of the audit tables.
"""
import argparse, glob, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

CLASSICAL = ["rf_pruned_full", "rf_strong_full", "xgb_full", "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced"]
PRIOR = ["tabpfn_2.5", "tabicl"]


def boot_ci(d, n=10000):
    d = np.asarray(d, float); b = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def signflip(d):
    d = np.asarray(d, float); o = abs(d.mean())
    return float(np.mean([abs(np.mean(d * np.array(x))) >= o - 1e-15 for x in itertools.product((-1, 1), repeat=len(d))]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = a.runs.split(",")
    mets = pd.concat([pd.read_csv(f) for r in runs for f in glob.glob(f"{r}/**/metrics.csv", recursive=True)], ignore_index=True)
    mets = mets.drop_duplicates(["cell", "split", "seed", "model", "test_set"], keep="last")
    cap = mets[mets.test_set == "capped"].copy(); cap.to_csv(out / "panel_per_seed.csv", index=False)
    cols = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "rare_recall", "rare_precision", "macro_ap_rare",
            "macro_ap_attack", "macro_ap_all", "attack_auprc", "attack_rocauc", "fit_s", "predict_s"]
    cols = [c for c in cols if c in cap.columns]
    g = cap.groupby(["split", "cell", "model"])[cols].agg(["mean", "std", "count"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]; g = g.reset_index()
    for m in ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "rare_recall", "macro_ap_rare", "attack_auprc"]:
        if f"{m}_mean" in g: g[f"{m}_rank"] = g.groupby(["split", "cell"])[f"{m}_mean"].rank(ascending=False, method="min")
    g.to_csv(out / "panel_summary.csv", index=False)
    # inversion statistics per split/cell
    inv = []
    for (sp, cell), h in g.groupby(["split", "cell"]):
        h = h[h.model.isin(CLASSICAL + PRIOR)]
        if len(h) < 4: continue
        acc_top = set(h.nlargest(3, "accuracy_mean").model); rr_bot = set(h.nsmallest(3, "rare_recall_mean").model)
        row = dict(split=sp, cell=cell, n_models=len(h), rho_acc_rare=float(stats.spearmanr(h.accuracy_mean, h.rare_recall_mean).statistic),
                   same_three=(acc_top == rr_bot) if len(h) == 8 else None, accuracy_leader=h.loc[h.accuracy_mean.idxmax()].model,
                   rare_leader=h.loc[h.rare_recall_mean.idxmax()].model, balacc_leader=h.loc[h.balanced_accuracy_mean.idxmax()].model,
                   mcc_leader=h.loc[h.mcc_mean.idxmax()].model)
        if "macro_ap_rare_mean" in h:
            row.update(rho_acc_aprare=float(stats.spearmanr(h.accuracy_mean, h.macro_ap_rare_mean).statistic),
                       rho_rare_aprare=float(stats.spearmanr(h.rare_recall_mean, h.macro_ap_rare_mean).statistic),
                       aprare_leader=h.loc[h.macro_ap_rare_mean.idxmax()].model, auprc_leader=h.loc[h.attack_auprc_mean.idxmax()].model)
        inv.append(row)
    pd.DataFrame(inv).to_csv(out / "inversion_stats.csv", index=False)
    # contrasts per split (rare recall and macro-AP rare)
    con = []
    for (sp, cell), h in cap.groupby(["split", "cell"]):
        for metric in ["rare_recall", "macro_ap_rare", "balanced_accuracy", "accuracy"]:
            if metric not in h: continue
            piv = h.pivot_table(index="seed", columns="model", values=metric)
            for pf in PRIOR:
                for cl in CLASSICAL:
                    if pf in piv and cl in piv:
                        d = (piv[pf] - piv[cl]).dropna().to_numpy()
                        if len(d) < 2: continue
                        lo, hi = boot_ci(d)
                        con.append(dict(split=sp, cell=cell, metric=metric, model=pf, comparator=cl, n=len(d), mean_delta=d.mean(),
                                        ci_lo=lo, ci_hi=hi, sd_diff=d.std(ddof=1), wins=int((d > 0).sum()),
                                        p_signflip=signflip(d) if len(d) <= 12 else np.nan))
    pd.DataFrame(con).to_csv(out / "contrast_by_split.csv", index=False)
    # PR curves of rare classes (weighted), seed-averaged on a recall grid
    from sklearn.metrics import precision_recall_curve
    grid = np.linspace(0, 1, 101); pr = []
    for f in sorted(f for r in runs for f in glob.glob(f"{r}/**/probs/*.npz", recursive=True)):
        name = Path(f).stem; z = np.load(f, allow_pickle=True)
        import re
        mm = re.match(r"^(cic_\w+?_V\d)_(row\d+|temporal|chrono|hostpair)_s(\d+)_(.+)$", name)
        if not mm: continue
        cell, split, seed, model = mm.group(1), mm.group(2), int(mm.group(3)), mm.group(4)
        classes = z["classes"]; y = z["y_true"]; P = z["proba"]; w = z["weights"]
        N = w.sum(); shares = {c: w[y == i].sum() / N for i, c in enumerate(classes)}
        for i, c in enumerate(classes):
            if str(c).lower() == "benign" or shares[c] >= 0.005 or (y == i).sum() == 0: continue
            p, r, _ = precision_recall_curve((y == i).astype(int), P[:, i], sample_weight=w)
            # precision at recall >= grid point (interpolated step)
            pi = np.array([p[r >= gr].max() if (r >= gr).any() else 0.0 for gr in grid])
            pr.append(dict(cell=cell, split=split, seed=seed, model=model, **{"class": str(c)}, **{f"p@{k}": v for k, v in zip(range(101), pi)}))
    if pr:
        prdf = pd.DataFrame(pr)
        prdf.groupby(["cell", "split", "model", "class"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index().to_csv(out / "pr_curves.csv", index=False)
    lk = glob.glob(f"{a.audit}/leakage_*.csv")
    if lk: pd.concat([pd.read_csv(f) for f in lk]).to_csv(out / "leakage_all.csv", index=False)
    for f in ["session_structure.csv", "split_manifest.csv", "split_per_class.csv"]:
        if Path(a.audit, f).is_file(): pd.read_csv(Path(a.audit, f)).to_csv(out / f, index=False)
    print("summaries written to", out, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
