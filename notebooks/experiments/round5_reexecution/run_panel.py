#!/usr/bin/env python3
"""Strict Axis-1 panel re-execution with probabilistic outputs under a chosen split design.

python run_panel.py --cell cic_ids2017_V3 --split row2026 --models xgb_full,tabpfn_2.5 --seeds 0,1 --out $S/runs/r5

Every (cell, split, seed, model) is written as soon as it is produced and skipped on
restart. Outputs under <out>/:
  metrics.csv      one row per run (accuracy, macro-F1, balanced accuracy, MCC, rare recall,
                   weighted APs, binary attack AUPRC/ROC-AUC, timings, hashes)
  per_class.csv    weighted per-class precision/recall/F1/support/AP
  confusion.csv    weighted confusion matrix (long format)
  probs/<cell>_<split>_s<seed>_<model>.npz   y_true, classes, proba, row weights
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--split", default="row2026")
    ap.add_argument("--models", default=",".join(C.CLASSICAL + C.FOUNDATION))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9"); ap.add_argument("--out", required=True)
    ap.add_argument("--full-validate-seeds", default="", help="seeds on which classical models are also scored on the full partition")
    a = ap.parse_args()
    out = Path(a.out) / a.cell; (out / "probs").mkdir(parents=True, exist_ok=True)   # one directory per cell: no concurrent appends, one AP column set
    MET, PCC, CMC = out / "metrics.csv", out / "per_class.csv", out / "confusion.csv"
    done = set()
    if MET.exists():
        p = pd.read_csv(MET); done = set(zip(p.cell, p.split, p.seed, p.model, p.test_set))
    which, grouped, use_pca = C.CELLS[a.cell]
    base, meta = C.load_cell_base(which, grouped)
    seeds = [int(s) for s in a.seeds.split(",")]; models = a.models.split(",")
    vseeds = {int(s) for s in a.full_validate_seeds.split(",") if s != ""}
    tr_idx, te_idx, man = C.make_split(base, meta, a.split)
    tr, te = base.loc[tr_idx], base.loc[te_idx]
    print(f"{a.cell} split={a.split} train={len(tr):,} test={len(te):,} hash={man['test_hash']} dropped={man['dropped_classes']!r}", flush=True)
    labs_all = sorted(base.Label.astype(str).unique())

    for seed in seeds:
        te_cap, weights = C.cap_test(te, seed)
        sel_cache = {}; tf_cache = {}
        for name in models:
            is_fm = name in C.FOUNDATION
            targets = [("capped", te_cap, weights)]
            if (not is_fm) and seed in vseeds: targets.append(("full", te, None))
            targets = [t for t in targets if (a.cell, a.split, seed, name, t[0]) not in done]
            if not targets: continue
            b = C.budget_of(name); key = "full" if b is None else f"bal{b}"
            if key not in sel_cache:
                sset = tr if b is None else C.balanced(tr, b, seed)
                if b is None and len(sset) > C.MAX_FULL_ROWS: sset = sset.sample(C.MAX_FULL_ROWS, random_state=seed)
                sel_cache[key] = sset
            sel = sel_cache[key]; ck = (key, tuple(t[0] for t in targets))
            if ck not in tf_cache: tf_cache[ck] = C.transform(sel, [t[1] for t in targets], use_pca)
            xtr, xtes = tf_cache[ck]
            ytr = sel["Label"].astype(str).reset_index(drop=True)
            try:
                C.set_seed(seed); m = C.make_model(name, seed); t0 = time.time()
                if name.startswith("xgb"):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder().fit(ytr); m.fit(xtr.to_numpy(), le.transform(ytr)); classes = le.classes_
                else:
                    m.fit(xtr.to_numpy(), ytr.to_numpy()); classes = np.asarray(m.classes_)
                ft = time.time() - t0
            except Exception as e:
                print(f"    {name} seed {seed}: FIT FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
            classes = np.asarray(classes).astype(str)
            for (ts, tdf, w), xte in zip(targets, xtes):
                t0 = time.time()
                try:
                    proba = C.chunked_proba(m, xte.to_numpy())
                except Exception as e:
                    print(f"    {name} seed {seed} {ts}: PREDICT FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
                pt = time.time() - t0
                y_true = tdf["Label"].astype(str).to_numpy()
                # align probability columns to the full label set of the cell
                P = np.zeros((len(xte), len(labs_all)), np.float32)
                for j, c in enumerate(classes):
                    if c in labs_all: P[:, labs_all.index(c)] = proba[:, j]
                y_pred = np.asarray(labs_all)[P.argmax(1)]
                wmap = w or {}
                cm = C.weighted_confusion(y_true, y_pred, labs_all, wmap)
                met, per, rare_idx = C.metrics_from_cm(cm, labs_all)
                yti = np.array([labs_all.index(c) for c in y_true]); wrow = np.array([wmap.get(c, 1.0) for c in y_true])
                met.update(C.ap_metrics(yti, P, wrow, labs_all, rare_idx))
                per["ap"] = [met.get("ap_" + str(c), np.nan) for c in labs_all]
                tag = dict(cell=a.cell, split=a.split, seed=seed, model=name, test_set=ts)
                extra = dict(budget=("full" if b is None else f"balanced{b}"), n_train=len(sel), n_test_rows=len(xte),
                             fit_s=ft, predict_s=pt, split_seed=man["split_seed"], test_hash=man["test_hash"],
                             dropped_classes=man["dropped_classes"], n_features=xtr.shape[1])
                pd.DataFrame([{**tag, **extra, **met}]).to_csv(MET, mode="a", index=False, header=not MET.exists())
                for c, v in reversed(list(tag.items())): per.insert(0, c, v)
                per.to_csv(PCC, mode="a", index=False, header=not PCC.exists())
                cml = pd.DataFrame([{**tag, "true_label": x, "pred_label": y, "count": float(cm[i, j])}
                                    for i, x in enumerate(labs_all) for j, y in enumerate(labs_all)])
                cml.to_csv(CMC, mode="a", index=False, header=not CMC.exists())
                if ts == "capped":
                    np.savez_compressed(out / "probs" / f"{a.cell}_{a.split}_s{seed}_{name}.npz", y_true=yti.astype(np.int16),
                                        classes=np.asarray(labs_all), proba=P, weights=wrow.astype(np.float32))
                print(f"    {name:20s} seed {seed} {ts:6s} acc={met['accuracy']:.4f} mF1={met['macro_f1']:.3f} "
                      f"rare={met['rare_recall']:.3f} bAcc={met['balanced_accuracy']:.3f} mcc={met['mcc']:.3f} "
                      f"APrare={met.get('macro_ap_rare', float('nan')):.3f} fit={ft:.0f}s pred={pt:.0f}s", flush=True)
            del m
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
