#!/usr/bin/env python3
"""Near-duplicate / memorisation audit between fitting and held-out rows.

For a cell, split design and seed: fit the budget transform on the fitting sample
(large natural 400k and matched balanced), standardise with the fitting-sample
statistics, and find for every capped held-out row its nearest fitting row
(Euclidean). Reports per class: median NN distance, share of held-out rows whose
nearest fitting row is (near-)identical (distance < 1e-6 and < 1e-2 after
standardisation), and the label agreement of the nearest neighbour (a 1-NN
classifier: the recall a pure memoriser would obtain).

python leakage_audit.py --cell cic_ids2017_V3 --split row2026 --seed 0 --out $S/audit
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def nn_search(xtr, xte, chunk=2048):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        A = torch.as_tensor(xtr, dtype=torch.float32, device=dev)
        a2 = (A * A).sum(1)
        d_out = np.empty(len(xte), np.float32); i_out = np.empty(len(xte), np.int64)
        for s in range(0, len(xte), chunk):
            B = torch.as_tensor(xte[s:s + chunk], dtype=torch.float32, device=dev)
            d2 = (B * B).sum(1, keepdim=True) + a2[None, :] - 2.0 * B @ A.T
            d2.clamp_(min=0); v, i = d2.min(1)
            d_out[s:s + chunk] = v.sqrt().cpu().numpy(); i_out[s:s + chunk] = i.cpu().numpy()
        return d_out, i_out
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(xtr)
        d, i = nn.kneighbors(xte); return d[:, 0], i[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--split", default="row2026")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    which, grouped, use_pca = C.CELLS[a.cell]
    base, meta = C.load_cell_base(which, grouped)
    tr_idx, te_idx, man = C.make_split(base, meta, a.split)
    tr, te = base.loc[tr_idx], base.loc[te_idx]
    te_cap, weights = C.cap_test(te, a.seed)
    rows = []
    for regime in ["large_natural", "balanced"]:
        if regime == "large_natural":
            sel = tr.sample(C.MAX_FULL_ROWS, random_state=a.seed) if len(tr) > C.MAX_FULL_ROWS else tr
        else:
            sel = C.balanced(tr, C.BALANCED, a.seed)
        xtr, xte = C.transform(sel, te_cap, use_pca)
        mu = xtr.mean(0).to_numpy(); sd = xtr.std(0).replace(0, 1).to_numpy()
        A = ((xtr.to_numpy() - mu) / sd).astype(np.float32); B = ((xte.to_numpy() - mu) / sd).astype(np.float32)
        t0 = time.time(); d, i = nn_search(A, B); print(f"{regime}: NN search {len(B):,} x {len(A):,} in {time.time()-t0:.0f}s", flush=True)
        ytr = sel.Label.astype(str).to_numpy(); yte = te_cap.Label.astype(str).to_numpy()
        same = ytr[i] == yte
        w = np.array([weights[c] for c in yte])
        for c in sorted(set(yte)):
            mk = yte == c
            rows.append(dict(cell=a.cell, split=a.split, seed=a.seed, regime=regime, **{"class": c}, n_test_rows=int(mk.sum()),
                             n_fit_rows_class=int((ytr == c).sum()), nn_dist_median=float(np.median(d[mk])),
                             nn_dist_q10=float(np.quantile(d[mk], 0.1)), share_nn_below_1e6=float((d[mk] < 1e-6).mean()),
                             share_nn_below_1e2=float((d[mk] < 1e-2).mean()), share_nn_below_1e1=float((d[mk] < 1e-1).mean()),
                             nn1_recall=float(same[mk].mean())))
        tot = dict(cell=a.cell, split=a.split, seed=a.seed, regime=regime, **{"class": "__all_weighted__"}, n_test_rows=len(yte),
                   n_fit_rows_class=len(ytr), nn_dist_median=float(np.median(d)), nn_dist_q10=float(np.quantile(d, 0.1)),
                   share_nn_below_1e6=float(np.average(d < 1e-6, weights=w)), share_nn_below_1e2=float(np.average(d < 1e-2, weights=w)),
                   share_nn_below_1e1=float(np.average(d < 1e-1, weights=w)), nn1_recall=float(np.average(same, weights=w)))
        rows.append(tot)
    df = pd.DataFrame(rows); f = out / f"leakage_{a.cell}_{a.split}_s{a.seed}.csv"; df.to_csv(f, index=False)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
