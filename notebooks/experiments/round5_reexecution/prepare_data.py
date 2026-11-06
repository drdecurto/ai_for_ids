#!/usr/bin/env python3
"""Acquire the raw benchmark tables, build the cleaned tables + metadata, and run the
split/session/duplicate audits.

  python prepare_data.py download     # login node (internet): HF mirrors, pinned UNSW revision
  python prepare_data.py build        # cleaned parquet tables (+ metadata) and count checks
  python prepare_data.py audit        # session statistics, split manifests, exact-duplicate audit
"""
import hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

AUDIT = C.S / "audit"


def download():
    from huggingface_hub import hf_hub_download
    raw_cic = C.RAW / "CIC-IDS2017" / "MachineLearningCVE"; raw_cic.mkdir(parents=True, exist_ok=True)
    for fn in C.CIC_FILES:
        if not (raw_cic / fn).is_file():
            p = hf_hub_download(repo_id=C.CIC_HF, filename=fn, repo_type="dataset")
            shutil.copy2(p, raw_cic / fn); print("fetched", fn, flush=True)
    z = C.RAW / "CIC-IDS2017" / "GeneratedLabelledFlows.zip"
    if not z.is_file():
        try:
            p = hf_hub_download(repo_id=C.CIC_GLF_HF[0], filename=C.CIC_GLF_HF[1], repo_type="dataset")
            shutil.copy2(p, z); print("fetched GeneratedLabelledFlows.zip", flush=True)
        except Exception as e:
            print("GeneratedLabelledFlows.zip not fetched:", repr(e)[:300], flush=True)
    raw_u = C.RAW / "CIC-UNSW"; raw_u.mkdir(parents=True, exist_ok=True)
    target = raw_u / "CICFlowMeter_out.csv"
    if not target.is_file() or target.stat().st_size != C.UNSW_EXPECTED_BYTES:
        p = hf_hub_download(repo_id=C.UNSW_HF[0], filename=C.UNSW_HF[2], repo_type="dataset", revision=C.UNSW_HF[1])
        shutil.copy2(p, target)
    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    print("UNSW bytes", target.stat().st_size, "sha256", h.hexdigest(), "expected OK:", h.hexdigest() == C.UNSW_EXPECTED_SHA256, flush=True)


def build():
    for which, grouped in [("cic", False), ("cic", True), ("unsw", None)]:
        data, meta = C.load_cell_base(which, grouped)
        print(which, grouped, data.shape, meta.columns.tolist(), flush=True)


def glf_session_stats():
    """Per-label session structure of CIC-IDS2017 from the GeneratedLabelledFlows release
    (hosts, host pairs, capture files), independent of the row alignment to the cleaned table;
    also checks that flow records are in chronological order within each aligned file."""
    import zipfile, re
    z = C.RAW / "CIC-IDS2017" / "GeneratedLabelledFlows.zip"
    if not z.is_file(): return [], []
    zf = zipfile.ZipFile(z); rows = []; order_rows = []
    for n in zf.namelist():
        if not n.endswith(".csv"): continue
        g = pd.read_csv(zf.open(n), encoding="latin-1", low_memory=False, usecols=lambda c: c.strip() in
                        {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"})
        g.columns = [c.strip() for c in g.columns]
        g["Label"] = g["Label"].astype(str).apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
        fname = Path(n).name
        ts = pd.to_datetime(g["Timestamp"], errors="coerce", dayfirst=True)
        d = ts.diff().dt.total_seconds().dropna()
        order_rows.append(dict(file=fname, rows=len(g), ts_parsed=float(ts.notna().mean()),
                               share_nondecreasing_steps=float((d >= 0).mean()), median_step_s=float(d.median())))
        for lab, h in g.groupby("Label"):
            pairs = h["Source IP"].astype(str) + ">" + h["Destination IP"].astype(str)
            rows.append(dict(file=fname, **{"class": lab}, rows=len(h), n_src_ip=h["Source IP"].nunique(),
                             n_dst_ip=h["Destination IP"].nunique(), n_host_pairs=pairs.nunique(),
                             top_pair_share=float(pairs.value_counts(normalize=True).iloc[0]),
                             n_src_ip_nonvictim=h["Source IP"].nunique()))
    df = pd.DataFrame(rows)
    agg = df.groupby("class").agg(rows=("rows", "sum"), n_files=("file", "nunique"), n_src_ip=("n_src_ip", "max"),
                                  n_dst_ip=("n_dst_ip", "max"), n_host_pairs=("n_host_pairs", "max"),
                                  top_pair_share=("top_pair_share", "min")).reset_index()
    agg.insert(0, "cell", "cic_ids2017_V1"); agg["n_capture_days"] = agg.n_files
    return agg.to_dict("records"), order_rows


def audit():
    AUDIT.mkdir(parents=True, exist_ok=True)
    sess, dup, manif, pcs = [], [], [], []
    cic_sess, cic_order = glf_session_stats()
    sess.extend(cic_sess)
    if cic_order: pd.DataFrame(cic_order).to_csv(AUDIT / "cic_flow_order_check.csv", index=False)
    for cell, (which, grouped, use_pca) in C.CELLS.items():
        if cell == "cic_unsw_V2": continue      # same rows as V1
        data, meta = C.load_cell_base(which, grouped)
        lab = data.Label.astype(str)
        # --- session structure per class (why a session-disjoint split is not well posed)
        if "src_ip" in meta:
            for c, g in meta.groupby(lab):
                pairs = g.src_ip + ">" + g.dst_ip
                top = pairs.value_counts(normalize=True)
                sess.append(dict(cell=cell, **{"class": c}, rows=len(g), n_src_ip=g.src_ip.nunique(), n_dst_ip=g.dst_ip.nunique(),
                                 n_host_pairs=pairs.nunique(), top_pair_share=float(top.iloc[0]), n_capture_days=g.capture_day.nunique(),
                                 n_files=g.src_file.nunique()))
        # --- splits: manifest + exact-duplicate audit (feature rows identical across partitions)
        feats = data.drop(columns=["Label"])
        hashes = pd.util.hash_pandas_object(feats, index=False).to_numpy()
        designs = ["row2026", "row2027", "row2028", "temporal", "chrono"] + (["hostpair"] if which == "unsw" else [])
        for design in designs:
            tr, te, man = C.make_split(data, meta, design)
            # label-conflicting duplicates: identical feature rows carrying different labels
            tr_h = hashes[tr]; te_h = hashes[te]
            tr_set = pd.Index(np.unique(tr_h))
            te_dup = tr_set.get_indexer(te_h) >= 0
            te_lab = lab.to_numpy()[te]
            tr_lab_by_hash = pd.Series(lab.to_numpy()[tr], index=tr_h).groupby(level=0).agg(lambda x: "|".join(sorted(set(x))))
            te_train_lab = tr_lab_by_hash.reindex(te_h).to_numpy()
            conflict = te_dup & (te_train_lab != te_lab)
            manif.append(dict(cell=cell, design=design, n_train=len(tr), n_test=len(te), test_hash=man["test_hash"],
                              dropped=man["dropped_classes"], benign_share_test=float(pd.Series(te_lab).str.lower().eq("benign").mean()),
                              test_rows_exact_dup_conflicting_label=int(conflict.sum()),
                              n_pairs_test=man.get("n_pairs_test"), n_pairs_total=man.get("n_pairs_total"),
                              test_rows_with_exact_train_duplicate=int(te_dup.sum()), share_test_exact_dup=float(te_dup.mean())))
            vc = pd.Series(te_lab).value_counts()
            for c in vc.index:
                mk = te_lab == c
                pcs.append(dict(cell=cell, design=design, **{"class": c}, n_test=int(mk.sum()), n_train=int((lab.to_numpy()[tr] == c).sum()),
                                share_test=float(mk.mean()), exact_dup_share=float(te_dup[mk].mean()),
                                exact_dup_conflicting_share=float(conflict[mk].mean())))
            # temporal ordering check: rank of test rows vs train rows within class/day
            if design == "temporal" and "timestamp_raw" in meta:
                m2 = meta.loc[np.concatenate([tr, te])]
                dup.append(dict(cell=cell, design=design, note="metadata available", n_meta=len(m2)))
        del data, meta, feats, hashes
    pd.DataFrame(sess).to_csv(AUDIT / "session_structure.csv", index=False)
    pd.DataFrame(manif).to_csv(AUDIT / "split_manifest.csv", index=False)
    pd.DataFrame(pcs).to_csv(AUDIT / "split_per_class.csv", index=False)
    print(pd.DataFrame(manif).to_string(), flush=True)
    if sess: print(pd.DataFrame(sess).to_string(), flush=True)


if __name__ == "__main__":
    {"download": download, "build": build, "audit": audit}[sys.argv[1]]()
