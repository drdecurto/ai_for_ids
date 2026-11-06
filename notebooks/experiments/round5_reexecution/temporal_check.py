"""Audit of the chronological ("temporal") design against parsed timestamps.

For CIC-IDS2017 the chronological split orders rows by capture file and row
position. This script attaches the flow-start Timestamp of every cleaned row
(GeneratedLabelledFlows: row-aligned for Monday-Wednesday; for Thursday/Friday
aligned after dropping the unlabelled GLF rows, verified on four probe columns),
disambiguates the 12-hour clock of the release (capture hours 08:00-17:00, so
hours 1-7 are PM), and then checks, for every (class, capture day) stratum of the
temporal split, whether the held-out rows follow the fitting rows in time:
  * max fitting timestamp vs min held-out timestamp,
  * share of held-out rows that start before the last fitting flow,
  * share of held-out rows that start before the 99th / 50th percentile of the
    fitting timestamps (how far the overlap reaches back),
  * the same for CIC-UNSW-NB15 (timestamps come with the CICFlowMeter output).
It also counts the rows affected by the whole-table median imputation of
Flow_Bytes/s and Flow_Packets/s in the vendored CIC cleaning.
Outputs: <out>/temporal_boundary_check.csv, <out>/temporal_boundary_summary.csv,
<out>/cic_imputation_counts.csv, <out>/cic_glf_alignment_thu_fri.json
"""
import argparse, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import common as C


def parse_cic_ts(s):
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    h = ts.dt.hour
    # 12-hour clock without AM/PM in the release: captures run ~08:00-17:00
    fix = ts + pd.to_timedelta(np.where(h.between(1, 7), 12, 0), unit="h")
    return pd.Series(fix, index=s.index)


def cic_timestamps(ml_frames):
    z = C.RAW / "CIC-IDS2017" / "GeneratedLabelledFlows.zip"
    zf = zipfile.ZipFile(z); names = zf.namelist(); out = []; rep = {}
    probe = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]
    for f, ml in zip(C.CIC_FILES, ml_frames):
        cand = [n for n in names if n.endswith(".csv") and Path(n).name.lower() == f.lower()] or \
               [n for n in names if n.endswith(".csv") and Path(n).name[:40].lower() == f[:40].lower()]
        g = pd.read_csv(zf.open(cand[0]), encoding="latin-1", low_memory=False)
        g.columns = [c.strip() for c in g.columns]
        mlc = ml.copy(); mlc.columns = [c.strip() for c in mlc.columns]
        def eq(a, b):
            return len(a) == len(b) and all(np.array_equal(pd.to_numeric(a[c], errors="coerce").fillna(-1).to_numpy(),
                                                            pd.to_numeric(b[c], errors="coerce").fillna(-1).to_numpy()) for c in probe)
        mode = "row"
        if not eq(g, mlc):
            g2 = g[g["Label"].notna() & (g["Label"].astype(str).str.strip() != "")].reset_index(drop=True)
            if eq(g2, mlc): g, mode = g2, "row_after_dropping_unlabelled"
            else:
                mode = "unaligned"; rep[f] = dict(mode=mode, glf=len(g), glf_labelled=len(g2), ml=len(mlc))
                out.append(pd.Series([pd.NaT] * len(mlc))); continue
        rep[f] = dict(mode=mode, glf=len(g), ml=len(mlc))
        out.append(parse_cic_ts(g["Timestamp"].astype(str)).reset_index(drop=True))
    return pd.concat(out, ignore_index=True), rep


def boundary_rows(cell, df, meta, ts, tr, te):
    rows = []
    lab = df.Label.astype(str).to_numpy(); day = meta.capture_day.astype(str).to_numpy()
    tsv = ts.to_numpy()
    is_tr = np.zeros(len(df), bool); is_tr[tr] = True
    is_te = np.zeros(len(df), bool); is_te[te] = True
    for (c, d) in sorted(set(zip(lab, day))):
        m = (lab == c) & (day == d)
        a = tsv[m & is_tr]; b = tsv[m & is_te]
        a = a[~pd.isna(a)]; b = b[~pd.isna(b)]
        if len(a) == 0 or len(b) == 0:
            rows.append(dict(cell=cell, cls=c, day=d, n_fit=int((m & is_tr).sum()), n_test=int((m & is_te).sum()), ts_available=False)); continue
        a = np.sort(a); b = np.sort(b)
        q99, q50 = a[int(0.99 * (len(a) - 1))], a[len(a) // 2]
        rows.append(dict(cell=cell, cls=c, day=d, n_fit=int((m & is_tr).sum()), n_test=int((m & is_te).sum()), ts_available=True,
                         fit_min=str(a[0]), fit_max=str(a[-1]), test_min=str(b[0]), test_max=str(b[-1]),
                         test_before_fit_max=float((b < a[-1]).mean()), test_before_fit_q99=float((b < q99).mean()),
                         test_before_fit_q50=float((b < q50).mean()), fit_after_test_min=float((a > b[0]).mean()),
                         overlap_minutes=float(max(0.0, (a[-1] - b[0]) / np.timedelta64(1, "m")))))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="temporal,chrono"); args = ap.parse_args()
    designs = args.designs.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allrows = []
    # ---- CIC-IDS2017 (V1 = fine labels; V3 shares rows and strata, only the labels differ)
    raw = C.RAW / "CIC-IDS2017" / "MachineLearningCVE"
    frames = [pd.read_csv(raw / f, encoding="latin-1") for f in C.CIC_FILES]
    ts_all, rep = cic_timestamps(frames)
    (out / "cic_glf_alignment_thu_fri.json").write_text(json.dumps(rep, indent=2)); print(rep, flush=True)
    # imputation counts on the raw concatenation (before the duplicate drop) and after it, as in build_cic
    data = pd.concat(frames, ignore_index=True)
    C._features_correction(data); C._ascii_correction(data, "Label")
    imp = []
    for stage, d in (("raw", data), ("after_duplicate_drop", data.drop_duplicates())):
        for col in ("Flow_Bytes/s", "Flow_Packets/s"):
            v = d[col].replace([np.inf, -np.inf], np.nan)
            imp.append(dict(stage=stage, column=col, rows=len(d), nan_or_inf=int(v.isna().sum()), share=float(v.isna().mean()),
                            by_label=json.dumps({k: int(x) for k, x in d.loc[v.isna(), "Label"].value_counts().items()})))
    pd.DataFrame(imp).to_csv(out / "cic_imputation_counts.csv", index=False); print(pd.DataFrame(imp).to_string(), flush=True)
    offs = {}; base = 0
    for f, fr in zip(C.CIC_FILES, frames): offs[f] = base; base += len(fr)
    del frames, data
    for grouped, cell in ((False, "cic_ids2017_V1"), (True, "cic_ids2017_V3")):
        df, meta = C.load_cell_base("cic", grouped)
        raw_index = meta.src_file.map(offs).to_numpy() + meta.row_pos.to_numpy()
        ts = ts_all.iloc[raw_index].reset_index(drop=True)
        for dsg in designs:
            tr, te, man = C.make_split(df, meta, dsg)
            allrows += [dict(design=dsg, **r) for r in boundary_rows(cell, df, meta, ts, tr, te)]
        del df, meta
    # ---- CIC-UNSW-NB15 (V2 differs from V1 only by the PCA transform; same rows and split)
    df, meta = C.load_cell_base("unsw", None)
    ts = C._parse_ts(meta.timestamp_raw)
    for dsg in designs:
        tr, te, man = C.make_split(df, meta, dsg)
        allrows += [dict(design=dsg, **r) for r in boundary_rows("cic_unsw_V1+V2", df, meta, ts, tr, te)]
    R = pd.DataFrame(allrows); R.to_csv(out / "temporal_boundary_check.csv", index=False)
    pd.set_option("display.width", 250); print(R.to_string(), flush=True)
    S = R[R.ts_available == True].groupby(["design", "cell"]).agg(strata=("cls", "size"), strata_clean=("test_before_fit_max", lambda x: int((x == 0).sum())),
                                                      test_before_fit_max_mean=("test_before_fit_max", "mean"),
                                                      test_before_fit_max_max=("test_before_fit_max", "max"),
                                                      test_before_fit_q99_max=("test_before_fit_q99", "max"),
                                                      test_before_fit_q50_max=("test_before_fit_q50", "max"),
                                                      overlap_minutes_max=("overlap_minutes", "max"))
    S.to_csv(out / "temporal_boundary_summary.csv"); print(S.to_string(), flush=True)


if __name__ == "__main__":
    main()
