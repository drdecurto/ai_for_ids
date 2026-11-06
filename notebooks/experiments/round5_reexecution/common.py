"""Round-5 shared pipeline for the IoT-IDS benchmark (second-platform re-execution).

Faithful transcription of the released protocol (vendored loaders of
revision4_experiments.ipynb, cell 6-8, and make_model of
axis1_ten_seed_campaign.ipynb), extended with
  * probabilistic outputs (predict_proba on the capped evaluation rows),
  * alternative held-out designs: other stratified row partitions and a
    chronological within-class/within-capture-day split,
  * row-level metadata (capture file, position, timestamp, hosts) kept beside
    the cleaned feature table for the leakage/session audit.
Nothing in the cleaning, budgets, cap or model settings is changed.
"""
from __future__ import annotations
import gc, hashlib, json, os, re, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

# ----------------------------------------------------------------------------- paths
S = Path(os.environ.get("IDS_SCRATCH", "/path/to/scratch/ids"))
DATA = S / "data"; RAW = DATA / "raw"; PROC = DATA / "processed"

CIC_FILES = ["Monday-WorkingHours.pcap_ISCX.csv", "Tuesday-WorkingHours.pcap_ISCX.csv",
             "Wednesday-workingHours.pcap_ISCX.csv", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Friday-WorkingHours-Morning.pcap_ISCX.csv",
             "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
CIC_HF = "c01dsnap/CIC-IDS2017"
CIC_GLF_HF = ("bencorn/CICIDS2017", "csvs/GeneratedLabelledFlows.zip")
UNSW_HF = ("bencorn/CIC-UNSW-NB15", "8b5e371bf1cfb32683a958f62acacc6be19b5c58", "CICFlowMeter_out.csv")
UNSW_EXPECTED_BYTES = 1_914_305_081
UNSW_EXPECTED_SHA256 = "025a94239f29521e154be3a65d65268404d4c58600b66b94a4235eae87dbd8b5"
EXPECTED = {("cic", False): (2_522_362, 15), ("cic", True): (2_522_362, 9), ("unsw", None): (3_540_241, 10)}

CIC_IDS2017_CLASSES_MAPPING = {
    'BENIGN': 'BENIGN', 'DDoS': 'DDoS', 'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS', 'PortScan': 'Port Scan', 'FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force',
    'Bot': 'Bot', 'Web Attack  Brute Force': 'Web Attack', 'Web Attack  Sql Injection': 'Web Attack',
    'Web Attack  XSS': 'Web Attack', 'Infiltration': 'Infiltration', 'Heartbleed': 'Heartbleed'}

CELLS = {"cic_ids2017_V1": ("cic", False, False), "cic_ids2017_V3": ("cic", True, False),
         "cic_unsw_V1": ("unsw", None, False), "cic_unsw_V2": ("unsw", None, True)}
CLASSICAL = ["rf_pruned_full", "rf_strong_full", "xgb_full", "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced"]
FOUNDATION = ["tabpfn_2.5", "tabicl"]

TEST_SIZE, FIXED_SEED = 0.30, 2026
KEEP_FRACTION, CAP_MIN = 0.05, 1500
RARE_SHARE = 0.005
BALANCED, MAX_FULL_ROWS = 2000, 400_000
PREDICT_CHUNK = 4096
META_COLS = ["src_file", "row_pos", "timestamp_raw", "src_ip", "dst_ip", "flow_id", "capture_day"]


def sha(a):
    return hashlib.sha1(np.asarray(a).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- cleaning (vendored)
def _features_correction(d):
    d.rename(columns={c: str(c).strip().replace(" ", "_") for c in d.columns}, inplace=True)

def _ascii_correction(d, col):
    d[col] = d[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', str(x)))

def _downcast(d):
    for col in d.columns:
        t = d[col].dtype
        if t != object:
            lo, hi = d[col].min(), d[col].max()
            if 'float' in str(t) and lo > np.finfo(np.float32).min and hi < np.finfo(np.float32).max:
                try: d[col] = d[col].astype(np.float32)
                except ValueError: pass
            elif 'int' in str(t) and lo > np.iinfo(np.int32).min and hi < np.iinfo(np.int32).max:
                d[col] = d[col].astype(np.int32)

def _drop_single_unique(d):
    nu = d.nunique(); return d[nu[nu > 1].index]

def check_counts(tag, d):
    rows, classes = EXPECTED[tag]
    msg = f"{tag}: {len(d):,} rows / {d['Label'].nunique()} classes; manuscript states {rows:,} / {classes}"
    assert len(d) == rows and d["Label"].nunique() == classes, msg
    print("  count check OK -", msg, flush=True)


def build_cic(grouped: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned CIC-IDS2017 table (exactly as src.data) + aligned metadata frame.

    The metadata (file, row position, and the Flow ID / IPs / Timestamp from the
    GeneratedLabelledFlows release when it aligns) is carried on the original
    index through the duplicate drop, so meta.loc[i] describes feature row i."""
    raw = RAW / "CIC-IDS2017" / "MachineLearningCVE"
    frames, metas = [], []
    for f in CIC_FILES:
        df = pd.read_csv(raw / f, encoding="latin-1")
        m = pd.DataFrame({"src_file": f, "row_pos": np.arange(len(df))})
        frames.append(df); metas.append(m)
    data = pd.concat(frames, ignore_index=True); meta = pd.concat(metas, ignore_index=True)
    glf = _cic_glf_metadata(frames)
    if glf is not None:
        meta = pd.concat([meta, glf], axis=1)
    del frames; gc.collect()
    _features_correction(data); _ascii_correction(data, "Label")
    data.drop_duplicates(inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in ("Flow_Bytes/s", "Flow_Packets/s"):
        if c in data.columns: data[c] = data[c].fillna(data[c].median())
    data.dropna(inplace=True)
    _downcast(data); data = _drop_single_unique(data)
    if grouped: data["Label"] = data["Label"].map(CIC_IDS2017_CLASSES_MAPPING)
    check_counts(("cic", grouped), data)
    meta = meta.loc[data.index].copy()
    meta["capture_day"] = meta.src_file.str.split("-").str[0]
    return data.reset_index(drop=True), meta.reset_index(drop=True)


def _cic_glf_metadata(ml_frames):
    """Flow ID / IPs / Timestamp from GeneratedLabelledFlows, aligned row by row to the
    MachineLearningCVE files. Returns None (with a recorded reason) if the release is
    absent or does not align; the chronological split then uses file order only."""
    z = RAW / "CIC-IDS2017" / "GeneratedLabelledFlows.zip"
    report = {"available": z.is_file(), "aligned": False, "files": {}}
    out = None
    if z.is_file():
        try:
            zf = zipfile.ZipFile(z); names = zf.namelist(); parts = []
            ok = True
            for f, ml in zip(CIC_FILES, ml_frames):
                cand = [n for n in names if n.endswith(".csv") and Path(n).name.split("-")[0].lower() == f.split("-")[0].lower()
                        and Path(n).name.replace("Infilteration", "Infilteration") == f]
                if not cand:   # tolerate encoding differences in the file name (never across different files)
                    cand = [n for n in names if n.endswith(".csv") and Path(n).name.lower() == f.lower()]
                if not cand:
                    cand = [n for n in names if n.endswith(".csv") and Path(n).name[:40].lower() == f[:40].lower()]
                if not cand: ok = False; report["files"][f] = "missing"; break
                g = pd.read_csv(zf.open(cand[0]), encoding="latin-1", low_memory=False)
                g.columns = [c.strip() for c in g.columns]
                mlc = ml.copy(); mlc.columns = [c.strip() for c in mlc.columns]
                same_len = len(g) == len(mlc)
                probe = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]
                match = same_len and all(np.array_equal(pd.to_numeric(g[c], errors="coerce").fillna(-1).to_numpy(),
                                                        pd.to_numeric(mlc[c], errors="coerce").fillna(-1).to_numpy())
                                         for c in probe)
                report["files"][f] = {"glf_rows": len(g), "ml_rows": len(mlc), "aligned": bool(match)}
                if not match: ok = False; break
                parts.append(pd.DataFrame({"timestamp_raw": g["Timestamp"].astype(str).to_numpy(),
                                           "src_ip": g["Source IP"].astype(str).to_numpy(),
                                           "dst_ip": g["Destination IP"].astype(str).to_numpy(),
                                           "flow_id": g["Flow ID"].astype(str).to_numpy()}))
            if ok:
                out = pd.concat(parts, ignore_index=True); report["aligned"] = True
        except Exception as e:
            report["error"] = repr(e)
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "cic_glf_alignment.json").write_text(json.dumps(report, indent=2))
    print("  GeneratedLabelledFlows alignment:", report["aligned"], flush=True)
    return out


def build_unsw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned CIC-UNSW-NB15 table (exactly as src.data; note: no duplicate drop) + metadata."""
    csv = RAW / "CIC-UNSW" / "CICFlowMeter_out.csv"
    df = pd.read_csv(csv, low_memory=False)
    labels = df["Label"].copy()
    meta = pd.DataFrame({"src_file": csv.name, "row_pos": np.arange(len(df))})
    for src, dst in [("Timestamp", "timestamp_raw"), ("Src IP", "src_ip"), ("Dst IP", "dst_ip"), ("Flow ID", "flow_id")]:
        if src in df.columns: meta[dst] = df[src].astype(str).to_numpy()
    data = df.select_dtypes(include=["number"]).copy(); data["Label"] = labels.to_numpy()
    del df; gc.collect()
    _features_correction(data); _downcast(data); data = _drop_single_unique(data)
    check_counts(("unsw", None), data)
    if "timestamp_raw" in meta:
        ts = pd.to_datetime(meta.timestamp_raw, errors="coerce", dayfirst=True)
        meta["capture_day"] = ts.dt.strftime("%Y-%m-%d").fillna("unknown")
    else:
        meta["capture_day"] = "all"
    return data.reset_index(drop=True), meta.reset_index(drop=True)


def load_cell_base(which: str, grouped):
    """Processed parquet cache of (features, metadata)."""
    PROC.mkdir(parents=True, exist_ok=True)
    tag = f"{which}_grouped{bool(grouped)}" if which == "cic" else "unsw"
    fp, mp = PROC / f"{tag}.parquet", PROC / f"{tag}_meta.parquet"
    if fp.is_file() and mp.is_file():
        data, meta = pd.read_parquet(fp), pd.read_parquet(mp)
    else:
        data, meta = build_cic(bool(grouped)) if which == "cic" else build_unsw()
        data.to_parquet(fp); meta.to_parquet(mp)
    for c in data.columns:
        if data[c].dtype == np.float64: data[c] = data[c].astype(np.float32)
    check_counts(("cic", bool(grouped)) if which == "cic" else ("unsw", None), data)
    return data, meta


# ----------------------------------------------------------------------------- protocol
def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s); np.random.seed(s)
    try:
        import torch, random; random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    except Exception: pass


def split_rows(df, split_seed=FIXED_SEED):
    """Stratified 70/30 row split (the released design with split_seed=2026)."""
    from sklearn.model_selection import train_test_split
    vc = df.Label.value_counts(); tiny = list(vc[vc < 2].index)
    if tiny: df = df[~df.Label.isin(tiny)]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=split_seed, stratify=df.Label.to_numpy())
    return df.index[tr], df.index[te], {"split": f"row{split_seed}", "split_seed": split_seed,
                                        "test_hash": sha(np.sort(te)), "dropped_classes": ";".join(tiny)}


def split_temporal(df, meta):
    """Chronological within-class, within-capture-day split: for every (class, day)
    stratum the first 70% of flow records (capture-file order) fit, the last 30%
    are held out. No held-out flow precedes a fitting flow of the same class and
    day, so interleaved records of one attack session cannot straddle the split."""
    order = np.lexsort((meta.row_pos.to_numpy(), meta.src_file.to_numpy()))   # file order = capture order
    rank = np.empty(len(df), int); rank[order] = np.arange(len(df))
    key = df.Label.astype(str).to_numpy()
    day = meta.capture_day.astype(str).to_numpy()
    strata = pd.Series(np.char.add(np.char.add(key, "|"), day))
    te_mask = np.zeros(len(df), bool)
    for _, pos in strata.groupby(strata).indices.items():
        pos = np.asarray(pos); pos = pos[np.argsort(rank[pos], kind="stable")]
        n_te = int(round(TEST_SIZE * len(pos)))
        if len(pos) >= 2 and n_te == 0: n_te = 1
        if n_te >= len(pos): n_te = len(pos) - 1
        te_mask[pos[len(pos) - n_te:]] = True
    # classes absent from one side (single-row classes) are dropped as in split_rows
    tr_lab = set(df.Label[~te_mask]); te_lab = set(df.Label[te_mask])
    keep = df.Label.isin(tr_lab & te_lab).to_numpy()
    tr = df.index[(~te_mask) & keep]; te = df.index[te_mask & keep]
    return tr, te, {"split": "temporal", "split_seed": None, "test_hash": sha(np.sort(te.to_numpy())),
                    "dropped_classes": ";".join(sorted(set(df.Label) - (tr_lab & te_lab)))}


def split_hostpair(df, meta, split_seed=FIXED_SEED):
    """Host-pair-disjoint split (sessions = source/destination host pairs): pairs are
    drawn in a seeded random order into the held-out side until it holds 30% of the
    rows; no held-out flow shares its host pair with any fitting flow. Feasible only
    where every class spans many host pairs (CIC-UNSW-NB15); classes left without rows
    on one side are dropped and recorded."""
    assert "src_ip" in meta and "dst_ip" in meta, "host columns are not available for this source"
    pair = (meta.src_ip.astype(str) + ">" + meta.dst_ip.astype(str)).to_numpy()
    uniq, inv, counts = np.unique(pair, return_inverse=True, return_counts=True)
    order = np.random.default_rng(split_seed).permutation(len(uniq))
    te_pairs = np.zeros(len(uniq), bool); acc = 0; target = TEST_SIZE * len(df)
    for k in order:
        if acc >= target: break
        te_pairs[k] = True; acc += counts[k]
    te_mask = te_pairs[inv]
    tr_lab = set(df.Label[~te_mask]); te_lab = set(df.Label[te_mask])
    keep = df.Label.isin(tr_lab & te_lab).to_numpy()
    tr = df.index[(~te_mask) & keep]; te = df.index[te_mask & keep]
    return tr, te, {"split": "hostpair", "split_seed": split_seed, "test_hash": sha(np.sort(te.to_numpy())),
                    "dropped_classes": ";".join(sorted(set(df.Label) - (tr_lab & te_lab))),
                    "n_pairs_total": int(len(uniq)), "n_pairs_test": int(te_pairs.sum())}


def _parse_ts(raw, cic=False):
    """Flow-start timestamps. The CIC-IDS2017 release writes a 12-hour clock without an
    AM/PM marker; captures run 08:00-17:10, so hours 1-7 are afternoon hours."""
    ts = pd.to_datetime(pd.Series(raw).astype(str), errors="coerce", dayfirst=True, format="mixed")
    if cic:
        h = ts.dt.hour
        ts = ts + pd.to_timedelta(np.where(h.between(1, 7), 12, 0), unit="h")
    return ts


def cic_timestamps(meta):
    """Per-row flow-start timestamps of the cleaned CIC-IDS2017 rows, from the
    GeneratedLabelledFlows release (row-aligned to MachineLearningCVE for Monday-Wednesday,
    and after dropping its unlabelled rows for the Thursday/Friday files). Cached."""
    fp = PROC / "cic_timestamps.parquet"
    if not fp.is_file():
        import zipfile as _zf
        raw = RAW / "CIC-IDS2017" / "MachineLearningCVE"
        z = _zf.ZipFile(RAW / "CIC-IDS2017" / "GeneratedLabelledFlows.zip"); names = z.namelist()
        probe = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]
        parts = []
        for f in CIC_FILES:
            ml = pd.read_csv(raw / f, encoding="latin-1"); ml.columns = [c.strip() for c in ml.columns]
            cand = [n for n in names if n.endswith(".csv") and Path(n).name.lower() == f.lower()] or \
                   [n for n in names if n.endswith(".csv") and Path(n).name[:40].lower() == f[:40].lower()]
            g = pd.read_csv(z.open(cand[0]), encoding="latin-1", low_memory=False); g.columns = [c.strip() for c in g.columns]
            def eq(a, b):
                return len(a) == len(b) and all(np.array_equal(pd.to_numeric(a[c], errors="coerce").fillna(-1).to_numpy(),
                                                                pd.to_numeric(b[c], errors="coerce").fillna(-1).to_numpy()) for c in probe)
            if not eq(g, ml):
                g = g[g["Label"].notna() & (g["Label"].astype(str).str.strip() != "")].reset_index(drop=True)
                assert eq(g, ml), f"GeneratedLabelledFlows does not align with {f}"
            parts.append(pd.DataFrame({"src_file": f, "row_pos": np.arange(len(ml)), "ts": _parse_ts(g["Timestamp"], cic=True).to_numpy()}))
        PROC.mkdir(parents=True, exist_ok=True); pd.concat(parts, ignore_index=True).to_parquet(fp)
    t = pd.read_parquet(fp)
    return meta[["src_file", "row_pos"]].merge(t, on=["src_file", "row_pos"], how="left")["ts"].to_numpy()


def split_chrono(df, meta):
    """Chronological within-class, within-capture-day split on parsed flow-start
    timestamps: for every (class, day) stratum the earliest 70% of flows fit and the
    latest 30% are held out (ties at the one-minute resolution broken by capture-file
    order). Unlike split_temporal, which uses file order, every held-out flow starts no
    earlier than every fitting flow of its class and day."""
    if "timestamp_raw" in meta and meta.src_file.nunique() == 1:     # CIC-UNSW-NB15: timestamps in the release
        ts = _parse_ts(meta.timestamp_raw).to_numpy()
    else:
        ts = cic_timestamps(meta)
    n_nat = int(pd.isna(ts).sum())
    tsi = pd.Series(ts).astype("int64").to_numpy()                   # NaT -> int64 min
    tsi = np.where(pd.isna(ts), np.iinfo(np.int64).max, tsi)          # unparsed rows go last (held out)
    forder = np.lexsort((meta.row_pos.to_numpy(), meta.src_file.to_numpy()))
    frank = np.empty(len(df), int); frank[forder] = np.arange(len(df))
    order = np.lexsort((frank, tsi)); rank = np.empty(len(df), int); rank[order] = np.arange(len(df))
    key = df.Label.astype(str).to_numpy(); day = meta.capture_day.astype(str).to_numpy()
    strata = pd.Series(np.char.add(np.char.add(key, "|"), day))
    te_mask = np.zeros(len(df), bool)
    for _, pos in strata.groupby(strata).indices.items():
        pos = np.asarray(pos); pos = pos[np.argsort(rank[pos], kind="stable")]
        n_te = int(round(TEST_SIZE * len(pos)))
        if len(pos) >= 2 and n_te == 0: n_te = 1
        if n_te >= len(pos): n_te = len(pos) - 1
        te_mask[pos[len(pos) - n_te:]] = True
    tr_lab = set(df.Label[~te_mask]); te_lab = set(df.Label[te_mask])
    keep = df.Label.isin(tr_lab & te_lab).to_numpy()
    tr = df.index[(~te_mask) & keep]; te = df.index[te_mask & keep]
    return tr, te, {"split": "chrono", "split_seed": None, "test_hash": sha(np.sort(te.to_numpy())),
                    "dropped_classes": ";".join(sorted(set(df.Label) - (tr_lab & te_lab))), "rows_without_timestamp": n_nat}


def make_split(df, meta, design: str):
    if design.startswith("row"): return split_rows(df, int(design[3:]))
    if design == "temporal": return split_temporal(df, meta)
    if design == "chrono": return split_chrono(df, meta)
    if design == "hostpair": return split_hostpair(df, meta)
    raise ValueError(design)


def balanced(df, n, seed):
    r = np.random.default_rng(seed)
    return pd.concat([p.iloc[r.choice(len(p), min(len(p), n), replace=False)]
                      for _, p in df.groupby("Label", sort=True)]).sample(frac=1, random_state=seed)


def cap_test(test, seed):
    r = np.random.default_rng(seed); keep = []; full = test.Label.value_counts()
    for c, p in test.groupby("Label", sort=True):
        k = min(len(p), max(CAP_MIN, int(np.ceil(len(p) * KEEP_FRACTION))))
        keep.append(p.iloc[r.choice(len(p), k, replace=False)] if k < len(p) else p)
    sub = pd.concat(keep); kept = sub.Label.value_counts()
    return sub, {c: float(full[c]) / float(kept[c]) for c in kept.index}


def transform(train, tests, use_pca=False):
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA
    single = isinstance(tests, pd.DataFrame); tests = [tests] if single else list(tests)
    xtr = train.drop(columns=["Label"]).select_dtypes(include=[np.number]).astype(np.float32)
    xtes = [t.reindex(columns=xtr.columns).astype(np.float32) for t in tests]
    imp = SimpleImputer(strategy="median"); a = imp.fit_transform(xtr); bs = [imp.transform(x) for x in xtes]
    vt = VarianceThreshold(0.0); a = vt.fit_transform(a); bs = [vt.transform(b) for b in bs]
    names = [c for c, k in zip(xtr.columns, vt.get_support()) if k]
    if use_pca:
        sc = StandardScaler(); a = sc.fit_transform(a); bs = [sc.transform(b) for b in bs]
        k = max(1, min(len(names) // 2, len(a) - 1))
        pca = IncrementalPCA(n_components=k, batch_size=max(1000, 2 * k))
        a = pca.fit_transform(a); bs = [pca.transform(b) for b in bs]; names = [f"PC{i+1}" for i in range(k)]
    F = lambda m: pd.DataFrame(np.asarray(m, np.float32), columns=names).reset_index(drop=True)
    outs = [F(b) for b in bs]
    return F(a), (outs[0] if single else outs)


def budget_of(n):
    if n.endswith("_balanced") or n in FOUNDATION: return BALANCED
    return None


# ----------------------------------------------------------------------------- models
def make_model(name, seed):
    from sklearn.ensemble import RandomForestClassifier
    if name.startswith("rf_pruned"):
        return RandomForestClassifier(n_estimators=15, max_depth=8, max_features=20, random_state=seed, n_jobs=-1)
    if name.startswith("rf_strong"):
        return RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
    if name.startswith("xgb"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="mlogloss")
    if name.startswith("tabpfn"):
        return TabPFN25(seed)
    if name == "tabicl":
        return TabICLWrap(seed)
    raise ValueError(name)


TABPFN_CONFIG = {"n_estimators": 4, "balance_probabilities": False, "average_before_softmax": False, "device": "auto",
                 "ignore_pretraining_limits": False, "inference_precision": "auto", "fit_mode": "fit_preprocessors",
                 "memory_saving_mode": True, "n_jobs": -1}
TABPFN_EXPERT = {"CLASS_SHIFT_METHOD": "shuffle", "FEATURE_SHIFT_METHOD": "shuffle", "FINGERPRINT_FEATURE": True,
                 "MAX_NUMBER_OF_CLASSES": 10, "MAX_NUMBER_OF_FEATURES": 500, "MAX_NUMBER_OF_SAMPLES": 100000,
                 "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 30, "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 4,
                 "OUTLIER_REMOVAL_STD": "auto", "POLYNOMIAL_FEATURES": 'no', "SUBSAMPLE_SAMPLES": None,
                 "USE_SKLEARN_16_DECIMAL_PRECISION": False}
TABPFN_MANY_CLASS = {"alphabet_size": 5, "n_estimators": None, "n_estimators_redundancy": 4}
TABICL_CONFIG = {"n_estimators": 16, "norm_methods": ["none", "power"], "feat_shuffle_method": "latin", "class_shift": True,
                 "outlier_threshold": 4.0, "softmax_temperature": 0.9, "average_logits": True, "use_hierarchical": True,
                 "batch_size": 8, "use_amp": True, "model_path": None, "allow_auto_download": True,
                 "checkpoint_version": "tabicl-classifier-v1.1-0506.ckpt", "device": None, "n_jobs": None,
                 "verbose": False, "inference_config": None}


class TabPFN25:
    """Same construction as src.models.trainable.tabpfn.TabPFNModel(version='2.5', use_default_config=False)."""
    def __init__(self, seed):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        import tabpfn.inference_config as ic
        if not hasattr(ic, "default_classifier_preprocessor_configs"):
            for n in ("v2_5_classifier_preprocessor_configs", "v2_classifier_preprocessor_configs"):
                if hasattr(ic, n): ic.default_classifier_preprocessor_configs = getattr(ic, n); break
        expert = ic.InferenceConfig(PREPROCESS_TRANSFORMS=ic.default_classifier_preprocessor_configs(), **TABPFN_EXPERT)
        self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, **TABPFN_CONFIG, random_state=seed,
                                                                 inference_config=expert)
        self.seed = seed; self.ext = None
        from sklearn.preprocessing import LabelEncoder; self.le = LabelEncoder()

    def fit(self, x, y):
        y = self.le.fit_transform(y)
        if len(np.unique(y)) > TABPFN_EXPERT["MAX_NUMBER_OF_CLASSES"]:
            from tabpfn_extensions.many_class import ManyClassClassifier
            self.ext = ManyClassClassifier(self.model, random_state=self.seed, **TABPFN_MANY_CLASS); self.ext.fit(x, y)
        else:
            self.model.fit(x, y)
        self.classes_ = self.le.classes_

    def predict_proba(self, x):
        est = self.ext if self.ext is not None else self.model
        p = est.predict_proba(x)
        cls = np.asarray(est.classes_)
        out = np.zeros((len(x), len(self.classes_)), np.float32); out[:, cls] = p
        return out


class TabICLWrap:
    def __init__(self, seed):
        from tabicl import TabICLClassifier
        self.model = TabICLClassifier(random_state=seed, **TABICL_CONFIG)
        from sklearn.preprocessing import LabelEncoder; self.le = LabelEncoder()
    def fit(self, x, y):
        self.model.fit(x, self.le.fit_transform(y)); self.classes_ = self.le.classes_
    def predict_proba(self, x):
        p = self.model.predict_proba(x); cls = np.asarray(self.model.classes_)
        out = np.zeros((len(x), len(self.classes_)), np.float32); out[:, cls] = p
        return out


def chunked_proba(model, x, chunk=PREDICT_CHUNK):
    out = []; n = len(x); t0 = time.time()
    for s in range(0, n, chunk):
        out.append(np.asarray(model.predict_proba(x[s:s + chunk]), np.float32))
        if s == 0 and n > chunk:
            print(f"      first {min(chunk, n)} rows in {time.time()-t0:.0f}s -> ~{(time.time()-t0)*n/min(chunk,n)/60:.1f} min "
                  f"for {n:,} rows", flush=True)
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    return np.concatenate(out)


# ----------------------------------------------------------------------------- metrics
def weighted_confusion(y_true, y_pred, labs, weights):
    idx = {c: i for i, c in enumerate(labs)}; cm = np.zeros((len(labs), len(labs)))
    w = np.array([weights.get(a, 1.0) for a in y_true])
    np.add.at(cm, (np.array([idx[a] for a in y_true]), np.array([idx[b] for b in y_pred])), w)
    return cm


def metrics_from_cm(cm, labs):
    tp = np.diag(cm); row = cm.sum(1); col = cm.sum(0); N = cm.sum()
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = row > 0
    cov = N * tp.sum() - (row * col).sum()
    den = np.sqrt(max(N**2 - (col**2).sum(), 0) * max(N**2 - (row**2).sum(), 0))
    ben = [i for i, c in enumerate(labs) if str(c).lower() == "benign"]
    share = row / N
    rare = [i for i in range(len(labs)) if i not in ben and share[i] < RARE_SHARE and row[i] > 0]
    per = pd.DataFrame({"class": labs, "precision": prec, "recall": rec, "f1": f1, "support": row,
                        "rare": [i in rare for i in range(len(labs))]})
    return dict(accuracy=tp.sum() / N, macro_f1=f1[present].mean(), balanced_accuracy=rec[present].mean(),
                mcc=(cov / den if den > 0 else 0.0), rare_recall=(rec[rare].mean() if rare else np.nan),
                rare_precision=(prec[rare].mean() if rare else np.nan), n_rare=len(rare)), per, rare


def ap_metrics(y_true_idx, proba, weights_row, labs, rare_idx):
    """Weighted one-vs-rest average precision per class, plus macro summaries and the
    binary attack-vs-benign AUPRC/ROC-AUC (score = 1 - p(benign))."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    aps = {}
    for i, c in enumerate(labs):
        yb = (y_true_idx == i).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): aps[c] = np.nan; continue
        aps[c] = float(average_precision_score(yb, proba[:, i], sample_weight=weights_row))
    ben = [i for i, c in enumerate(labs) if str(c).lower() == "benign"]
    att = [i for i in range(len(labs)) if i not in ben]
    out = {"ap_" + str(c): v for c, v in aps.items()}
    out["macro_ap_rare"] = float(np.nanmean([aps[labs[i]] for i in rare_idx])) if rare_idx else np.nan
    out["macro_ap_attack"] = float(np.nanmean([aps[labs[i]] for i in att]))
    out["macro_ap_all"] = float(np.nanmean(list(aps.values())))
    if ben:
        yb = np.isin(y_true_idx, att).astype(int); s = 1.0 - proba[:, ben[0]]
        out["attack_auprc"] = float(average_precision_score(yb, s, sample_weight=weights_row))
        out["attack_rocauc"] = float(roc_auc_score(yb, s, sample_weight=weights_row))
    return out
