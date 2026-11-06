from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "results" / "round5" / "record"
MN5 = ROOT / "results" / "round5" / "mn5"
STRICT = ["rf_pruned_full", "rf_strong_full", "xgb_full", "rf_pruned_balanced",
          "rf_strong_balanced", "xgb_balanced", "tabpfn_2.5", "tabicl"]


def test_record_layer_complete() -> None:
    per = pd.read_csv(REC / "metrics_extended_per_seed.csv")
    assert len(per) == 320 and set(per.model) == set(STRICT) and set(per.seed) == set(range(10))
    summ = pd.read_csv(REC / "metrics_extended_summary.csv")
    assert len(summ) == 32
    # accuracy and MCC select the same leader in every cell; balanced accuracy and rare recall too
    for cell, g in summ.groupby("cell"):
        assert g.loc[g.accuracy_mean.idxmax()].model == g.loc[g.mcc_mean.idxmax()].model
        assert g.loc[g.balanced_accuracy_mean.idxmax()].model == g.loc[g.rare_recall_mean.idxmax()].model


def test_baseline_and_metric_families() -> None:
    b = pd.read_csv(REC / "baseline_and_inversion_magnitude.csv").set_index("cell")
    assert np.isclose(b.loc["cic_ids2017_V1", "constant_benign_accuracy"], 0.831158, atol=1e-5)
    assert np.isclose(b.loc["cic_unsw_V1", "constant_benign_accuracy"], 0.974696, atol=1e-5)
    assert (b.accuracy_cost_of_recall_leader.between(0.004, 0.012)).all()
    assert (b.rare_recall_gain_of_recall_leader.between(0.21, 0.44)).all()
    c = pd.read_csv(REC / "metric_rank_correlations.csv")
    mcc = c[c.metric == "mcc"].set_index("cell")
    assert np.isclose(mcc.loc["cic_ids2017_V1", "rho_with_accuracy"], 1.0)
    assert np.isclose(mcc.loc["cic_ids2017_V3", "rho_with_accuracy"], 1.0)
    bal = c[c.metric == "balanced_accuracy"]
    assert (bal.rho_with_rare_recall >= 0.95).all() and (bal.rho_with_accuracy < 0).all()
    s = json.loads((REC / "round5_record_summary.json").read_text())
    assert s["inversion_permutation"]["p_two_sided"] < 0.05


def test_effect_sizes_and_mde() -> None:
    e = pd.read_csv(REC / "effect_sizes_axis1.csv")
    rr = e[e.metric == "rare_recall"]
    assert len(rr) == 48 and (rr.p_holm_6 < 0.05).all()
    assert int((rr.mean_delta.abs() < rr.mde80).sum()) == 4
    a2 = pd.read_csv(REC / "effect_sizes_axis2.csv")
    assert len(a2) == 12
    kimi = a2[a2.model == "Kimi-K3"].set_index("source")
    assert np.isclose(kimi.loc["grouped CIC-IDS2017", "mean_delta"], 0.0280, atol=5e-4)
    assert np.isclose(kimi.loc["CIC-UNSW-NB15 V1", "mean_delta"], 0.0400, atol=5e-4)
    assert (kimi.mean_delta <= kimi.mde80 + 1e-3).all()
    fa = pd.read_csv(REC / "effect_sizes_factorial.csv")
    assert len(fa) == 8 and (fa[fa.contrast.str.startswith("balance")].d_z > 2).all()


def test_mn5_layer_if_present() -> None:
    if not (MN5 / "panel_summary.csv").is_file():
        return
    ps = pd.read_csv(MN5 / "panel_summary.csv")
    ref = ps[ps.split == "row2026"]
    assert set(ref.model) == set(STRICT) and len(ref) == 32
    rep = pd.read_csv(MN5 / "platform_replication.csv")
    assert rep.d_rare_recall.abs().max() < 0.03 and rep.d_accuracy.abs().max() < 0.005
    inv = pd.read_csv(MN5 / "inversion_stats.csv")
    assert (inv.rho_acc_rare < 0).all()
    assert inv[inv.split == "row2026"].same_three.astype(bool).all()


def test_chronological_split_is_chronological() -> None:
    """The timestamp-ordered design must hold out, in every class x capture-day stratum,
    only flows that start no earlier than every fitting flow (temporal_check.py)."""
    fp = MN5 / "temporal_boundary_check.csv"
    if not fp.is_file():
        return
    b = pd.read_csv(fp)
    c = b[(b.design == "chrono") & (b.ts_available == True)]
    assert len(c) >= 60 and (c.test_before_fit_max == 0).all() and (c.overlap_minutes == 0).all()
    # the file-order holdout kept in the record is documented as not chronological for benign
    t = b[(b.design == "temporal") & (b.ts_available == True)]
    assert (t.test_before_fit_max > 0).any()
