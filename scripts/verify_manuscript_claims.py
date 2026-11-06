#!/usr/bin/env python3
"""Verify the numerical claims that define the released manuscript.

This script is intentionally redundant with the unit tests: it emits a readable
verification report and fails immediately if a manuscript-defining quantity no
longer matches the persisted run record.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "results" / "derived"
R4DER = ROOT / "results" / "round4" / "derived"
AX1 = ROOT / "results" / "axis1"


def close(actual: float, expected: float, *, atol: float = 5e-10) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{actual!r} != {expected!r}")


def row(df: pd.DataFrame, **where: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in where.items():
        mask &= df[key].eq(value)
    out = df.loc[mask]
    if len(out) != 1:
        raise AssertionError(f"Expected one row for {where}, found {len(out)}")
    return out.iloc[0]



# --------------------------------------------------------------------------- round 5
R5 = ROOT / "results" / "round5"


def verify_round5() -> dict:
    """Round-5 quantities quoted in the manuscript (abstract, Findings, Sections 4.1, 4.4, 5.3,
    5.4, 6.1) checked against the released record-derived and re-execution summaries."""
    rec, mn5 = R5 / "record", R5 / "mn5"
    cells = ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1", "cic_unsw_V2"]

    # constant-benign baselines, headroom, accuracy cost and recall gain of the recall leader
    base = pd.read_csv(rec / "baseline_and_inversion_magnitude.csv").set_index("cell")
    close(round(base.loc["cic_ids2017_V1", "constant_benign_accuracy"], 3), 0.831, atol=5e-4)
    close(round(base.loc["cic_unsw_V1", "constant_benign_accuracy"], 3), 0.975, atol=5e-4)
    assert base.loc[["cic_ids2017_V1", "cic_ids2017_V3"], "panel_headroom_min"].min() >= 0.79
    assert base.loc[["cic_ids2017_V1", "cic_ids2017_V3"], "panel_headroom_max"].max() <= 0.995
    assert (base.loc[["cic_unsw_V1", "cic_unsw_V2"], "panel_headroom_min"] < 0).all()
    assert [round(x, 3) for x in (base.accuracy_cost_of_recall_leader.min(), base.accuracy_cost_of_recall_leader.max())] == [0.005, 0.011]
    assert [round(x, 2) for x in (base.rare_recall_gain_of_recall_leader.min(), base.rare_recall_gain_of_recall_leader.max())] == [0.22, 0.43]

    # metric families: balanced accuracy / G-mean follow rare recall, kappa / MCC follow accuracy
    corr = pd.read_csv(rec / "metric_rank_correlations.csv")
    rr = corr[corr.metric == "rare_recall"].set_index("cell").rho_with_accuracy
    assert all(-0.48 <= rr[c] <= -0.42 for c in cells)
    for m in ("balanced_accuracy", "gmean"):
        q = corr[corr.metric == m].set_index("cell")
        assert (q.rho_with_rare_recall >= 0.88).all() and (q.rho_with_accuracy < 0).all()
    for m in ("kappa", "mcc"):
        q = corr[corr.metric == m].set_index("cell")
        assert (q.loc[["cic_ids2017_V1", "cic_ids2017_V3"], "rho_with_accuracy"] == 1.0).all()
        assert (q.loc[["cic_unsw_V1", "cic_unsw_V2"], "rho_with_accuracy"] >= 0.76).all()
    assert (corr[corr.metric == "macro_f1"].rho_with_accuracy > 0).all()
    summ = json.loads((rec / "round5_record_summary.json").read_text())
    close(round(summ["inversion_permutation"]["p_two_sided"], 3), 0.015, atol=5e-4)

    # effect sizes and minimum detectable effects
    ax1 = pd.read_csv(rec / "effect_sizes_axis1.csv")
    rare = ax1[ax1.metric == "rare_recall"] if "metric" in ax1 else ax1
    assert len(rare) == 48 and (rare.wins == 10).sum() == 40
    close(round(rare.sd_diff.median(), 3), 0.019, atol=5e-4)
    assert (rare.mean_delta.abs() > rare.mde80).sum() == 44
    ax2 = pd.read_csv(rec / "effect_sizes_axis2.csv")
    kimi = ax2[ax2.model.str.startswith("Kimi")].sort_values("source", ascending=False)
    assert [round(x, 3) for x in kimi.mean_delta] == [0.028, 0.040]
    assert [round(x, 3) for x in kimi.mde80] == [0.032, 0.040]
    assert (kimi.p_bonferroni_6 > 0.05).all()
    fac = pd.read_csv(rec / "effect_sizes_factorial.csv")
    bal = fac[fac.contrast.str.startswith("balance")]
    assert [round(bal.mean_delta.min(), 2), round(bal.mean_delta.max(), 2)] == [0.26, 0.58]
    assert (fac[fac.contrast.str.startswith("size under balanced")].mean_delta <= 0).all()

    # cost and throughput extrapolations (corrected Nemotron record; both hour measures)
    app = pd.read_csv(rec / "llm_applicability.csv")
    nem = app[app.model.str.startswith("Nemotron") & app.source.str.contains("CIC-IDS2017")].iloc[0]
    close(round(nem.usd_per_1000, 2), 9.95, atol=5e-3)
    cic = app[app.source.str.contains("CIC-IDS2017")]; unsw = app[~app.source.str.contains("CIC-IDS2017")]
    assert 2300 <= cic.usd_full_partition.min() and cic.usd_full_partition.max() <= 28900
    assert 190 <= cic.hours_full_partition_at_endpoint_latency.min() and cic.hours_full_partition_at_endpoint_latency.max() <= 1120
    assert 260 <= cic.hours_full_partition_at_campaign_rate.min() and cic.hours_full_partition_at_campaign_rate.max() <= 3830
    assert 3500 <= unsw.usd_full_partition.min() and unsw.usd_full_partition.max() <= 46700
    assert 225 <= unsw.hours_full_partition_at_endpoint_latency.min() and unsw.hours_full_partition_at_endpoint_latency.max() <= 2700
    assert 280 <= unsw.hours_full_partition_at_campaign_rate.min() and unsw.hours_full_partition_at_campaign_rate.max() <= 14420

    # recommended operating points
    op = pd.read_csv(rec / "recommended_operating_points.csv")
    acc = op[op.objective == "maximise accuracy"].set_index("cell").model
    assert acc["cic_unsw_V2"] == "rf_strong_full" and (acc.drop("cic_unsw_V2") == "xgb_full").all()
    assert (op[op.objective == "fewest false alarms"].model == "rf_pruned_full").all()
    cov = op[op.objective == "maximise rare-class coverage"].set_index("cell").model
    assert cov["cic_ids2017_V1"] == cov["cic_ids2017_V3"] == "xgb_balanced" and cov["cic_unsw_V1"] == cov["cic_unsw_V2"] == "tabpfn_2.5"

    # re-execution: reference replication, chronological and host-pair designs
    inv = pd.read_csv(mn5 / "inversion_stats.csv").set_index(["split", "cell"]).rho_acc_rare
    assert [round(inv[("row2026", c)], 3) for c in cells] == [-0.429, -0.452, -0.429, -0.476]
    assert [round(inv[("chrono", c)], 3) for c in cells] == [-0.214, -0.119, -0.429, -0.476]
    assert [round(inv[("hostpair", c)], 2) for c in cells[2:]] == [-0.45, -0.48]
    cs = pd.read_csv(mn5 / "contrast_by_split.csv")
    nat = ["rf_pruned_full", "rf_strong_full", "xgb_full"]
    q = cs[(cs.split == "chrono") & (cs.metric == "rare_recall") & cs.comparator.isin(nat) & cs.model.isin(["tabpfn_2.5", "tabicl"])]
    assert len(q) == 24 and (q.wins == 10).all() and q.mean_delta.min() >= 0.14
    ps = pd.read_csv(mn5 / "panel_summary.csv")
    ch = ps[ps.split == "chrono"].set_index(["cell", "model"]).rare_recall_mean
    assert round(ch[("cic_ids2017_V1", "tabicl")], 3) == 0.764 and round(ch[("cic_ids2017_V3", "tabicl")], 3) == 0.823
    bc = pd.read_csv(mn5 / "temporal_boundary_check.csv")
    c = bc[(bc.design == "chrono") & (bc.ts_available == True)]
    assert len(c) == 62 and (c.test_before_fit_max == 0).all()
    return {"round5_axis1_contrasts": 48, "round5_all_ten_seed_contrasts": 40, "round5_chrono_strata_clean": 62,
            "round5_chrono_rho": [float(round(inv[("chrono", c)], 3)) for c in cells]}

def main() -> None:
    core = pd.read_csv(DER / "axis1_core_summary.csv")
    rank = pd.read_csv(DER / "axis1_rank_inversion.csv")
    pairs = pd.read_csv(DER / "axis1_fm_vs_classical.csv")
    sb = pd.read_csv(DER / "size_balance_contrasts.csv")
    op = pd.read_csv(DER / "operating_points_selected_summary.csv")
    cost = pd.read_csv(DER / "cost_crossovers_summary.csv")
    tabstar = pd.read_csv(DER / "tabstar_named_vs_generic.csv")
    many = pd.read_csv(DER / "manyclass_summary.csv")
    axis2 = pd.read_csv(DER / "axis2_closed_label_summary.csv")
    axis2_unsw = pd.read_csv(R4DER / "axis2_second_variant_summary.csv")
    axis2_cross = pd.read_csv(R4DER / "axis2_cross_variant_ordering.csv")
    mi = pd.read_csv(DER / "axis2_mi_llm_effects.csv")
    cap = pd.read_csv(DER / "cap_validation_matching_hash_summary.csv")
    provenance = pd.read_csv(AX1 / "provenance.csv")

    assert len(core) == 32
    assert set(core["model"]) == {
        "rf_pruned_full", "rf_strong_full", "xgb_full",
        "rf_pruned_balanced", "rf_strong_balanced", "xgb_balanced",
        "tabpfn_2.5", "tabicl",
    }
    assert not core["model"].str.contains("tabstar", case=False).any()

    expected_rank = {
        "cic_ids2017_V1": -0.4523809523809524,
        "cic_ids2017_V3": -0.4523809523809524,
        "cic_unsw_V1": -0.4285714285714286,
        "cic_unsw_V2": -0.4761904761904762,
    }
    for cell, expected in expected_rank.items():
        r = row(rank, cell=cell)
        close(float(r.spearman), expected)
        assert bool(r.same_three)

    counts = pairs["direction"].value_counts().to_dict()
    assert counts == {"FM all 10": 40, "classical all 10": 4, "mixed": 4}

    key_core = {
        ("cic_ids2017_V3", "xgb_full"): (0.998898, 0.743059),
        ("cic_ids2017_V3", "rf_pruned_full"): (0.994867, 0.334706),
        ("cic_ids2017_V3", "xgb_balanced"): (0.992953, 0.971874),
        ("cic_unsw_V2", "rf_strong_full"): (0.984479, 0.264371),
        ("cic_unsw_V2", "tabpfn_2.5"): (0.973171, 0.695497),
    }
    for (cell, model), (acc, rec) in key_core.items():
        r = row(core, cell=cell, model=model)
        close(float(r.accuracy), acc, atol=5e-7)
        close(float(r.rare_recall), rec, atol=5e-7)

    expected_sb = {
        "balance at small": 0.509131249220103,
        "balance at large": 0.2599285291258865,
        "size at natural": 0.24213834585430946,
        "size at balanced": -0.007064374239907023,
    }
    for contrast, expected in expected_sb.items():
        r = row(sb, contrast=contrast)
        close(float(r.mean_delta), expected)
        close(float(r.p_holm_4), 0.0078125)

    # Selected operating-point counts and alert burdens quoted in the paper.
    r = row(op, cell="cic_ids2017_V1", model="xgb_balanced")
    assert (int(r.recovered), int(r.high_recall_low_precision), int(r.missed)) == (8, 2, 1)
    close(float(r.alerts_per_tp_pooled), 1.35, atol=0.006)
    r = row(op, cell="cic_ids2017_V1", model="tabpfn_2.5")
    assert (int(r.recovered), int(r.high_recall_low_precision), int(r.missed)) == (6, 4, 1)
    close(float(r.alerts_per_tp_pooled), 1.65, atol=0.006)
    r = row(op, cell="cic_ids2017_V1", model="tabstar")
    assert (int(r.recovered), int(r.high_recall_low_precision), int(r.missed)) == (6, 3, 2)
    close(float(r.alerts_per_tp_pooled), 2.71, atol=0.006)

    # Cost crossovers quoted in the main text.
    expected_cost = {
        ("uniform", "cic_ids2017_V1", "xgb_full"): (0.040675, 24.6),
        ("uniform", "cic_unsw_V1", "xgb_full"): (0.386470, 2.6),
        ("severity weighted", "cic_ids2017_V1", "xgb_full"): (0.0220, 45.4),
        ("severity weighted", "cic_unsw_V1", "xgb_full"): (0.185168, 5.4),
        ("uniform", "cic_ids2017_V1", "tabpfn_2.5"): (2.029, 1 / 2.029),
        ("uniform", "cic_unsw_V1", "tabpfn_2.5"): (1.465, 1 / 1.465),
    }
    for (scenario, cell, model), (rho, reciprocal_or_rounded) in expected_cost.items():
        r = row(cost, scenario=scenario, cell=cell, model=model)
        close(float(r.rho_median_positive), rho, atol=0.001)
        if model == "xgb_full":
            close(float(r.one_over_rho), reciprocal_or_rounded, atol=0.12)

    r = row(tabstar, metric="rare_recall")
    close(float(r.mean_delta_named_minus_generic), 0.010475, atol=5e-7)
    close(float(r.p_raw), 0.75390625)
    assert float(many.rare_recall.max() - many.rare_recall.min()) < 0.0062

    rf = row(axis2, model="RandomForest")
    kimi = row(axis2, model="Kimi-K3")
    glm = row(axis2, model="GLM-5.2")
    close(float(rf.accuracy), 0.8040650406504065)
    close(float(rf.macro_f1_closed), 0.8055125534282566)
    close(float(kimi.accuracy), 0.8182926829268292)
    close(float(kimi.macro_f1_closed), 0.8335459279893437)
    close(float(kimi.mean_delta), 0.0280333745610871)
    close(float(kimi.p_bonferroni_6), 0.1171875)
    assert int(kimi.wins) == 9
    close(float(glm.mean_delta), 0.0132245451645882)
    close(float(glm.p_bonferroni_6), 0.12890625)
    main_axis2_cost = float(axis2.loc[axis2.model != "RandomForest", "usd"].sum())
    close(main_axis2_cost, 218.3418699)

    unsw_rf = row(axis2_unsw, model="RandomForest")
    unsw_kimi = row(axis2_unsw, model="Kimi-K3")
    unsw_glm = row(axis2_unsw, model="GLM-5.2")
    unsw_qwen = row(axis2_unsw, model="Qwen3.5-397B-A17B")
    close(float(unsw_rf.macro_f1_closed), 0.4332027742226584)
    close(float(unsw_kimi.macro_f1_closed), 0.47318489242454714)
    close(float(unsw_kimi.mean_delta), 0.03998211820188868)
    close(float(unsw_kimi.p_bonferroni_6), 0.10546875)
    assert int(unsw_kimi.wins) == 9
    close(float(unsw_glm.mean_delta), 0.03282746585398405)
    close(float(unsw_glm.p_bonferroni_6), 0.234375)
    assert int(unsw_glm.wins) == 8
    close(float(unsw_qwen.accuracy), 0.419)
    close(float(unsw_qwen.macro_f1_closed), 0.39579892193686994)
    close(float(unsw_qwen.mean_delta), -0.03740385228578844)
    close(float(unsw_qwen.p_bonferroni_6), 0.22265625)
    assert int(unsw_qwen.wins) == 2
    close(float(unsw_qwen.parse_rate), 1.0)
    assert int(unsw_qwen.unparsed) == 0
    second_axis2_cost = float(axis2_unsw.loc[axis2_unsw.model != "RandomForest", "usd"].sum())
    close(second_axis2_cost, 293.68404155)
    close(float(axis2_cross.loc[0, "spearman_rank_correlation"]), 1.0)
    combined_axis2_cost = main_axis2_cost + second_axis2_cost
    close(combined_axis2_cost, 512.02591145)

    r = row(mi, metric="macro_f1", regime="development_only")
    close(float(r.mean_delta), 0.011865, atol=5e-7)
    close(float(r.p_raw), 0.27734375)
    r = row(mi, metric="parse_rate", regime="development_only")
    close(float(r.mean_delta), -0.032817, atol=5e-7)
    close(float(r.p_raw), 0.005859375)

    # The cap-validation audit deliberately retains only matching parent hashes.
    assert set(cap.loc[cap.cell.str.startswith("cic_ids2017"), "n"]) == {9}
    assert set(cap.loc[cap.cell.str.startswith("cic_unsw"), "n"]) == {30}

    assert len(provenance) == 4
    assert set(provenance.rows) == {2522362, 3540241}
    assert provenance.raw_source_sha256.isna().all()

    round5 = verify_round5()

    report = {
        "status": "PASS",
        **round5,
        "strict_axis1_records": 320,
        "strict_axis1_summary_rows": 32,
        "pairwise_direction_counts": counts,
        "axis2_grouped_cic_llm_cost_usd": main_axis2_cost,
        "axis2_cic_unsw_llm_cost_usd": second_axis2_cost,
        "axis2_combined_llm_cost_usd": combined_axis2_cost,
        "axis2_cross_variant_spearman": float(axis2_cross.loc[0, "spearman_rank_correlation"]),
        "raw_source_checksums_available": False,
    }
    out = DER / "manuscript_claim_verification.json"
    out.write_text(json.dumps(report, indent=2))

    # Refresh the complete results manifest after writing this verification file.
    import hashlib
    manifest_path = DER / "release_file_manifest_sha256.csv"
    manifest_rows = []
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and path != manifest_path and path.name != ".DS_Store":
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            manifest_rows.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": h.hexdigest(),
            })
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
