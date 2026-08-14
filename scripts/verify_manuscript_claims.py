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
    close(float(axis2.loc[axis2.model != "RandomForest", "usd"].sum()), 218.3418699)

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

    report = {
        "status": "PASS",
        "strict_axis1_records": 320,
        "strict_axis1_summary_rows": 32,
        "pairwise_direction_counts": counts,
        "axis2_llm_cost_usd": float(axis2.loc[axis2.model != "RandomForest", "usd"].sum()),
        "raw_source_checksums_available": False,
    }
    out = DER / "manuscript_claim_verification.json"
    out.write_text(json.dumps(report, indent=2))

    # Refresh the complete results manifest after writing this verification file.
    import hashlib
    manifest_path = DER / "release_file_manifest_sha256.csv"
    manifest_rows = []
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and path != manifest_path:
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
