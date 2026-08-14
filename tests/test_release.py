from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "results" / "derived"


def test_required_artifacts_exist() -> None:
    required = [
        "verification_summary.json",
        "axis1_core_summary.csv",
        "axis1_fm_vs_classical.csv",
        "axis1_rank_inversion.csv",
        "size_balance_contrasts.csv",
        "axis2_closed_label_summary.csv",
        "axis2_paired_tests.csv",
        "release_file_manifest_sha256.csv",
        "manuscript_claim_verification.json",
        "table_size_balance_combined.tex",
        "table_axis2_mi_combined.tex",
        "table_axis2_calibration.tex",
        "table_dataset_cells.tex",
        "table_cap_validation.tex",
        "table_statistical_audit.json",
    ]
    for name in required:
        assert (DER / name).is_file(), name


def test_axis1_core_record_and_direction_counts() -> None:
    summary = json.loads((DER / "verification_summary.json").read_text())
    assert summary["axis1_core_records"] == 320
    assert summary["pairwise_counts"] == {
        "FM all 10": 40,
        "classical all 10": 4,
        "mixed": 4,
    }


def test_rank_inversion_values() -> None:
    rank = pd.read_csv(DER / "axis1_rank_inversion.csv").set_index("cell")
    expected = {
        "cic_ids2017_V1": -0.4523809523809524,
        "cic_ids2017_V3": -0.4523809523809524,
        "cic_unsw_V1": -0.4285714285714286,
        "cic_unsw_V2": -0.4761904761904762,
    }
    assert rank["same_three"].all()
    for cell, value in expected.items():
        assert np.isclose(rank.loc[cell, "spearman"], value)


def test_factorial_contrasts() -> None:
    d = pd.read_csv(DER / "size_balance_contrasts.csv").set_index("contrast")
    expected = {
        "balance at small": 0.509131249220103,
        "balance at large": 0.2599285291258865,
        "size at natural": 0.24213834585430946,
        "size at balanced": -0.007064374239907023,
    }
    for contrast, value in expected.items():
        assert np.isclose(d.loc[contrast, "mean_delta"], value)


def test_axis2_closed_label_numbers() -> None:
    d = pd.read_csv(DER / "axis2_closed_label_summary.csv").set_index("model")
    assert np.isclose(d.loc["RandomForest", "macro_f1_closed"], 0.8055125534282566)
    assert np.isclose(d.loc["Kimi-K3", "macro_f1_closed"], 0.8335459279893437)
    assert np.isclose(d.loc["Kimi-K3", "p_bonferroni_6"], 0.1171875)
    llm_cost = d.drop(index="RandomForest")["usd"].sum()
    assert np.isclose(llm_cost, 218.3418699)


def test_no_tabstar_in_strict_core() -> None:
    d = pd.read_csv(DER / "axis1_core_summary.csv")
    assert not d["model"].str.contains("tabstar", case=False).any()
    assert len(d) == 32


def test_independent_table_audit_passes() -> None:
    report = json.loads((DER / "table_statistical_audit.json").read_text())
    assert report["status"] == "PASS"
    # The standalone repository has 15 numerical table checks (Tables 2--16).
    # A 16th manuscript-literal check runs when the manuscript source is supplied
    # beside the repository in the complete submission package.
    assert len(report["checks"]) >= 15
    assert all(item["status"] == "PASS" for item in report["checks"])
