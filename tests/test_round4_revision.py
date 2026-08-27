from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "round4" / "raw"
DER = ROOT / "results" / "round4" / "derived"
GEN = ROOT / "results" / "round4" / "generated"


def test_completed_raw_grids() -> None:
    mc = pd.read_csv(RAW / "exp4_manyclass.csv")
    unsw = pd.read_csv(RAW / "exp5_size_by_balance_unsw.csv")
    rich = pd.read_csv(RAW / "exp6_axis2_datarich.csv")
    rich_pc = pd.read_csv(RAW / "exp6_axis2_datarich_perclass.csv")

    assert len(mc) == 40
    assert len(unsw) == 40
    assert len(rich) == 40
    assert len(rich_pc) == 360
    assert mc["wrapper_engaged"].astype(bool).all()
    assert set(mc["seed"]) == set(range(10))
    assert set(unsw["seed"]) == set(range(10))
    assert set(rich["seed"]) == set(range(10))


def test_completed_second_axis2_variant() -> None:
    summary = json.loads((DER / "round4_summary.json").read_text())
    check = summary["axis2_second_variant"]
    assert summary["input_checks"]["axis2_second_variant_complete"] is True
    assert summary["input_checks"]["axis2_second_variant_raw_records"] == 70
    assert check["status"] == "PASS"
    assert check["n_models"] == 6 and check["n_seeds"] == 10
    assert check["n_eval_per_seed"] == 300 and check["stage_a_all_clean"]
    assert np.isclose(check["anchor_macro_f1"], 0.4332027742226584)
    assert np.isclose(check["kimi_macro_f1"], 0.47318489242454714)
    assert np.isclose(check["kimi_delta"], 0.03998211820188868)
    assert np.isclose(check["kimi_p_adjusted"], 0.10546875)
    assert np.isclose(check["glm_delta"], 0.03282746585398405)
    assert np.isclose(check["glm_p_adjusted"], 0.234375)
    assert np.isclose(check["spearman_hosted_ordering_vs_cic"], 1.0)
    assert np.isclose(check["qwen_accuracy"], 0.419)
    assert np.isclose(check["qwen_macro_f1"], 0.39579892193686994)
    assert np.isclose(check["qwen_delta"], -0.03740385228578844)
    assert np.isclose(check["qwen_p_adjusted"], 0.22265625)
    assert check["qwen_wins"] == 2
    assert np.isclose(check["qwen_parse_rate"], 1.0)
    assert check["qwen_unparsed"] == 0
    assert np.isclose(check["total_cost_usd"], 293.68404155)
    assert np.isclose(summary["combined_axis2_total_cost"], 512.02591145)
    raw_dir = RAW / "axis2_second_variant"
    assert (raw_dir / "raw_generations.jsonl").is_file()
    assert sum(1 for line in (raw_dir / "raw_generations.jsonl").read_text().splitlines() if line.strip()) == 70


def test_threshold_sensitivity() -> None:
    summary = json.loads((DER / "round4_summary.json").read_text())
    threshold = summary["threshold"]
    assert threshold["n_cell_thresholds"] == 16
    assert threshold["all_negative"]
    assert threshold["all_same_three"]
    assert np.isclose(threshold["rho_min"], -0.4761904761904762)
    assert np.isclose(threshold["rho_max"], -0.4285714285714286)


def test_two_source_factorial_effects() -> None:
    d = pd.read_csv(DER / "factorial_two_source_contrasts.csv")
    expected = {
        ("CIC-IDS2017 V3", "balance at small budget"): 0.509131249220103,
        ("CIC-IDS2017 V3", "balance at large budget"): 0.2599285291258865,
        ("CIC-UNSW-NB15 V1", "balance at small budget"): 0.5791858496369546,
        ("CIC-UNSW-NB15 V1", "balance at large budget"): 0.3195285794823816,
    }
    for key, value in expected.items():
        row = d[(d["dataset"] == key[0]) & (d["contrast"] == key[1]) & (d["metric"] == "rare_recall")]
        assert len(row) == 1
        assert np.isclose(row.iloc[0]["mean_delta"], value)
        assert row.iloc[0]["wins"] == 10
        assert np.isclose(row.iloc[0]["p_holm_4"], 0.0078125)


def test_manyclass_and_axis2_references() -> None:
    summary = json.loads((DER / "round4_summary.json").read_text())
    assert np.isclose(summary["manyclass"]["spread"], 0.003029947127971111)
    rich = pd.DataFrame(summary["axis2_datarich"]["summary"])
    xgb = rich[(rich.reference == "development-rich") & (rich.learner == "XGBoost")]
    rf = rich[(rich.reference == "development-rich") & (rich.learner == "strong RF")]
    assert np.isclose(xgb.iloc[0].macro_f1, 0.9189766895195495)
    assert np.isclose(rf.iloc[0].macro_f1, 0.911166765910564)
    assert np.isclose(summary["current_axis2_total_cost"], 218.3418699)
    assert np.isclose(summary["combined_axis2_total_cost"], 512.02591145)


def test_revision_tables_exist() -> None:
    required = [
        "table_core_panel_round4.tex",
        "table_threshold_sensitivity_round4.tex",
        "table_factorial_means_round4.tex",
        "table_factorial_effects_round4.tex",
        "table_manyclass_round4.tex",
        "table_precision_aware_round4.tex",
        "table_rare_perclass_cic_v1_round4.tex",
        "table_cost_sensitivity_round4.tex",
        "table_axis2_datarich_round4.tex",
        "table_axis2_cost_round4.tex",
        "table_axis2_two_variants_round4.tex",
        "table_legacy_omitted_cells_round4.tex",
        "table_severity_vectors_round4.tex",
    ]
    for name in required:
        assert (GEN / name).is_file(), name


def test_qwen_rerun_audit_is_complete_and_localised() -> None:
    """Only the two interrupted Qwen3.5 jobs differ from the pre-rerun record."""
    audit = RAW / "axis2_second_variant" / "rerun_audit"
    canonical = RAW / "axis2_second_variant"
    assert (audit / "README.md").is_file()
    assert (audit / "qwen_seed2_seed3_comparison.csv").is_file()

    old = pd.read_csv(audit / "pre_rerun_results.csv").sort_values(["model", "seed"]).reset_index(drop=True)
    new = pd.read_csv(canonical / "results.csv").sort_values(["model", "seed"]).reset_index(drop=True)
    assert old[["model", "seed"]].equals(new[["model", "seed"]])
    compare_cols = [c for c in new.columns if c not in {"latency_median_s", "wall_min", "usd"}]
    changed = []
    for i in range(len(new)):
        if not old.loc[i, compare_cols].equals(new.loc[i, compare_cols]):
            changed.append((str(new.loc[i, "model"]), int(new.loc[i, "seed"])))
    assert set(changed) == {("Qwen3.5-397B-A17B", 2), ("Qwen3.5-397B-A17B", 3)}

    old_q = old[(old.model == "Qwen3.5-397B-A17B") & old.seed.isin([2, 3])]
    new_q = new[(new.model == "Qwen3.5-397B-A17B") & new.seed.isin([2, 3])]
    assert (old_q.parse_rate == 0).all() and (old_q.errors == 300).all()
    assert (new_q.parse_rate == 1).all() and (new_q.errors == 0).all()

    def records(path: Path) -> dict[tuple[str, int], dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {(str(r["model"]), int(r["seed"])): r for r in rows}

    old_raw = records(audit / "pre_rerun_raw_generations.jsonl")
    new_raw = records(canonical / "raw_generations.jsonl")
    assert old_raw.keys() == new_raw.keys()
    raw_changed = {key for key in new_raw if old_raw[key] != new_raw[key]}
    assert raw_changed == {("Qwen3.5-397B-A17B", 2), ("Qwen3.5-397B-A17B", 3)}
    for key in raw_changed:
        assert all(v is None for v in old_raw[key]["y_pred"])
        assert all(v is not None for v in new_raw[key]["y_pred"])
