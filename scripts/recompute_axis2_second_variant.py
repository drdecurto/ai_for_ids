#!/usr/bin/env python3
"""Recompute and validate the completed second Axis-2 variant.

The experiment applies the frozen six-model, five-shot Axis-2 protocol to
CIC-UNSW-NB15 V1 over ten paired seeds.  This script rescales nothing and does
not trust the supplied aggregate metrics: it reconstructs accuracy, closed-label
macro-F1, parse rate, paired differences, exact sign-flip tests, bootstrap
intervals, costs, and cross-variant ordering from the released raw generations.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_ORDER = [
    "RandomForest", "Kimi-K3", "GLM-5.2", "MiniMax-M3",
    "Qwen3.5-397B-A17B", "DeepSeek-V4-Pro", "Nemotron-3-Ultra-550b-a55b",
]
DISPLAY = {
    "RandomForest": "Random Forest",
    "Kimi-K3": "Kimi-K3",
    "GLM-5.2": "GLM-5.2",
    "MiniMax-M3": "MiniMax-M3",
    "Qwen3.5-397B-A17B": "Qwen3.5",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "Nemotron-3-Ultra-550b-a55b": "Nemotron-3 Ultra",
}


def signflip(values: Sequence[float]) -> float:
    d = np.asarray(values, dtype=float)
    observed = abs(float(d.mean()))
    null = [abs(float(np.mean(d * np.asarray(s))))
            for s in itertools.product((-1.0, 1.0), repeat=len(d))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 50000,
                 seed: int = 0) -> tuple[float, float]:
    d = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assert_close(a: float, b: float, atol: float = 5e-10) -> None:
    if not np.isclose(a, b, atol=atol, rtol=0):
        raise AssertionError(f"{a!r} != {b!r}")


def _write_table(main: pd.DataFrame, second: pd.DataFrame, path: Path) -> None:
    """Create a one-row-per-model cross-source table for the manuscript."""
    main_idx = main.set_index("model")
    second_idx = second.set_index("model")
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Grouped CIC-IDS2017} & \multicolumn{3}{c}{CIC-UNSW-NB15 $V_1$} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & Acc./$F_1$ & $\Delta F_1$ & $p_{\rm adj}$ & Acc./$F_1$ & $\Delta F_1$ & $p_{\rm adj}$ & USD/1k (CIC/UNSW) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        m = main_idx.loc[model]
        u = second_idx.loc[model]
        if model == "RandomForest":
            md, mp, ud, up = "--", "--", "--", "--"
            c1 = c2 = 0.0
        else:
            md, mp = f"{float(m.mean_delta):+.3f}", f"{float(m.p_bonferroni_6):.4f}"
            ud, up = f"{float(u.mean_delta):+.3f}", f"{float(u.p_bonferroni_6):.4f}"
            c1 = 1000 * float(m.usd) / 2460
            c2 = float(u.usd_per_1000)
        lines.append(
            f"{DISPLAY[model]} & {float(m.accuracy):.3f}/{float(m.macro_f1_closed):.3f} & "
            f"{md} & {mp} & {float(u.accuracy):.3f}/{float(u.macro_f1_closed):.3f} & "
            f"{ud} & {up} & {c1:.2f}/{c2:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recompute(root: Path | str | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    raw_dir = root / "results" / "round4" / "raw" / "axis2_second_variant"
    out_dir = root / "results" / "round4" / "derived"
    gen_dir = root / "results" / "round4" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    source_results = pd.read_csv(raw_dir / "results.csv")
    source_tests = pd.read_csv(raw_dir / "paired_tests.csv")
    stage = pd.read_csv(raw_dir / "stage_a_probe.csv")
    chosen = json.loads((raw_dir / "stage_a_chosen_config.json").read_text())
    raw = read_jsonl(raw_dir / "raw_generations.jsonl")

    assert len(source_results) == 70
    assert len(raw) == 70
    assert source_results.duplicated(["seed", "model"]).sum() == 0
    assert set(source_results.seed.astype(int)) == set(range(10))
    assert set(source_results.model) == set(MODEL_ORDER)
    assert len(stage) == 6 and (stage.parsed.astype(int) == 5).all()
    assert (stage.clean.astype(int) == 5).all() and stage.frozen.astype(bool).all()
    assert set(chosen) == set(MODEL_ORDER) - {"RandomForest"}

    rows: list[dict] = []
    for record in raw:
        y_true = np.asarray(record["y_true"], dtype=str)
        y_pred_raw = record["y_pred"]
        assert len(y_true) == len(y_pred_raw) == 300
        labels = sorted(set(y_true))
        assert len(labels) == 10
        y_pred = np.asarray([p if p is not None else "__unparsed__" for p in y_pred_raw], dtype=str)
        rows.append({
            "seed": int(record["seed"]),
            "model": record["model"],
            "config": record.get("config", ""),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1_closed": float(f1_score(y_true, y_pred, labels=labels,
                                                average="macro", zero_division=0)),
            "parse_rate": float(np.mean([p is not None for p in y_pred_raw])),
            "unparsed": int(sum(p is None for p in y_pred_raw)),
            "n_eval": int(len(y_true)),
        })
    per_seed = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)

    # Verify that the independently reconstructed scores match the run ledger.
    ledger = source_results.rename(columns={"macro_f1": "macro_f1_closed"})
    merged = per_seed.merge(ledger[["seed", "model", "accuracy", "macro_f1_closed", "parse_rate", "usd", "latency_median_s"]],
                            on=["seed", "model"], suffixes=("_raw", "_ledger"), validate="one_to_one")
    for metric in ["accuracy", "macro_f1_closed", "parse_rate"]:
        if not np.allclose(merged[f"{metric}_raw"], merged[f"{metric}_ledger"], atol=1e-12, rtol=0):
            raise AssertionError(f"raw/ledger mismatch for {metric}")
    per_seed.to_csv(out_dir / "axis2_second_variant_per_seed.csv", index=False)

    rf = per_seed[per_seed.model == "RandomForest"].set_index("seed")
    summary_rows: list[dict] = []
    for model in MODEL_ORDER:
        g = per_seed[per_seed.model == model].set_index("seed")
        source = source_results[source_results.model == model]
        row = {
            "model": model,
            "config": str(source.config.dropna().iloc[0]) if source.config.notna().any() else "matched anchor",
            "accuracy": float(g.accuracy.mean()),
            "macro_f1_closed": float(g.macro_f1_closed.mean()),
            "parse_rate": float(g.parse_rate.mean()),
            "unparsed": int(g.unparsed.sum()),
            "usd": float(source.usd.sum()),
            "latency_median_s": float(source.latency_median_s.median()),
            "n_scored": int(g.n_eval.sum()),
        }
        row["usd_per_1000"] = 1000 * row["usd"] / row["n_scored"] if row["n_scored"] else 0.0
        if model == "RandomForest":
            row.update(mean_delta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       p_raw=np.nan, p_bonferroni_6=np.nan, wins=np.nan)
        else:
            d = g.loc[rf.index, "macro_f1_closed"].to_numpy() - rf.macro_f1_closed.to_numpy()
            lo, hi = bootstrap_ci(d)
            p = signflip(d)
            row.update(mean_delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
                       p_raw=p, p_bonferroni_6=min(1.0, 6 * p),
                       wins=int((d > 0).sum()))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "axis2_second_variant_summary.csv", index=False)

    paired = summary[summary.model != "RandomForest"][
        ["model", "accuracy", "macro_f1_closed", "parse_rate", "mean_delta",
         "ci_lo", "ci_hi", "wins", "p_raw", "p_bonferroni_6"]
    ].copy()
    paired.to_csv(out_dir / "axis2_second_variant_paired_tests.csv", index=False)

    # Verify independently reconstructed paired quantities against the supplied
    # run summary; differences in column naming only are permitted.
    source_tests = source_tests.set_index("model")
    for r in paired.itertuples():
        s = source_tests.loc[r.model]
        for a, b in [
            (r.accuracy, s.accuracy), (r.macro_f1_closed, s.macro_f1),
            (r.parse_rate, s.parse), (r.mean_delta, s.delta),
            (r.ci_lo, s.lo), (r.ci_hi, s.hi), (r.p_raw, s.p_raw),
            (r.p_bonferroni_6, s.p_bonf),
        ]:
            _assert_close(float(a), float(b), atol=5e-10)
        assert int(r.wins) == int(s.wins)

    main = pd.read_csv(root / "results" / "derived" / "axis2_closed_label_summary.csv")
    common_models = [m for m in MODEL_ORDER if m != "RandomForest"]
    main_order = main.set_index("model").loc[common_models, "macro_f1_closed"].rank(ascending=False)
    second_order = summary.set_index("model").loc[common_models, "macro_f1_closed"].rank(ascending=False)
    spearman = float(main_order.corr(second_order, method="spearman"))
    ordering = pd.DataFrame({
        "model": common_models,
        "cic_macro_f1": main.set_index("model").loc[common_models, "macro_f1_closed"].to_numpy(),
        "unsw_macro_f1": summary.set_index("model").loc[common_models, "macro_f1_closed"].to_numpy(),
        "cic_rank": main_order.to_numpy(),
        "unsw_rank": second_order.to_numpy(),
    })
    ordering["spearman_rank_correlation"] = spearman
    ordering.to_csv(out_dir / "axis2_cross_variant_ordering.csv", index=False)

    cost = summary[summary.model != "RandomForest"][
        ["model", "usd", "n_scored", "usd_per_1000", "latency_median_s"]
    ].copy()
    cost.to_csv(out_dir / "axis2_second_variant_cost.csv", index=False)

    _write_table(main, summary, gen_dir / "table_axis2_two_variants_round4.tex")

    result = {
        "status": "PASS",
        "dataset": "CIC-UNSW-NB15 V1",
        "n_models": 6,
        "n_seeds": 10,
        "n_eval_per_seed": 300,
        "n_raw_records": len(raw),
        "stage_a_all_clean": True,
        "anchor_accuracy": float(summary.loc[summary.model == "RandomForest", "accuracy"].iloc[0]),
        "anchor_macro_f1": float(summary.loc[summary.model == "RandomForest", "macro_f1_closed"].iloc[0]),
        "kimi_macro_f1": float(summary.loc[summary.model == "Kimi-K3", "macro_f1_closed"].iloc[0]),
        "kimi_delta": float(summary.loc[summary.model == "Kimi-K3", "mean_delta"].iloc[0]),
        "kimi_p_adjusted": float(summary.loc[summary.model == "Kimi-K3", "p_bonferroni_6"].iloc[0]),
        "glm_delta": float(summary.loc[summary.model == "GLM-5.2", "mean_delta"].iloc[0]),
        "glm_p_adjusted": float(summary.loc[summary.model == "GLM-5.2", "p_bonferroni_6"].iloc[0]),
        "qwen_accuracy": float(summary.loc[summary.model == "Qwen3.5-397B-A17B", "accuracy"].iloc[0]),
        "qwen_macro_f1": float(summary.loc[summary.model == "Qwen3.5-397B-A17B", "macro_f1_closed"].iloc[0]),
        "qwen_delta": float(summary.loc[summary.model == "Qwen3.5-397B-A17B", "mean_delta"].iloc[0]),
        "qwen_p_adjusted": float(summary.loc[summary.model == "Qwen3.5-397B-A17B", "p_bonferroni_6"].iloc[0]),
        "qwen_wins": int(summary.loc[summary.model == "Qwen3.5-397B-A17B", "wins"].iloc[0]),
        "qwen_parse_rate": float(summary.loc[summary.model == "Qwen3.5-397B-A17B", "parse_rate"].iloc[0]),
        "qwen_unparsed": int(summary.loc[summary.model == "Qwen3.5-397B-A17B", "unparsed"].iloc[0]),
        "spearman_hosted_ordering_vs_cic": spearman,
        "total_cost_usd": float(cost.usd.sum()),
    }
    (out_dir / "axis2_second_variant_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(recompute(), indent=2))
