#!/usr/bin/env python3
"""Generate the round-5 LaTeX table fragments from analysis/out (record-only) and, when
present, from the second-platform run summaries (analysis/mn5/*.csv).

python make_round5_tables.py --out paper/manuscript_round5/tables
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT_A = HERE / "out"; MN5 = HERE / "mn5"   # overridden by --record / --mn5
CELL = {"cic_ids2017_V1": "CIC-IDS2017 $V_1$", "cic_ids2017_V3": "CIC-IDS2017 $V_3$",
        "cic_unsw_V1": "CIC-UNSW-NB15 $V_1$", "cic_unsw_V2": "CIC-UNSW-NB15 $V_2$"}
MODEL = {"rf_pruned_full": "RF-P/L", "rf_strong_full": "RF-S/L", "xgb_full": "XGB/L", "rf_pruned_balanced": "RF-P/B",
         "rf_strong_balanced": "RF-S/B", "xgb_balanced": "XGB/B", "tabpfn_2.5": "TabPFN~2.5", "tabicl": "TabICL"}
ORDER = list(MODEL)
CROSS = Path("/mnt/user-data/uploads/ai_for_ids/results/derived/cost_crossovers_summary.csv")
if not CROSS.is_file(): CROSS = Path("results/derived/cost_crossovers_summary.csv")


def f(x, d=3):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{d}f}"


def w(path, s):
    path.write_text(s); print("wrote", path.name)


def table_baseline(O):
    b = pd.read_csv(OUT_A / "baseline_and_inversion_magnitude.csv")
    rows = []
    for _, r in b.iterrows():
        rows.append(f"{CELL[r.cell]} & {f(r.constant_benign_accuracy,4)} & {f(r.panel_accuracy_min,4)}--{f(r.panel_accuracy_max,4)} & "
                    f"{f(r.panel_headroom_min,2)}--{f(r.panel_headroom_max,2)} & {MODEL[r.accuracy_leader]} & {MODEL[r.recall_leader]} & "
                    f"${f(r.accuracy_cost_of_recall_leader,4)}$ & $+{f(r.rare_recall_gain_of_recall_leader,3)}$ & "
                    f"{r.missed_rare_per_100k_accuracy_leader:.0f}\\,/\\,{r.missed_rare_per_100k_recall_leader:.0f} & "
                    f"{r.false_alarms_per_100k_accuracy_leader:.0f}\\,/\\,{r.false_alarms_per_100k_recall_leader:.0f} \\\\")
    s = ("\\begin{tabular}{@{}lccclcrrrr@{}}\n\\toprule\n"
         "Cell & Constant-benign acc. & Panel accuracy range & Headroom retained & Acc.\\ leader & Recall leader & "
         "$\\Delta$acc. & $\\Delta R_{\\mathrm{rare}}$ & Missed rare / $10^5$ & False alarms / $10^5$ \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_baseline_magnitude.tex", s)


def table_metric_families(O):
    c = pd.read_csv(OUT_A / "metric_rank_correlations.csv")
    mets = [("balanced_accuracy", "Balanced accuracy"), ("gmean", "G-mean"), ("macro_f1", "Macro-$F_1$"),
            ("mcc", "MCC"), ("kappa", "Cohen's $\\kappa$")]
    rows = []
    for m, lab in mets:
        g = c[c.metric == m].set_index("cell")
        rows.append(lab + " & " + " & ".join(f"${g.loc[k].rho_with_accuracy:+.3f}$ / ${g.loc[k].rho_with_rare_recall:+.3f}$ & {MODEL[g.loc[k].leader]}"
                                           for k in CELL) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{v}}}" for v in CELL.values())
    sub = " & ".join(["$\\rho_s$ (acc.\\,/\\,$R_{\\mathrm{rare}}$) & leader"] * 4)
    s = ("\\begin{tabular}{@{}l" + "cl" * 4 + "@{}}\n\\toprule\nSummary & " + head + " \\\\\n & " + sub + " \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_metric_families.tex", s)


def table_extended(O):
    s = pd.read_csv(OUT_A / "metrics_extended_summary.csv")
    mn5 = MN5 / "panel_summary_row2026.csv"
    ap = pd.read_csv(mn5).set_index(["cell", "model"]) if mn5.is_file() else None
    rows = []
    for cell in CELL:
        rows.append(f"\\multicolumn{{{'8' if ap is not None else '6'}}}{{@{{}}l}}{{\\emph{{{CELL[cell]}}}}} \\\\")
        g = s[s.cell == cell].set_index("model")
        for m in ORDER:
            r = g.loc[m]
            line = (f"{MODEL[m]} & {f(r.accuracy_mean,4)} & {f(r.balanced_accuracy_mean)} & {f(r.mcc_mean)} & {f(r.gmean_mean)} & "
                    f"{f(r.rare_recall_mean)}")
            if ap is not None and (cell, m) in ap.index:
                a = ap.loc[(cell, m)]
                line += f" & {f(a.macro_ap_rare_mean)} & {f(a.attack_auprc_mean)}"
            elif ap is not None:
                line += " & -- & --"
            rows.append(line + " \\\\")
    cols = "Configuration & Accuracy & Bal.\\ acc. & MCC & G-mean & $R_{\\mathrm{rare}}$"
    if ap is not None: cols += " & Macro-AP (rare) & Attack AUPRC"
    s_ = ("\\begin{tabular}{@{}l" + "c" * (7 if ap is not None else 5) + "@{}}\n\\toprule\n" + cols + " \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_extended_metrics.tex", s_)


def table_effect_sizes(O):
    e = pd.read_csv(OUT_A / "effect_sizes_axis1.csv"); e = e[e.metric == "rare_recall"]
    rows = []
    for cell in CELL:
        for pf in ["tabpfn_2.5", "tabicl"]:
            g = e[(e.cell == cell) & (e.model == pf)].set_index("comparator")
            for cl in ORDER[:6]:
                r = g.loc[cl]
                rows.append(f"{CELL[cell]} & {MODEL[pf]} & {MODEL[cl]} & ${r.mean_delta:+.3f}$ & $[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]$ & "
                            f"{f(r.sd_diff)} & {r.d_z:+.1f} & {f(r.mde80)} & {int(r.wins)} & {r.p_holm_6:.3f} \\\\")
    s = ("\\begin{tabular}{@{}lllrcccccc@{}}\n\\toprule\nCell & Model & Comparator & $\\Delta R_{\\mathrm{rare}}$ & 95\\% CI & SD$_\\Delta$ & $d_z$ & MDE$_{80}$ & Wins & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_effect_sizes_axis1.tex", s)
    # compact summary per cell/model
    rows = []
    for cell in CELL:
        for pf in ["tabpfn_2.5", "tabicl"]:
            g = e[(e.cell == cell) & (e.model == pf)]
            rows.append(f"{CELL[cell]} & {MODEL[pf]} & {g.mean_delta.min():+.3f} to {g.mean_delta.max():+.3f} & {g.sd_diff.median():.3f} & "
                        f"{g.mde80.min():.3f}--{g.mde80.max():.3f} & {int((g.mean_delta.abs() < g.mde80).sum())} & {int((g.p_holm_6 < 0.05).sum())}/6 \\\\")
    s = ("\\begin{tabular}{@{}llcccc c@{}}\n\\toprule\nCell & Model & Range of $\\Delta R_{\\mathrm{rare}}$ & Median SD$_\\Delta$ & MDE$_{80}$ range & $|\\Delta|<$ MDE$_{80}$ & Holm $<0.05$ \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_effect_sizes_axis1_compact.tex", s)
    a2 = pd.read_csv(OUT_A / "effect_sizes_axis2.csv")
    rows = []
    for src in ["grouped CIC-IDS2017", "CIC-UNSW-NB15 V1"]:
        g = a2[a2.source == src].sort_values("mean_delta", ascending=False)
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{src.replace('V1', '$V_1$')}}}}} \\\\")
        for _, r in g.iterrows():
            name = r.model.replace("Qwen3.5-397B-A17B", "Qwen3.5").replace("Nemotron-3-Ultra-550b-a55b", "Nemotron-3 Ultra")
            rows.append(f"{name} & ${r.mean_delta:+.3f}$ & $[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]$ & {f(r.sd_diff)} & {r.d_z:+.2f} & {f(r.mde80)} & {int(r.wins)} & {r.p_bonferroni_6:.3f} \\\\")
    s = ("\\begin{tabular}{@{}lrcccccc@{}}\n\\toprule\nModel & $\\Delta F_1$ & 95\\% CI & SD$_\\Delta$ & $d_z$ & MDE$_{80}$ & Wins & $p_{\\mathrm{adj}}$ \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_effect_sizes_axis2.tex", s)
    fa = pd.read_csv(OUT_A / "effect_sizes_factorial.csv")
    rows = [f"{r.dataset.replace('V3','$V_3$').replace('V1','$V_1$')} & {r.contrast} & ${r.mean_delta:+.3f}$ & $[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]$ & {f(r.sd_diff)} & {r.d_z:+.1f} & {f(r.mde80)} & {int(r.wins)} \\\\"
            for _, r in fa.iterrows()]
    s = ("\\begin{tabular}{@{}llrccccc@{}}\n\\toprule\nCell & Contrast & $\\Delta R_{\\mathrm{rare}}$ & 95\\% CI & SD$_\\Delta$ & $d_z$ & MDE$_{80}$ & Wins \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_effect_sizes_factorial.tex", s)


def table_llm_applicability(O):
    a = pd.read_csv(OUT_A / "llm_applicability.csv")
    rows = []
    for src in ["grouped CIC-IDS2017", "CIC-UNSW-NB15 V1"]:
        g = a[a.source == src].sort_values("usd_per_1000")
        n = int(g.full_partition_rows.iloc[0])
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{src.replace('V1', '$V_1$')}: {n:,} held-out flows}}}} \\\\")
        for _, r in g.iterrows():
            name = r.model.replace("Qwen3.5-397B-A17B", "Qwen3.5").replace("Nemotron-3-Ultra-550b-a55b", "Nemotron-3 Ultra")
            rows.append(f"{name} & {r.usd_per_1000:.2f} & {r.lat:.2f} & {r.seconds_per_row_wall:.2f} & {r.usd_full_partition:,.0f} & {r.hours_full_partition_at_endpoint_latency:,.0f} & {r.hours_full_partition_at_campaign_rate:,.0f} \\\\")
    s = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\nModel & USD / 1,000 rows & Median latency (s) & Wall s / row & USD, full partition & Hours (latency) & Hours (campaign rate) \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_llm_applicability.tex", s)


def table_operating_points(O):
    r = pd.read_csv(OUT_A / "recommended_operating_points.csv")
    cross = pd.read_csv(CROSS) if CROSS.is_file() else None
    rows = []
    objs = [("maximise accuracy", "Highest accuracy"), ("fewest false alarms", "Fewest false alarms"), ("maximise rare-class coverage", "Highest rare-class coverage"),
            ("balanced aggregate (macro-F1)", "Balanced aggregate (macro-$F_1$)")]
    for cell in CELL:
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{CELL[cell]}}}}} \\\\")
        for key, lab in objs:
            x = r[(r.cell == cell) & (r.objective == key)].iloc[0]
            rho = "--"
            if cross is not None:
                c = cross[(cross.scenario == "uniform") & (cross.cell == cell) & (cross.model == x.model)]
                if len(c): rho = f"{c.rho_median_positive.iloc[0]:.2f}"
            rows.append(f"{lab} & {MODEL[x.model]} & {f(x.accuracy,4)} & {f(x.rare_recall)} & {f(x.macro_f1)} & {x.false_alarms_per_100k:.0f} / {x.missed_rare_per_100k:.0f} & {rho} \\\\")
    s = ("\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nObjective & Recommended & Accuracy & $R_{\\mathrm{rare}}$ & Macro-$F_1$ & FA / missed rare per $10^5$ & $\\rho^*$ vs RF-P/L \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_operating_points.tex", s)


# ----------------------------------------------------------------------------- second-platform tables
DESIGN = {"row2026": "Row (ref.)", "row2027": "Row 2027", "row2028": "Row 2028", "chrono": "Chronological", "temporal": "Capture-order", "hostpair": "Host-pair"}
PAPER_DESIGNS = ["row2026", "row2027", "row2028", "chrono", "hostpair"]   # the file-order holdout stays in the record only


def table_sessions(O):
    fp = MN5 / "session_structure.csv"
    if not fp.is_file(): return
    s = pd.read_csv(fp); s = s[s["class"].notna() & (s["class"].astype(str) != "nan")]
    rows = []
    for cell in ["cic_ids2017_V1", "cic_unsw_V1"]:
        g = s[s.cell == cell].sort_values("rows", ascending=False)
        if g.empty: continue
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{'CIC-IDS2017 (native taxonomy)' if cell.startswith('cic_ids') else 'CIC-UNSW-NB15'}}}}} \\\\")
        for _, r in g.iterrows():
            rows.append(f"{r['class']} & {int(r.rows):,} & {int(r.n_src_ip)} & {int(r.n_dst_ip)} & {int(r.n_host_pairs)} & {r.top_pair_share:.2f} & {int(r.n_capture_days)} \\\\")
    t = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\nClass & Flows (raw release) & Source hosts & Destination hosts & Host pairs & Top-pair share & Capture files \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_sessions.tex", t)


def table_leakage(O):
    fm, fl = MN5 / "split_manifest.csv", MN5 / "leakage_all.csv"
    if not fm.is_file(): return
    m = pd.read_csv(fm); lk = pd.read_csv(fl) if fl.is_file() else None
    rows = []
    for cell in ["cic_ids2017_V1", "cic_ids2017_V3", "cic_unsw_V1"]:
        g = m[m.cell == cell]
        if g.empty: continue
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{CELL[cell]}{' (also $V_2$ rows)' if cell=='cic_unsw_V1' else ''}}}}} \\\\")
        for _, r in g.iterrows():
            if r.design not in PAPER_DESIGNS or r.design in ("row2027", "row2028"): continue   # same duplicate shares as the reference; no NN audit
            cells = [DESIGN.get(r.design, r.design), f"{100*r.share_test_exact_dup:.2f}\\%"]
            cells.append(f"{100*r.test_rows_exact_dup_conflicting_label/r.n_test:.2f}\\%" if "test_rows_exact_dup_conflicting_label" in r and pd.notna(r.get("test_rows_exact_dup_conflicting_label")) else "--")
            for regime in ["large_natural", "balanced"]:
                v = None
                if lk is not None:
                    q = lk[(lk.cell == cell) & (lk.split == r.design) & (lk.regime == regime) & (lk["class"] == "__all_weighted__")]
                    if len(q): v = q.iloc[0]
                cells += ([f"{100*v.share_nn_below_1e6:.2f}\\%", f"{v.nn1_recall:.3f}"] if v is not None else ["--", "--"])
            dropped = r.dropped if isinstance(r.dropped, str) and r.dropped else "none"
            cells.append(dropped.replace("_", "\\_"))
            rows.append(" & ".join(cells) + " \\\\")
    t = ("\\begin{tabular}{@{}lcc cc cc l@{}}\n\\toprule\n & & & \\multicolumn{2}{c}{Large natural fit} & \\multicolumn{2}{c}{Balanced fit} & \\\\\n"
         "Design & Exact dup.\\ & Conflicting label & Near-identical & 1-NN acc.\\ & Near-identical & 1-NN acc.\\ & Dropped classes \\\\\n\\midrule\n"
         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_leakage.tex", t)


def table_splits(O):
    fp = MN5 / "panel_summary.csv"
    if not fp.is_file(): return
    ps = pd.read_csv(fp); rec = pd.read_csv(OUT_A / "metrics_extended_summary.csv")
    designs = [d for d in PAPER_DESIGNS if d in set(ps.split)]
    head = "Configuration & Record & " + " & ".join(DESIGN[d] for d in designs) + " & " + " & ".join(DESIGN[d] for d in designs) + " \\\\"
    rows = []
    for cell in CELL:
        rows.append(f"\\multicolumn{{{2+2*len(designs)}}}{{@{{}}l}}{{\\emph{{{CELL[cell]}}}}} \\\\")
        for mdl in ORDER:
            r0 = rec[(rec.cell == cell) & (rec.model == mdl)]
            line = [MODEL[mdl], f(r0.rare_recall_mean.iloc[0]) if len(r0) else "--"]
            for metric in ["rare_recall_mean", "macro_ap_rare_mean"]:
                for d in designs:
                    q = ps[(ps.cell == cell) & (ps.model == mdl) & (ps.split == d)]
                    line.append(f(q[metric].iloc[0]) if len(q) and metric in q else "--")
            rows.append(" & ".join(line) + " \\\\")
    t = ("\\begin{tabular}{@{}ll" + "c" * (2 * len(designs)) + "@{}}\n\\toprule\n & & \\multicolumn{" + str(len(designs)) + "}{c}{$R_{\\mathrm{rare}}$ (re-execution)} & \\multicolumn{"
         + str(len(designs)) + "}{c}{Macro-AP$_{\\mathrm{rare}}$} \\\\\n" + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w(O / "round5_splits.tex", t)
    inv = MN5 / "inversion_stats.csv"
    if inv.is_file():
        iv = pd.read_csv(inv); rows = []
        for cell in CELL:
            for d in designs:
                q = iv[(iv.cell == cell) & (iv.split == d)]
                if not len(q): continue
                r = q.iloc[0]
                rows.append(f"{CELL[cell]} & {DESIGN[d]} & {int(r.n_models)} & ${r.rho_acc_rare:+.3f}$ & {'yes' if r.same_three==True else ('no' if r.same_three==False else '--')} & "
                            f"{MODEL.get(r.accuracy_leader, r.accuracy_leader)} & {MODEL.get(r.rare_leader, r.rare_leader)} & "
                            f"{MODEL.get(r.get('aprare_leader', ''), r.get('aprare_leader', '--'))} & {MODEL.get(r.mcc_leader, r.mcc_leader)} \\\\")
        t = ("\\begin{tabular}{@{}llcccllll@{}}\n\\toprule\nCell & Design & Configs & $\\rho_s$(acc., $R_{\\mathrm{rare}}$) & Same three & Acc.\\ leader & $R_{\\mathrm{rare}}$ leader & Macro-AP$_{\\mathrm{rare}}$ leader & MCC leader \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        w(O / "round5_inversion_by_design.tex", t)


def main():
    global OUT_A, MN5, CROSS
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=str(OUT_A)); ap.add_argument("--mn5", default=str(MN5))
    ap.add_argument("--crossovers", default=str(CROSS)); a = ap.parse_args()
    OUT_A, MN5, CROSS = Path(a.record), Path(a.mn5), Path(a.crossovers)
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    table_baseline(O); table_metric_families(O); table_extended(O); table_effect_sizes(O)
    table_llm_applicability(O); table_operating_points(O)
    table_sessions(O); table_leakage(O); table_splits(O)


if __name__ == "__main__":
    main()
