"""
Write every manuscript table as a LaTeX fragment in tables/, reading only
results/summary.json (produced by analyze.py).  No number is typed by hand.
"""
import os, json
import numpy as np
from common import RES_DIR, TAB_DIR, CFG, DEP_YEARS, BASES

S = json.load(open(os.path.join(RES_DIR, "summary.json")))
YRS = [str(y) for y in DEP_YEARS]
BN = {"RandomForest": "Random forest", "HistGB": "Gradient boosting", "LogReg": "Logistic regression"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f4(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def n0(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def g(which, variant, regime, year, metric, stat="mean"):
    try:
        node = S[f"main_{which}"][variant][regime][str(year)]
    except KeyError:
        return None
    if metric.startswith("pooled_") or metric in ("frac_configs_below_nominal", "n_samples", "n_mal", "n_ben"):
        return node.get(metric)
    v = node.get(metric)
    return None if v is None else v.get(stat)


def write(name, body):
    with open(os.path.join(TAB_DIR, f"{name}.tex"), "w") as f:
        f.write(body.strip() + "\n")
    print("wrote", name)


def table(label, caption, colspec, header, rows, size="\\small", extra_after_header="", colsep=None):
    lines = [r"\begin{table}[htbp]", r"\centering", size]
    if colsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{colsep}}}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header
    lines.append(r"\midrule")
    if extra_after_header:
        lines.append(extra_after_header)
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ accounting
def t_accounting():
    A, E = S["accounting"]["real"], S["accounting"]["emu"]
    def row(label, key, total_only=False):
        r = [label, n0(A[key + "|malware"]) if key + "|malware" in A else "",
             n0(A[key + "|benign"]) if key + "|benign" in A else "", n0(A[key]), n0(E[key])]
        return " & ".join(r) + r" \\"
    rows = [
        " & ".join(["Records in the distributed CSV files", n0(A["raw_malware"]), n0(A["raw_benign"]),
                    n0(A["raw_records"]), n0(E["raw_records"])]) + r" \\",
        r"\midrule",
        row(r"Earliest-modification date not a calendar date (\texttt{1980-00-00})", "invalid_date"),
        row(r"Non-numeric feature value (\texttt{None})", "non_numeric_feature"),
        " & ".join(["Parsed records", "", "", n0(A["parsed_records"]), n0(E["parsed_records"])]) + r" \\",
        row("Duplicate SHA-256 (second occurrence)", "duplicate_sha256"),
        row("Date in 1980 (ZIP-format epoch)", "outside_2008_2020:epoch_1980"),
        row("Date before 2008", "outside_2008_2020:pre_2008"),
        row("Date after 2020", "outside_2008_2020:post_2020"),
        r"\midrule",
        row("Retained, 2008--2020", "kept"),
    ]
    O = S["overlap"]
    cap = (r"Complete filtering and de-duplication of the two KronoDroid subsets. Each record receives exactly one "
           r"status, applied in the order listed, so the columns sum to the raw totals. The emulator subset shares "
           rf"{n0(O['shared'])} of its {n0(O['emu_kept'])} retained applications ({pct(O['shared_frac_of_emu'])}\%) with "
           r"the real-device subset (Section~\ref{sec:emulator}).")
    write("tab_accounting", table("tab:accounting", cap, r">{\raggedright\arraybackslash}p{7.4cm}rrrr",
                                  [r" & \multicolumn{3}{c}{Real device} & Emulator \\",
                                   r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
                                   r"Stage & Malware & Benign & Total & Total \\"], rows, size="\\footnotesize"))


def t_composition():
    A = S["accounting"]["real"]["per_year"]
    rows = []
    for y in range(2008, 2021):
        v = A[str(y)]
        if y == 2013:
            rows.append(r"\midrule")
        rows.append(f"{y} & {n0(v['n'])} & {n0(v['malware'])} & {n0(v['benign'])} & {v['malware'] / v['n']:.3f} \\\\")
    cap = (r"Temporal composition of the retained real-device corpus. Years 2008--2012 form the training and "
           r"calibration window; 2013--2020 are the deployment years. The malware rate changes by an order of "
           r"magnitude between adjacent years.")
    write("tab_composition", table("tab:composition", cap, "lrrrr",
                                   [r"Year & Samples & Malware & Benign & Malware rate \\"], rows))


# ------------------------------------------------------------------ in-window
def t_inwindow():
    rows = []
    for b in BASES + ["All"]:
        if b == "All":
            node = S["main_real"]
            lab = r"\midrule All detectors"
        else:
            node = S["per_base_real"][b]
            lab = BN[b]
        sc = node["SCP"]["static"]["in-window"]
        mo = node["Mondrian"]["static"]["in-window"]
        rows.append(f"{lab} & {f3(sc['coverage']['mean'])} ({f3(sc['coverage']['sd'])}) & "
                    f"[{f3(sc['coverage']['min'])}, {f3(sc['coverage']['max'])}] & {f3(sc['cov_mal']['mean'])} & "
                    f"{f3(sc['cov_ben']['mean'])} & {f3(mo['cov_mal']['mean'])} & {f3(mo['cov_ben']['mean'])} & "
                    f"{f3(sc['frac_empty']['mean'])} & {f3(sc['pooled_acc_single'])} \\\\")
    n_test = S["meta_real"]["RandomForest"]["n_test"]
    cap = (rf"Validity on the held-out in-window test split ({n0(n_test)} samples from 2008--2012 never used for "
           r"fitting or calibration), nominal level 0.90. Coverage is mean (standard deviation) and range over seeds; "
           r"class-conditional columns give coverage among malware and among benign samples; the last two columns "
           r"give the empty-set rate and the accuracy of accepted (singleton) decisions under split conformal.")
    write("tab_inwindow", table("tab:inwindow", cap, "lcccccccc",
                                [r" & \multicolumn{4}{c}{Split conformal} & \multicolumn{2}{c}{Mondrian} & & \\",
                                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
                                 r"Detector & Coverage & Range & Malware & Benign & Malware & Benign & Empty & Acc. acc. \\"],
                                rows, size="\\footnotesize", colsep="4pt"))


# ------------------------------------------------------------------ coverage
def t_coverage():
    cols = [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate"), ("EnbPI", "batch_year")]
    rows = []
    iw = ["In-window (held-out)", f3(g("real", "SCP", "static", "in-window", "coverage")),
          f3(g("real", "Mondrian", "static", "in-window", "coverage")), "--", "--",
          f3(g("real", "EnbPI", "static", "in-window", "coverage"))]
    rows.append(" & ".join(iw) + r" \\")
    rows.append(r"\midrule")
    for y in YRS:
        rows.append(" & ".join([y] + [f3(g("real", v, r, y, "coverage")) for v, r in cols]) + r" \\")
    rows.append(r"\midrule")
    rows.append(" & ".join(["SD across configurations, 2020"] + [f"({f3(g('real', v, r, '2020', 'coverage', 'sd'))})" for v, r in cols]) + r" \\")
    def first_breach(v, r):
        for y in YRS:
            c = g("real", v, r, y, "coverage")
            if c is not None and c < 0.80:
                return y
        return "none"
    rows.append(" & ".join([r"First year with mean coverage $<0.80$"] + [first_breach(v, r) for v, r in cols]) + r" \\")
    cap = (r"Realised marginal coverage by deployment year, nominal $1-\alpha = 0.90$, mean over configurations "
           r"(three base detectors; ten seeds, five for EnbPI). Conformal risk control with the miscoverage loss "
           r"returns exactly the split-conformal threshold in every configuration and is not listed separately. "
           r"ACI receives each label before the next sample is scored; EnbPI receives a year's labels at the end of "
           r"that year (Section~\ref{sec:feedback} compares them at matched feedback).")
    write("tab_coverage", table("tab:coverage", cap, "lccccc",
                                [r"Year & SCP (= CRC) & Mondrian & Weighted CP & ACI & EnbPI \\"], rows))


def t_accepted():
    blocks = [("SCP", "static", "SCP"), ("ACI", "immediate", "ACI"), ("EnbPI", "batch_year", "EnbPI")]
    rows = []
    for y in ["in-window"] + YRS:
        cells = ["In-window" if y == "in-window" else y]
        for v, r, _ in blocks:
            if y == "in-window" and v == "ACI":
                cells += ["--", "--", "--"]
                continue
            rr = "static" if (y == "in-window" and v == "EnbPI") else r
            cells += [f3(g("real", v, rr, y, "coverage")), f3(g("real", v, rr, y, "frac_single")),
                      f3(g("real", v, rr, y, "pooled_acc_single"))]
        rows.append(" & ".join(cells) + r" \\")
        if y == "in-window":
            rows.append(r"\midrule")
    cap = (r"Marginal coverage against the quality of automatically accepted decisions. Acceptance is the share "
           r"of samples that receive a singleton set; accepted accuracy is the share of those singletons that contain "
           r"the true label (pooled over configurations). Marginal coverage counts every empty set as a miss, so it "
           r"falls far faster than the accuracy of the decisions the detector actually accepts.")
    write("tab_accepted", table("tab:accepted", cap, "lccccccccc",
                                [r" & \multicolumn{3}{c}{Split conformal} & \multicolumn{3}{c}{ACI (immediate feedback)} & \multicolumn{3}{c}{EnbPI (yearly batch)} \\",
                                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                                 r"Year & Coverage & Accept. & Acc. acc. & Coverage & Accept. & Acc. acc. & Coverage & Accept. & Acc. acc. \\"],
                                rows, size="\\footnotesize"))


def t_errors():
    blocks = [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate")]
    rows = []
    for y in YRS:
        cells = [y, n0(g("real", "SCP", "static", y, "n_ben"))]
        for v, r in blocks:
            cells += [f3(g("real", v, r, y, "pooled_fpr_acc")), f3(g("real", v, r, y, "pooled_fnr_acc")),
                      f3(g("real", v, r, y, "wrong_single_rate"))]
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Errors among automatically accepted decisions. FPR is the share of accepted benign samples "
           r"labelled malware; FNR is the share of accepted malware labelled benign (both pooled over configurations); "
           r"`Silent' is the share of the whole stream that receives a wrong singleton, the only miscoverage that "
           r"is not flagged for review. Benign counts per year are small after 2016, so the late-year FPR values "
           r"rest on few samples.")
    write("tab_errors", table("tab:errors", cap, "lrcccccccccccc",
                              [r" & & \multicolumn{3}{c}{Split conformal} & \multicolumn{3}{c}{Mondrian} & \multicolumn{3}{c}{Weighted CP} & \multicolumn{3}{c}{ACI} \\",
                               r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
                               r"Year & Benign & FPR & FNR & Silent & FPR & FNR & Silent & FPR & FNR & Silent & FPR & FNR & Silent \\"],
                              rows, size="\\scriptsize"))


def t_setcomp():
    blocks = [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate"), ("EnbPI", "batch_year")]
    rows = []
    for y in YRS:
        cells = [y]
        for v, r in blocks:
            cells += [f3(g("real", v, r, y, "frac_empty")), f3(g("real", v, r, y, "frac_full"))]
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Set composition: share of the stream receiving the empty set ($|C|=0$, rejection) and the full set "
           r"($|C|=2$, ambiguity). Under the operating policy both are escalated, so their sum is the mandatory "
           r"abstention rate.")
    write("tab_setcomp", table("tab:setcomp", cap, "lcccccccccc",
                               [r" & \multicolumn{2}{c}{SCP} & \multicolumn{2}{c}{Mondrian} & \multicolumn{2}{c}{Weighted CP} & \multicolumn{2}{c}{ACI} & \multicolumn{2}{c}{EnbPI} \\",
                                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
                                r"Year & Empty & Full & Empty & Full & Empty & Full & Empty & Full & Empty & Full \\"],
                               rows, size="\\footnotesize"))


def t_capture():
    rows = []
    for y in ["in-window"] + YRS:
        e = g("real", "SCP", "static", y, "frac_empty")
        nov = g("real", "SCP", "static", y, "novelty_auc_empty_vs_single")
        rows.append(" & ".join(["In-window" if y == "in-window" else y, f3(e),
                                f3(g("real", "SCP", "static", y, "pooled_argmax_acc_empty")),
                                f3(g("real", "SCP", "static", y, "pooled_argmax_acc_single")),
                                f3(g("real", "SCP", "static", y, "argmax_acc")),
                                f3(g("real", "SCP", "static", y, "pooled_err_recall_empty")),
                                "--" if y == "in-window" else f3(nov)]) + r" \\")
        if y == "in-window":
            rows.append(r"\midrule")
    cap = (r"What the split-conformal empty set contains. Base-detector accuracy is the accuracy of the "
           r"arg-max label, computed separately on rejected (empty-set) and accepted (singleton) samples; error "
           r"capture is the share of all arg-max errors that fall in the empty set; the last column is the AUC "
           r"with which a calibration-versus-deployment domain classifier ranks rejected above accepted samples "
           r"(0.5 means rejected samples are no more deployment-like than accepted ones).")
    write("tab_capture", table("tab:capture", cap, "lcccccc",
                               [r" & & \multicolumn{3}{c}{Base-detector accuracy} & & \\",
                                r"\cmidrule(lr){3-5}",
                                r"Year & Empty-set rate & Rejected & Accepted & All & Error capture & Novelty AUC \\"],
                               rows))


# ------------------------------------------------------------------ prior shift
def t_classcond():
    D = S["decomposition_real"]["SCP"]
    rows = []
    for y in YRS:
        rows.append(" & ".join([y, f3(g("real", "SCP", "static", y, "n_mal") / g("real", "SCP", "static", y, "n_samples")),
                                f3(g("real", "SCP", "static", y, "cov_mal")), f3(g("real", "SCP", "static", y, "cov_ben")),
                                f3(g("real", "Mondrian", "static", y, "cov_mal")), f3(g("real", "Mondrian", "static", y, "cov_ben")),
                                f"{D[y]['total']:+.3f}", f"{D[y]['composition']:+.3f}", f"{D[y]['within']:+.3f}"]) + r" \\")
    iw = S["main_real"]["SCP"]["static"]["in-window"]
    iwm = S["main_real"]["Mondrian"]["static"]["in-window"]
    rows.insert(0, " & ".join(["In-window", f3(iw["n_mal"] / iw["n_samples"]), f3(iw["cov_mal"]["mean"]),
                               f3(iw["cov_ben"]["mean"]), f3(iwm["cov_mal"]["mean"]), f3(iwm["cov_ben"]["mean"]),
                               "", "", ""]) + r" \\" + "\n" + r"\midrule")
    lo, hi = D["2013"]["envelope_lo"], D["2013"]["envelope_hi"]
    cap = (r"Class-conditional coverage and an exact decomposition of the change in split-conformal coverage "
           r"relative to the in-window held-out split. The composition term is the change that the observed shift "
           r"in malware share would produce if each class kept its in-window coverage; the within-class term is the "
           r"remainder, caused by changes in the class-conditional score distributions. The two terms sum exactly to "
           rf"the total. Under pure prior shift split-conformal coverage could only move within [{f3(lo)}, {f3(hi)}], "
           r"the range of its in-window class-conditional coverages.")
    write("tab_classcond", table("tab:classcond", cap, "lcccccccc",
                                 [r" & & \multicolumn{2}{c}{SCP coverage} & \multicolumn{2}{c}{Mondrian coverage} & \multicolumn{3}{c}{Change in SCP coverage} \\",
                                  r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
                                  r"Year & Malware share & Malware & Benign & Malware & Benign & Total & Composition & Within-class \\"],
                                 rows, size="\\footnotesize"))


def t_fixedprior():
    R = S["resampled_real"]
    rows = []
    iwv = ["In-window", "--", "--"] + [f3(g("real", f"FixedPrior{p}", "static", "in-window", "std_cov")) for p in CFG["fixed_priors"]] + ["--", "--"]
    rows.append(" & ".join(iwv) + r" \\")
    rows.append(r"\midrule")
    for y in YRS:
        cells = [y, f3(g("real", "LSW-oracle", "static", y, "coverage")), f3(g("real", "LSW-BBSE", "static", y, "coverage"))]
        cells += [f3(g("real", f"FixedPrior{p}", "static", y, "std_cov")) for p in CFG["fixed_priors"]]
        for p in CFG["resampled_stream"]["priors"]:
            cells.append(f3(R[str(p)]["ACI"][y]["coverage"]["mean"]))
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Controlled experiments that remove prior shift. Label-shift weighted conformal prediction reweights "
           r"calibration points to the deployment-year class prior, either the true prior (oracle) or one estimated "
           r"from unlabelled predictions (BBSE); both are exactly valid under pure prior shift. The fixed-prior "
           r"protocol holds the malware share at $\pi^\ast$ in calibration and in every deployment year, so no prior "
           r"shift remains by construction. The last two columns run ACI on deployment streams resampled within each "
           r"year to a fixed malware share (twenty replicates per configuration).")
    pri = CFG["fixed_priors"]
    write("tab_fixedprior", table("tab:fixedprior", cap, "lccccccc",
                                  [r" & \multicolumn{2}{c}{Label-shift weighted CP} & \multicolumn{3}{c}{Fixed-prior SCP, $\pi^\ast$} & \multicolumn{2}{c}{ACI, resampled $\pi^\ast$} \\",
                                   r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
                                   r"Year & Oracle prior & BBSE prior & " + " & ".join(f"{p:.2f}" for p in pri) + " & " +
                                   " & ".join(f"{p:.2f}" for p in CFG["resampled_stream"]["priors"]) + r" \\"],
                                  rows, size="\\footnotesize"))


# ------------------------------------------------------------------ feedback
REG = [("immediate", "Immediate"), ("delay7", "Delay 7 days"), ("delay30", "Delay 30 days"),
       ("delay90", "Delay 90 days"), ("delay180", "Delay 180 days"), ("delay365", "Delay 365 days"),
       ("batch_year", "End-of-year batch"), ("partial0.5", "Partial, 50\\%"), ("partial0.2", "Partial, 20\\%"),
       ("partial0.1", "Partial, 10\\%"), ("partial0.05", "Partial, 5\\%")]


def yr_stats(which, v, r, metric):
    vals = [g(which, v, r, y, metric) for y in YRS]
    vals = [x for x in vals if x is not None]
    return (min(vals), float(np.mean(vals)), max(vals)) if vals else (None, None, None)


def t_feedback():
    meths = [("ACI", r"ACI, $\gamma=0.01$"), ("ACI_g0.001", r"ACI, $\gamma=0.001$"), ("RecalSCP", "Recalibrated SCP"), ("EnbPI", "EnbPI")]
    rows = []
    for r, lab in REG:
        cells = [lab]
        for v, _ in meths:
            mn, _, mx = yr_stats("real", v, r, "coverage")
            ab = yr_stats("real", v, r, "abstention")[1]
            cells += [f"{f3(mn)}--{f3(mx)}" if mn is not None else "--", f3(ab)]
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Online procedures compared at matched label availability. For each feedback regime the table gives "
           r"the range of yearly mean coverage over 2013--2020 and the mean abstention (empty plus full sets). "
           r"Delays are calendar days from the sample date to the arrival of its label; partial feedback reveals "
           r"each label independently with the stated probability. Recalibrated SCP applies EnbPI's sliding-window "
           r"threshold to the single base detector.")
    hdr = (" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in meths) + r" \\")
    write("tab_feedback", table("tab:feedback", cap, "lcccccccc",
                                [hdr, r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
                                 r"Feedback & Coverage & Abst. & Coverage & Abst. & Coverage & Abst. & Coverage & Abst. \\"],
                                rows, size="\\footnotesize"))


def t_aciyears():
    rows = []
    for y in YRS:
        node = S["main_real"]["ACI"]["immediate"][y]
        c = node["coverage"]
        rows.append(" & ".join([y, n0(node["n_samples"]), f3(c["mean"]), f4(c["sd"]), f"[{f3(c['min'])}, {f3(c['max'])}]",
                                f"{node['frac_configs_below_nominal']:.2f}",
                                f3(S["gc_window_bound_real"][y])]) + r" \\")
    sp = S["aci_span_real"]["ACI|immediate"]
    rows.append(r"\midrule")
    rows.append(" & ".join(["2013--2020", n0(sp["T"]), f3(sp["stats"]["mean"]), f4(sp["stats"]["sd"]),
                            f"[{f3(sp['stats']['min'])}, {f3(sp['stats']['max'])}]", "--", f4(sp["gc_bound"])]) + r" \\")
    cap = (r"Year-by-year coverage of ACI with immediate feedback ($\gamma = 0.01$), across the thirty "
           r"configurations. The last column is the largest deviation of a window's empirical miscoverage from "
           r"$\alpha$ that the ACI update permits over a window of that length, $(1+\gamma)/(\gamma T)$; for the "
           r"whole 2013--2020 stream it is the long-run bound of \citet{gibbs2021adaptive}. The guarantee is "
           r"informative over the full stream and in large years, and vacuous in 2019 and 2020.")
    write("tab_aciyears", table("tab:aciyears", cap, "lrccccc",
                                [r"Period & $T$ & Mean coverage & SD & Range & Share $<0.90$ & Window bound \\"], rows))


# ------------------------------------------------------------------ burden
def t_burden():
    vs = [("SCP", "static", "Split conformal", 0), ("Mondrian", "static", "Mondrian", 0), ("WCP", "static", "Weighted CP", 0),
          ("ACI", "immediate", "ACI, immediate", 1), ("ACI", "partial0.1", r"ACI, 10\% labels", 1),
          ("RecalSCP", "batch_year", "Recalibrated SCP, batch", 1), ("EnbPI", "batch_year", "EnbPI, batch", 1)]
    rows = []
    for period, ys in (("2020", ["2020"]), ("Mean 2013--2020", YRS)):
        rows.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{period}}}}} \\")
        for v, r, lab, _ in vs:
            def m(metric):
                vals = [g("real", v, r, y, metric) for y in ys]
                vals = [x for x in vals if x is not None]
                return float(np.mean(vals)) if vals else None
            def pm(metric):
                vals = [g("real", v, r, y, metric) for y in ys]
                vals = [x for x in vals if x is not None]
                return float(np.mean(vals)) if vals else None
            ab, ex_q, ex_r = m("abstention"), m("extra_defer_retro"), m("extra_defer_rolling")
            if ab is None:
                continue
            lab_used = m("labels_used")
            acc_r = m("acc_after_rolling")
            rows.append(" & ".join([lab, f3(lab_used), f3(ab), f3(ex_q), f3(ab + ex_q), f3(ex_r), f3(ab + ex_r),
                                    f3(acc_r), f3(m("oracle_budget"))]) + r" \\")
        if period == "2020":
            rows.append(r"\midrule")
    cap = (r"Human-review burden under the operating policy (escalate every non-singleton set; defer additional "
           r"singletons, least confident first, until accepted accuracy reaches 0.90). `Labels' is the share of the "
           r"stream whose ground truth the procedure consumes for adaptation. The retrospective deferral uses the "
           r"period's own labels to decide where to stop and is a lower bound; the rolling policy fixes the deferral "
           r"threshold before each year from the previous year's labelled decisions and is implementable. The "
           r"last column is the oracle alert budget of Section~\ref{sec:metrics} (true-label ordering), kept only "
           r"for comparison. Totals add abstention and deferral, which are disjoint.")
    write("tab_burden", table("tab:burden", cap, "lcccccccc",
                              [r" & & & \multicolumn{2}{c}{Retrospective} & \multicolumn{3}{c}{Rolling policy} & \\",
                               r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
                               r"Procedure & Labels & Abstention & Deferral & Total & Deferral & Total & Acc. acc. & Oracle budget \\"],
                              rows, size="\\scriptsize"))


# ------------------------------------------------------------------ robustness
def t_emu():
    cols = [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate"), ("EnbPI", "batch_year")]
    rows = []
    rows.append(" & ".join(["In-window (held-out)", f3(g("emu", "SCP", "static", "in-window", "coverage")),
                            f3(g("emu", "Mondrian", "static", "in-window", "coverage")), "--", "--",
                            f3(g("emu", "EnbPI", "static", "in-window", "coverage")), f3(g("emu", "SCP", "static", "in-window", "pooled_acc_single")), "--"]) + r" \\")
    rows.append(r"\midrule")
    for y in YRS:
        rows.append(" & ".join([y] + [f3(g("emu", v, r, y, "coverage")) for v, r in cols] +
                               [f3(g("emu", "SCP", "static", y, "pooled_acc_single")),
                                f3(g("emu", "ACI", "immediate", y, "pooled_acc_single"))]) + r" \\")
    O = S["overlap"]
    cap = (r"Replication on the KronoDroid emulator subset. These are largely the same applications as in the "
           rf"real-device subset ({pct(O['shared_frac_of_emu'])}\% of retained emulator samples share a SHA-256 with a "
           r"real-device sample, and their static features are identical), with dynamic features collected "
           r"independently in an emulator; the table therefore tests robustness to the dynamic-feature collection "
           r"environment, not to an independent sample of applications.")
    write("tab_emu", table("tab:emu", cap, "lccccccc",
                           [r" & \multicolumn{5}{c}{Marginal coverage} & \multicolumn{2}{c}{Accepted accuracy} \\",
                            r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
                            r"Year & SCP & Mondrian & Weighted CP & ACI & EnbPI & SCP & ACI \\"], rows, size="\\footnotesize"))


def t_sensitivity():
    rows = []
    def worst(which, v, r):
        vals = [g(which, v, r, y, "coverage") for y in YRS]
        return min(vals)
    for a in CFG["alpha_sweep"]:
        sv = "SCP" if a == CFG["alpha"] else f"SCP_a{a}"
        mv = "Mondrian" if a == CFG["alpha"] else f"Mondrian_a{a}"
        av = "ACI" if a == CFG["alpha"] else f"ACI_a{a}"
        rows.append(" & ".join([rf"$\alpha={a:.2f}$", f3(g("real", sv, "static", "in-window", "coverage")),
                                f3(g("real", sv, "static", "2020", "coverage")), f3(worst("real", sv, "static")),
                                f3(g("real", mv, "static", "2020", "coverage")), f3(worst("real", mv, "static")),
                                f3(g("real", av, "immediate", "2020", "coverage")), f3(worst("real", av, "immediate")),
                                f3(np.mean([g("real", av, "immediate", y, "frac_full") for y in YRS]))]) + r" \\")
    cap = (r"Sensitivity to the nominal level. Columns give in-window held-out coverage, coverage in 2020 and the "
           r"worst yearly coverage over 2013--2020, and for ACI the mean full-set share; the target is $1-\alpha$ in "
           r"each row.")
    write("tab_alpha", table("tab:alpha", cap, "lcccccccc",
                             [r" & \multicolumn{3}{c}{Split conformal} & \multicolumn{2}{c}{Mondrian} & \multicolumn{3}{c}{ACI (immediate)} \\",
                              r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
                              r"Level & In-window & 2020 & Worst & 2020 & Worst & 2020 & Worst & Mean full \\"], rows, size="\\footnotesize"))
    # gamma sweep and feature groups
    rows = []
    for gm in CFG["aci"]["gamma_sweep"]:
        v = "ACI" if gm == CFG["aci"]["gamma"] else f"ACI_g{gm}"
        cov = [g("real", v, "immediate", y, "coverage") for y in YRS]
        rows.append(" & ".join([rf"$\gamma={gm}$", f3(min(cov)), f3(max(cov)),
                                f3(np.mean([g("real", v, "immediate", y, "frac_full") for y in YRS])),
                                f3(np.mean([g("real", v, "immediate", y, "frac_empty") for y in YRS])),
                                f3(np.mean([g("real", v, "immediate", y, "pooled_acc_single") for y in YRS]))]) + r" \\")
    cap = (r"Sensitivity of ACI (immediate feedback) to the step size $\gamma$: range of yearly coverage, mean "
           r"full-set and empty-set shares, and mean accepted accuracy over 2013--2020.")
    write("tab_gamma", table("tab:gamma", cap, "lccccc",
                             [r"Step size & Worst year & Best year & Mean full & Mean empty & Mean acc. acc. \\"], rows))
    A = S.get("ablation_real")
    if A:
        rows = []
        for grp, lab in (("static", "Static only"), ("dynamic", "Dynamic only")):
            iw = A[grp]["SCP"]["in-window"]
            rows.append(" & ".join([lab, f3(iw["argmax_acc"]["mean"]), f3(iw["coverage"]["mean"])] +
                                   [f3(A[grp]["SCP"][y]["coverage"]["mean"]) for y in ("2013", "2015", "2018", "2020")] +
                                   [f3(A[grp]["SCP"]["2020"]["frac_empty"]["mean"]), f3(A[grp]["SCP"]["2020"]["pooled_acc_single"]),
                                    f3(min(A[grp]["ACI"][y]["coverage"]["mean"] for y in YRS))]) + r" \\")
        allr = ["All features", f3(g("real", "SCP", "static", "in-window", "argmax_acc")), f3(g("real", "SCP", "static", "in-window", "coverage"))]
        allr += [f3(g("real", "SCP", "static", y, "coverage")) for y in ("2013", "2015", "2018", "2020")]
        allr += [f3(g("real", "SCP", "static", "2020", "frac_empty")), f3(g("real", "SCP", "static", "2020", "pooled_acc_single")),
                 f3(min(g("real", "ACI", "immediate", y, "coverage") for y in YRS))]
        rows.insert(0, " & ".join(allr) + r" \\")
        cap = (r"Feature-group ablation on the real-device subset (three detectors, five seeds for the ablated rows). "
               r"Static features (permissions, intent filters, metadata) are identical across the two KronoDroid "
               r"subsets for shared applications; dynamic features are system-call counts.")
        write("tab_features", table("tab:features", cap, "lccccccccc",
                                    [r" & \multicolumn{2}{c}{In-window} & \multicolumn{4}{c}{SCP coverage} & \multicolumn{2}{c}{SCP, 2020} & ACI \\",
                                     r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
                                     r"Features & Accuracy & Coverage & 2013 & 2015 & 2018 & 2020 & Empty & Acc. acc. & Worst year \\"],
                                    rows, size="\\footnotesize"))


def t_family():
    F = S.get("family_real")
    if not F:
        return
    rows = []
    for r in F["families"]:
        name = r["family"].replace("&", r"\&").replace("_", r"\_")
        rows.append(" & ".join([name, n0(r["n"]), n0(r["n_window"]), f3(r["coverage_split"]), f3(r["empty_split"]),
                                f3(r["wrong_single_split"]), f3(r["acc_single_split"]), f3(r["coverage_aci"]),
                                f3(r["full_aci"])]) + r" \\")
    cap = (rf"Malware-class coverage for the fourteen largest deployment families ({n0(F['top14_total'])} of "
           rf"{n0(F['n_deployment_malware'])} deployment malware samples), pooled over 2013--2020, means over three "
           r"detectors and five seeds. Every family also occurs in the 2008--2012 window (`Window' gives its count "
           r"there). For split conformal the miss rate splits into the empty-set rate and the silent wrong-singleton "
           r"rate.")
    write("tab_family", table("tab:family", cap, "lrrcccccc",
                              [r" & \multicolumn{2}{c}{Samples} & \multicolumn{4}{c}{Split conformal} & \multicolumn{2}{c}{ACI} \\",
                               r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
                               r"Family & Deployment & Window & Coverage & Empty & Silent & Acc. acc. & Coverage & Full \\"],
                              rows, size="\\footnotesize", colsep="4pt"))


# ------------------------------------------------------------------ diagnostic
def t_cwdv():
    D = S.get("diagnostic_real")
    if not D:
        return
    rows = []
    for b in BASES:
        c = D["calendar"][b]
        obs = sorted(set(x for x in c["observed_first_breach_080"] if x is not None))
        rows.append(" & ".join([BN[b], f3(c["velocity"]["mean"]),
                                f"{c['predicted_breach_080']['mean']:.2f}" if c["predicted_breach_080"] else "--",
                                ", ".join(str(o) for o in obs) if obs else "none"]) + r" \\")
    cap = (r"Calendar-time CWDV: coverage-decay velocity estimated inside the calibration window and the implied "
           r"first year of coverage below 0.80, against the observed first breach year of the same detector "
           r"(means over five seeds; observed years listed when seeds differ).")
    write("tab_cwdv", table("tab:cwdv", cap, "lccc",
                            [r"Base detector & Velocity (coverage/year) & Predicted breach of 0.80 & Observed \\"], rows))


PRED = [("CWDVD_KS", "CWDV-D (score KS), raw", "yes"), ("CWDVD_KS+rolling", "CWDV-D (score KS), rolling offset", "yes$^\\dagger$"),
        ("CWDVD_KS+LOYO", "CWDV-D (score KS), LOYO offset", "no"),
        ("CWDVD_DOM", "CWDV-D (domain AUC), raw", "yes"), ("CWDVD_DOM+rolling", "CWDV-D (domain AUC), rolling", "yes$^\\dagger$"),
        ("NE", "Non-empty rate $\\times$ accuracy, raw", "yes"), ("NE+rolling", "Non-empty rate, rolling", "yes$^\\dagger$"),
        ("ATC", "ATC, raw", "yes"), ("ATC+rolling", "ATC, rolling", "yes$^\\dagger$"),
        ("DoC", "DoC, raw", "yes"), ("DoC+rolling", "DoC, rolling", "yes$^\\dagger$"),
        ("Persistence", "Persistence (last year's coverage)", "yes$^\\dagger$")]


def t_diag(which="real"):
    D = S.get(f"diagnostic_{which}")
    if not D:
        return
    rows = []
    for k, lab, pro in PRED:
        P = D["predictors"][k]
        rows.append(" & ".join([lab, pro] + [f3(P["per_base"][b]["mae"]) for b in BASES] +
                               [f3(P["mae"]), f"{P['spearman_pooled']:+.3f}",
                                f"{P['per_base']['LogReg']['spearman']:+.3f}" if P["per_base"]["LogReg"]["spearman"] is not None else "--"]) + r" \\")
    cap = (r"Accuracy of label-free coverage predictors on the deployment years: mean absolute error per base "
           r"detector and pooled, pooled Spearman correlation between predicted and realised coverage, and the "
           r"within-configuration Spearman correlation for logistic regression. `Prosp.' marks predictors "
           r"that use only information available before the year is scored ($^\dagger$ also requires the labels of "
           r"earlier deployment years). The LOYO offset uses other deployment years, including later ones, and is "
           r"shown only to quantify the value of hindsight.")
    if which == "emu":
        cap = "Emulator subset. " + cap
    write(f"tab_diag_{which}", table(f"tab:diag{which}", cap, "llcccccc",
                                     [r" & & \multicolumn{4}{c}{Mean absolute error} & \multicolumn{2}{c}{Spearman} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}", r"Predictor & Prosp. & RF & GB & LR & Pooled & Pooled & LR \\"],
                                     rows, size="\\footnotesize", colsep="4pt"))


def t_warning(which="real"):
    D = S.get(f"diagnostic_{which}")
    if not D:
        return
    rows = []
    for k, lab, pro in PRED:
        w = D["predictors"][k]["warning"]["0.8"]
        rows.append(" & ".join([lab, str(w["breach"]), str(w["hit"]), str(w["miss"]), str(w["false_alarm"]),
                                f3(w["sensitivity"]), f3(w["false_alarm_rate"]), f3(w["precision"])]) + r" \\")
    cap = (r"Coverage predictors used as a warning system: an alarm is raised for a deployment year when predicted "
           r"split-conformal coverage is below 0.80, and a breach occurs when realised coverage is below 0.80. Counts "
           r"are over configuration-years (three detectors, five seeds, eight years).")
    write(f"tab_warning_{which}", table(f"tab:warning{which}", cap, "lrrrrccc",
                                        [r"Predictor & Breaches & Detected & Missed & False alarms & Sensitivity & FA rate & Precision \\"],
                                        rows, size="\\footnotesize", colsep="4pt"))


def t_repro():
    rows = [
        r"Tables~\ref{tab:accounting}, \ref{tab:composition}; overlap & \texttt{build\_dataset.py real|emu|overlap} & \texttt{accounting\_*.json}, \texttt{overlap\_real\_emu.json} \\",
        r"Tables~\ref{tab:inwindow}--\ref{tab:gamma} & \texttt{run\_audit.py real} & \texttt{audit\_real\_*.json} \\",
        r"EnbPI entries of Tables~\ref{tab:coverage}, \ref{tab:accepted}, \ref{tab:setcomp}, \ref{tab:feedback}, \ref{tab:burden} & \texttt{run\_enbpi.py real} & \texttt{enbpi\_real.json} \\",
        r"Table~\ref{tab:features} & \texttt{run\_feature\_ablation.py} & \texttt{ablation\_real.json} \\",
        r"Table~\ref{tab:emu} & \texttt{run\_audit.py emu}, \texttt{run\_enbpi.py emu} & \texttt{audit\_emu\_*.json}, \texttt{enbpi\_emu.json} \\",
        r"Table~\ref{tab:family} & \texttt{run\_family.py} & \texttt{family\_real.json} \\",
        r"Tables~\ref{tab:cwdv}--\ref{tab:diagemu} & \texttt{run\_diagnostic.py real|emu} & \texttt{diagnostic\_*.json} \\",
        r"Library-version sensitivity (Section~\ref{sec:implementation}) & \texttt{run\_audit.py real} under a second scikit-learn release; \texttt{compare\_versions.py} & \texttt{alt\_sklearn\_1.6.1/}, \texttt{version\_sensitivity.json} \\",
        r"\midrule",
        r"Aggregation of all per-run outputs & \texttt{analyze.py} & \texttt{summary.json} \\",
        r"Numbers quoted in the text & \texttt{make\_numbers.py} & \texttt{derived.json} \\",
        r"All tables and figures & \texttt{make\_tables.py}, \texttt{make\_figures.py} & \texttt{tables/}, \texttt{figures/} \\",
        r"Independent recomputation of headline results & \texttt{verify\_numbers.py} & console \\",
    ]
    cap = (r"Reproducibility index. The first block lists the experiment scripts and their per-run outputs; the "
           r"second lists the scripts that turn those outputs into every number quoted in the text and every table "
           r"and figure. \texttt{run\_all.sh} executes the whole chain except the library-version comparison, "
           r"which needs a second Python environment (see the package README).")
    write("tab_repro", table("tab:repro", cap, r">{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{5.0cm}",
                             [r"Content & Script & Output \\"], rows, size="\\footnotesize"))


if __name__ == "__main__":
    t_accounting(); t_composition(); t_inwindow(); t_coverage(); t_accepted(); t_errors(); t_setcomp()
    t_capture(); t_classcond(); t_fixedprior(); t_feedback(); t_aciyears(); t_burden(); t_emu()
    t_sensitivity(); t_family(); t_cwdv(); t_diag("real"); t_diag("emu"); t_warning("real"); t_repro()
