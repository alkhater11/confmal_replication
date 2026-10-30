"""
All manuscript figures, MATLAB-style (default MATLAB colour order, boxed axes,
inward ticks, light solid grid, sans-serif type), saved as PDF and EPS.
Reads results/summary.json and the per-run diagnostic file only.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cycler import cycler
from common import RES_DIR, FIG_DIR, CFG, DEP_YEARS, BASES

ML = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560),
      (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330), (0.6350, 0.0780, 0.1840)]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "black", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#dedede", "grid.linestyle": "-", "grid.linewidth": 0.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.prop_cycle": cycler(color=ML), "lines.linewidth": 1.3, "lines.markersize": 4.5,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black", "legend.framealpha": 1.0,
    "legend.borderpad": 0.4, "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
S = json.load(open(os.path.join(RES_DIR, "summary.json")))
YRS = [str(y) for y in DEP_YEARS]
X = np.array(DEP_YEARS)
MK = ["o", "s", "d", "^", "v", ">", "<", "p"]
NAMES = {("SCP", "static"): "Split conformal (= CRC)", ("Mondrian", "static"): "Mondrian",
         ("WCP", "static"): "Weighted CP", ("ACI", "immediate"): "ACI (immediate labels)",
         ("EnbPI", "batch_year"): "EnbPI (yearly labels)"}
MAIN = list(NAMES.keys())


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.eps"), bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def series(which, v, r, metric, stat="mean", base=None):
    node = S[f"per_base_{which}"][base] if base else S[f"main_{which}"]
    out = []
    for y in YRS:
        try:
            m = node[v][r][y]
            val = m.get(metric) if metric.startswith("pooled_") else (m[metric][stat] if metric in m else None)
        except KeyError:
            val = None
        out.append(np.nan if val is None else val)
    return np.array(out, dtype=float)


def style(ax, ylabel=None, ylim=None, xlabel="Deployment year"):
    ax.set_xticks(X)
    ax.set_xlim(X[0] - 0.5, X[-1] + 0.5)
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def fig_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for i, key in enumerate(MAIN):
        v, r = key
        c = series("real", v, r, "coverage")
        a = series("real", v, r, "pooled_acc_single")
        axes[0].plot(X, c, marker=MK[i], color=ML[i], mfc="none", label=NAMES[key])
        axes[1].plot(X, a, marker=MK[i], color=ML[i], mfc="none", label=NAMES[key])
    for ax in axes:
        ax.axhline(0.90, color="black", ls="--", lw=0.9)
    lo = np.nanmin([np.nanmin(series("real", v, r, "pooled_acc_single")) for v, r in MAIN])
    lo = min(0.6, np.floor(lo * 10) / 10)
    style(axes[0], "Marginal coverage", (0.0, 1.02))
    style(axes[1], "Accuracy of accepted (singleton) decisions", (lo, 1.0))
    axes[0].set_title("(a) Marginal coverage")
    axes[1].set_title("(b) Accepted-decision accuracy")
    axes[0].legend(loc="lower left", ncol=1)
    fig.tight_layout()
    save(fig, "fig1_coverage_accepted")


def fig_bybase():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, b in zip(axes, BASES):
        for i, key in enumerate(MAIN):
            v, r = key
            ax.plot(X, series("real", v, r, "coverage", base=b), marker=MK[i], color=ML[i], mfc="none",
                    label=NAMES[key], ms=3.5, lw=1.1)
        ax.axhline(0.90, color="black", ls="--", lw=0.9)
        style(ax, None, (0.0, 1.02))
        ax.set_title({"RandomForest": "Random forest", "HistGB": "Gradient boosting", "LogReg": "Logistic regression"}[b])
        ax.set_xticks(X[::2])
    axes[0].set_ylabel("Marginal coverage")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig2_coverage_by_base")


def fig_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    w = 0.7
    for ax, (v, r, title) in zip(axes, [("SCP", "static", "(a) Split conformal"), ("ACI", "immediate", "(b) ACI, immediate labels")]):
        single = series("real", v, r, "frac_single")
        wrong = series("real", v, r, "wrong_single_rate")
        empty = series("real", v, r, "frac_empty")
        full = series("real", v, r, "frac_full")
        correct = single - wrong
        b1 = ax.bar(X, correct, w, color=ML[0], edgecolor="black", lw=0.4, label="Correct singleton")
        b2 = ax.bar(X, wrong, w, bottom=correct, color=ML[6], edgecolor="black", lw=0.4, label="Wrong singleton (silent)")
        b3 = ax.bar(X, empty, w, bottom=correct + wrong, color=ML[2], edgecolor="black", lw=0.4, label="Empty set (rejected)")
        b4 = ax.bar(X, full, w, bottom=correct + wrong + empty, color="#bfbfbf", edgecolor="black", lw=0.4,
                    hatch="////", label="Full set (ambiguous)")
        style(ax, None, (0, 1.0))
        ax.set_title(title)
    axes[0].set_ylabel("Share of deployment stream")
    axes[1].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig3_set_composition")


def fig_capture():
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    acc_e = series("real", "SCP", "static", "pooled_argmax_acc_empty")
    acc_s = series("real", "SCP", "static", "pooled_argmax_acc_single")
    cap = series("real", "SCP", "static", "pooled_err_recall_empty")
    ax.plot(X, acc_s, marker="o", color=ML[0], mfc="none", label="Detector accuracy, accepted")
    ax.plot(X, acc_e, marker="s", color=ML[1], mfc="none", label="Detector accuracy, rejected")
    ax.plot(X, cap, marker="d", color=ML[4], mfc="none", ls="-.", label="Share of errors captured by rejection")
    style(ax, "Proportion", (0.0, 1.02))
    ax.legend(loc="lower left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig4_rejection_content")


def fig_prior():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    D = S["decomposition_real"]["SCP"]
    comp = np.array([D[y]["composition"] for y in YRS])
    within = np.array([D[y]["within"] for y in YRS])
    tot = np.array([D[y]["total"] for y in YRS])
    w = 0.36
    axes[0].bar(X - w / 2, comp, w, color=ML[2], edgecolor="black", lw=0.4, label="Class composition")
    axes[0].bar(X + w / 2, within, w, color=ML[0], edgecolor="black", lw=0.4, label="Within-class change")
    axes[0].plot(X, tot, "k_", ms=10, mew=1.6, label="Total change")
    axes[0].axhline(0, color="black", lw=0.6)
    style(axes[0], "Change in coverage vs. in-window", (-0.55, 0.05))
    axes[0].set_title("(a) Decomposition, split conformal")
    axes[0].legend(loc="lower left", fontsize=6.8)
    curves = [("SCP", "static", "coverage", "Split conformal (natural prior)"),
              ("LSW-oracle", "static", "coverage", "Label-shift CP, oracle prior"),
              ("LSW-BBSE", "static", "coverage", "Label-shift CP, BBSE prior"),
              ("FixedPrior0.1", "static", "std_cov", r"Fixed prior $\pi^\ast$=0.10"),
              ("FixedPrior0.5", "static", "std_cov", r"Fixed prior $\pi^\ast$=0.50"),
              ("FixedPrior0.9", "static", "std_cov", r"Fixed prior $\pi^\ast$=0.90")]
    for i, (v, r, mt, lab) in enumerate(curves):
        axes[1].plot(X, series("real", v, r, mt), marker=MK[i], color=ML[i], mfc="none", label=lab,
                     ls="-" if i < 3 else "--")
    axes[1].axhline(0.90, color="black", ls=":", lw=0.9)
    style(axes[1], "Coverage", (0.0, 1.02))
    axes[1].set_title("(b) Coverage with prior shift removed")
    axes[1].legend(loc="lower left", fontsize=6.2)
    fig.tight_layout()
    save(fig, "fig5_prior_shift")


def fig_feedback():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    regs = [("immediate", "Immediate"), ("delay30", "Delay 30 d"), ("delay90", "Delay 90 d"),
            ("delay365", "Delay 365 d"), ("batch_year", "Yearly batch"), ("partial0.1", "10% of labels")]
    for i, (r, lab) in enumerate(regs):
        axes[0].plot(X, series("real", "ACI", r, "coverage"), marker=MK[i], color=ML[i], mfc="none", label=lab)
        axes[1].plot(X, series("real", "ACI", r, "frac_full"), marker=MK[i], color=ML[i], mfc="none", label=lab)
    axes[0].axhline(0.90, color="black", ls="--", lw=0.9)
    style(axes[0], "Marginal coverage", (0.0, 1.02))
    style(axes[1], "Share of full sets {benign, malware}", (0.0, 1.02))
    axes[0].set_title(r"(a) ACI coverage, $\gamma=0.01$")
    axes[1].set_title(r"(b) ACI full-set share, $\gamma=0.01$")
    axes[0].legend(loc="lower left", fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig6_feedback")


def fig_burden():
    vs = [("SCP", "static", "SCP"), ("Mondrian", "static", "Mondrian"), ("WCP", "static", "Weighted CP"),
          ("ACI", "immediate", "ACI"), ("ACI", "partial0.1", "ACI 10%"), ("RecalSCP", "batch_year", "Recal. SCP"),
          ("EnbPI", "batch_year", "EnbPI")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (period, ys) in zip(axes, [("2020", ["2020"]), ("Mean 2013-2020", YRS)]):
        ab, ex, lab_used, names = [], [], [], []
        for v, r, nm in vs:
            try:
                ab.append(np.mean([S["main_real"][v][r][y]["abstention"]["mean"] for y in ys]))
                ex.append(np.mean([S["main_real"][v][r][y]["extra_defer_rolling"]["mean"] for y in ys]))
                lab_used.append(np.mean([S["main_real"][v][r][y]["labels_used"]["mean"] for y in ys]))
                names.append(nm)
            except KeyError:
                continue
        xi = np.arange(len(names))
        ax.bar(xi, ab, 0.62, color=ML[0], edgecolor="black", lw=0.4, label="Mandatory abstention")
        ax.bar(xi, ex, 0.62, bottom=ab, color=ML[2], edgecolor="black", lw=0.4, label="Extra deferral (rolling policy)")
        ax.plot(xi, lab_used, "kD", mfc="white", ms=4.5, label="Labels consumed")
        ax.set_xticks(xi)
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"({'a' if period == '2020' else 'b'}) {period}")
    axes[0].set_ylabel("Share of deployment stream")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig7_burden")


def fig_cwdv():
    D = S.get("diagnostic_real")
    if not D:
        return
    runs = json.load(open(os.path.join(RES_DIR, "diagnostic_real.json")))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, thr, title in zip(axes, ("0.8", "0.7"), ("(a) Threshold 0.80", "(b) Threshold 0.70")):
        for i, b in enumerate(BASES):
            rb = [r for r in runs if r["base"] == b]
            pred = [r["predicted_breach_year"][thr] for r in rb if r["predicted_breach_year"][thr]]
            obs = [r["observed_first_breach"][thr] for r in rb if r["observed_first_breach"][thr]]
            jit = np.linspace(-0.15, 0.15, len(pred)) if len(pred) > 1 else np.zeros(len(pred))
            ax.plot(np.full(len(pred), i) + jit - 0.12, pred, MK[i], color=ML[i], mfc="none", ms=5.5)
            if obs:
                ax.plot([i + 0.18], [np.median(obs)], "k_", ms=14, mew=2.0)
        ax.set_xticks(range(len(BASES)))
        ax.set_xticklabels(["Random\nforest", "Gradient\nboosting", "Logistic\nregression"])
        ax.set_xlim(-0.6, len(BASES) - 0.4)
        ax.set_ylim(2011.0, 2016.6)
        ax.set_title(title)
    axes[0].set_ylabel("First year with coverage below threshold")
    axes[1].legend(handles=[Line2D([0], [0], marker="o", color="gray", mfc="none", ls="", label="CWDV prediction (per seed)"),
                            Line2D([0], [0], marker="_", color="black", ms=12, mew=2.0, ls="", label="Observed (same detector)")],
                   loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig9_cwdv_calendar")


def fig_diag():
    p = os.path.join(RES_DIR, "diagnostic_real.json")
    if not os.path.exists(p):
        return
    runs = json.load(open(p))["runs"]
    panels = [("CWDVD_KS", "(a) CWDV-D, raw"), ("CWDVD_KS+rolling", "(b) CWDV-D, rolling offset"),
              ("NE", "(c) Non-empty-rate baseline")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, (k, title) in zip(axes, panels):
        for i, b in enumerate(BASES):
            a, pr = [], []
            for r in runs:
                if r["base"] != b:
                    continue
                for t in YRS:
                    a.append(r["deployment"][t]["actual"]); pr.append(r["deployment"][t]["pred"][k])
            ax.plot(a, pr, MK[i], color=ML[i], mfc="none", ms=3.5, ls="",
                    label={"RandomForest": "RF", "HistGB": "GB", "LogReg": "LR"}[b])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.axvline(0.80, color="gray", lw=0.6, ls=":"); ax.axhline(0.80, color="gray", lw=0.6, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Realised coverage")
        ax.set_title(title)
    axes[0].set_ylabel("Predicted coverage")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig10_diagnostic_scatter")


def fig_family():
    F = S.get("family_real")
    if not F:
        return
    fams = F["families"]
    names = [f["family"] for f in fams]
    cov = np.array([f["coverage_split"] for f in fams])
    empty = np.array([f["empty_split"] for f in fams])
    wrong = np.array([f["wrong_single_split"] for f in fams])
    aci = np.array([f["coverage_aci"] for f in fams])
    yi = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(yi, cov, 0.68, color=ML[0], edgecolor="black", lw=0.4, label="Covered (SCP)")
    ax.barh(yi, wrong, 0.68, left=cov, color=ML[6], edgecolor="black", lw=0.4, label="Wrong singleton (silent)")
    ax.barh(yi, empty, 0.68, left=cov + wrong, color=ML[2], edgecolor="black", lw=0.4, label="Empty set (rejected)")
    ax.plot(aci, yi, "kD", mfc="white", ms=4, label="ACI coverage")
    ax.axvline(0.90, color="black", ls="--", lw=0.9)
    ax.set_yticks(yi)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of family's deployment samples (2013-2020)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=4, fontsize=6.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig8_family")


if __name__ == "__main__":
    fig_coverage(); fig_bybase(); fig_decomposition(); fig_capture(); fig_prior(); fig_feedback()
    fig_burden(); fig_family(); fig_cwdv(); fig_diag()
