"""
Aggregate every result file into results/summary.json, the single source of
all numbers quoted in the manuscript, its tables and its figures.

Conventions
  * coverage-type metrics are averaged over configurations (base detector x
    seed); because every configuration scores the same samples in a given
    year this equals pooling;
  * accepted-decision accuracy, false-positive and false-negative rates among
    accepted decisions, and error-capture ratios are pooled over
    configurations (ratio of summed counts), which weights configurations by
    the number of decisions they actually accept;
  * spreads are standard deviations, minima and maxima across configurations.
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from common import RES_DIR, DATA_DIR, CFG, DEP_YEARS, BASES, load_json, save_json

MEAN_METRICS = ["coverage", "cov_mal", "cov_ben", "frac_empty", "frac_single", "frac_full",
                "abstention", "avg_set_size", "wrong_single_rate", "argmax_acc", "labels_used",
                "extra_defer_retro", "extra_defer_rolling", "extra_defer_static", "oracle_budget",
                "std_cov", "novelty_auc_empty_vs_single", "acc_after_rolling", "acc_after_retro",
                "acc_after_static", "alpha_t_mean", "alpha_t_start", "alpha_t_end", "wcp_ess",
                "wcp_frac_inf", "pi_target", "pi_true", "acc_single"]
GAMMA = CFG["aci"]["gamma"]


def load_audit(which):
    recs, meta, res = [], [], []
    for b in BASES:
        p = os.path.join(RES_DIR, f"audit_{which}_{b}.json")
        if os.path.exists(p):
            d = load_json(p)
            recs += d["records"]; meta += d["meta"]; res += d["resampled"]
    return recs, meta, res


def stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def aggregate(records, keyf):
    groups = defaultdict(list)
    for r in records:
        groups[keyf(r)].append(r)
    out = {}
    for k, rs in groups.items():
        g = {}
        for mtr in MEAN_METRICS:
            s = stats([r.get(mtr) for r in rs])
            if s is not None:
                g[mtr] = s
        tot = lambda f: float(sum(r.get(f, 0) or 0 for r in rs))
        n_single = tot("n_single")
        g["pooled_acc_single"] = tot("n_correct_single") / n_single if n_single else None
        g["pooled_fpr_acc"] = tot("FP") / (tot("FP") + tot("TN")) if (tot("FP") + tot("TN")) else None
        g["pooled_fnr_acc"] = tot("FN") / (tot("FN") + tot("TP")) if (tot("FN") + tot("TP")) else None
        g["sum_FP"], g["sum_TN"], g["sum_FN"], g["sum_TP"] = tot("FP"), tot("TN"), tot("FN"), tot("TP")
        n_empty = sum(round((r["frac_empty"] or 0) * r["n"]) for r in rs)
        n_errors = tot("n_errors")
        g["pooled_argmax_acc_empty"] = (n_empty - tot("err_in_empty")) / n_empty if n_empty else None
        err_single = n_errors - tot("err_in_empty") - tot("err_in_full")
        g["pooled_argmax_acc_single"] = (n_single - err_single) / n_single if n_single else None
        g["pooled_err_recall_empty"] = tot("err_in_empty") / n_errors if n_errors else None
        g["pooled_err_recall_abstain"] = (tot("err_in_empty") + tot("err_in_full")) / n_errors if n_errors else None
        cov = [r["coverage"] for r in rs]
        g["frac_configs_below_nominal"] = float(np.mean([c < 1 - CFG["alpha"] for c in cov]))
        g["n_samples"] = rs[0]["n"]
        g["n_mal"] = rs[0]["n_mal"]
        g["n_ben"] = rs[0]["n_ben"]
        out[k] = g
    return out


def nest(flat):
    """{(a, b, c): v} -> {a: {b: {c: v}}} with string keys."""
    out = {}
    for k, v in flat.items():
        d = out
        for part in k[:-1]:
            d = d.setdefault(str(part), {})
        d[str(k[-1])] = v
    return out


def decomposition(records):
    """Exact split of the coverage change into class-composition and within-class terms."""
    by_cfg = defaultdict(dict)
    for r in records:
        by_cfg[(r["base"], r["seed"], r["variant"], r["regime"])][str(r["year"])] = r
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (b, s, v, rg), yrs in by_cfg.items():
        if "in-window" not in yrs or v not in ("SCP", "Mondrian"):
            continue
        iw = yrs["in-window"]
        pi_in = iw["n_mal"] / iw["n"]
        cm_in, cb_in, c_in = iw["cov_mal"], iw["cov_ben"], iw["coverage"]
        for ty in DEP_YEARS:
            r = yrs.get(str(ty))
            if r is None:
                continue
            pi_t = r["n_mal"] / r["n"]
            cb_t = r["cov_ben"] if r["cov_ben"] is not None else cb_in
            comp = (pi_t - pi_in) * (cm_in - cb_in)
            within = pi_t * (r["cov_mal"] - cm_in) + (1 - pi_t) * (cb_t - cb_in)
            d = res[v][str(ty)]
            d["total"].append(r["coverage"] - c_in)
            d["composition"].append(comp)
            d["within"].append(within)
            d["envelope_lo"].append(min(cm_in, cb_in))
            d["envelope_hi"].append(max(cm_in, cb_in))
            d["check"].append(r["coverage"] - c_in - comp - within)
    out = {}
    for v, yrs in res.items():
        out[v] = {ty: {k: float(np.mean(vals)) for k, vals in d.items()} for ty, d in yrs.items()}
        for ty in out[v]:
            out[v][ty]["max_abs_check"] = float(np.max(np.abs(res[v][ty]["check"])))
    return out


def aci_span(records, variant="ACI", regime="immediate"):
    """Coverage of ACI pooled over the whole 2013-2020 stream, per configuration."""
    per = defaultdict(lambda: [0.0, 0])
    for r in records:
        if r["variant"] == variant and r["regime"] == regime and r["year"] != "in-window":
            k = (r["base"], r["seed"])
            per[k][0] += r["coverage"] * r["n"]; per[k][1] += r["n"]
    covs = [a / n for a, n in per.values()]
    T = list(per.values())[0][1] if per else 0
    return dict(stats=stats(covs), T=T, gc_bound=(max(CFG["alpha"], 1 - CFG["alpha"]) + GAMMA) / (GAMMA * T) if T else None)


def resampled_summary(res):
    g = defaultdict(lambda: defaultdict(list))
    for r in res:
        k = (r["prior"], r["method"], r["year"])
        for f in ("coverage", "cov_mal", "cov_ben", "frac_empty", "frac_full", "acc_single"):
            if r.get(f) is not None:
                g[k][f].append(r[f])
    out = {}
    for k, d in g.items():
        out[k] = {f: dict(mean=float(np.mean(v)), p025=float(np.percentile(v, 2.5)),
                          p975=float(np.percentile(v, 97.5))) for f, v in d.items()}
    return nest(out)


def diagnostic_summary(which):
    p = os.path.join(RES_DIR, f"diagnostic_{which}.json")
    if not os.path.exists(p):
        return None
    d = load_json(p)
    runs = d["runs"]
    preds = list(runs[0]["deployment"][str(DEP_YEARS[0])]["pred"].keys())
    out = dict(blocks=d["blocks"], predictors={}, calendar={}, divergence_spearman={})
    for name in preds:
        per_base = defaultdict(lambda: dict(mae=[], mse=[], sp=[]))
        pooled_p, pooled_a = [], []
        warn = {str(t): dict(breach=0, hit=0, miss=0, false_alarm=0, quiet=0) for t in CFG["diagnostic"]["breach_thresholds"]}
        for r in runs:
            a = np.array([r["deployment"][str(t)]["actual"] for t in DEP_YEARS])
            pr = np.array([r["deployment"][str(t)]["pred"][name] for t in DEP_YEARS])
            per_base[r["base"]]["mae"].append(float(np.mean(np.abs(pr - a))))
            per_base[r["base"]]["mse"].append(float(np.mean(pr - a)))
            per_base[r["base"]]["sp"].append(float(spearmanr(pr, a).correlation) if np.std(pr) > 0 else np.nan)
            pooled_p += list(pr); pooled_a += list(a)
            for thr in CFG["diagnostic"]["breach_thresholds"]:
                w = warn[str(thr)]
                for pi, ai in zip(pr, a):
                    if ai < thr:
                        w["breach"] += 1
                        w["hit" if pi < thr else "miss"] += 1
                    else:
                        w["false_alarm" if pi < thr else "quiet"] += 1
        for thr, w in warn.items():
            w["sensitivity"] = w["hit"] / w["breach"] if w["breach"] else None
            nb = w["false_alarm"] + w["quiet"]
            w["false_alarm_rate"] = w["false_alarm"] / nb if nb else None
            w["precision"] = w["hit"] / (w["hit"] + w["false_alarm"]) if (w["hit"] + w["false_alarm"]) else None
        out["predictors"][name] = dict(
            per_base={b: dict(mae=float(np.mean(v["mae"])), mean_signed_error=float(np.mean(v["mse"])),
                              spearman=float(np.nanmean(v["sp"])) if not np.all(np.isnan(v["sp"])) else None)
                      for b, v in per_base.items()},
            mae=float(np.mean(np.abs(np.array(pooled_p) - np.array(pooled_a)))),
            spearman_pooled=float(spearmanr(pooled_p, pooled_a).correlation),
            warning=warn)
    for b in BASES:
        rb = [r for r in runs if r["base"] == b]
        out["calendar"][b] = dict(
            velocity=stats([r["velocity"] for r in rb]),
            predicted_breach_080=stats([r["predicted_breach_year"]["0.8"] for r in rb if r["predicted_breach_year"]["0.8"]]),
            observed_first_breach_080=[r["observed_first_breach"]["0.8"] for r in rb],
            beta_KS=stats([r["beta_KS"] for r in rb]), beta_DOM=stats([r["beta_DOM"] for r in rb]),
            qhat=stats([r["qhat"] for r in rb]), cov_cal=stats([r["cov_cal"] for r in rb]))
        for div in ("KS", "DOM"):
            sp = []
            for r in rb:
                dv = [r["deployment"][str(t)][div] for t in DEP_YEARS]
                a = [r["deployment"][str(t)]["actual"] for t in DEP_YEARS]
                sp.append(float(spearmanr(dv, a).correlation))
            out["divergence_spearman"].setdefault(div, {})[b] = stats(sp)
    out["actual_by_year"] = {str(t): stats([r["deployment"][str(t)]["actual"] for r in runs]) for t in DEP_YEARS}
    # exact composition / within-class decomposition of the within-window pseudo-block coverage change
    pdec = defaultdict(lambda: defaultdict(list))
    for r in runs:
        cc = r.get("cal_class")
        if not cc:
            continue
        for i, b in enumerate(r["pseudo_blocks"]):
            cb = b["cov_ben"] if b["cov_ben"] is not None else cc["cov_ben"]
            cm = b["cov_mal"] if b["cov_mal"] is not None else cc["cov_mal"]
            comp = (b["pi"] - cc["pi"]) * (cc["cov_mal"] - cc["cov_ben"])
            within = b["pi"] * (cm - cc["cov_mal"]) + (1 - b["pi"]) * (cb - cc["cov_ben"])
            pdec[r["base"]][i].append((b["coverage"] - r["cov_cal"], comp, within, b["pi"], b["t_year"]))
    out["pseudo_decomposition"] = {
        b: [dict(total=float(np.mean([v[0] for v in vals])), composition=float(np.mean([v[1] for v in vals])),
                 within=float(np.mean([v[2] for v in vals])), pi=float(np.mean([v[3] for v in vals])),
                 t_year=float(np.mean([v[4] for v in vals]))) for i, vals in sorted(d.items())]
        for b, d in pdec.items()}
    if runs and runs[0].get("cal_class"):
        out["cal_block_pi"] = float(np.mean([r["cal_class"]["pi"] for r in runs]))
    return out


def main():
    S = {}
    S["accounting"] = {w: load_json(os.path.join(DATA_DIR, f"accounting_{w}.json")) for w in ("real", "emu")}
    S["overlap"] = load_json(os.path.join(DATA_DIR, "overlap_real_emu.json"))
    S["config"] = CFG
    for which in ("real", "emu"):
        recs, meta, res = load_audit(which)
        if not recs:
            continue
        enb = os.path.join(RES_DIR, f"enbpi_{which}.json")
        enb_recs = load_json(enb)["records"] if os.path.exists(enb) else []
        allr = recs + enb_recs
        S[f"main_{which}"] = nest(aggregate(allr, lambda r: (r["variant"], r["regime"], r["year"])))
        S[f"per_base_{which}"] = nest(aggregate(allr, lambda r: (r["base"], r["variant"], r["regime"], r["year"])))
        S[f"decomposition_{which}"] = decomposition(recs)
        S[f"resampled_{which}"] = resampled_summary(res)
        S[f"aci_span_{which}"] = {f"{v}|{rg}": aci_span(recs, v, rg) for v, rg in
                                  [("ACI", "immediate"), ("ACI", "partial0.1"), ("ACI", "delay90"), ("ACI_g0.001", "delay90")]}
        S[f"meta_{which}"] = {b: dict(q_scp=stats([m["q_scp"] for m in meta if m["base"] == b]),
                                      q_mondrian_ben=stats([m["q_mondrian"][0] for m in meta if m["base"] == b]),
                                      q_mondrian_mal=stats([m["q_mondrian"][1] for m in meta if m["base"] == b]),
                                      crc_equals_scp=all(m["crc_equals_scp"] for m in meta if m["base"] == b),
                                      in_window_argmax_acc=stats([m["in_window_argmax_acc"] for m in meta if m["base"] == b]),
                                      fit_seconds=stats([m["fit_seconds"] for m in meta if m["base"] == b]),
                                      seconds=stats([m["seconds"] for m in meta if m["base"] == b]),
                                      n_fit=meta[0]["n_fit"], n_cal=meta[0]["n_cal"], n_test=meta[0]["n_test"],
                                      bbse={str(t): dict(true=float(np.mean([m["bbse"][str(t)]["true"] for m in meta if m["base"] == b])),
                                                         est=stats([m["bbse"][str(t)]["bbse"] for m in meta if m["base"] == b]))
                                            for t in DEP_YEARS})
                                  for b in BASES}
        S[f"gc_window_bound_{which}"] = {str(t): (1 + GAMMA) / (GAMMA * S[f"main_{which}"]["SCP"]["static"][str(t)]["n_samples"])
                                         for t in DEP_YEARS}
        S[f"diagnostic_{which}"] = diagnostic_summary(which)
        enbm = load_json(enb)["meta"] if os.path.exists(enb) else []
        S[f"enbpi_meta_{which}"] = dict(seconds=stats([m["seconds"] for m in enbm]),
                                        in_window_cov=stats([m["in_window_cov"] for m in enbm]))
    fam = os.path.join(RES_DIR, "family_real.json")
    if os.path.exists(fam):
        S["family_real"] = load_json(fam)
    abl = os.path.join(RES_DIR, "ablation_real.json")
    if os.path.exists(abl):
        A = load_json(abl)
        for r in A:
            r.setdefault("regime", "static")
        S["ablation_real"] = nest(aggregate(A, lambda r: (r["group"], r["variant"], r["year"])))
    save_json(S, os.path.join(RES_DIR, "summary.json"))
    print("summary written:", sorted(S.keys()))


if __name__ == "__main__":
    main()
