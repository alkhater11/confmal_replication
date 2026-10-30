"""
Main audit (version 2).

For every base detector and seed:
  * the 2008-2012 window is split at random into fit / calibration / in-window
    test (60/25/15); the in-window test split is never used for fitting or
    calibration and gives an independent exchangeable check of validity;
  * static procedures (no deployment labels): split conformal, conformal risk
    control (miscoverage loss), Mondrian, weighted CP (covariate-shift weights
    from a domain classifier), label-shift weighted CP with oracle and with
    BBSE-estimated deployment priors, and the fixed-prior protocol;
  * online procedures under explicit feedback regimes: ACI (immediate, yearly
    batch, calendar delays, partial labels, step-size sweep) and sliding-window
    recalibrated split conformal (immediate and yearly batch);
  * sensitivity to the nominal level;
  * fixed-class-ratio resampled deployment streams for ACI and split conformal.
Every record carries coverage, class-conditional coverage, set composition,
accepted-decision quality, error capture by abstention and burden accounting.

Usage: python3 run_audit.py real|emu [RandomForest HistGB LogReg]
"""
import sys, time, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import (CFG, ALPHA, DEP_YEARS, BASES, RES_DIR, load_subset, window_mask,
                    window_split, Detector, save_json)
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
BASE_LIST = sys.argv[2:] if len(sys.argv) > 2 else BASES
SEEDS = ([int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS")
         else CFG["seeds_main"])
OUT_TAG = os.environ.get("OUT_TAG", "")
TARGET = CFG["deferral_target_accuracy"]
GAMMA = CFG["aci"]["gamma"]
W_RECAL = CFG["recalibration_window"]

D = load_subset(WHICH)
X, y, years, dates = D["X"], D["y"], D["years"], D["dates"]
win = window_mask(years)
dep = np.isin(years, DEP_YEARS)
Xw, yw, dw = X[win], y[win], dates[win]
Xd, yd, yrd, dd = X[dep], y[dep], years[dep], dates[dep]
year_masks = {ty: yrd == ty for ty in DEP_YEARS}
print(f"[DATA:{WHICH}] window {len(yw)} (malware {yw.mean():.3f}), deployment {len(yd)}", flush=True)

for base in BASE_LIST:
    records, meta, resampled = [], [], []
    t_base = time.time()
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        fit, cal, test = window_split(len(yw), seed)
        det = Detector(base, seed).fit(Xw[fit], yw[fit])
        fit_s = time.time() - t0
        S_cal, y_cal = det.scores(Xw[cal]), yw[cal]
        s_cal_true = S_cal[np.arange(len(y_cal)), y_cal]
        S_iw, y_iw = det.scores(Xw[test]), yw[test]
        S_dep = det.scores(Xd)
        cal_sorted = C.SortedScores(s_cal_true)
        q_scp = cal_sorted.q(ALPHA)
        q_crc = C.crc_threshold(s_cal_true, ALPHA)
        q_mond = C.mondrian_thresholds(s_cal_true, y_cal, ALPHA)
        pi_cal = np.array([np.mean(y_cal == 0), np.mean(y_cal == 1)])
        yhat_cal = np.argmin(S_cal, axis=1)
        tau_static = C.choose_tau(S_cal <= q_scp, y_cal, S_cal, TARGET)
        prev = {}

        def add(variant, regime, year, mask, ylab, S, extra=None, novelty=None,
                labels_used=0.0, burden=True):
            m = C.set_metrics(mask, ylab, S, novelty)
            m.update(dataset=WHICH, base=base, seed=seed, variant=variant, regime=regime,
                     year=year, labels_used=labels_used)
            if burden and year != "in-window":
                key = (variant, regime)
                tau_roll = C.choose_tau(*prev[key], TARGET) if key in prev else tau_static
                ex_r, acc_r, _ = C.singleton_deferral(mask, ylab, S, TARGET, tau=tau_roll)
                ex_q, acc_q, _ = C.singleton_deferral(mask, ylab, S, TARGET, tau=None)
                ex_s, acc_s, _ = C.singleton_deferral(mask, ylab, S, TARGET, tau=tau_static)
                covered = mask[np.arange(len(ylab)), ylab]
                m.update(extra_defer_rolling=ex_r, acc_after_rolling=acc_r,
                         extra_defer_retro=ex_q, acc_after_retro=acc_q,
                         extra_defer_static=ex_s, acc_after_static=acc_s,
                         oracle_budget=C.oracle_alert_budget(S, ylab, covered, 1 - ALPHA))
                prev[key] = (mask, ylab, S)
            if extra:
                m.update(extra)
            records.append(m)

        # ---------------- in-window held-out test (exchangeable reference)
        add("SCP", "static", "in-window", S_iw <= q_scp, y_iw, S_iw)
        add("CRC", "static", "in-window", S_iw <= q_crc, y_iw, S_iw)
        add("Mondrian", "static", "in-window", S_iw <= q_mond[None, :], y_iw, S_iw)
        for a in CFG["alpha_sweep"]:
            if a != ALPHA:
                add(f"SCP_a{a}", "static", "in-window", S_iw <= cal_sorted.q(a), y_iw, S_iw)
                add(f"Mondrian_a{a}", "static", "in-window",
                    S_iw <= C.mondrian_thresholds(s_cal_true, y_cal, a)[None, :], y_iw, S_iw)
        q_fixed = {}
        for pstar in CFG["fixed_priors"]:
            q_fp = C.label_shift_thresholds(s_cal_true, y_cal, pi_cal, [1 - pstar, pstar], ALPHA)
            q_fixed[pstar] = q_fp
            mk = S_iw <= q_fp[None, :]
            cm, cb = mk[y_iw == 1, 1].mean(), mk[y_iw == 0, 0].mean()
            add(f"FixedPrior{pstar}", "static", "in-window", mk, y_iw, S_iw,
                extra=dict(std_cov=pstar * cm + (1 - pstar) * cb, pstar=pstar))

        # ---------------- static procedures per deployment year
        bbse = {}
        for ty in DEP_YEARS:
            m = year_masks[ty]
            S_t, y_t, X_t = S_dep[m], yd[m], Xd[m]
            nsub = min(CFG["weighted_cp"]["max_per_side"], len(y_cal), int(m.sum()))
            ci = rng.choice(len(y_cal), nsub, replace=False)
            ti = rng.choice(int(m.sum()), nsub, replace=False)
            Xdom = np.vstack([Xw[cal][ci], X_t[ti]])
            ydom = np.concatenate([np.zeros(nsub), np.ones(nsub)])
            dsc = StandardScaler().fit(Xdom)
            dom = LogisticRegression(max_iter=CFG["weighted_cp"]["domain_max_iter"],
                                     C=CFG["weighted_cp"]["domain_C"]).fit(dsc.transform(Xdom), ydom)
            p_cal = np.clip(dom.predict_proba(dsc.transform(Xw[cal]))[:, 1], 1e-6, 1 - 1e-6)
            p_te = np.clip(dom.predict_proba(dsc.transform(X_t))[:, 1], 1e-6, 1 - 1e-6)
            w_cal = p_cal / (1 - p_cal)
            q_w = C.weighted_quantile_thresholds(s_cal_true, w_cal, p_te / (1 - p_te), ALPHA)
            ess = float(w_cal.sum() ** 2 / np.sum(w_cal ** 2))
            pi_t = np.array([np.mean(y_t == 0), np.mean(y_t == 1)])
            pi_hat = C.bbse_prior(yhat_cal, y_cal, np.argmin(S_t, axis=1))
            bbse[ty] = dict(true=pi_t[1], bbse=pi_hat[1])
            q_lso = C.label_shift_thresholds(s_cal_true, y_cal, pi_cal, pi_t, ALPHA)
            q_lsb = C.label_shift_thresholds(s_cal_true, y_cal, pi_cal, pi_hat, ALPHA)

            add("SCP", "static", ty, S_t <= q_scp, y_t, S_t, novelty=p_te)
            add("CRC", "static", ty, S_t <= q_crc, y_t, S_t)
            add("Mondrian", "static", ty, S_t <= q_mond[None, :], y_t, S_t)
            add("WCP", "static", ty, S_t <= q_w[:, None], y_t, S_t,
                extra=dict(wcp_q_median=float(np.median(q_w[np.isfinite(q_w)])) if np.isfinite(q_w).any() else None,
                           wcp_frac_inf=float(np.mean(~np.isfinite(q_w))), wcp_ess=ess))
            add("LSW-oracle", "static", ty, S_t <= q_lso[None, :], y_t, S_t,
                extra=dict(pi_target=pi_t[1]))
            add("LSW-BBSE", "static", ty, S_t <= q_lsb[None, :], y_t, S_t,
                extra=dict(pi_target=pi_hat[1], pi_true=pi_t[1]))
            for pstar, q_fp in q_fixed.items():
                mk = S_t <= q_fp[None, :]
                cm = mk[y_t == 1, 1].mean() if (y_t == 1).any() else np.nan
                cb = mk[y_t == 0, 0].mean() if (y_t == 0).any() else np.nan
                add(f"FixedPrior{pstar}", "static", ty, mk, y_t, S_t, burden=False,
                    extra=dict(std_cov=pstar * cm + (1 - pstar) * cb, pstar=pstar))
            for a in CFG["alpha_sweep"]:
                if a != ALPHA:
                    add(f"SCP_a{a}", "static", ty, S_t <= cal_sorted.q(a), y_t, S_t, burden=False)
                    add(f"Mondrian_a{a}", "static", ty,
                        S_t <= C.mondrian_thresholds(s_cal_true, y_cal, a)[None, :], y_t, S_t, burden=False)

        # ---------------- online procedures over the whole 2013-2020 stream
        def add_stream(variant, regime, mask, consumed, burden=True, extra_fn=None):
            for ty in DEP_YEARS:
                m = year_masks[ty]
                ex = extra_fn(m) if extra_fn else None
                add(variant, regime, ty, mask[m], yd[m], S_dep[m], extra=ex,
                    labels_used=float(consumed[m].mean()), burden=burden)

        feedback = [("immediate", dict(regime="immediate")),
                    ("batch_year", dict(regime="batch_year"))]
        feedback += [(f"delay{d}", dict(regime="delay", delay_days=d)) for d in CFG["aci"]["delays_days"]]
        feedback += [(f"partial{r}", dict(regime="partial", partial_rate=r)) for r in CFG["aci"]["partial_rates"]]
        for g_name, g in (("ACI", GAMMA), ("ACI_g0.001", 0.001)):
            for name, kw in feedback:
                prng = np.random.default_rng(1000 * seed + 7)
                mk, a_used, cons = C.run_aci(S_dep, yd, dd, yrd, cal_sorted, ALPHA, g, rng=prng, **kw)
                add_stream(g_name, name, mk, cons,
                           extra_fn=lambda mm, a_used=a_used: dict(alpha_t_mean=float(a_used[mm].mean()),
                                                                  alpha_t_start=float(a_used[mm][0]),
                                                                  alpha_t_end=float(a_used[mm][-1])))
        for g in CFG["aci"]["gamma_sweep"]:
            if g not in (GAMMA, 0.001):
                mk, a_used, cons = C.run_aci(S_dep, yd, dd, yrd, cal_sorted, ALPHA, g, regime="immediate")
                add_stream(f"ACI_g{g}", "immediate", mk, cons, burden=False)
        for a in CFG["alpha_sweep"]:
            if a != ALPHA:
                mk, a_used, cons = C.run_aci(S_dep, yd, dd, yrd, cal_sorted, a, GAMMA, regime="immediate")
                add_stream(f"ACI_a{a}", "immediate", mk, cons, burden=False)
        cal_order = np.argsort(dw[cal], kind="stable")
        init = s_cal_true[cal_order]
        for name, kw in feedback:
            prng = np.random.default_rng(1000 * seed + 7)
            mk, cons = C.run_recalibration(S_dep, yd, yrd, init, W_RECAL, ALPHA, dates=dd, rng=prng, **kw)
            add_stream("RecalSCP", name, mk, cons)

        # ---------------- fixed-class-ratio resampled streams
        rs = CFG["resampled_stream"]
        for pstar in rs["priors"]:
            for r in range(rs["replicates"]):
                rr = np.random.default_rng(100000 + 1000 * seed + 10 * r + int(round(100 * pstar)))
                parts = []
                for ty in DEP_YEARS:
                    pos = np.where(year_masks[ty] & (yd == 1))[0]
                    neg = np.where(year_masks[ty] & (yd == 0))[0]
                    n_t = int(year_masks[ty].sum())
                    n_pos = int(round(pstar * n_t))
                    sel = np.concatenate([rr.choice(pos, n_pos, replace=True),
                                          rr.choice(neg, n_t - n_pos, replace=True)])
                    parts.append(np.sort(sel))
                idx = np.concatenate(parts)
                S_r, y_r, yr_r, d_r = S_dep[idx], yd[idx], yrd[idx], dd[idx]
                mk_aci, _, _ = C.run_aci(S_r, y_r, d_r, yr_r, cal_sorted, ALPHA, GAMMA, regime="immediate")
                mk_scp = S_r <= q_scp
                for meth, mk in (("ACI", mk_aci), ("SCP", mk_scp)):
                    for ty in DEP_YEARS:
                        mm = yr_r == ty
                        mt, yt = mk[mm], y_r[mm]
                        cov = mt[np.arange(len(yt)), yt]
                        sz = mt.sum(axis=1)
                        single = sz == 1
                        lab = C.included_label(mt)
                        resampled.append(dict(base=base, seed=seed, prior=pstar, rep=r, method=meth, year=ty,
                                              coverage=float(cov.mean()),
                                              cov_mal=float(cov[yt == 1].mean()),
                                              cov_ben=float(cov[yt == 0].mean()),
                                              frac_empty=float((sz == 0).mean()),
                                              frac_full=float((sz == 2).mean()),
                                              acc_single=float((lab[single] == yt[single]).mean()) if single.any() else None))

        meta.append(dict(base=base, seed=seed, n_fit=len(fit), n_cal=len(cal), n_test=len(test),
                         fit_seconds=fit_s, q_scp=q_scp, q_crc=q_crc, q_mondrian=q_mond.tolist(),
                         crc_equals_scp=bool(q_crc == q_scp), pi_cal=pi_cal[1],
                         tau_static=tau_static, bbse=bbse,
                         q_fixed={str(k): v.tolist() for k, v in q_fixed.items()},
                         in_window_argmax_acc=float(np.mean(np.argmin(S_iw, 1) == y_iw)),
                         seconds=time.time() - t0))
        iw = [r for r in records if r["seed"] == seed and r["year"] == "in-window" and r["variant"] == "SCP"][0]
        print(f"[{WHICH}:{base} seed={seed}] fit={fit_s:.1f}s q_scp={q_scp:.4f} crc==scp={q_crc == q_scp} "
              f"in-window held-out cov={iw['coverage']:.4f} (n={iw['n']}) total={time.time() - t0:.0f}s",
              flush=True)
        save_json(dict(records=records, meta=meta, resampled=resampled),
                  os.path.join(RES_DIR, f"audit_{WHICH}_{base}{OUT_TAG}.json"))
    print(f"[DONE {WHICH}:{base}] {len(records)} records in {time.time() - t_base:.0f}s", flush=True)
