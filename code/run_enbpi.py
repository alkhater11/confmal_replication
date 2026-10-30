"""
EnbPI (Xu & Xie 2021) adapted to binary classification, version 2.

A bootstrap ensemble of B reduced-capacity members is fitted on the fit and
calibration splits of the 2008-2012 window (the in-window test split is held
out); out-of-bag aggregation gives leave-one-out scores that initialise a
sliding window of the W most recent labelled scores.  Feedback regimes are
the same as those given to ACI and to recalibrated split conformal in
run_audit.py, so comparisons can be made at matched label availability:
  batch_year  labels of year t are appended after year t is scored (original)
  immediate   each label is appended before the next point is scored
  delayN      each label is appended N calendar days after the sample date
  partialR    each label is appended immediately with probability R

Usage: python3 run_enbpi.py real|emu
"""
import sys, time, os
import numpy as np
from common import (CFG, ALPHA, DEP_YEARS, BASES, RES_DIR, load_subset, window_mask,
                    window_split, Detector, save_json)
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
SEEDS = ([int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS")
         else CFG["seeds_enbpi"])
B, W = CFG["enbpi"]["B"], CFG["enbpi"]["window"]
TARGET = CFG["deferral_target_accuracy"]

D = load_subset(WHICH)
X, y, years, dates = D["X"], D["y"], D["years"], D["dates"]
win = window_mask(years)
dep = np.isin(years, DEP_YEARS)
Xw, yw, dw = X[win], y[win], dates[win]
Xd, yd, yrd, ddp = X[dep], y[dep], years[dep], dates[dep]
year_masks = {ty: yrd == ty for ty in DEP_YEARS}

records, meta = [], []
for base in BASES:
    for seed in SEEDS:
        t0 = time.time()
        fit, cal, test = window_split(len(yw), seed)
        tr = np.sort(np.concatenate([fit, cal]))          # date order preserved
        Xtr, ytr = Xw[tr], yw[tr]
        n = len(ytr)
        rng = np.random.default_rng(seed)
        P_oob = np.zeros((n, 2)); cnt = np.zeros(n)
        P_dep = np.zeros((len(yd), 2)); P_iw = np.zeros((len(test), 2))
        for b in range(B):
            bi = rng.integers(0, n, n)
            det = Detector(base, seed * 100 + b, reduced=True).fit(Xtr[bi], ytr[bi])
            oob = np.ones(n, dtype=bool); oob[np.unique(bi)] = False
            if oob.any():
                P_oob[oob] += det.proba(Xtr[oob]); cnt[oob] += 1
            P_dep += det.proba(Xd); P_iw += det.proba(Xw[test])
        valid = cnt > 0
        P_oob[valid] /= cnt[valid][:, None]
        S_dep, S_iw = 1.0 - P_dep / B, 1.0 - P_iw / B
        s_oob = (1.0 - P_oob[np.arange(n), ytr])[valid]          # date-ordered
        init = s_oob[-W:]
        q0 = C.SlidingQuantile(init, W).q(ALPHA)
        y_iw = yw[test]
        mk_iw = S_iw <= q0
        tau_static = C.choose_tau(mk_iw, y_iw, S_iw, TARGET)
        m_iw = C.set_metrics(mk_iw, y_iw, S_iw)
        m_iw.update(dataset=WHICH, base=base, seed=seed, variant="EnbPI", regime="static",
                    year="in-window", labels_used=0.0)
        records.append(m_iw)
        feedback = [("immediate", dict(regime="immediate")),
                    ("batch_year", dict(regime="batch_year"))]
        feedback += [(f"delay{d}", dict(regime="delay", delay_days=d)) for d in CFG["aci"]["delays_days"]]
        feedback += [(f"partial{r}", dict(regime="partial", partial_rate=r)) for r in CFG["aci"]["partial_rates"]]
        for regime, kw in feedback:
            prng = np.random.default_rng(1000 * seed + 7)
            mk, cons = C.run_recalibration(S_dep, yd, yrd, init, W, ALPHA, dates=ddp, rng=prng, **kw)
            prev = None
            for ty in DEP_YEARS:
                m = year_masks[ty]
                mt, yt, St = mk[m], yd[m], S_dep[m]
                rec = C.set_metrics(mt, yt, St)
                tau_roll = C.choose_tau(*prev, TARGET) if prev is not None else tau_static
                ex_r, acc_r, _ = C.singleton_deferral(mt, yt, St, TARGET, tau=tau_roll)
                ex_q, acc_q, _ = C.singleton_deferral(mt, yt, St, TARGET, tau=None)
                ex_s, acc_s, _ = C.singleton_deferral(mt, yt, St, TARGET, tau=tau_static)
                covered = mt[np.arange(len(yt)), yt]
                rec.update(dataset=WHICH, base=base, seed=seed, variant="EnbPI", regime=regime, year=ty,
                           labels_used=float(cons[m].mean()), extra_defer_rolling=ex_r, acc_after_rolling=acc_r,
                           extra_defer_retro=ex_q, acc_after_retro=acc_q,
                           extra_defer_static=ex_s, acc_after_static=acc_s,
                           oracle_budget=C.oracle_alert_budget(St, yt, covered, 1 - ALPHA))
                records.append(rec)
                prev = (mt, yt, St)
        meta.append(dict(base=base, seed=seed, n_train=n, n_oob=int(valid.sum()), q_initial=q0,
                         in_window_cov=m_iw["coverage"], seconds=time.time() - t0))
        print(f"[EnbPI:{WHICH}:{base} s{seed}] {time.time() - t0:.0f}s q0={q0:.4f} "
              f"in-window held-out cov={m_iw['coverage']:.4f}", flush=True)
        save_json(dict(records=records, meta=meta), os.path.join(RES_DIR, f"enbpi_{WHICH}.json"))
print(f"[DONE] {len(records)} EnbPI records", flush=True)
