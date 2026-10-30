"""
Pre-deployment validity diagnostics, version 2: calendar CWDV, divergence-
indexed CWDV-D, label-free baselines, and a strictly chronological
evaluation of their usefulness as a warning system.

Construction (calibration window only): the 2008-2012 window is ordered by
date and cut into K = 10 equal-count blocks; blocks 1-4 fit the detector,
block 5 calibrates split conformal, blocks 6-10 are pseudo-deployment blocks.

Predictors of split-conformal coverage in deployment year t (all label-free
at the moment year t is scored):
  CWDV-D(KS)   c0 - beta * KS(score of predicted label, cal vs t)
  CWDV-D(dom)  same with a feature-space divergence: held-out AUC of a
               domain classifier separating calibration block from year t
  NE           observed non-empty-set rate in t times calibration-block
               singleton accuracy
  ATC          share of year-t points whose predicted-label score is below
               the threshold that reproduces calibration coverage (Garg et al.)
  DoC          calibration coverage minus the drop in mean confidence
               (Guillory et al.)
Label-requiring reference: persistence (last year's realised coverage).

Offset corrections:
  rolling  add the mean error of years 2013..t-1 (needs only past labels;
           prospective)
  LOYO     add the mean error of all other years (uses later years; NOT
           prospective, reported for comparison with the original submission)

Usage: python3 run_diagnostic.py real|emu
"""
import sys, os, time
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import CFG, ALPHA, DEP_YEARS, BASES, RES_DIR, load_subset, window_mask, Detector, save_json
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
SEEDS = CFG["seeds_diagnostic"]
P = CFG["diagnostic"]
D = load_subset(WHICH)
X, y, years, dates = D["X"], D["y"], D["years"], D["dates"]
win = window_mask(years)
Xw, yw, dw = X[win], y[win], dates[win]
order = np.argsort(dw, kind="stable")
Xw, yw, dw = Xw[order], yw[order], dw[order]
blocks = np.array_split(np.arange(len(yw)), P["K_blocks"])
fit_idx = np.concatenate(blocks[:P["fit_blocks"]])
cal_idx = np.concatenate(blocks[P["fit_blocks"]:P["fit_blocks"] + P["cal_blocks"]])
pseudo = blocks[P["fit_blocks"] + P["cal_blocks"]:]
dep_masks = {ty: years == ty for ty in DEP_YEARS}
block_info = [dict(n=len(b), start=datetime.fromordinal(int(dw[b[0]])).strftime("%Y-%m-%d"),
                   end=datetime.fromordinal(int(dw[b[-1]])).strftime("%Y-%m-%d"),
                   malware_rate=float(yw[b].mean())) for b in blocks]


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    g = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, g, side="right") / len(a)
                               - np.searchsorted(b, g, side="right") / len(b))))


def domain_auc(Xa, Xb, rng, max_side=2000):
    """Two-fold held-out AUC of a classifier separating Xa from Xb (0.5 = indistinguishable)."""
    na, nb = min(max_side, len(Xa)), min(max_side, len(Xb))
    A = Xa[rng.choice(len(Xa), na, replace=False)]
    Bm = Xb[rng.choice(len(Xb), nb, replace=False)]
    Z = np.vstack([A, Bm]); lab = np.concatenate([np.zeros(na), np.ones(nb)])
    fold = rng.random(len(lab)) < 0.5
    scores = np.zeros(len(lab))
    for f in (True, False):
        tr, te = fold == f, fold != f
        if len(np.unique(lab[tr])) < 2 or len(np.unique(lab[te])) < 2:
            return np.nan
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=400, C=0.1).fit(sc.transform(Z[tr]), lab[tr])
        scores[te] = clf.predict_proba(sc.transform(Z[te]))[:, 1]
    return float(roc_auc_score(lab, scores))


def linfit(xs, ys):
    A = np.vstack([np.asarray(xs), np.ones(len(xs))]).T
    slope, icept = np.linalg.lstsq(A, np.asarray(ys), rcond=None)[0]
    return float(slope), float(icept)


out = []
for base in BASES:
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        det = Detector(base, seed).fit(Xw[fit_idx], yw[fit_idx])
        Sc = det.scores(Xw[cal_idx]); yc = yw[cal_idx]
        s_cal = Sc[np.arange(len(yc)), yc]
        qhat = C.SortedScores(s_cal).q(ALPHA)
        m_cal = Sc.min(axis=1); conf_cal = 1 - m_cal
        mk_cal = Sc <= qhat
        cov_cal = float(mk_cal[np.arange(len(yc)), yc].mean())
        sz = mk_cal.sum(axis=1)
        lab = C.included_label(mk_cal)
        acc_single_cal = float((lab[sz == 1] == yc[sz == 1]).mean())
        full_cal = float((sz == 2).mean())
        t_atc = float(np.quantile(m_cal, cov_cal))

        def label_free(Sx, Xx):
            mx = Sx.min(axis=1)
            szx = (Sx <= qhat).sum(axis=1)
            return dict(KS=ks_stat(m_cal, mx), DOM=domain_auc(Xw[cal_idx], Xx, rng),
                        NE=float((szx == 1).mean()) * acc_single_cal + float((szx == 2).mean()),
                        ATC=float(np.mean(mx <= t_atc)),
                        DoC=cov_cal - (float(conf_cal.mean()) - float((1 - mx).mean())))

        # pseudo-deployment blocks (calibration window only)
        cov_vec_cal = mk_cal[np.arange(len(yc)), yc]
        cal_class = dict(pi=float(yc.mean()), cov_mal=float(cov_vec_cal[yc == 1].mean()),
                         cov_ben=float(cov_vec_cal[yc == 0].mean()))
        pb = []
        for b in pseudo:
            Sb = det.scores(Xw[b])
            lf = label_free(Sb, Xw[b])
            cvb = Sb[np.arange(len(b)), yw[b]] <= qhat
            lf["coverage"] = float(cvb.mean())
            lf["pi"] = float(yw[b].mean())
            lf["cov_mal"] = float(cvb[yw[b] == 1].mean()) if (yw[b] == 1).any() else None
            lf["cov_ben"] = float(cvb[yw[b] == 0].mean()) if (yw[b] == 0).any() else None
            lf["t_year"] = float(np.median(dw[b])) / 365.2425
            pb.append(lf)
        cal_t = float(np.median(dw[cal_idx])) / 365.2425
        slope_t, icept_t = linfit([p["t_year"] - cal_t for p in pb], [p["coverage"] for p in pb])
        velocity = -slope_t
        cal_year = datetime.fromordinal(int(np.median(dw[cal_idx]))).year
        breach_pred = {}
        for thr in P["breach_thresholds"]:
            breach_pred[str(thr)] = (float(cal_year + max(((1 - ALPHA) - thr) / velocity, 0.0))
                                     if velocity > 1e-9 else None)
        sl_ks, ic_ks = linfit([p["KS"] for p in pb], [p["coverage"] for p in pb])
        sl_dm, ic_dm = linfit([p["DOM"] for p in pb], [p["coverage"] for p in pb])

        dep = {}
        for ty in DEP_YEARS:
            m = dep_masks[ty]
            St = det.scores(X[m])
            lf = label_free(St, X[m])
            actual = float(np.mean(St[np.arange(int(m.sum())), y[m]] <= qhat))
            dep[ty] = dict(actual=actual, KS=lf["KS"], DOM=lf["DOM"],
                           pred=dict(CWDVD_KS=float(np.clip(ic_ks + sl_ks * lf["KS"], 0, 1)),
                                     CWDVD_DOM=float(np.clip(ic_dm + sl_dm * lf["DOM"], 0, 1)),
                                     NE=lf["NE"], ATC=lf["ATC"], DoC=float(np.clip(lf["DoC"], 0, 1))))
        # persistence: last year's realised coverage (requires last year's labels)
        for i, ty in enumerate(DEP_YEARS):
            dep[ty]["pred"]["Persistence"] = cov_cal if i == 0 else dep[DEP_YEARS[i - 1]]["actual"]
        # offset corrections
        for name in list(dep[DEP_YEARS[0]]["pred"].keys()):
            if name == "Persistence":
                continue
            errs = {ty: dep[ty]["actual"] - dep[ty]["pred"][name] for ty in DEP_YEARS}
            for i, ty in enumerate(DEP_YEARS):
                past = [errs[s] for s in DEP_YEARS[:i]]
                others = [errs[s] for s in DEP_YEARS if s != ty]
                dep[ty]["pred"][name + "+rolling"] = float(np.clip(dep[ty]["pred"][name] + (np.mean(past) if past else 0.0), 0, 1))
                dep[ty]["pred"][name + "+LOYO"] = float(np.clip(dep[ty]["pred"][name] + np.mean(others), 0, 1))
        obs_first = {}
        for thr in P["breach_thresholds"]:
            br = [ty for ty in DEP_YEARS if dep[ty]["actual"] < thr]
            obs_first[str(thr)] = br[0] if br else None
        out.append(dict(dataset=WHICH, base=base, seed=seed, qhat=qhat, cov_cal=cov_cal, cal_class=cal_class,
                        acc_single_cal=acc_single_cal, full_cal=full_cal, t_atc=t_atc,
                        pseudo_blocks=pb, velocity=velocity, cal_year=cal_year,
                        predicted_breach_year=breach_pred, observed_first_breach=obs_first,
                        beta_KS=-sl_ks, icept_KS=ic_ks, beta_DOM=-sl_dm, icept_DOM=ic_dm,
                        deployment=dep))
        print(f"[DIAG:{WHICH}:{base} s{seed}] {time.time() - t0:.0f}s velocity={velocity:.4f} "
              f"beta_KS={-sl_ks:.3f} beta_DOM={-sl_dm:.3f} "
              f"MAE KS={np.mean([abs(dep[t]['pred']['CWDVD_KS'] - dep[t]['actual']) for t in DEP_YEARS]):.4f} "
              f"NE={np.mean([abs(dep[t]['pred']['NE'] - dep[t]['actual']) for t in DEP_YEARS]):.4f}", flush=True)
        save_json(dict(blocks=block_info, runs=out), os.path.join(RES_DIR, f"diagnostic_{WHICH}.json"))
print("[DONE]", flush=True)
