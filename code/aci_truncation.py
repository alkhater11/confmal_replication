"""
Effect of the alpha_t truncation used in version 1 of the code.

Version 1 clipped the ACI level to [1e-4, 0.999] after every update. With
n calibration points the threshold at level alpha_t is the
ceil((n+1)(1 - alpha_t))-th order statistic, so a floor of 1e-4 > 1/(n+1)
caps the threshold at the largest calibration score: the full-set state
(alpha_t <= 0, threshold +inf) that the guarantee relies on can never be
reached, and confident errors cannot be covered. Version 2 uses the standard
untruncated update.

This script re-runs ACI with immediate feedback in both forms on the thirty
main configurations, changing nothing but the truncation. The untruncated run
must reproduce the main audit exactly; the script checks this.

Usage: python3 aci_truncation.py [real|emu]
Writes results/aci_truncation_<subset>.json.
"""
import json, os, sys, time
import numpy as np
from common import CFG, ALPHA, DEP_YEARS, BASES, RES_DIR, load_subset, window_mask, window_split, Detector, save_json
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
GAMMA = CFG["aci"]["gamma"]
LO, HI = 1e-4, 0.999


def aci_truncated(S, y, cal_sorted, alpha, gamma, lo, hi):
    """Immediate-feedback ACI with the version-1 clipping of alpha_t."""
    n = len(y)
    mask = np.zeros((n, 2), dtype=bool)
    a_t = alpha
    s_sorted, ncal = cal_sorted.s, cal_sorted.n
    for i in range(n):
        k = int(np.ceil((ncal + 1) * (1.0 - a_t)))
        q = np.inf if k > ncal else (-np.inf if k < 1 else s_sorted[k - 1])
        c0, c1 = S[i, 0] <= q, S[i, 1] <= q
        mask[i, 0], mask[i, 1] = c0, c1
        err = 0.0 if (c1 if y[i] == 1 else c0) else 1.0
        a_t = float(np.clip(a_t + gamma * (alpha - err), lo, hi))
    return mask


D = load_subset(WHICH)
X, y, years, dates = D["X"], D["y"], D["years"], D["dates"]
win = window_mask(years)
dep = np.isin(years, DEP_YEARS)
Xw, yw = X[win], y[win]
Xd, yd, yrd, dd = X[dep], y[dep], years[dep], dates[dep]

main = {}
for b in BASES:
    for r in json.load(open(os.path.join(RES_DIR, f"audit_{WHICH}_{b}.json")))["records"]:
        if r["variant"] == "ACI" and r["regime"] == "immediate":
            main[(b, int(r["seed"]), str(r["year"]))] = r["coverage"]

rows, max_check = [], 0.0
for base in BASES:
    for seed in CFG["seeds_main"]:
        t0 = time.time()
        fit, cal, _ = window_split(len(yw), seed)
        det = Detector(base, seed).fit(Xw[fit], yw[fit])
        S_cal, y_cal = det.scores(Xw[cal]), yw[cal]
        cal_sorted = C.SortedScores(S_cal[np.arange(len(y_cal)), y_cal])
        S_dep = det.scores(Xd)
        mk_std, _, _ = C.run_aci(S_dep, yd, dd, yrd, cal_sorted, ALPHA, GAMMA, regime="immediate")
        mk_tr = aci_truncated(S_dep, yd, cal_sorted, ALPHA, GAMMA, LO, HI)
        for ty in DEP_YEARS:
            m = yrd == ty
            row = dict(base=base, seed=seed, year=ty, n=int(m.sum()))
            for name, mk in (("standard", mk_std), ("truncated", mk_tr)):
                mt, yt = mk[m], yd[m]
                row[f"coverage_{name}"] = float(mt[np.arange(len(yt)), yt].mean())
                row[f"frac_full_{name}"] = float((mt.sum(axis=1) == 2).mean())
                row[f"frac_empty_{name}"] = float((mt.sum(axis=1) == 0).mean())
            max_check = max(max_check, abs(row["coverage_standard"] - main[(base, seed, str(ty))]))
            rows.append(row)
        print(f"[{WHICH}:{base} seed={seed}] {time.time() - t0:.0f}s", flush=True)

summary = {}
for ty in DEP_YEARS:
    rr = [r for r in rows if r["year"] == ty]
    summary[str(ty)] = {k: float(np.mean([r[k] for r in rr])) for k in
                        ("coverage_standard", "coverage_truncated", "frac_full_standard", "frac_full_truncated",
                         "frac_empty_standard", "frac_empty_truncated")}
out = dict(subset=WHICH, gamma=GAMMA, clip=[LO, HI], n_configurations=len(BASES) * len(CFG["seeds_main"]),
           max_abs_diff_standard_vs_main_audit=max_check, by_year=summary, rows=rows)
save_json(out, os.path.join(RES_DIR, f"aci_truncation_{WHICH}.json"))
print(f"standard run reproduces the main audit to {max_check:.2e}")
for ty in DEP_YEARS:
    s = summary[str(ty)]
    print(f"{ty}: standard {s['coverage_standard']:.4f} (full {s['frac_full_standard']:.3f})   "
          f"truncated {s['coverage_truncated']:.4f} (full {s['frac_full_truncated']:.3f})")
