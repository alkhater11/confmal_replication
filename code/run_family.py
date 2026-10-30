"""
Per-family breakdown over the pooled 2013-2020 deployment span (malware only,
so coverage here is malware-class coverage).  For the fourteen largest
deployment families we report split-conformal and ACI coverage together with
the two components of split-conformal miscoverage: rejection (empty set,
visible without labels) and wrong singletons (silent errors).

Usage: python3 run_family.py real
"""
import sys, os
import numpy as np
from collections import Counter
from common import CFG, ALPHA, DEP_YEARS, BASES, RES_DIR, load_subset, window_mask, window_split, Detector, save_json
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
SEEDS = CFG["seeds_family"]
D = load_subset(WHICH)
X, y, years, dates, fams = D["X"], D["y"], D["years"], D["dates"], D["fams"]
win = window_mask(years)
dep = np.isin(years, DEP_YEARS)
Xw, yw = X[win], y[win]
Xd, yd, yrd, dd, fd = X[dep], y[dep], years[dep], dates[dep], fams[dep]
win_fams = set(f for f in fams[win & (y == 1)] if f)
dep_counts = Counter(f for f in fd[yd == 1] if f)
top = [f for f, _ in dep_counts.most_common(14)]
n_dep_mal = int((yd == 1).sum())
print(f"[FAMILIES] window {len(win_fams)} families; deployment {len(dep_counts)}; "
      f"top-14 cover {sum(dep_counts[f] for f in top)}/{n_dep_mal}", flush=True)
acc = {f: dict(scp=[], aci=[], empty=[], wrong=[], acc_single=[], aci_empty=[], aci_full=[]) for f in top}
for base in BASES:
    for seed in SEEDS:
        fit, cal, test = window_split(len(yw), seed)
        det = Detector(base, seed).fit(Xw[fit], yw[fit])
        Sc = det.scores(Xw[cal]); yc = yw[cal]
        cs = C.SortedScores(Sc[np.arange(len(yc)), yc])
        q = cs.q(ALPHA)
        S = det.scores(Xd)
        mk = S <= q
        mk_aci, _, _ = C.run_aci(S, yd, dd, yrd, cs, ALPHA, CFG["aci"]["gamma"], regime="immediate")
        for f in top:
            m = fd == f
            mt, yt = mk[m], yd[m]
            sz = mt.sum(axis=1)
            lab = C.included_label(mt)
            ma = mk_aci[m]; sza = ma.sum(axis=1)
            acc[f]["scp"].append(float(mt[np.arange(len(yt)), yt].mean()))
            acc[f]["empty"].append(float((sz == 0).mean()))
            acc[f]["wrong"].append(float(((sz == 1) & (lab != yt)).mean()))
            acc[f]["acc_single"].append(float((lab[sz == 1] == yt[sz == 1]).mean()) if (sz == 1).any() else np.nan)
            acc[f]["aci"].append(float(ma[np.arange(len(yt)), yt].mean()))
            acc[f]["aci_empty"].append(float((sza == 0).mean()))
            acc[f]["aci_full"].append(float((sza == 2).mean()))
    print(f"[{base}] done", flush=True)
out = []
for f in top:
    a = acc[f]
    first_year = int(yrd[(fd == f)].min())
    yrs = Counter(int(t) for t in yrd[fd == f])
    out.append(dict(family=f, n=dep_counts[f], seen_in_window=f in win_fams,
                    n_window=int(np.sum((fams[win] == f) & (yw == 1))),
                    years=dict(sorted(yrs.items())),
                    coverage_split=float(np.mean(a["scp"])), coverage_split_sd=float(np.std(a["scp"])),
                    empty_split=float(np.mean(a["empty"])), wrong_single_split=float(np.mean(a["wrong"])),
                    acc_single_split=float(np.nanmean(a["acc_single"])),
                    coverage_aci=float(np.mean(a["aci"])), coverage_aci_sd=float(np.std(a["aci"])),
                    empty_aci=float(np.mean(a["aci_empty"])), full_aci=float(np.mean(a["aci_full"]))))
    r = out[-1]
    print(f"  {f[:30]:30s} n={r['n']:5d} seen={r['seen_in_window']!s:5s} SCP={r['coverage_split']:.4f} "
          f"empty={r['empty_split']:.3f} wrong={r['wrong_single_split']:.3f} ACI={r['coverage_aci']:.4f}", flush=True)
save_json(dict(n_window_families=len(win_fams), n_deployment_families=len(dep_counts),
               n_deployment_malware=n_dep_mal, top14_total=sum(dep_counts[f] for f in top),
               families=out), os.path.join(RES_DIR, f"family_{WHICH}.json"))
print("[SAVED]", flush=True)
