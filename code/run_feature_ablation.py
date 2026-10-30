"""
Feature-group ablation on the real-device subset: split conformal, Mondrian
and ACI (immediate feedback) with static features only and with dynamic
(system-call) features only.  Because the emulator subset shares the static
features of almost all its applications with the real-device subset, this
ablation shows how much of the audit's behaviour is carried by the feature
group that the two subsets have in common.

Usage: python3 run_feature_ablation.py real
"""
import sys, os
import numpy as np
from common import (CFG, ALPHA, DEP_YEARS, RES_DIR, BASES, load_subset, window_mask, window_split,
                    Detector, save_json, dynamic_static_indices)
import conformal_lib as C

WHICH = sys.argv[1] if len(sys.argv) > 1 else "real"
SEEDS = CFG["seeds_ablation"]
D = load_subset(WHICH)
X, y, years, dates = D["X"], D["y"], D["years"], D["dates"]
dyn_idx, stat_idx = dynamic_static_indices(D["feature_names"])
win = window_mask(years)
dep = np.isin(years, DEP_YEARS)
yw = y[win]
yd, yrd, dd = y[dep], years[dep], dates[dep]
records = []
for group, cols in (("static", stat_idx), ("dynamic", dyn_idx)):
    Xw, Xd = X[win][:, cols], X[dep][:, cols]
    for base in BASES:
        for seed in SEEDS:
            fit, cal, test = window_split(len(yw), seed)
            det = Detector(base, seed).fit(Xw[fit], yw[fit])
            Sc = det.scores(Xw[cal]); yc = yw[cal]
            sc_true = Sc[np.arange(len(yc)), yc]
            cs = C.SortedScores(sc_true)
            q = cs.q(ALPHA)
            qm = C.mondrian_thresholds(sc_true, yc, ALPHA)
            S_iw, y_iw = det.scores(Xw[test]), yw[test]
            S = det.scores(Xd)
            mk_aci, _, _ = C.run_aci(S, yd, dd, yrd, cs, ALPHA, CFG["aci"]["gamma"], regime="immediate")
            rec = C.set_metrics(S_iw <= q, y_iw, S_iw)
            rec.update(group=group, base=base, seed=seed, variant="SCP", year="in-window")
            records.append(rec)
            for ty in DEP_YEARS:
                m = yrd == ty
                for v, mk in (("SCP", S[m] <= q), ("Mondrian", S[m] <= qm[None, :]), ("ACI", mk_aci[m])):
                    rec = C.set_metrics(mk, yd[m], S[m])
                    rec.update(group=group, base=base, seed=seed, variant=v, year=ty)
                    records.append(rec)
            print(f"[ablation {group} {base} s{seed}] in-window acc "
                  f"{np.mean(np.argmin(S_iw, 1) == y_iw):.4f}", flush=True)
    save_json(records, os.path.join(RES_DIR, f"ablation_{WHICH}.json"))
print("[DONE]", flush=True)
