"""
Library-version sensitivity of the main audit.

The main audit (run_audit.py real) is run twice with identical code, data,
configuration and seeds, once under each of two scikit-learn releases, and the
per-configuration results are compared here.

  python3 compare_versions.py <results_dir_A> <label_A> <results_dir_B> <label_B>

<results_dir_*> each hold audit_real_<detector>.json for the three detectors.
Writes results/version_sensitivity.json, which make_numbers.py reads.

The second set of results is shipped in results/alt_sklearn_1.6.1/; the
package README (section 7) gives the commands that regenerate it.
"""
import json, os, sys
import numpy as np
from common import RES_DIR, BASES, DEP_YEARS

DIR_A, LAB_A, DIR_B, LAB_B = sys.argv[1:5]
TREES = [b for b in BASES if b != "LogReg"]
YEARS = [str(t) for t in DEP_YEARS]


def load(d):
    recs, meta = {}, {}
    for b in BASES:
        J = json.load(open(os.path.join(d, f"audit_real_{b}.json")))
        for r in J["records"]:
            recs[(r["variant"], r["regime"], r["base"], int(r["seed"]), str(r["year"]))] = r
        for m in J["meta"]:
            meta[(m["base"], int(m["seed"]))] = m
    return recs, meta


RA, MA = load(DIR_A)
RB, MB = load(DIR_B)
common = sorted(set(MA) & set(MB))
seeds = sorted(set(s for _, s in common))
out = {"A": LAB_A, "B": LAB_B, "n_configurations": len(common), "seeds": seeds, "procedures": {}}

for v, rg in [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate")]:
    node = {}
    for b in BASES:
        d_cfg, worst = [], None
        for y in ["in-window"] + YEARS:
            for s in seeds:
                k = (v, rg, b, s, y)
                if k in RA and k in RB:
                    dd = abs(RA[k]["coverage"] - RB[k]["coverage"])
                    d_cfg.append(dd)
                    if worst is None or dd > worst[0]:
                        worst = (dd, y, s, RA[k]["n"])
        node[b] = {"max_abs_diff_config": max(d_cfg) if d_cfg else None,
                   "worst": {"diff": worst[0], "year": worst[1], "seed": worst[2], "n": worst[3]} if worst else None}
    # yearly means over all configurations
    dm = {}
    for y in ["in-window"] + YEARS:
        a = [RA[(v, rg, b, s, y)]["coverage"] for b in BASES for s in seeds if (v, rg, b, s, y) in RA]
        c = [RB[(v, rg, b, s, y)]["coverage"] for b in BASES for s in seeds if (v, rg, b, s, y) in RB]
        if a and len(a) == len(c):
            dm[y] = float(np.mean(a) - np.mean(c))
    node["mean_diff_by_year"] = dm
    node["max_abs_mean_diff"] = max(abs(x) for x in dm.values()) if dm else None
    node["max_abs_diff_trees"] = max(node[b]["max_abs_diff_config"] for b in TREES)
    node["max_abs_diff_logreg"] = node["LogReg"]["max_abs_diff_config"]
    out["procedures"][f"{v}|{rg}"] = node

# thresholds and in-window accuracy
out["q_scp_max_abs_diff"] = {b: max(abs(MA[(b, s)]["q_scp"] - MB[(b, s)]["q_scp"]) for s in seeds) for b in BASES}
out["in_window_acc_max_abs_diff"] = {b: max(abs(MA[(b, s)]["in_window_argmax_acc"] - MB[(b, s)]["in_window_argmax_acc"])
                                            for s in seeds) for b in BASES}
json.dump(out, open(os.path.join(RES_DIR, "version_sensitivity.json"), "w"), indent=1)
P = out["procedures"]["SCP|static"]
print(f"SCP: trees max |d| per configuration {P['max_abs_diff_trees']:.3f}, "
      f"LogReg {P['max_abs_diff_logreg']:.3f}, yearly means {P['max_abs_mean_diff']:.3f}")
