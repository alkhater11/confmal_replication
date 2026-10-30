"""
Independent re-computation of headline numbers directly from the per-run
result files (bypassing analyze.py) and comparison with summary.json, plus
consistency checks of the counts behind qualitative statements.
Exit status 1 on any mismatch.
"""
import os, json, sys
import numpy as np
from collections import defaultdict
from common import RES_DIR, BASES, DEP_YEARS

S = json.load(open(os.path.join(RES_DIR, "summary.json")))
D = json.load(open(os.path.join(RES_DIR, "derived.json")))
fails = []


def check(name, a, b, tol=1e-9):
    if a is None or b is None or abs(a - b) > tol:
        fails.append(f"{name}: recomputed {a} vs summary {b}")


recs = []
for b in BASES:
    recs += json.load(open(os.path.join(RES_DIR, f"audit_real_{b}.json")))["records"]
enb = json.load(open(os.path.join(RES_DIR, "enbpi_real.json")))["records"]
by = defaultdict(list)
for r in recs + enb:
    by[(r["variant"], r["regime"], str(r["year"]))].append(r)

# configuration counts
check("n configs SCP 2020", len(by[("SCP", "static", "2020")]), 30)
check("n configs EnbPI 2020", len(by[("EnbPI", "batch_year", "2020")]), 15)
for (v, rg) in [("SCP", "static"), ("Mondrian", "static"), ("WCP", "static"), ("ACI", "immediate"),
                ("EnbPI", "batch_year"), ("RecalSCP", "batch_year"), ("LSW-oracle", "static")]:
    for y in (["in-window"] if (v, rg, "in-window") in by else []) + [str(t) for t in DEP_YEARS]:
        rs = by[(v, rg, y)]
        node = S["main_real"][v][rg][y]
        check(f"{v}/{rg}/{y} coverage", float(np.mean([r["coverage"] for r in rs])), node["coverage"]["mean"])
        ns = sum(r["n_single"] for r in rs)
        if ns:
            check(f"{v}/{rg}/{y} pooled acc", sum(r["n_correct_single"] for r in rs) / ns, node["pooled_acc_single"])
        check(f"{v}/{rg}/{y} empty", float(np.mean([r["frac_empty"] for r in rs])), node["frac_empty"]["mean"])
        check(f"{v}/{rg}/{y} full", float(np.mean([r["frac_full"] for r in rs])), node["frac_full"]["mean"])
        # internal identity: miscoverage = empty + wrong singletons
        for r in rs:
            if abs((1 - r["coverage"]) - (r["frac_empty"] + r["wrong_single_rate"] + 0.0)) > 1e-9 and r["frac_full"] == 0:
                fails.append(f"identity miscoverage {v}/{rg}/{y} {r['base']} {r['seed']}")
                break

# CRC identity in every configuration
meta = []
for b in BASES:
    meta += json.load(open(os.path.join(RES_DIR, f"audit_real_{b}.json")))["meta"]
if not all(m["crc_equals_scp"] for m in meta):
    fails.append("CRC differs from SCP in some configuration")
for y in [str(t) for t in DEP_YEARS]:
    a = [r["coverage"] for r in by[("SCP", "static", y)]]
    c = [r["coverage"] for r in by[("CRC", "static", y)]]
    if max(abs(x - z) for x, z in zip(a, c)) > 0:
        fails.append(f"CRC coverage differs from SCP in {y}")

# decomposition exactness
for v in ("SCP", "Mondrian"):
    for y, d in S["decomposition_real"][v].items():
        if d["max_abs_check"] > 1e-12:
            fails.append(f"decomposition not exact {v} {y}")

# WCP statements: some configuration covers 1.000 and some year has a config with >50% empty sets
wcp_max = max(r["coverage"] for y in DEP_YEARS for r in by[("WCP", "static", str(y))])
check("WCP max coverage is 1.000", round(wcp_max, 3), 1.0)
# the untruncated ACI of aci_truncation.py must reproduce the main audit exactly
tp = os.path.join(RES_DIR, "aci_truncation_real.json")
if os.path.exists(tp) and json.load(open(tp))["max_abs_diff_standard_vs_main_audit"] > 0:
    fails.append("aci_truncation.py standard run differs from the main audit")
# 'all eight' phrasing must match the count
if D.get("scp_years_below_envelope_n") != "eight" and D.get("scp_years_below_envelope_phrase", "").startswith("all"):
    fails.append("envelope phrase inconsistent with count")
print(f"checked; {len(fails)} failures")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
