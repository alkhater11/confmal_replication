"""
Derived numbers quoted in the manuscript prose (results/derived.json).
Every value is computed from results/summary.json, the per-run result files
or the processed data; none is typed by hand.
"""
import os, json, csv, hashlib
from collections import Counter
import numpy as np
from common import RES_DIR, DATA_DIR, KD_DIR, CFG, DEP_YEARS, BASES, load_subset, window_mask

S = json.load(open(os.path.join(RES_DIR, "summary.json")))
YRS = [str(y) for y in DEP_YEARS]
D = {}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def m(which, v, r, y, metric, stat="mean"):
    node = S[f"main_{which}"][v][r][str(y)]
    if metric.startswith("pooled_") or metric in ("n_samples", "n_mal", "n_ben", "frac_configs_below_nominal"):
        return node[metric]
    return node[metric][stat]


def over_years(which, v, r, metric, stat="mean"):
    return [m(which, v, r, y, metric, stat) for y in YRS]


def first_breach(which, v, r, thr=0.80):
    for y in YRS:
        if m(which, v, r, y, "coverage") < thr:
            return int(y)
    return None


# ------------------------------------------------------------ data facts
d = load_subset("real")
yrs = d["years"]
win = window_mask(yrs)
D["n_window"] = int(win.sum())
keys = [hashlib.md5(row.tobytes()).hexdigest() for row in d["X"]]
groups = {}
for i, k in enumerate(keys):
    groups.setdefault(k, []).append(i)
dup = [g for g in groups.values() if len(g) > 1]
D["featdup_groups"] = len(dup)
D["featdup_rows"] = sum(len(g) for g in dup)
D["featdup_conflicting"] = sum(1 for g in dup if len(set(d["y"][g])) > 1)
D["featdup_crossing"] = sum(1 for g in dup if win[g].any() and (~win[g]).any())
fam_counts = Counter(f for f, l in zip(d["fams"], d["y"]) if l == 1 and f)
for i, (f, c) in enumerate(fam_counts.most_common(5), 1):
    D[f"fam{i}_name"] = f
    D[f"fam{i}_n"] = c

csv.field_size_limit(10 ** 8)
drop = set(CFG["dropped_columns"])
cols_bad = Counter()
for fn in ("real_malware_v1.csv", "real_legitimate_v1.csv"):
    p = os.path.join(KD_DIR, fn)
    if not os.path.exists(p):
        continue
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        fc = [c for c in rdr.fieldnames if c not in drop]
        for row in rdr:
            if (row.get("EarliestModDate") or "").strip() in ("", "1980-00-00"):
                continue
            for c in fc:
                v = row.get(c, "")
                if v in ("", None):
                    continue
                try:
                    float(v)
                except ValueError:
                    cols_bad[c] += 1
                    break
if cols_bad:
    top, n_top = cols_bad.most_common(1)[0]
    total = sum(cols_bad.values())
    D["none_column_note"] = (f"in the \\texttt{{{top}}} count in every case" if n_top == total
                             else f"in the \\texttt{{{top}}} count in {n_top} of {total} cases")
else:
    D["none_column_note"] = "see the selection records"

O = S["overlap"]["per_year"]
eo = [O[y]["emu_only_malware"] + O[y]["emu_only_benign"] for y in YRS]
D["emu_only_min_year"], D["emu_only_max_year"] = min(eo), max(eo)
D["kd_commit"] = CFG["kronodroid"]["commit"][:12]

# ------------------------------------------------------------ timing
meta = S["meta_real"]
D["fit_rf"] = meta["RandomForest"]["fit_seconds"]["mean"]
D["fit_gb"] = meta["HistGB"]["fit_seconds"]["mean"]
D["fit_lr"] = meta["LogReg"]["fit_seconds"]["mean"]
D["cfg_seconds_min"] = min(meta[b]["seconds"]["mean"] for b in BASES)
D["cfg_seconds_max"] = max(meta[b]["seconds"]["mean"] for b in BASES)
D["enbpi_seconds"] = S["enbpi_meta_real"]["seconds"]["mean"] if S["enbpi_meta_real"]["seconds"] else 0.0

# ------------------------------------------------------------ audit facts
acc = over_years("real", "SCP", "static", "pooled_acc_single")
D["scp_accacc_min"], D["scp_accacc_max"] = min(acc), max(acc)
D["scp_accacc_min_year"] = YRS[int(np.argmin(acc))]
sil = over_years("real", "SCP", "static", "wrong_single_rate")
D["scp_silent_min"], D["scp_silent_max"] = min(sil), max(sil)
D["scp_silent_max_year"] = YRS[int(np.argmax(sil))]
rej = over_years("real", "SCP", "static", "pooled_argmax_acc_empty")
nmaj = sum(1 for v in rej if v > 0.5)
D["scp_rej_majority_correct_years"] = WORDS[nmaj]
D["scp_rej_acc_max"], D["scp_rej_acc_min"] = max(rej), min(rej)
D["scp_rej_acc_max_year"] = YRS[int(np.argmax(rej))]
cap = over_years("real", "SCP", "static", "pooled_err_recall_empty")
D["scp_capture_min"], D["scp_capture_max"] = min(cap), max(cap)
nov = over_years("real", "SCP", "static", "novelty_auc_empty_vs_single")
D["scp_nov_min"], D["scp_nov_max"] = min(nov), max(nov)
D["scp_nov_below_half"] = WORDS[sum(1 for v in nov if v < 0.5)]
D["scp_first_breach"] = first_breach("real", "SCP", "static")
D["mond_first_breach"] = first_breach("real", "Mondrian", "static")
D["wcp_first_breach"] = first_breach("real", "WCP", "static")
D["wcp_cfg_min"] = min(over_years("real", "WCP", "static", "coverage", "min"))
D["wcp_sd_max"] = max(over_years("real", "WCP", "static", "coverage", "sd"))
D["wcp_cov_min"], D["wcp_cov_max"] = min(over_years("real", "WCP", "static", "coverage")), max(over_years("real", "WCP", "static", "coverage"))
ess = over_years("real", "WCP", "static", "wcp_ess")
D["wcp_ess_min"], D["wcp_ess_max"] = min(ess), max(ess)
D["wcp_ess_max_ceil"] = float(np.ceil(max(ess) / 10.0) * 10)
ess_med = [S["main_real"]["WCP"]["static"][y]["wcp_ess"]["mean"] for y in YRS]
D["wcp_full_mean"] = float(np.mean(over_years("real", "WCP", "static", "frac_full")))
dec = S["decomposition_real"]["SCP"]
D["comp_abs_max"] = max(abs(dec[y]["composition"]) for y in YRS)
D["within_abs_max"] = max(abs(dec[y]["within"]) for y in YRS)
D["comp_share_max"] = max(abs(dec[y]["composition"]) / abs(dec[y]["total"]) for y in YRS if abs(dec[y]["total"]) > 0.1)
D["envelope_lo"], D["envelope_hi"] = dec["2013"]["envelope_lo"], dec["2013"]["envelope_hi"]
decm = S["decomposition_real"]["Mondrian"]
D["mond_within_abs_max"] = max(abs(decm[y]["within"]) for y in YRS)
cm = over_years("real", "SCP", "static", "cov_mal")
D["scp_covmal_2020"] = cm[-1]
D["acc_argmax_iw"] = S["main_real"]["SCP"]["static"]["in-window"]["argmax_acc"]["mean"]
aa = over_years("real", "SCP", "static", "argmax_acc")
D["acc_argmax_2018"], D["acc_argmax_2019"], D["acc_argmax_2020"] = aa[5], aa[6], aa[7]
D["acc_argmax_min"] = min(aa)
D["need_full_2020"] = max(0.0, 0.90 - aa[7])
D["need_full_2018"] = max(0.0, 0.90 - aa[5])
bb = S["meta_real"]["RandomForest"]["bbse"]
D["bbse_err_max_early"] = max(abs(np.mean([S["meta_real"][b]["bbse"][y]["est"]["mean"] for b in BASES]) - bb[y]["true"]) for y in ["2013", "2014", "2015", "2016", "2017"])
D["bbse_2018_est"] = float(np.mean([S["meta_real"][b]["bbse"]["2018"]["est"]["mean"] for b in BASES]))
D["bbse_2020_est"] = float(np.mean([S["meta_real"][b]["bbse"]["2020"]["est"]["mean"] for b in BASES]))
D["bbse_2018_true"], D["bbse_2020_true"] = bb["2018"]["true"], bb["2020"]["true"]

ac = over_years("real", "ACI", "immediate", "coverage")
D["aci_imm_cov_min"], D["aci_imm_cov_max"] = min(ac), max(ac)
D["aci_imm_cfg_min"] = min(over_years("real", "ACI", "immediate", "coverage", "min"))
D["aci_imm_cfg_min_year"] = YRS[int(np.argmin(over_years("real", "ACI", "immediate", "coverage", "min")))]
D["aci_imm_abst"] = float(np.mean(over_years("real", "ACI", "immediate", "abstention")))
D["aci_d7_abst"] = float(np.mean(over_years("real", "ACI", "delay7", "abstention")))
D["aci_d30_abst"] = float(np.mean(over_years("real", "ACI", "delay30", "abstention")))
D["aci_imm_accacc_min"] = min(over_years("real", "ACI", "immediate", "pooled_acc_single"))
D["aci_imm_accacc_min_year"] = YRS[int(np.argmin(over_years("real", "ACI", "immediate", "pooled_acc_single")))]
for reg in ["immediate", "batch_year", "delay7", "delay30", "delay90", "delay180", "delay365",
            "partial0.5", "partial0.2", "partial0.1", "partial0.05"]:
    for v, tag in (("ACI", "aci"), ("ACI_g0.001", "acig001"), ("RecalSCP", "recal"), ("EnbPI", "enbpi")):
        try:
            c = over_years("real", v, reg, "coverage")
            a = over_years("real", v, reg, "abstention")
        except KeyError:
            continue
        D[f"{tag}_{reg}_covmin"], D[f"{tag}_{reg}_covmax"] = min(c), max(c)
        D[f"{tag}_{reg}_covmean"] = float(np.mean(c))
        D[f"{tag}_{reg}_abst"] = float(np.mean(a))
for k, sp in S["aci_span_real"].items():
    if sp["stats"]:
        D["span_" + k.replace("|", "_")] = sp["stats"]["mean"]

# ------------------------------------------------------------ further prose numbers
pb = S["per_base_real"]
D["lr_scp_min"] = min(pb["LogReg"]["SCP"]["static"][y]["coverage"]["mean"] for y in YRS)
D["lr_scp_max"] = max(pb["LogReg"]["SCP"]["static"][y]["coverage"]["mean"] for y in YRS)
D["mond_beats_scp_all"] = all(pb[b]["Mondrian"]["static"][y]["coverage"]["mean"] > pb[b]["SCP"]["static"][y]["coverage"]["mean"] for b in BASES for y in YRS)
D["scp_accacc_2013_2017_min"] = min(m("real", "SCP", "static", y, "pooled_acc_single") for y in YRS[:5])
D["q_scp_min"] = min(S["meta_real"][b]["q_scp"]["min"] for b in BASES)    # over all thirty configurations
D["q_scp_max"] = max(S["meta_real"][b]["q_scp"]["max"] for b in BASES)
D["comp_abs_min"] = min(abs(dec[y]["composition"]) for y in YRS)
R = S["resampled_real"]
res_vals = [R[str(p)]["ACI"][y]["coverage"]["mean"] for p in CFG["resampled_stream"]["priors"] for y in YRS]
D["res_aci_min"], D["res_aci_max"] = min(res_vals), max(res_vals)
recal_delay_abst = [D[f"recal_delay{dd}_abst"] for dd in CFG["aci"]["delays_days"] if f"recal_delay{dd}_abst" in D]
if recal_delay_abst:
    D["recal_abst_delay_min"], D["recal_abst_delay_max"] = min(recal_delay_abst), max(recal_delay_abst)


def burden(v, r, ys):
    return float(np.mean([m("real", v, r, y, "abstention") + m("real", v, r, y, "extra_defer_rolling") for y in ys]))


D["burden_scp_2020"], D["burden_aci_2020"] = burden("SCP", "static", ["2020"]), burden("ACI", "immediate", ["2020"])
D["burden_scp_mean"], D["burden_aci_mean"] = burden("SCP", "static", YRS), burden("ACI", "immediate", YRS)
D["burden_ratio_2020"] = D["burden_scp_2020"] / D["burden_aci_2020"]
D["burden_ratio_mean"] = D["burden_scp_mean"] / D["burden_aci_mean"]
# pooled split-conformal coverage over 2013-2020 (weighted by year size)
nn = [m("real", "SCP", "static", y, "n_samples") for y in YRS]
D["scp_pooled_cov"] = float(np.sum([m("real", "SCP", "static", y, "coverage") * n for y, n in zip(YRS, nn)]) / np.sum(nn))
# emulator
if "main_emu" in S:
    D["emu_scp_first_breach"] = first_breach("emu", "SCP", "static")
    ea = over_years("emu", "ACI", "immediate", "coverage")
    D["emu_aci_min"], D["emu_aci_max"] = min(ea), max(ea)
    D["emu_scp_accacc_min"] = min(over_years("emu", "SCP", "static", "pooled_acc_single"))
    ew = over_years("emu", "WCP", "static", "coverage")
    D["emu_wcp_min"], D["emu_wcp_max"] = min(ew), max(ew)
    dece = S["decomposition_emu"]["SCP"]
    D["emu_within_abs_max"] = max(abs(dece[y]["within"]) for y in YRS)
    D["emu_comp_abs_max"] = max(abs(dece[y]["composition"]) for y in YRS)
# families
F = S.get("family_real")
if F:
    fams = F["families"]
    covs = [f["coverage_split"] for f in fams]
    D["fam_below_nominal"] = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "all fourteen"}.get(
        sum(1 for c in covs if c < 0.90), str(sum(1 for c in covs if c < 0.90)))
    i_min = int(np.argmin(covs))
    D["fam_min_cov"], D["fam_min_name"] = covs[i_min], fams[i_min]["family"]
    D["fam_max_cov"], D["fam_max_name"] = max(covs), fams[int(np.argmax(covs))]["family"]
    D["fam_aci_min"] = min(f["coverage_aci"] for f in fams)
    D["fam_aci_max"] = max(f["coverage_aci"] for f in fams)
    n5 = sum(1 for f in fams if f["wrong_single_split"] > 0.05)
    D["fam_silent_over5"] = WORDS.get(n5, str(n5)) if n5 <= 8 else str(n5)
    D["fam_all_seen"] = all(f["seen_in_window"] for f in fams)

json.dump(D, open(os.path.join(RES_DIR, "derived.json"), "w"), indent=1, default=float)
print(f"{len(D)} derived numbers written")


# ------------------------------------------------------------ counts behind qualitative statements
def yearlist(ys):
    ys = [str(y) for y in ys]
    if not ys:
        return "no year"
    if len(ys) == 1:
        return ys[0]
    return ", ".join(ys[:-1]) + " and " + ys[-1]


acc_scp = over_years("real", "SCP", "static", "pooled_acc_single")
acc_aci = over_years("real", "ACI", "immediate", "pooled_acc_single")
D["aci_acc_lower_years_n"] = WORDS[sum(1 for a, b in zip(acc_aci, acc_scp) if a < b)]
D["aci_acc_below_target_years"] = yearlist([y for y, a in zip(YRS, acc_aci) if a < 0.90])
D["scp_acc_below_target_years"] = yearlist([y for y, a in zip(YRS, acc_scp) if a < 0.90])
cov_scp = over_years("real", "SCP", "static", "coverage")
D["scp_years_below_envelope_n"] = WORDS[sum(1 for c in cov_scp if c < D["envelope_lo"])]
_nb = sum(1 for c in cov_scp if c < D["envelope_lo"])
D["scp_years_below_envelope_phrase"] = ("all eight deployment years lie" if _nb == len(YRS)
                                        else f"{WORDS[_nb]} of the eight deployment years lie")
D["scp_years_far_below_envelope"] = yearlist([y for y, c in zip(YRS, cov_scp) if c < D["envelope_lo"] - 0.05])
D["aci_years_majority_below_n"] = WORDS[sum(1 for y in YRS if m("real", "ACI", "immediate", y, "frac_configs_below_nominal") > 0.5)]
for reg in ("delay30", "delay90", "delay180", "delay365", "batch_year"):
    fulls = over_years("real", "ACI", reg, "frac_full")
    empties = over_years("real", "ACI", reg, "frac_empty")
    D[f"aci_{reg}_years_full90"] = WORDS[sum(1 for f in fulls if f > 0.9)]
    D[f"aci_{reg}_years_empty40"] = WORDS[sum(1 for e in empties if e > 0.4)]
    D[f"aci_{reg}_full_min"] = min(fulls)
lsw = over_years("real", "LSW-oracle", "static", "coverage")
D["lsw_minus_scp_min"] = min(a - b for a, b in zip(lsw, cov_scp))
D["lsw_minus_scp_max"] = max(a - b for a, b in zip(lsw, cov_scp))
D["acc_single_mean_2019_scp"] = m("real", "SCP", "static", "2019", "acc_single")
json.dump(D, open(os.path.join(RES_DIR, "derived.json"), "w"), indent=1, default=float)
print(f"{len(D)} derived numbers written (final)")


# ------------------------------------------------------------ diagnostic prose numbers
for which in ("real", "emu"):
    DG = S.get(f"diagnostic_{which}")
    if not DG:
        continue
    cal = DG["calendar"]
    p80 = [cal[b]["predicted_breach_080"]["mean"] for b in BASES if cal[b]["predicted_breach_080"]]
    D[f"{which}_cwdv_pred080_min"], D[f"{which}_cwdv_pred080_max"] = min(p80), max(p80)
    obs80 = sorted(set(y for b in BASES for y in cal[b]["observed_first_breach_080"] if y))
    D[f"{which}_cwdv_obs080"] = yearlist(obs80)
    vel = [cal[b]["velocity"]["mean"] for b in BASES]
    D[f"{which}_cwdv_vel_min"], D[f"{which}_cwdv_vel_max"] = min(vel), max(vel)
    pdc = DG.get("pseudo_decomposition", {})
    late = [blk for b in BASES for blk in pdc.get(b, [])[-2:]]
    if late:
        D[f"{which}_pseudo_comp_late_min"] = min(x["composition"] for x in late)
        D[f"{which}_pseudo_comp_late_max"] = max(x["composition"] for x in late)
        D[f"{which}_pseudo_within_late_min"] = min(x["within"] for x in late)
        D[f"{which}_pseudo_within_late_max"] = max(x["within"] for x in late)
        D[f"{which}_pseudo_pi_late_min"] = min(x["pi"] for x in late)
        D[f"{which}_pseudo_total_late_min"] = min(x["total"] for x in late)
    D[f"{which}_cal_block_pi"] = DG.get("cal_block_pi")
    runs = json.load(open(os.path.join(RES_DIR, f"diagnostic_{which}.json")))["runs"]
    # thresholds 0.70: predicted vs observed per detector
    for b in BASES:
        rb = [r for r in runs if r["base"] == b]
        pr = [r["predicted_breach_year"]["0.7"] for r in rb if r["predicted_breach_year"]["0.7"]]
        ob = [r["observed_first_breach"]["0.7"] for r in rb if r["observed_first_breach"]["0.7"]]
        D[f"{which}_cwdv_pred070_{b}"] = float(np.mean(pr)) if pr else None
        D[f"{which}_cwdv_obs070_{b}"] = int(np.median(ob)) if ob else None
    P = DG["predictors"]
    for k in ("CWDVD_KS", "CWDVD_KS+rolling", "CWDVD_KS+LOYO", "NE", "NE+rolling", "ATC", "ATC+rolling", "DoC", "Persistence", "CWDVD_DOM", "CWDVD_DOM+rolling"):
        t = k.replace("+", "_")
        D[f"{which}_{t}_mae"] = P[k]["mae"]
        D[f"{which}_{t}_sp"] = P[k]["spearman_pooled"]
        w = P[k]["warning"]["0.8"]
        D[f"{which}_{t}_sens"] = w["sensitivity"]
        D[f"{which}_{t}_far"] = w["false_alarm_rate"]
        D[f"{which}_{t}_fa"] = w["false_alarm"]
        D[f"{which}_{t}_miss"] = w["miss"]
        D[f"{which}_{t}_breach"] = w["breach"]
        D[f"{which}_{t}_nonbreach"] = w["false_alarm"] + w["quiet"]
        for b in BASES:
            D[f"{which}_{t}_mae_{b}"] = P[k]["per_base"][b]["mae"]
            D[f"{which}_{t}_sp_{b}"] = P[k]["per_base"][b]["spearman"]
            D[f"{which}_{t}_mse_{b}"] = P[k]["per_base"][b]["mean_signed_error"]
    for div in ("KS", "DOM"):
        for b in BASES:
            v = DG["divergence_spearman"][div][b]
            D[f"{which}_divsp_{div}_{b}"] = v["mean"] if v else None
    # NE over-prediction in the confident-error year 2018 (mean over configurations)
    D[f"{which}_ne_overpred_2018"] = float(np.mean([r["deployment"]["2018"]["pred"]["NE"] - r["deployment"]["2018"]["actual"] for r in runs]))
    D[f"{which}_ne_miss_years"] = yearlist(sorted(set(t for r in runs for t in DEP_YEARS
                                                    if r["deployment"][str(t)]["actual"] < 0.8 and r["deployment"][str(t)]["pred"]["NE"] >= 0.8)))


def pair_phrase(a, b, unit, pct=False):
    """'X<unit> each' when the two rounded values coincide, else 'X<unit> and Y<unit> respectively'."""
    fa, fb = (f"{100 * a:.0f}", f"{100 * b:.0f}") if pct else (f"{a:d}", f"{b:d}")
    return f"{fa}{unit} each" if fa == fb else f"{fa}{unit} and {fb}{unit} respectively"


D["real_roll_sens_phrase"] = pair_phrase(D["real_NE_rolling_sens"], D["real_ATC_rolling_sens"], r"\%", pct=True)
D["emu_roll_fa_phrase"] = pair_phrase(int(D["emu_NE_rolling_fa"]), int(D["emu_ATC_rolling_fa"]), " false alarms")
json.dump(D, open(os.path.join(RES_DIR, "derived.json"), "w"), indent=1, default=float)
print(f"{len(D)} derived numbers written (with diagnostics)")

# family thresholds
F = S.get("family_real")
if F:
    covs = [f["coverage_split"] for f in F["families"]]
    n88 = sum(1 for c in covs if c < 0.88)
    D["fam_below_088"] = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}.get(n88, WORDS.get(n88, str(n88)))

# library-version sensitivity (compare_versions.py; optional second environment)
vp = os.path.join(RES_DIR, "version_sensitivity.json")
if os.path.exists(vp):
    V = json.load(open(vp))
    P = V["procedures"]["SCP|static"]
    D["ver_A"], D["ver_B"] = V["A"], V["B"]
    D["ver_trees_max"] = P["max_abs_diff_trees"]
    tb = max((b for b in BASES if b != "LogReg"), key=lambda b: P[b]["max_abs_diff_config"])
    D["ver_trees_worst_year"] = P[tb]["worst"]["year"]
    D["ver_trees_worst_n"] = P[tb]["worst"]["n"]
    D["ver_lr_max"] = P["max_abs_diff_logreg"]
    D["ver_mean_max"] = P["max_abs_mean_diff"]
    D["ver_aci_mean_max"] = V["procedures"]["ACI|immediate"]["max_abs_mean_diff"]
    D["ver_inwindow_mean_diff"] = abs(P["mean_diff_by_year"]["in-window"])

# effect of the version-1 alpha_t truncation on ACI (aci_truncation.py)
ap = os.path.join(RES_DIR, "aci_truncation_real.json")
if os.path.exists(ap):
    A = json.load(open(ap))
    by = A["by_year"]
    D["acitr_2018_truncated"] = by["2018"]["coverage_truncated"]
    D["acitr_2018_standard"] = by["2018"]["coverage_standard"]
    D["acitr_2018_full_truncated"] = by["2018"]["frac_full_truncated"]
    D["acitr_2018_full_standard"] = by["2018"]["frac_full_standard"]
    D["acitr_min_truncated"] = min(v["coverage_truncated"] for v in by.values())
    D["acitr_min_truncated_year"] = min(by, key=lambda t: by[t]["coverage_truncated"])
    D["acitr_years_below_090_truncated"] = yearlist([int(t) for t in by if by[t]["coverage_truncated"] < 0.90])
    D["acitr_check"] = A["max_abs_diff_standard_vs_main_audit"]
json.dump(D, open(os.path.join(RES_DIR, "derived.json"), "w"), indent=1, default=float)
