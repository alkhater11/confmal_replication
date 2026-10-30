"""
Build temporally ordered, leakage-controlled, de-duplicated KronoDroid matrices
and a complete per-sample selection record.

Usage:  python3 build_dataset.py real|emu|overlap

Filtering order (each row receives exactly one status):
  1. invalid_date         earliest-modification date missing or unparseable
                          (KronoDroid writes the literal string 1980-00-00)
  2. non_numeric_feature  a feature cell that is neither empty nor numeric
                          (KronoDroid writes the literal string None)
  3. duplicate_sha256     second and later occurrences of the same SHA-256
  4. outside_2008_2020    valid date outside 2008-2020; sub-reason recorded
                          as epoch_1980 (ZIP epoch), pre_2008 or post_2020
  5. kept

Leakage control: Detection_Ratio and Scanners encode the antivirus verdict,
TimesSubmitted is post-hoc submission telemetry and NrContactedIps is not
reliably available at scoring time; all are dropped with identifiers,
labels and timestamps (config.json, "dropped_columns").
"""
import csv, sys, json, os
from collections import Counter, OrderedDict
from datetime import datetime
import numpy as np
from common import ROOT, CFG, DATA_DIR, RECORD_DIR, KD_DIR, save_json

csv.field_size_limit(10 ** 8)
DROP = set(CFG["dropped_columns"])
Y0, Y1 = CFG["year_range"]


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def read_file(fname, feature_cols=None):
    path = os.path.join(KD_DIR, fname)
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames
        if feature_cols is None:
            feature_cols = [c for c in header if c not in DROP]
        datecol = next(c for c in CFG["date_columns"] if c in header)
        for row in rdr:
            raw = row.get(datecol, "")
            d = parse_date(raw)
            rec = dict(sha256=(row.get("sha256") or "").strip().lower(),
                       label=int(float(row["Malware"])),
                       family=(row.get("MalFamily") or "").strip(),
                       raw_date=(raw or "").strip(), date=d, x=None, status=None)
            if d is None:
                rec["status"] = "invalid_date"
                rows.append(rec)
                continue
            vals, ok = [], True
            for c in feature_cols:
                v = row.get(c, "")
                try:
                    vals.append(float(v) if v not in ("", None) else 0.0)
                except (ValueError, TypeError):
                    ok = False
                    break
            if not ok:
                rec["status"] = "non_numeric_feature"
            else:
                rec["x"] = vals
            rows.append(rec)
    return rows, feature_cols


def build(which):
    mal_f, ben_f = (("real_malware_v1.csv", "real_legitimate_v1.csv") if which == "real"
                    else ("emu_malware_v1.csv", "emu_legitimate_v1.csv"))
    rows_m, cols = read_file(mal_f)
    rows_b, _ = read_file(ben_f, feature_cols=cols)
    for r in rows_m:
        r["source_file"] = mal_f
    for r in rows_b:
        r["source_file"] = ben_f
    rows = rows_m + rows_b
    seen = set()
    for r in rows:
        if r["status"] is not None:
            continue
        if r["sha256"] in seen:
            r["status"] = "duplicate_sha256"
            continue
        seen.add(r["sha256"])
        yr = r["date"].year
        if yr == 1980:
            r["status"] = "outside_2008_2020:epoch_1980"
        elif yr < Y0:
            r["status"] = "outside_2008_2020:pre_2008"
        elif yr > Y1:
            r["status"] = "outside_2008_2020:post_2020"
        else:
            r["status"] = "kept"

    acc = OrderedDict()
    acc["raw_records"] = len(rows)
    acc["raw_malware"] = len(rows_m)
    acc["raw_benign"] = len(rows_b)
    st = Counter(r["status"] for r in rows)
    st_lab = Counter((r["status"], r["label"]) for r in rows)
    for k in ["invalid_date", "non_numeric_feature", "duplicate_sha256",
              "outside_2008_2020:epoch_1980", "outside_2008_2020:pre_2008",
              "outside_2008_2020:post_2020", "kept"]:
        acc[k] = st.get(k, 0)
        acc[k + "|malware"] = st_lab.get((k, 1), 0)
        acc[k + "|benign"] = st_lab.get((k, 0), 0)
    acc["parsed_records"] = acc["raw_records"] - acc["invalid_date"] - acc["non_numeric_feature"]
    acc["outside_2008_2020_total"] = sum(acc[k] for k in acc if k.startswith("outside_2008_2020:") and "|" not in k)
    acc["n_features"] = len(cols)
    assert acc["raw_records"] == sum(st.values())

    kept = [r for r in rows if r["status"] == "kept"]
    order = np.argsort(np.array([r["date"].toordinal() for r in kept]), kind="stable")
    kept = [kept[i] for i in order]
    X = np.array([r["x"] for r in kept], dtype=np.float32)
    y = np.array([r["label"] for r in kept], dtype=np.int8)
    dates = np.array([r["date"].toordinal() for r in kept], dtype=np.int64)
    years = np.array([r["date"].year for r in kept], dtype=np.int64)
    fams = np.array([r["family"] for r in kept], dtype=object)
    sha = np.array([r["sha256"] for r in kept], dtype=object)
    np.savez_compressed(os.path.join(DATA_DIR, f"{which}_data.npz"), X=X, y=y, dates=dates,
                        years=years, fams=fams, sha256=sha,
                        feature_names=np.array(cols, dtype=object))
    per_year = OrderedDict()
    for yr in range(Y0, Y1 + 1):
        m = years == yr
        per_year[str(yr)] = dict(n=int(m.sum()), malware=int(y[m].sum()),
                                 benign=int(m.sum() - y[m].sum()))
    acc["per_year"] = per_year
    acc["kept_malware_families"] = len(set(f for f, l in zip(fams, y) if l == 1 and f))
    save_json(acc, os.path.join(DATA_DIR, f"accounting_{which}.json"))

    with open(os.path.join(RECORD_DIR, f"sample_selection_{which}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "subset", "source_file", "label", "family", "raw_date", "year", "status"])
        for r in rows:
            w.writerow([r["sha256"], which, r["source_file"], r["label"], r["family"], r["raw_date"],
                        r["date"].year if r["date"] is not None else "", r["status"]])
    print(json.dumps({k: v for k, v in acc.items() if k != "per_year"}, indent=1))
    for yr, v in per_year.items():
        print(f"  {yr}: n={v['n']:6d} malware={v['malware']:6d} benign={v['benign']:6d}")


def overlap():
    """APK-hash overlap between the two KronoDroid subsets (kept samples)."""
    a = np.load(os.path.join(DATA_DIR, "real_data.npz"), allow_pickle=True)
    b = np.load(os.path.join(DATA_DIR, "emu_data.npz"), allow_pickle=True)
    ra = {s: (int(yr), int(l)) for s, yr, l in zip(a["sha256"], a["years"], a["y"])}
    rb = {s: (int(yr), int(l)) for s, yr, l in zip(b["sha256"], b["years"], b["y"])}
    inter = set(ra) & set(rb)
    out = OrderedDict(real_kept=len(ra), emu_kept=len(rb), shared=len(inter),
                      real_only=len(set(ra) - set(rb)), emu_only=len(set(rb) - set(ra)),
                      shared_frac_of_emu=len(inter) / len(rb), shared_frac_of_real=len(inter) / len(ra),
                      label_mismatch_on_shared=sum(ra[s][1] != rb[s][1] for s in inter),
                      year_mismatch_on_shared=sum(ra[s][0] != rb[s][0] for s in inter))
    py = OrderedDict()
    for yr in range(Y0, Y1 + 1):
        e_y = [s for s, v in rb.items() if v[0] == yr]
        py[str(yr)] = dict(emu=len(e_y), shared=sum(s in ra for s in e_y),
                           emu_only_malware=sum(1 for s in e_y if s not in ra and rb[s][1] == 1),
                           emu_only_benign=sum(1 for s in e_y if s not in ra and rb[s][1] == 0))
    out["per_year"] = py
    # static-feature identity on shared applications
    fa, fb = list(a["feature_names"]), list(b["feature_names"])
    dyn_end = fa.index("ACCEPT_HANDOVER")
    stat = [c for c in fa[dyn_end:] if c in fb]
    dyn = [c for c in fa[:dyn_end] if c in fb]
    ia = {s: i for i, s in enumerate(a["sha256"])}
    ib = {s: i for i, s in enumerate(b["sha256"])}
    sh = sorted(inter)
    A_ = a["X"][[ia[s] for s in sh]]
    B_ = b["X"][[ib[s] for s in sh]]
    sa = [fa.index(c) for c in stat]; sb = [fb.index(c) for c in stat]
    da = [fa.index(c) for c in dyn]; db = [fb.index(c) for c in dyn]
    stat_diff = (A_[:, sa] != B_[:, sb]).any(axis=1)
    dyn_diff = (A_[:, da] != B_[:, db]).any(axis=1)
    out["n_static_compared"] = len(stat)
    out["n_dynamic_compared"] = len(dyn)
    out["shared_with_any_static_difference"] = float(stat_diff.mean())
    out["shared_with_any_dynamic_difference"] = float(dyn_diff.mean())
    fam_a = {s: f for s, f in zip(a["sha256"], a["fams"])}
    fam_b = {s: f for s, f in zip(b["sha256"], b["fams"])}
    out["shared_family_label_agreement"] = float(np.mean([fam_a[s] == fam_b[s] for s in sh]))
    save_json(out, os.path.join(DATA_DIR, "overlap_real_emu.json"))
    with open(os.path.join(RECORD_DIR, "overlap_real_emu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "in_real", "in_emu"])
        for s in sorted(set(ra) | set(rb)):
            w.writerow([s, int(s in ra), int(s in rb)])
    print(json.dumps({k: v for k, v in out.items() if k != "per_year"}, indent=1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "real"
    if which == "overlap":
        overlap()
    else:
        build(which)
