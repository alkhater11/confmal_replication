# Conformal audit of evolving malware detectors — replication package, version 2.0.0

Code, configuration, seed lists, per-sample selection records and result files
for the revised manuscript

> From drift detection to calibrated guarantees: an audit of non-exchangeable
> conformal prediction for evolving malware detectors, with a pre-deployment
> validity diagnostic (International Journal of Information Security, revision).

Every result, table and figure in the manuscript is produced by the scripts in
`code/` from the four public KronoDroid CSV files, and no result is typed by
hand: `analyze.py` aggregates the per-run outputs into `results/summary.json`,
`make_numbers.py` derives the figures quoted in the prose
(`results/derived.json`), and `make_tables.py` / `make_figures.py` write every
table fragment and figure.

## 1. Data

KronoDroid, https://github.com/aleguma/kronodroid, commit
`c6ec342167bc449967a802824d068900ac8120c5`. SHA-256 digests of the four archives
used are recorded in `config.json`.

    git clone https://github.com/aleguma/kronodroid
    mkdir -p kd/extracted
    for f in kronodroid/real_device/*.zip kronodroid/emulator/*.zip; do
        unzip -o -q "$f" -d kd/extracted/
    done

The scripts look for the CSV files in `kd/extracted/` (override with the
environment variable `KRONODROID_DIR`).

## 2. Environment

Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2
(see `requirements.txt`). Per-configuration results depend on the scikit-learn
release, while means over configurations are robust; section 7 quantifies this.
`run_all.sh` sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; the
results are identical with multi-threaded libraries (checked), but single
threads avoid oversubscription when two jobs share two cores.

## 3. Reproduce everything

    ./run_all.sh            # about half an hour on two CPU cores

or step by step (from `code/`):

| Step | Script | Output |
|---|---|---|
| Parse, filter, de-duplicate; selection records | `build_dataset.py real`, `build_dataset.py emu` | `data/*_data.npz`, `data/accounting_*.json`, `records/sample_selection_*.csv` |
| SHA-256 overlap of the two subsets | `build_dataset.py overlap` | `data/overlap_real_emu.json`, `records/overlap_real_emu.csv` |
| Unit tests on synthetic data | `test_conformal_lib.py` | console |
| Main audit (all static procedures, ACI and recalibrated SCP under all feedback regimes, sensitivity runs, fixed-prior resampled streams) | `run_audit.py real|emu [detector]` | `results/audit_<subset>_<detector>.json` |
| EnbPI under all feedback regimes | `run_enbpi.py real|emu` | `results/enbpi_<subset>.json` |
| Diagnostics (CWDV, CWDV-D, baselines, chronological evaluation) | `run_diagnostic.py real|emu` | `results/diagnostic_<subset>.json` |
| Per-family breakdown | `run_family.py real` | `results/family_real.json` |
| Static-only / dynamic-only ablation | `run_feature_ablation.py real` | `results/ablation_real.json` |
| Effect of the version-1 truncation of the ACI level (response letter) | `aci_truncation.py real` | `results/aci_truncation_real.json` |
| Aggregation | `analyze.py` | `results/summary.json` |
| Prose numbers | `make_numbers.py` | `results/derived.json` |
| Tables, figures (PDF and EPS) | `make_tables.py`, `make_figures.py` | `tables/*.tex`, `figures/*.pdf|eps` |
| Independent recomputation of headline results | `verify_numbers.py` | console (exit status 1 on any mismatch) |
| Library-version sensitivity (optional, Section 5.2) | `compare_versions.py` | `results/version_sensitivity.json` |

`run_all.sh` runs every step except the last, which needs a second Python
environment (Section 7 below).

## 4. Protocol in one paragraph

Window 2008-2012 (43,249 samples) split at random into fit / calibration /
held-out in-window test (60/25/15), ten seeds (`config.json: seeds_main`;
five for EnbPI, diagnostics, families and ablation). Deployment years
2013-2020 scored in temporal order. Detectors are never refitted. Static
procedures use no deployment labels. Online procedures (ACI, EnbPI,
recalibrated split conformal) are run under identical feedback regimes:
immediate, calendar delays of 7/30/90/180/365 days, end-of-year batch, and
random partial labelling (50/20/10/5%).

## 5. Data accounting (real-device subset)

| Stage | Records |
|---|---|
| Distributed CSV records | 78,137 |
| Earliest-modification date is the literal `1980-00-00` | -1,513 |
| Non-numeric feature (`None` in the `Activities` count) | -617 |
| Parsed | 76,007 |
| Duplicate SHA-256 (second occurrence) | -4 |
| Outside 2008-2020 (6,901 ZIP-epoch 1980, 495 pre-2008, 2 post-2020) | -7,398 |
| Retained | 68,605 |

`records/sample_selection_real.csv` gives the status of every record.

## 6. Notes on corrections made in version 2

* In-window coverage is measured on a held-out split. Coverage on the
  calibration points themselves equals k/n by construction and is not a check.
* The split-conformal threshold is the k-th order statistic,
  k = ceil((n+1)(1-alpha)).
* Conformal risk control with the miscoverage loss is computed from its own
  definition; it equals the split-conformal threshold in every configuration,
  as theory predicts.
* ACI uses the standard untruncated update (alpha_t <= 0 gives the full set,
  alpha_t >= 1 the empty set). Version 1 clipped alpha_t to [1e-4, 0.999];
  `aci_truncation.py` re-runs ACI with that clipping, nothing else changed,
  to measure its effect.
* The alert budget of version 1 (true-label ordering) is retained only as an
  oracle lower bound; the burden accounting counts mandatory abstentions,
  retrospective and rolling (prospective) deferrals, and labels consumed.
* The emulator subset shares 98.8% of its retained applications with the
  real-device subset (identical static features), so it is treated as a
  re-measurement under an independent dynamic-feature collection, not as an
  independent sample.

## 7. Library-version sensitivity (optional)

Per-configuration results of the tree ensembles depend on the scikit-learn
release. To measure this, the main audit is re-run with identical code, data,
configuration and seeds under scikit-learn 1.6.1, and the two sets of results
are compared. The results of that run are shipped in
`results/alt_sklearn_1.6.1/`; to regenerate them:

    python3 -m venv --system-site-packages venv_sk16
    venv_sk16/bin/pip install scikit-learn==1.6.1
    mkdir -p alt && cp config.json alt/ && ln -s "$PWD/data" alt/data
    for d in RandomForest HistGB LogReg; do
        OMP_NUM_THREADS=1 CONFMAL_ROOT=$PWD/alt venv_sk16/bin/python code/run_audit.py real $d
    done
    mkdir -p results/alt_sklearn_1.6.1 && cp alt/results/audit_real_*.json results/alt_sklearn_1.6.1/
    cd code && python3 compare_versions.py ../results 1.9.1 ../results/alt_sklearn_1.6.1 1.6.1
    python3 make_numbers.py
