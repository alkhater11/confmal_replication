"""
Shared configuration, paths, data loading, splits and base detectors.

Paths resolve relative to the package root (the directory holding
config.json) unless overridden by the environment variables CONFMAL_ROOT
and KRONODROID_DIR, so the package runs from any location.
"""
import json, os, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.environ.get("CONFMAL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, "config.json")) as _f:
    CFG = json.load(_f)
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
RECORD_DIR = os.path.join(ROOT, "records")
FIG_DIR = os.path.join(ROOT, "figures")
TAB_DIR = os.path.join(ROOT, "tables")
KD_DIR = os.environ.get("KRONODROID_DIR", os.path.join(ROOT, "kd", "extracted"))
for _d in (DATA_DIR, RES_DIR, RECORD_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

ALPHA = CFG["alpha"]
DEP_YEARS = CFG["deployment_years"]
BASES = list(CFG["base_detectors"].keys())


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_subset(which):
    d = np.load(os.path.join(DATA_DIR, f"{which}_data.npz"), allow_pickle=True)
    return dict(X=d["X"], y=d["y"].astype(int), years=d["years"], dates=d["dates"],
                fams=d["fams"], sha256=d["sha256"], feature_names=list(d["feature_names"]))


def window_mask(years):
    a, b = CFG["window_years"]
    return (years >= a) & (years <= b)


def window_split(n_win, seed):
    """Random fit / calibration / in-window test split of the 2008-2012 window."""
    fr = CFG["window_split_fractions"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_win)
    n_fit = int(round(fr["fit"] * n_win))
    n_cal = int(round(fr["calibration"] * n_win))
    return idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]


def make_model(name, seed, reduced=False):
    p = CFG["base_detectors"][name]
    if name == "RandomForest":
        n = CFG["enbpi"]["RandomForest_trees"] if reduced else p["n_estimators"]
        return RandomForestClassifier(n_estimators=n, n_jobs=1, random_state=seed,
                                      class_weight=p["class_weight"],
                                      min_samples_leaf=p["min_samples_leaf"])
    if name == "HistGB":
        n = CFG["enbpi"]["HistGB_iter"] if reduced else p["max_iter"]
        return HistGradientBoostingClassifier(max_iter=n, random_state=seed)
    if name == "LogReg":
        return LogisticRegression(max_iter=p["max_iter"], C=p["C"], random_state=seed)
    raise ValueError(name)


class Detector:
    """Base detector returning non-conformity scores s(x, y) = 1 - p_hat(y | x)."""

    def __init__(self, name, seed, reduced=False):
        self.name, self.seed = name, seed
        self.model = make_model(name, seed, reduced)
        self.scaler = None

    def fit(self, X, y):
        if self.name == "LogReg":
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.model.fit(X, y)
        return self

    def proba(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def scores(self, X):
        return 1.0 - self.proba(X)


def dynamic_static_indices(feature_names):
    """System-call counts precede the first permission column in KronoDroid."""
    k = feature_names.index("ACCEPT_HANDOVER")
    return np.arange(k), np.arange(k, len(feature_names))
