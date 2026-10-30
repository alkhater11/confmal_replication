"""
Conformal procedures, online feedback regimes and evaluation metrics for the
drifting-malware audit (version 2).

Scores are s(x, y) = 1 - p_hat(y | x).  A label y enters the prediction set
when s(x, y) <= threshold.  For binary classification |C(x)| is 0, 1 or 2.

Threshold conventions
  * split conformal: q_hat = s_(k), the k-th smallest calibration score,
    k = ceil((n + 1)(1 - alpha)); q_hat = +inf when k > n.
  * a threshold of +inf includes every label; -inf includes none.
"""
import bisect
from collections import deque
import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- thresholds
class SortedScores:
    """Calibration scores kept sorted so any conformal level is O(1)."""

    def __init__(self, scores):
        self.s = np.sort(np.asarray(scores, dtype=float))
        self.n = len(self.s)

    def q(self, alpha):
        if alpha <= 0:
            return np.inf
        if alpha >= 1:
            return -np.inf
        k = int(np.ceil((self.n + 1) * (1.0 - alpha)))
        if k > self.n:
            return np.inf
        if k < 1:
            return -np.inf
        return float(self.s[k - 1])


def split_conformal_threshold(cal_true, alpha):
    return SortedScores(cal_true).q(alpha)


def crc_threshold(cal_true, alpha, B=1.0):
    """
    Conformal risk control (Angelopoulos et al.) with the miscoverage loss:
    lambda_hat = inf{lam : n/(n+1) * R_n(lam) + B/(n+1) <= alpha}.
    For this loss the solution coincides with the split-conformal threshold;
    it is computed independently here so the identity is checked, not assumed.
    """
    s = np.sort(np.asarray(cal_true, dtype=float))
    n = len(s)
    cand = np.unique(s)
    count_gt = n - np.searchsorted(s, cand, side="right")
    ok = (n / (n + 1.0)) * (count_gt / n) + B / (n + 1.0) <= alpha
    if not ok.any():
        return np.inf
    return float(cand[np.argmax(ok)])


def mondrian_thresholds(cal_true, cal_y, alpha, n_classes=2):
    return np.array([SortedScores(cal_true[cal_y == k]).q(alpha) for k in range(n_classes)])


def weighted_quantile_thresholds(cal_scores, cal_w, test_w, alpha):
    """Tibshirani et al. (2019): per-test-point weighted conformal quantile."""
    order = np.argsort(cal_scores)
    s_sorted = cal_scores[order]
    cum_w = np.cumsum(cal_w[order])
    total = cum_w[-1]
    target = (1.0 - alpha) * (total + np.asarray(test_w))
    idx = np.searchsorted(cum_w, target, side="left")
    out = np.full(len(idx), np.inf)
    ok = idx < len(s_sorted)
    out[ok] = s_sorted[idx[ok]]
    return out


def label_shift_thresholds(cal_true, cal_y, pi_cal, pi_target, alpha):
    """
    Weighted conformal prediction under label shift (Podkopaev & Ramdas 2021):
    calibration point i has weight w(Y_i) = pi_target(Y_i) / pi_cal(Y_i) and the
    test point has weight w(y) for candidate label y, giving one threshold per
    candidate label.  Exact when P(X | Y) is unchanged and the weights are known.
    """
    w_class = np.array([pi_target[0] / pi_cal[0], pi_target[1] / pi_cal[1]])
    w = w_class[cal_y]
    order = np.argsort(cal_true)
    s_sorted = cal_true[order]
    cum_w = np.cumsum(w[order])
    tot = cum_w[-1]
    q = np.full(2, np.inf)
    for yc in (0, 1):
        idx = np.searchsorted(cum_w, (1.0 - alpha) * (tot + w_class[yc]), side="left")
        if idx < len(s_sorted):
            q[yc] = s_sorted[idx]
    return q


def bbse_prior(yhat_src, y_src, yhat_tgt):
    """Black-box shift estimation (Lipton et al. 2018) of the target class prior."""
    C = np.array([[np.mean((yhat_src == i) & (y_src == j)) for j in (0, 1)] for i in (0, 1)])
    mu = np.array([np.mean(yhat_tgt == 0), np.mean(yhat_tgt == 1)])
    w = np.linalg.lstsq(C, mu, rcond=None)[0]
    w = np.clip(w, 0.0, None)
    pi_src = np.array([np.mean(y_src == 0), np.mean(y_src == 1)])
    pi_t = w * pi_src
    if pi_t.sum() <= 0:
        return pi_src
    pi_t = pi_t / pi_t.sum()
    return np.clip(pi_t, 1e-4, 1 - 1e-4) / np.clip(pi_t, 1e-4, 1 - 1e-4).sum()


# ------------------------------------------------------------- online regimes
def run_aci(S, y, dates, years, cal_sorted, alpha, gamma, regime="immediate",
            delay_days=0, partial_rate=1.0, rng=None):
    """
    Adaptive conformal inference (Gibbs & Candes 2021), standard form without
    clipping: alpha_t <= 0 gives the full set and alpha_t >= 1 the empty set.

    Feedback regimes (which labels reach the update, and when):
      immediate   label of point t is revealed before point t+1 is scored
      delay       label revealed delay_days calendar days after the sample date
      batch_year  all labels of a deployment year revealed at the end of the year
      partial     each label revealed immediately with probability partial_rate
    Returns the prediction-set mask, the alpha_t used for each point, and a
    boolean array marking which labels were consumed by the update.
    """
    n = len(y)
    mask = np.zeros((n, 2), dtype=bool)
    alpha_used = np.empty(n)
    consumed = np.zeros(n, dtype=bool)
    a_t = alpha
    s_sorted, ncal = cal_sorted.s, cal_sorted.n
    pending = deque()          # (reveal_key, err)
    S0, S1 = S[:, 0], S[:, 1]
    reveal = None
    if regime == "partial":
        reveal = rng.random(n) < partial_rate
    for i in range(n):
        # apply all feedback that has arrived before point i is scored
        if regime == "delay":
            di = dates[i]
            while pending and pending[0][0] <= di:
                a_t += gamma * (alpha - pending.popleft()[1])
        elif regime == "batch_year":
            yi = years[i]
            while pending and pending[0][0] < yi:
                a_t += gamma * (alpha - pending.popleft()[1])
        # threshold at level a_t
        if a_t <= 0:
            q = np.inf
        elif a_t >= 1:
            q = -np.inf
        else:
            k = int(np.ceil((ncal + 1) * (1.0 - a_t)))
            q = np.inf if k > ncal else (-np.inf if k < 1 else s_sorted[k - 1])
        c0 = S0[i] <= q
        c1 = S1[i] <= q
        mask[i, 0] = c0
        mask[i, 1] = c1
        alpha_used[i] = a_t
        err = 0.0 if (c1 if y[i] == 1 else c0) else 1.0
        if regime == "immediate":
            a_t += gamma * (alpha - err)
            consumed[i] = True
        elif regime == "partial":
            if reveal[i]:
                a_t += gamma * (alpha - err)
                consumed[i] = True
        elif regime == "delay":
            pending.append((dates[i] + delay_days, err))
            consumed[i] = True
        elif regime == "batch_year":
            pending.append((years[i], err))
            consumed[i] = True
    return mask, alpha_used, consumed


class SlidingQuantile:
    """Sorted sliding window of the W most recent labelled scores."""

    def __init__(self, init_scores, W):
        self.W = W
        init = list(np.asarray(init_scores, dtype=float)[-W:])
        self.fifo = deque(init)
        self.sorted = sorted(init)

    def add(self, s):
        self.fifo.append(s)
        bisect.insort(self.sorted, s)
        if len(self.fifo) > self.W:
            old = self.fifo.popleft()
            del self.sorted[bisect.bisect_left(self.sorted, old)]

    def q(self, alpha):
        n = len(self.sorted)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            return np.inf
        return self.sorted[k - 1]


def run_recalibration(S, y, years, init_scores, W, alpha, regime="batch_year",
                      dates=None, delay_days=0, partial_rate=1.0, rng=None):
    """
    Split conformal with a sliding calibration window of the W most recently
    labelled true-label scores (the thresholding mechanism of EnbPI, applied to
    any score matrix).  The feedback regimes are those of run_aci:
    'immediate', 'batch_year', 'delay' (needs dates) and 'partial' (needs rng).
    Returns the prediction-set mask and the boolean array of consumed labels.
    """
    n = len(y)
    mask = np.zeros((n, 2), dtype=bool)
    consumed = np.zeros(n, dtype=bool)
    win = SlidingQuantile(init_scores, W)
    q = win.q(alpha)
    s_true = S[np.arange(n), y]
    pending = deque()
    reveal = rng.random(n) < partial_rate if regime == "partial" else None
    for i in range(n):
        added = False
        if regime == "batch_year":
            while pending and pending[0][0] < years[i]:
                win.add(pending.popleft()[1]); added = True
        elif regime == "delay":
            while pending and pending[0][0] <= dates[i]:
                win.add(pending.popleft()[1]); added = True
        if added:
            q = win.q(alpha)
        mask[i] = S[i] <= q
        if regime == "immediate" or (regime == "partial" and reveal[i]):
            win.add(s_true[i]); q = win.q(alpha); consumed[i] = True
        elif regime == "batch_year":
            pending.append((years[i], s_true[i])); consumed[i] = True
        elif regime == "delay":
            pending.append((dates[i] + delay_days, s_true[i])); consumed[i] = True
    return mask, consumed


# ------------------------------------------------------------------- metrics
def included_label(mask):
    """Label inside a singleton set (undefined elsewhere)."""
    return np.where(mask[:, 1] & ~mask[:, 0], 1, 0)


def set_metrics(mask, y, S, novelty=None):
    """Coverage, set composition, accepted-decision quality and error capture."""
    n = len(y)
    ar = np.arange(n)
    sizes = mask.sum(axis=1)
    covered = mask[ar, y]
    single, empty, full = sizes == 1, sizes == 0, sizes == 2
    lab = included_label(mask)
    yhat = np.argmin(S, axis=1)                      # base-detector decision
    err = yhat != y
    TP = int(np.sum(single & (lab == 1) & (y == 1)))
    FP = int(np.sum(single & (lab == 1) & (y == 0)))
    TN = int(np.sum(single & (lab == 0) & (y == 0)))
    FN = int(np.sum(single & (lab == 0) & (y == 1)))
    n_single = int(single.sum())

    def frac(a, b):
        return float(a) / float(b) if b > 0 else np.nan

    out = dict(
        n=n, n_mal=int((y == 1).sum()), n_ben=int((y == 0).sum()),
        coverage=float(covered.mean()),
        cov_mal=frac(covered[y == 1].sum(), (y == 1).sum()),
        cov_ben=frac(covered[y == 0].sum(), (y == 0).sum()),
        frac_empty=float(empty.mean()), frac_single=float(single.mean()),
        frac_full=float(full.mean()), abstention=float(1.0 - single.mean()),
        avg_set_size=float(sizes.mean()),
        n_single=n_single, n_correct_single=TP + TN,
        acc_single=frac(TP + TN, n_single),
        wrong_single_rate=frac(FP + FN, n),
        TP=TP, FP=FP, TN=TN, FN=FN,
        fpr_acc=frac(FP, FP + TN), fnr_acc=frac(FN, FN + TP),
        argmax_acc=float(np.mean(~err)),
        argmax_acc_empty=frac(np.sum(empty & ~err), empty.sum()),
        argmax_acc_single=frac(np.sum(single & ~err), single.sum()),
        argmax_acc_full=frac(np.sum(full & ~err), full.sum()),
        n_errors=int(err.sum()),
        err_in_empty=int(np.sum(empty & err)), err_in_full=int(np.sum(full & err)),
        err_recall_empty=frac(np.sum(empty & err), err.sum()),
        err_recall_abstain=frac(np.sum(~single & err), err.sum()),
        mean_conf_empty=frac(np.sum((1 - S.min(axis=1))[empty]), empty.sum()),
        mean_conf_single=frac(np.sum((1 - S.min(axis=1))[single]), single.sum()),
    )
    if novelty is not None and empty.any() and single.any():
        sel = empty | single
        out["novelty_auc_empty_vs_single"] = float(roc_auc_score(empty[sel].astype(int), novelty[sel]))
    return out


def oracle_alert_budget(S, y, covered, target):
    """
    Alert budget as defined in the original submission: defer points in order
    of decreasing TRUE-label score until coverage on the retained remainder
    reaches target.  Uses deployment labels twice (ordering and stopping), so
    it is a retrospective lower bound, not an implementable policy.
    """
    n = len(y)
    s_true = S[np.arange(n), y]
    order = np.argsort(-s_true, kind="stable")
    cov_sorted = covered[order].astype(float)
    suffix = np.concatenate([np.cumsum(cov_sorted[::-1])[::-1], [0.0]])
    rem = n - np.arange(n)
    ok = suffix[:n] / rem >= target
    return float(np.argmax(ok) / n) if ok.any() else 1.0


def singleton_deferral(mask, y, S, target, tau=None):
    """
    Additional deferral among singletons, least confident first, measured by
    the score of the included label.  With tau=None the stopping point is the
    smallest deferral giving accepted accuracy >= target (retrospective: uses
    the period's labels).  With a numeric tau, singletons whose included-label
    score exceeds tau are deferred (prospective: tau fixed in advance).
    Returns (extra_fraction_of_stream, accepted_accuracy_after, accepted_fraction_after).
    """
    n = len(y)
    sizes = mask.sum(axis=1)
    single = sizes == 1
    lab = included_label(mask)
    s_inc = S[np.arange(n), lab]
    correct = single & (lab == y)
    idx = np.where(single)[0]
    if len(idx) == 0:
        return 0.0, np.nan, 0.0
    if tau is None:
        order = idx[np.argsort(-s_inc[idx], kind="stable")]   # least confident first
        c = correct[order].astype(float)
        suffix = np.concatenate([np.cumsum(c[::-1])[::-1], [0.0]])
        rem = len(order) - np.arange(len(order))
        ok = suffix[:len(order)] / rem >= target
        k = int(np.argmax(ok)) if ok.any() else len(order)
        kept = order[k:]
    else:
        kept = idx[s_inc[idx] <= tau]
        k = len(idx) - len(kept)
    acc = float(correct[kept].mean()) if len(kept) else np.nan
    return k / n, acc, len(kept) / n


def choose_tau(mask, y, S, target):
    """
    Largest included-label score threshold such that accepted accuracy on the
    supplied (historical, labelled) data is >= target while retaining as many
    singletons as possible.  Returns +inf when no deferral is needed.
    """
    n = len(y)
    single = mask.sum(axis=1) == 1
    lab = included_label(mask)
    s_inc = S[np.arange(n), lab]
    idx = np.where(single)[0]
    if len(idx) == 0:
        return np.inf
    order = idx[np.argsort(s_inc[idx], kind="stable")]       # most confident first
    c = (lab[order] == y[order]).astype(float)
    acc_prefix = np.cumsum(c) / np.arange(1, len(order) + 1)
    good = np.where(acc_prefix >= target)[0]
    if len(good) == 0:
        return -np.inf
    k = good[-1]
    if k == len(order) - 1:
        return np.inf
    return float(s_inc[order[k]])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
