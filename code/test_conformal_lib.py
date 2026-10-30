"""
Unit checks for conformal_lib on synthetic data with known answers.
Run: python3 test_conformal_lib.py   (prints PASS lines; raises on failure)
"""
import numpy as np
import conformal_lib as C


def synth_scores(n, rng, pi=0.3, sep=1.5):
    """Binary problem with Gaussian class-conditional logits; returns S, y."""
    y = (rng.random(n) < pi).astype(int)
    z = rng.normal(loc=np.where(y == 1, sep, -sep), scale=1.0)
    p1 = 1 / (1 + np.exp(-z))
    S = np.column_stack([p1, 1 - p1])          # s(x,0)=1-p0=p1, s(x,1)=1-p1
    return S, y


def test_split_and_crc():
    rng = np.random.default_rng(0)
    covs = []
    for rep in range(200):
        S_c, y_c = synth_scores(2000, rng)
        S_t, y_t = synth_scores(2000, rng)
        s_true = S_c[np.arange(len(y_c)), y_c]
        q = C.split_conformal_threshold(s_true, 0.1)
        assert q == C.crc_threshold(s_true, 0.1), "CRC (miscoverage loss) must equal split conformal"
        covs.append(np.mean(S_t[np.arange(len(y_t)), y_t] <= q))
    m = np.mean(covs)
    assert 0.895 <= m <= 0.910, m
    # ties: rounded scores
    s = np.round(rng.random(5000), 2)
    assert C.split_conformal_threshold(s, 0.1) == C.crc_threshold(s, 0.1)
    print(f"PASS split conformal exchangeable coverage {m:.4f}; CRC identical incl. ties")


def test_mondrian_and_label_shift():
    rng = np.random.default_rng(1)
    cov_scp, cov_lsw, cov_mon_pos, cov_mon_neg = [], [], [], []
    for rep in range(200):
        S_c, y_c = synth_scores(3000, rng, pi=0.3)
        S_t, y_t = synth_scores(3000, rng, pi=0.9)      # pure prior shift
        s_true = S_c[np.arange(len(y_c)), y_c]
        q = C.split_conformal_threshold(s_true, 0.1)
        cov_scp.append(np.mean(S_t[np.arange(len(y_t)), y_t] <= q))
        pi_cal = np.array([np.mean(y_c == 0), np.mean(y_c == 1)])
        qq = C.label_shift_thresholds(s_true, y_c, pi_cal, [0.1, 0.9], 0.1)
        cov_lsw.append(np.mean(S_t[np.arange(len(y_t)), y_t] <= qq[y_t]))
        qm = C.mondrian_thresholds(s_true, y_c, 0.1)
        st = S_t[np.arange(len(y_t)), y_t]
        cov_mon_pos.append(np.mean(st[y_t == 1] <= qm[1]))
        cov_mon_neg.append(np.mean(st[y_t == 0] <= qm[0]))
    assert 0.89 <= np.mean(cov_lsw) <= 0.915, np.mean(cov_lsw)
    assert 0.89 <= np.mean(cov_mon_pos) <= 0.915 and 0.89 <= np.mean(cov_mon_neg) <= 0.915
    print(f"PASS pure prior shift: SCP {np.mean(cov_scp):.4f}, label-shift CP {np.mean(cov_lsw):.4f}, "
          f"Mondrian per class {np.mean(cov_mon_pos):.4f}/{np.mean(cov_mon_neg):.4f}")


def test_bbse():
    rng = np.random.default_rng(2)
    S_c, y_c = synth_scores(20000, rng, pi=0.3)
    S_t, y_t = synth_scores(20000, rng, pi=0.8)
    pi = C.bbse_prior(np.argmin(S_c, 1), y_c, np.argmin(S_t, 1))
    assert abs(pi[1] - 0.8) < 0.02, pi
    print(f"PASS BBSE prior estimate {pi[1]:.4f} (true 0.80)")


def test_aci():
    rng = np.random.default_rng(3)
    S_c, y_c = synth_scores(3000, rng, sep=2.0)
    cal = C.SortedScores(S_c[np.arange(len(y_c)), y_c])
    parts = [synth_scores(2000, rng, sep=s) for s in (2.0, 1.2, 0.6, 0.3)]   # worsening drift
    S = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    T = len(y)
    dates = np.arange(T) // 10
    years = np.repeat(np.arange(4), 2000)
    mk, a_used, cons = C.run_aci(S, y, dates, years, cal, 0.1, 0.01, regime="immediate")
    cov = mk[np.arange(T), y].mean()
    bound = (max(0.1, 0.9) + 0.01) / (0.01 * T)
    assert abs((1 - cov) - 0.1) <= bound + 1e-9, (cov, bound)
    mk0, _, _ = C.run_aci(S, y, dates, years, cal, 0.1, 0.01, regime="delay", delay_days=0)
    assert np.array_equal(mk, mk0), "delay of 0 days must equal immediate feedback when dates tie"
    mkb, _, consb = C.run_aci(S, y, dates, years, cal, 0.1, 0.01, regime="batch_year")
    mkp, _, consp = C.run_aci(S, y, dates, years, cal, 0.1, 0.01, regime="partial", partial_rate=0.2,
                              rng=np.random.default_rng(0))
    assert abs(consp.mean() - 0.2) < 0.02
    # within year 0 the batch regime never updates: identical to static split conformal
    q = cal.q(0.1)
    assert np.array_equal(mkb[:2000], S[:2000] <= q)
    print(f"PASS ACI long-run coverage {cov:.4f} within bound {bound:.4f}; delay0==immediate; "
          f"batch year-1 == split; partial consumes {consp.mean():.3f}")


def test_recalibration_and_deferral():
    rng = np.random.default_rng(4)
    S, y = synth_scores(4000, rng)
    years = np.repeat([0, 1], 2000)
    init = S[np.arange(4000), y][:1000]
    mk, cons = C.run_recalibration(S, y, years, init, 500, 0.1, regime="batch_year")
    sl = C.SlidingQuantile(init, 500)
    assert np.array_equal(mk[:2000], S[:2000] <= sl.q(0.1))
    dates = np.arange(4000) // 5
    mi, _ = C.run_recalibration(S, y, years, init, 500, 0.1, regime="immediate")
    md, _ = C.run_recalibration(S, y, years, init, 500, 0.1, regime="delay", dates=dates, delay_days=0)
    assert np.array_equal(mi, md), "recalibration: delay 0 must equal immediate"
    mask = S <= 0.3
    ex, acc, kept = C.singleton_deferral(mask, y, S, 0.99, tau=None)
    assert acc >= 0.99 or np.isnan(acc)
    tau = C.choose_tau(mask, y, S, 0.99)
    ex2, acc2, _ = C.singleton_deferral(mask, y, S, 0.99, tau=tau)
    assert acc2 >= 0.99 - 1e-12, acc2
    print(f"PASS recalibration batch logic; deferral retro {ex:.3f}->acc {acc:.4f}; tau-policy in-sample acc {acc2:.4f}")


if __name__ == "__main__":
    test_split_and_crc()
    test_mondrian_and_label_shift()
    test_bbse()
    test_aci()
    test_recalibration_and_deferral()
    print("ALL TESTS PASSED")
