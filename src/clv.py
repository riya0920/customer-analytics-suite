"""BG/NBD and Gamma-Gamma, written out because `lifetimes` is not installed.

Writing them out is not a hardship here -- it forces the ASSUMPTIONS into view,
and the assumptions are what a screener asks about:

BG/NBD assumes
  - while active, a customer purchases according to a POISSON process with
    their own rate lambda  (so inter-purchase times are exponential and
    MEMORYLESS -- no seasonality, no "due for a purchase")
  - lambda is Gamma(r, alpha) distributed across customers
  - after each purchase a customer drops out with their own probability p
  - p is Beta(a, b) distributed across customers

The memorylessness is the assumption that breaks first in retail. A
subscription-like buyer (every 30 days, reliably) and a seasonal buyer (twice a
year, in season) both violate it -- the first because their hazard is periodic
rather than flat, the second because their rate is not constant. Neither is
"customer churned"; both look like it to this model.

Gamma-Gamma assumes monetary value is independent of frequency, which is
testable and IS tested in run_analytics.py rather than assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1


# --------------------------------------------------------------------------
# RFM summary
# --------------------------------------------------------------------------
def summarise(txn: np.ndarray, cal_end: float, obs_end: float | None = None):
    """Customer-level (x, t_x, T, monetary) from raw transactions.

    x    = number of REPEAT purchases in the calibration window
    t_x  = time of the last repeat purchase
    T    = length of the observation window since first purchase
    """
    n = int(txn[:, 0].max()) + 1
    x = np.zeros(n)
    t_x = np.zeros(n)
    T = np.zeros(n)
    monetary = np.zeros(n)
    first = np.full(n, np.nan)

    order = np.argsort(txn[:, 1], kind="stable")
    txn = txn[order]
    cal = txn[txn[:, 1] <= cal_end]
    for c in range(n):
        rows = cal[cal[:, 0] == c]
        if len(rows) == 0:
            continue
        t0 = rows[0, 1]
        first[c] = t0
        repeats = rows[1:]
        x[c] = len(repeats)
        t_x[c] = (repeats[-1, 1] - t0) if len(repeats) else 0.0
        T[c] = cal_end - t0
        monetary[c] = repeats[:, 2].mean() if len(repeats) else 0.0
    return dict(x=x, t_x=t_x, T=T, monetary=monetary, first=first)


def holdout_counts(txn: np.ndarray, cal_end: float, obs_end: float,
                   n_customers: int):
    hold = txn[(txn[:, 1] > cal_end) & (txn[:, 1] <= obs_end)]
    counts = np.zeros(n_customers)
    spend = np.zeros(n_customers)
    for c in range(n_customers):
        rows = hold[hold[:, 0] == c]
        counts[c] = len(rows)
        spend[c] = rows[:, 2].sum()
    return counts, spend


# --------------------------------------------------------------------------
# BG/NBD
# --------------------------------------------------------------------------
def _bgnbd_nll(params, x, t_x, T):
    r, alpha, a, b = np.exp(params)     # exp keeps everything positive
    ln_A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_A2 = betaln(a, b + x) - betaln(a, b)
    ln_A3 = -(r + x) * np.log(alpha + T)
    # the second term exists only for customers with at least one repeat
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_A4 = np.where(
            x > 0,
            np.log(a) - np.log(np.maximum(b + x - 1, 1e-12))
            - (r + x) * np.log(alpha + t_x),
            -np.inf)
    m = np.maximum(ln_A3, ln_A4)
    mix = m + np.log(np.exp(ln_A3 - m) + np.where(x > 0, np.exp(ln_A4 - m), 0.0))
    ll = ln_A1 + ln_A2 + mix
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -ll.sum()


class BGNBD:
    def fit(self, x, t_x, T):
        best = None
        for start in ([0.0, 0.0, 0.0, 0.0], [-0.5, 1.0, -0.5, 0.5],
                      [0.5, 2.0, 0.0, 1.0]):
            res = minimize(_bgnbd_nll, start, args=(x, t_x, T), method="Nelder-Mead",
                           options=dict(maxiter=6000, xatol=1e-6, fatol=1e-6))
            if best is None or res.fun < best.fun:
                best = res
        self.r, self.alpha, self.a, self.b = np.exp(best.x)
        self.nll_ = best.fun
        return self

    def p_alive(self, x, t_x, T):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ((self.alpha + T) / (self.alpha + t_x)) ** (self.r + x)
            denom = 1.0 + np.where(x > 0,
                                   (self.a / np.maximum(self.b + x - 1, 1e-12)) * ratio,
                                   0.0)
        return 1.0 / denom

    def expected_purchases(self, t, x, t_x, T):
        """E[Y(t) | x, t_x, T] -- expected repeat purchases in the NEXT t days."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        first = (a + b + x - 1) / (a - 1) if a > 1 else (a + b + x - 1) / 1e-6
        z = t / (alpha + T + t)
        term = 1.0 - ((alpha + T) / (alpha + T + t)) ** (r + x) * \
            hyp2f1(r + x, b + x, a + b + x - 1, z)
        return first * term * self.p_alive(x, t_x, T)


# --------------------------------------------------------------------------
# Gamma-Gamma
# --------------------------------------------------------------------------
def _gg_nll(params, x, m):
    p, q, v = np.exp(params)
    mask = x > 0
    xx, mm = x[mask], m[mask]
    ll = (gammaln(p * xx + q) - gammaln(p * xx) - gammaln(q)
          + q * np.log(v) + (p * xx - 1) * np.log(mm) + (p * xx) * np.log(xx)
          - (p * xx + q) * np.log(v + xx * mm))
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -ll.sum()


class GammaGamma:
    def fit(self, x, monetary):
        best = None
        for start in ([0.0, 0.0, 3.0], [1.0, 1.0, 5.0], [-0.5, 0.5, 4.0]):
            res = minimize(_gg_nll, start, args=(x, monetary), method="Nelder-Mead",
                           options=dict(maxiter=6000))
            if best is None or res.fun < best.fun:
                best = res
        self.p, self.q, self.v = np.exp(best.x)
        return self

    def expected_value(self, x, monetary):
        """Shrinks an individual's observed mean toward the population mean, and
        the shrinkage is stronger the fewer purchases they have made -- which is
        the entire reason to use this rather than the raw average."""
        pop = self.p * self.v / (self.q - 1)
        w = (self.p * x) / (self.p * x + self.q - 1)
        return np.where(x > 0, w * monetary + (1 - w) * pop, pop)
