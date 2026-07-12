"""Cluster-aware statistics for event-study backtests.

Invariants this module guarantees:
  * Every statistic treats the SESSION (trading day), not the trade, as
    the unit of independence. Trades within a session share regime,
    news, and volatility; counting them as independent observations is
    the single most common way a backtest manufactures significance.
  * Permutation nulls flip whole sessions, never individual trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _session_frame(returns, sessions) -> pd.DataFrame:
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sessions)
    if len(r) != len(s):
        raise ValueError("returns and sessions must align")
    return pd.DataFrame({"r": r, "s": s})


def session_t(returns, sessions, min_sessions: int = 40):
    """t-statistic of per-session MEAN returns.

    Returns (t, n_sessions). NaN if fewer than `min_sessions` sessions
    or zero dispersion — a thin or degenerate sample must never yield
    a confident number.
    """
    m = _session_frame(returns, sessions).groupby("s")["r"].mean().to_numpy()
    if len(m) < min_sessions or m.std(ddof=1) == 0:
        return float("nan"), len(m)
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def session_majority(returns, sessions) -> float:
    """Share of sessions whose mean return agrees with the overall sign.

    A real edge shows up in the typical session; a mean carried by a
    handful of outlier days does not.
    """
    df = _session_frame(returns, sessions)
    m = df.groupby("s")["r"].mean()
    sign = 1.0 if float(np.mean(df["r"])) >= 0 else -1.0
    return float((np.sign(m) == sign).mean())


def signflip_p(returns, sessions, n_perm: int = 10_000, seed: int = 0,
               two_sided: bool = True) -> float:
    """Session-block sign-flip permutation p-value for the mean return.

    Null: each session's aggregate P&L is symmetric around zero. Flips
    are applied to per-session SUMS so that heavy-trading sessions keep
    their weight in the null exactly as they have it in the statistic.
    """
    df = _session_frame(returns, sessions)
    sums = df.groupby("s")["r"].sum().to_numpy()
    n = len(df)
    real = float(df["r"].mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    null = (flips @ sums) / n
    if two_sided:
        return float((np.abs(null) >= abs(real)).mean())
    return float((null >= real).mean())


def maxt_adjusted_p(cells: dict, n_perm: int = 10_000, seed: int = 0):
    """Westfall-Young maxT family-wise adjusted p-values.

    `cells` maps cell name -> (returns, sessions). All cells share one
    permutation stream: in each draw, every session receives ONE sign
    flip applied consistently across all cells that traded it, the
    statistic (|mean return|) is recomputed per cell, and each cell is
    compared against the MAXIMUM statistic across the family.

    This controls the family-wise error rate exactly like Bonferroni,
    but under the battery's actual correlation structure: correlated
    cells pay a smaller penalty; independent cells converge to the
    Sidak/Bonferroni penalty. There is no free lunch — only removal of
    Bonferroni's worst-case slack.

    Returns {cell: adjusted_p}.
    """
    prepared = {}
    all_sessions = set()
    for name, (returns, sessions) in cells.items():
        df = _session_frame(returns, sessions)
        g = df.groupby("s")["r"].sum()
        prepared[name] = (g.index.to_numpy(), g.to_numpy(), len(df),
                          abs(float(df["r"].mean())))
        all_sessions.update(g.index.tolist())
    order = {s: i for i, s in enumerate(sorted(all_sessions))}

    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(order)))
    max_stat = np.zeros(n_perm)
    for name, (sess, sums, n, _real) in prepared.items():
        cols = np.array([order[s] for s in sess])
        null = np.abs(flips[:, cols] @ sums) / n
        np.maximum(max_stat, null, out=max_stat)
    return {name: float((max_stat >= real).mean())
            for name, (_s, _v, _n, real) in prepared.items()}


def drop_k_mean(returns, k: int = 10) -> float:
    """Mean after removing the k best trades.

    If the sign flips, the 'edge' is a lottery ticket: a tail of fat
    winners carrying an otherwise losing distribution.
    """
    r = np.sort(np.asarray(returns, dtype=float))[::-1]
    if k >= len(r):
        return float("nan")
    return float(r[k:].mean())
