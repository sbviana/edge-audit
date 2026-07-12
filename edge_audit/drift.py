"""Drift attribution: is the signal skill, or a long position in disguise?

The most persistent artifact in directional backtests is market drift.
Any long-tilted signal evaluated over a rising market shows a positive
mean; wrap it in enough machinery and it looks like alpha. Two tools:

  * arm_excess — compare each arm (long trades, short trades) against
    the SAME-SIDE unconditional benchmark over the full window. A real
    signal times its side better than random entry on that side; a
    drift proxy shows a positive long arm and a bleeding short arm.
  * matched_exposure — rebuild the strategy's P&L using its exposure
    share but random timing; if that reproduces the total, the signal
    added nothing beyond exposure.
"""

from __future__ import annotations

import numpy as np


def arm_excess(trade_sides, trade_rets, all_rets) -> dict:
    """Per-arm mean vs same-side unconditional mean.

    trade_sides, trade_rets : the strategy's trades (ret signed by side)
    all_rets : UNSIGNED (long-convention) returns of every eligible
        opportunity in the window, same horizon and units.

    Returns {"long": excess, "short": excess, "both_positive": bool}.
    A cell should only be called skill if BOTH arms beat their
    benchmarks — otherwise the mean is exposure, not selection.
    """
    sides = np.asarray(trade_sides, dtype=float)
    rets = np.asarray(trade_rets, dtype=float)
    base = np.asarray(all_rets, dtype=float)
    base = base[np.isfinite(base)]
    out = {}
    for s, name in ((1.0, "long"), (-1.0, "short")):
        m = sides == s
        out[name] = float(rets[m].mean() - (s * base).mean()) \
            if m.any() and len(base) else float("nan")
    out["both_positive"] = all(
        np.isfinite(v) and v > 0
        for k, v in out.items() if k in ("long", "short"))
    return out


def matched_exposure(sides, all_rets, n_draws: int = 1000, seed: int = 0):
    """Null distribution of strategy means under exposure-matched
    random timing.

    Draws random side assignments with the SAME long/short/flat counts
    as the strategy and computes the mean signed return of each draw
    over the same opportunity set. Returns (null_means, exceedance_p)
    where exceedance_p = share of draws whose mean >= the strategy's.
    """
    sides = np.asarray(sides, dtype=float)
    base = np.asarray(all_rets, dtype=float)
    if len(sides) != len(base):
        raise ValueError("sides and all_rets must align per opportunity")
    real = float((sides * base).mean())
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_draws)
    for i in range(n_draws):
        nulls[i] = float((rng.permutation(sides) * base).mean())
    return nulls, float((nulls >= real).mean())
