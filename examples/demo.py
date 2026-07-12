"""edge-audit demo: one fake edge, one real edge, same jury.

Generates a synthetic random-walk market with realistic sessions, then:

  1. MINES a "strategy" the way overfit research does — try 50 random
     signals, keep the best in-sample performer — and shows the full
     gate battery (with maxT multiplicity over everything that was
     tried) correctly refusing it.
  2. Plants a small TRUE edge in a second synthetic market and shows
     the same battery passing it.

Run: python examples/demo.py
"""

import numpy as np
import pandas as pd

from edge_audit import (GateSpec, arm_excess, build_sessions,
                        evaluate_cell, freeze_registration,
                        maxt_adjusted_p, nonoverlap_chain)

N_SESS, BARS = 700, 390          # ~2.8 years of 1-minute days
H = 15                           # holding horizon (bars)
SPEC = GateSpec(n_min=200, t_min=2.5, alpha=0.05, majority_min=0.52,
                cost_per_trade=12.0, point_value=50.0)


def make_market(seed, edge_signal=None, edge_bps=0.0):
    """Random-walk closes; optionally inject drift after signal bars."""
    rng = np.random.default_rng(seed)
    n = N_SESS * BARS
    steps = rng.normal(0, 0.25, n)
    sig = np.zeros(n, dtype=bool)
    if edge_signal is not None:
        sig = edge_signal(rng, n)
        # future drift: the H bars AFTER a signal get a tiny lift
        lift = np.zeros(n)
        idx = np.flatnonzero(sig)
        for t in idx:
            lift[t + 1: t + 1 + H] += edge_bps
        steps += lift
    close = 5000 + np.cumsum(steps)
    days = np.repeat(pd.date_range("2023-01-02", periods=N_SESS,
                                   freq="B"), BARS)
    minutes = np.tile(pd.timedelta_range("9h31min", periods=BARS,
                                         freq="min"), N_SESS)
    ts = pd.DatetimeIndex(days) + pd.TimedeltaIndex(minutes)
    return pd.DataFrame({"ts": ts, "close": close}), sig


def trades_for(signal_mask, bars, sess, last):
    ev = np.flatnonzero(signal_mask)
    tr = nonoverlap_chain(ev, np.ones(len(ev)), bars["close"], last, H)
    return tr, sess[tr["entry_idx"].to_numpy()]


def main():
    print(__doc__)

    # ---------- Part 1: the mined mirage ----------
    bars, _ = make_market(seed=1)
    sess, last = build_sessions(bars["ts"])
    close = bars["close"].to_numpy()
    rng = np.random.default_rng(2)

    print("PART 1 — mining 50 random signals on a pure random walk, "
          "keeping the best...")
    candidates = {}
    for i in range(50):
        mask = rng.random(len(bars)) < 0.002        # ~random 'setups'
        tr, tsess = trades_for(mask, bars, sess, last)
        if len(tr) >= SPEC.n_min:
            candidates[f"rule{i:02d}"] = (tr["ret"].to_numpy(), tsess)
    best = max(candidates, key=lambda k: candidates[k][0].mean())
    r_best, s_best = candidates[best]
    print(f"best in-sample rule: {best}  mean "
          f"{r_best.mean() * SPEC.point_value:+.2f} $/tr gross over "
          f"{len(r_best)} trades — looks tradeable!")

    # honest multiplicity: adjust over EVERYTHING that was tried
    adj = maxt_adjusted_p(candidates, n_perm=2000, seed=3)
    res = evaluate_cell(r_best, s_best, SPEC, adjusted_p=adj[best])
    print("\nverdict on the mined rule (maxT over all 50 tries):")
    print(res.report())
    assert not res.passed, "a mined mirage must not pass"

    # ---------- Part 2: a real (planted) edge ----------
    print("\nPART 2 — same jury, but the market now contains a real "
          "0.03pt/bar post-signal drift...")

    def signal(rng2, n):
        return rng2.random(n) < 0.004

    bars2, sig2 = make_market(seed=4, edge_signal=signal, edge_bps=0.03)
    sess2, last2 = build_sessions(bars2["ts"])

    reg = freeze_registration(
        {"hypothesis": "planted-signal continuation", "horizon": H,
         "direction": "long", "gates": "house PASS bar", "cells": 1},
        "examples/registrations.jsonl")
    print(f"registered before evaluation: sha256 {reg['sha256'][:16]}...")

    tr2, tsess2 = trades_for(sig2, bars2, sess2, last2)
    c2 = bars2["close"].to_numpy()
    exc = arm_excess(tr2["side"], tr2["ret"],
                     all_rets=(c2[H:] - c2[:-H])[::H])
    res2 = evaluate_cell(tr2["ret"].to_numpy(), tsess2, SPEC,
                         adjusted_p=None, arm_excess_result=None)
    print("\nverdict on the planted true edge (single registered cell):")
    print(res2.report())
    print(f"long-arm excess vs unconditional drift: "
          f"{exc['long'] * SPEC.point_value:+.2f} $/tr")
    assert res2.passed, "a real edge of this size should pass"

    print("\nSame statistics, opposite verdicts — the difference is the "
          "protocol, not the optimism.")


if __name__ == "__main__":
    main()
