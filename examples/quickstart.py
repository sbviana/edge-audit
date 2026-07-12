"""Copy-paste-runnable quickstart: evaluate one signal end to end.

Replace the synthetic-data block with your own bars and the toy signal
with your own event indices — everything downstream stays identical.

Run: python examples/quickstart.py
"""

import numpy as np
import pandas as pd

from edge_audit import (GateSpec, arm_excess, build_sessions,
                        evaluate_cell, freeze_registration,
                        maxt_adjusted_p, nonoverlap_chain)

# ---------------------------------------------------------------------------
# 1. YOUR DATA. Any DataFrame with one row per bar works. Required:
#      ts     — bar timestamp (used only to derive trading sessions)
#      close  — close price
#    Bars must be sorted by ts. Intraday or daily both work.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n_days, bars_per_day = 400, 390
ts = (pd.DatetimeIndex(np.repeat(
        pd.date_range("2024-01-02", periods=n_days, freq="B"), bars_per_day))
      + pd.TimedeltaIndex(np.tile(
        pd.timedelta_range("9h31min", periods=bars_per_day, freq="min"),
        n_days)))
bars = pd.DataFrame({"ts": ts,
                     "close": 5000 + rng.normal(0, 0.25,
                                                len(ts)).cumsum()})

# ---------------------------------------------------------------------------
# 2. REGISTER before you look. Commit the log file; the VCS timestamp
#    is your proof the spec predates the results.
# ---------------------------------------------------------------------------
HORIZON = 15
spec_entry = freeze_registration(
    {"hypothesis": "demo: fade a 5-bar down move", "horizon": HORIZON,
     "direction": "long after 5 consecutive down bars",
     "cells": ["H=15"],
     "prediction": "P(pass) ~ 0.05 — this is a random walk"},
    "examples/registrations.jsonl")
print(f"registered: sha256 {spec_entry['sha256'][:16]}...")

# ---------------------------------------------------------------------------
# 3. YOUR SIGNAL = integer bar indices + sides (+1 long / -1 short).
#    Compute it from bars[: t+1] information only — then PROVE that
#    with edge_audit.causality_violations on your feature function.
# ---------------------------------------------------------------------------
down = (bars["close"].diff() < 0).to_numpy()
runs = pd.Series(down).rolling(5).sum().to_numpy()
event_idx = np.flatnonzero(runs == 5)
sides = np.ones(len(event_idx))

# ---------------------------------------------------------------------------
# 4. Sessions, non-overlapping trades, and the verdict.
# ---------------------------------------------------------------------------
sess, sess_last = build_sessions(bars["ts"])
trades = nonoverlap_chain(event_idx, sides, bars["close"], sess_last,
                          horizon=HORIZON, no_entry_final=15)
trade_sessions = sess[trades["entry_idx"].to_numpy()]
print(f"{len(trades)} non-overlapping trades "
      f"across {pd.Series(trade_sessions).nunique()} sessions")

# multiplicity: list EVERY cell you evaluated, not just this one
adjusted = maxt_adjusted_p({"H=15": (trades["ret"].to_numpy(),
                                     trade_sessions)})

# drift attribution: unconditional same-horizon opportunities
c = bars["close"].to_numpy()
all_moves = (c[HORIZON:] - c[:-HORIZON])[::HORIZON]
excess = arm_excess(trades["side"], trades["ret"], all_moves)

gates = GateSpec(n_min=200, t_min=2.5, alpha=0.05, majority_min=0.52,
                 cost_per_trade=12.0,   # YOUR realistic round trip, $
                 point_value=50.0)      # $ per point per contract
result = evaluate_cell(trades["ret"].to_numpy(), trade_sessions, gates,
                       adjusted_p=adjusted["H=15"],
                       arm_excess_result=None)  # needs a short arm too
print("\n" + result.report())
print(f"long-arm excess vs drift: {excess['long'] * 50:+.2f} $/tr")
print("\n(This signal is noise on a random walk — the expected verdict "
      "is FAIL. Swap in your data and your signal; the jury stays.)")
