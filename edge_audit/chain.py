"""Event chaining with hard session walls.

Invariants this module guarantees:
  * No trade crosses a session boundary — entries near the close are
    refused, exits are clamped to the session's last bar.
  * Events never overlap: while one trade is open, later signals are
    skipped. Overlapping event studies count the same move many times
    and inflate n by an order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sessions(timestamps) -> tuple[np.ndarray, np.ndarray]:
    """From bar timestamps, return (session_id, session_last_idx) per bar.

    A session is a calendar date; session_last_idx[i] is the index of
    the final bar of bar i's session — the hard wall for exits.
    """
    ts = pd.to_datetime(pd.Series(list(timestamps)))
    sess = pd.factorize(ts.dt.date)[0]
    pos = pd.Series(np.arange(len(ts)))
    last = pos.groupby(sess).transform("max").to_numpy()
    return sess, last


def nonoverlap_chain(event_idx, sides, close, session_last, horizon: int,
                     no_entry_final: int = 15) -> pd.DataFrame:
    """Greedy earliest-first non-overlapping close-to-close event chain.

    Parameters
    ----------
    event_idx : sorted bar indices where the signal fires
    sides : +1 / -1 per event (0 events are skipped)
    close : close price per bar
    session_last : per-bar index of the session's final bar
    horizon : exit at min(t + horizon, session_last[t])
    no_entry_final : refuse entries within this many bars of the close

    Returns a DataFrame with entry_idx, exit_idx, side, ret (signed,
    in price points).
    """
    close = np.asarray(close, dtype=float)
    session_last = np.asarray(session_last)
    rows = []
    last_exit = -1
    for t, sgn in zip(np.asarray(event_idx), np.asarray(sides)):
        t = int(t)
        if t <= last_exit or sgn == 0:
            continue
        if session_last[t] - t < no_entry_final:
            continue
        x = min(t + horizon, int(session_last[t]))
        if x <= t:
            continue
        rows.append((t, x, int(sgn), float(sgn * (close[x] - close[t]))))
        last_exit = x
    return pd.DataFrame(rows, columns=["entry_idx", "exit_idx", "side",
                                       "ret"])
