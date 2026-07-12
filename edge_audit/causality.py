"""Lookahead detection and leak-safe splitting.

The deadliest backtest bug is invisible in the results: a feature that
reads one bar of the future turns noise into a money machine. The only
reliable defense is mechanical: recompute the feature on truncated
history and demand bit-identical values.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd


def causality_violations(feature_fn: Callable[[pd.DataFrame], pd.Series],
                         bars: pd.DataFrame, n_checks: int = 25,
                         min_history: int = 300, seed: int = 0,
                         atol: float = 1e-10) -> list:
    """Truncation test: feature values must not change when the future
    is deleted.

    For random cut points t, computes feature_fn(bars[:t+1]) and
    compares its LAST value with feature_fn(bars) at position t. Any
    difference means the feature at t used data after t.

    Returns a list of (t, full_value, truncated_value) violations —
    empty list == causal (at the tested points).
    """
    full = np.asarray(feature_fn(bars), dtype=float)
    if len(full) != len(bars):
        raise ValueError("feature_fn must return one value per bar")
    rng = np.random.default_rng(seed)
    cuts = rng.integers(min_history, len(bars) - 1, size=n_checks)
    bad = []
    for t in sorted(set(int(c) for c in cuts)):
        trunc = np.asarray(feature_fn(bars.iloc[: t + 1]), dtype=float)
        a, b = full[t], trunc[-1]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, atol=atol, equal_nan=False):
            bad.append((t, float(a), float(b)))
    return bad


def walkforward_splits(n: int, train_min: int, test_size: int,
                       embargo: int) -> Iterator[tuple]:
    """Expanding-window walk-forward splits with an embargo gap.

    Yields (train_end, test_start, test_end) index triples where
    test_start = train_end + embargo. The embargo must be at least the
    longest feature lookback, or the last training rows leak into the
    first test rows through overlapping windows.

    Never use random K-fold on time series: shuffling puts the future
    in the training set by construction.
    """
    if embargo < 0 or train_min <= 0 or test_size <= 0:
        raise ValueError("invalid split parameters")
    train_end = train_min
    while train_end + embargo + test_size <= n:
        yield train_end, train_end + embargo, train_end + embargo + test_size
        train_end += test_size
