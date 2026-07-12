"""Tests: each guard must catch the artifact it exists to catch."""

import numpy as np
import pandas as pd
import pytest

from edge_audit import (GateSpec, arm_excess, build_sessions,
                        causality_violations, evaluate_cell,
                        freeze_registration, matched_exposure,
                        maxt_adjusted_p, nonoverlap_chain, session_majority,
                        session_t, signflip_p, verify_registration,
                        walkforward_splits)
from edge_audit.stats import drop_k_mean


def _fake_sessions(n_sess=100, per_sess=8, mean=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(mean, 1.0, n_sess * per_sess)
    s = np.repeat(np.arange(n_sess), per_sess)
    return r, s


def test_session_t_null_is_small():
    r, s = _fake_sessions(mean=0.0)
    t, ns = session_t(r, s)
    assert ns == 100 and abs(t) < 3


def test_session_t_detects_signal():
    r, s = _fake_sessions(mean=0.5, seed=1)
    t, _ = session_t(r, s)
    assert t > 3


def test_session_t_refuses_thin_samples():
    t, ns = session_t([1.0, 2.0], [0, 1])
    assert np.isnan(t)


def test_clustering_deflates_within_session_correlation():
    """1 shared shock per session, duplicated over trades: trade-level t
    explodes, session t stays honest."""
    rng = np.random.default_rng(2)
    shocks = rng.normal(0.1, 1.0, 60)
    r = np.repeat(shocks, 50)                 # 3000 'trades'
    s = np.repeat(np.arange(60), 50)
    naive_t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    t, _ = session_t(r, s)
    assert abs(naive_t) > 2 * abs(t)


def test_signflip_p_calibration():
    r, s = _fake_sessions(mean=0.0, seed=3)
    assert signflip_p(r, s, n_perm=2000, seed=0) > 0.05


def test_maxt_leq_bonferroni_and_geq_raw():
    cells = {}
    for i in range(5):
        r, s = _fake_sessions(mean=0.15, seed=10 + i)
        cells[f"c{i}"] = (r, s)
    adj = maxt_adjusted_p(cells, n_perm=2000, seed=1)
    for name, (r, s) in cells.items():
        raw = signflip_p(r, s, n_perm=2000, seed=1)
        assert adj[name] >= raw - 0.02            # adjusted never < raw
        assert adj[name] <= min(1.0, raw * 5) + 0.05   # never > Bonferroni


def test_maxt_correlated_cells_pay_less():
    """Five IDENTICAL cells: maxT penalty collapses to ~1x (adjusted ==
    raw), where Bonferroni would charge 5x."""
    r, s = _fake_sessions(mean=0.1, seed=42)      # borderline effect
    cells = {f"c{i}": (r, s) for i in range(5)}
    adj = maxt_adjusted_p(cells, n_perm=4000, seed=2)
    raw = signflip_p(r, s, n_perm=4000, seed=2)
    assert raw > 0, "test needs a nonzero raw p to compare penalties"
    assert abs(adj["c0"] - raw) < 0.02            # identical cells: no charge
    assert adj["c0"] < min(1.0, raw * 5)          # strictly beats Bonferroni


def test_chain_respects_session_wall_and_overlap():
    ts = pd.date_range("2024-01-02 09:31", periods=100, freq="min") \
        .append(pd.date_range("2024-01-03 09:31", periods=100, freq="min"))
    sess, last = build_sessions(ts)
    close = np.arange(200, dtype=float)
    ev = np.array([10, 12, 95, 110])
    tr = nonoverlap_chain(ev, np.ones(4), close, last, horizon=20,
                          no_entry_final=15)
    assert 12 not in tr["entry_idx"].tolist()          # overlap skipped
    assert 95 not in tr["entry_idx"].tolist()          # too near close
    assert (tr["exit_idx"] <= np.asarray(last)[tr["entry_idx"]]).all()


def test_arm_excess_strips_drift():
    """A long-biased no-skill strategy in a drifting market shows a big
    RAW mean (the mirage) but ~zero long-arm EXCESS (the truth)."""
    rng = np.random.default_rng(5)
    base = rng.normal(0.5, 1.0, 20000)                  # strong uptrend
    sides = np.ones(4000)                               # always long, no skill
    rets = rng.choice(base, 4000)
    out = arm_excess(sides, rets, base)
    assert rets.mean() > 0.4                            # looks like an edge
    assert abs(out["long"]) < 0.1                       # excess says: drift
    assert np.isnan(out["short"])                       # no short arm...
    assert not out["both_positive"]                     # ...so no skill claim


def test_matched_exposure_absorbs_exposure_only_strategy():
    rng = np.random.default_rng(6)
    base = rng.normal(0.05, 1.0, 2000)
    sides = np.ones(2000)
    _, p = matched_exposure(sides, base, n_draws=200, seed=0)
    assert p > 0.3                                     # nothing beyond beta


def test_causality_catches_lookahead():
    bars = pd.DataFrame({"close": np.random.default_rng(7).normal(
        0, 1, 1000).cumsum()})

    def leaky(df):
        return df["close"].rolling(5).mean().shift(-1)   # reads the future

    def causal(df):
        return df["close"].rolling(5).mean()

    assert causality_violations(leaky, bars, n_checks=10) != []
    assert causality_violations(causal, bars, n_checks=10) == []


def test_walkforward_embargo():
    splits = list(walkforward_splits(1000, train_min=300, test_size=100,
                                     embargo=50))
    assert splits and all(ts - te == 50 for te, ts, _ in splits)


def test_gate_conjunction():
    r, s = _fake_sessions(n_sess=120, per_sess=5, mean=0.6, seed=8)
    spec = GateSpec(n_min=200, t_min=2.5, alpha=0.05, majority_min=0.52,
                    cost_per_trade=0.1, point_value=1.0)
    res = evaluate_cell(r, s, spec)
    assert res.passed
    spec_hard = GateSpec(n_min=200, t_min=2.5, alpha=0.05,
                         majority_min=0.52, cost_per_trade=10.0)
    assert not evaluate_cell(r, s, spec_hard).passed   # costs kill it


def test_registry_freeze_and_verify(tmp_path):
    spec = {"hypothesis": "x", "horizon": 15}
    entry = freeze_registration(spec, tmp_path / "reg.jsonl")
    assert verify_registration(spec, entry)
    assert not verify_registration({"hypothesis": "x", "horizon": 16},
                                   entry)
    with pytest.raises(ValueError):
        freeze_registration({"hypothesis": "y", "pnl": 999},
                            tmp_path / "reg.jsonl")


def test_drop_k():
    r = np.array([100.0] + [-1.0] * 99)
    assert drop_k_mean(r, 1) < 0 < r.mean()
