"""The PASS bar: a conjunction of independent guards.

A cell passes only if EVERY leg holds. Each leg kills a different
artifact:

  n_min           — thin samples cannot certify anything
  t_min           — session-clustered effect size (not trade-counted)
  alpha           — family-wise multiplicity (maxT preferred; the
                    caller supplies adjusted p from stats.maxt_adjusted_p,
                    or raw p with bonferroni_m for the crude version)
  net > 0         — after realistic costs, at the WORST honest anchor
  majority_min    — the typical session must agree with the mean
  both_arms       — drift attribution (see drift.arm_excess)

No leg may be waived after results are known. Weakening a gate for a
cell you have already seen is not analysis, it is shopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edge_audit.stats import session_majority, session_t, signflip_p


@dataclass(frozen=True)
class GateSpec:
    n_min: int = 200
    t_min: float = 2.5
    alpha: float = 0.05
    majority_min: float = 0.52
    cost_per_trade: float = 0.0        # same units as returns*point_value
    point_value: float = 1.0
    bonferroni_m: int = 1              # used only if adjusted_p not given


@dataclass
class GateResult:
    n: int
    t: float
    p_adjusted: float
    mean_gross: float
    mean_net: float
    majority: float
    legs: dict = field(default_factory=dict)
    passed: bool = False

    def report(self) -> str:
        legs = "  ".join(f"{k}={v}" for k, v in self.legs.items())
        return (f"n={self.n}  t_cl={self.t:+.2f}  p_adj={self.p_adjusted:.4f}"
                f"  gross={self.mean_gross:+.4f}  net={self.mean_net:+.4f}"
                f"  maj={self.majority:.3f}\n{legs}  ==> "
                f"{'PASS' if self.passed else 'FAIL'}")


def evaluate_cell(returns, sessions, spec: GateSpec,
                  adjusted_p: float | None = None,
                  arm_excess_result: dict | None = None,
                  n_perm: int = 10_000, seed: int = 0) -> GateResult:
    """Evaluate one pre-registered cell against the full PASS bar.

    `returns` are per-trade returns in POINTS; economics use
    spec.point_value and spec.cost_per_trade. If `adjusted_p` is given
    (e.g. from maxt_adjusted_p) it is used directly; otherwise a
    session sign-flip p is computed and multiplied by bonferroni_m.
    If `arm_excess_result` (from drift.arm_excess) is provided, its
    both_positive flag becomes a gate leg.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    t, _ns = session_t(r, sessions)
    maj = session_majority(r, sessions)
    if adjusted_p is None:
        p = signflip_p(r, sessions, n_perm=n_perm, seed=seed)
        p_adj = min(1.0, p * spec.bonferroni_m)
    else:
        p_adj = float(adjusted_p)
    gross = float(r.mean()) * spec.point_value
    net = gross - spec.cost_per_trade

    legs = {
        "n>=n_min": n >= spec.n_min,
        "t>=t_min": bool(np.isfinite(t) and abs(t) >= spec.t_min),
        "p_adj<alpha": p_adj < spec.alpha,
        "net>0": net > 0,
        "majority": maj >= spec.majority_min,
    }
    if arm_excess_result is not None:
        legs["both_arm_excess>0"] = bool(arm_excess_result["both_positive"])
    res = GateResult(n=n, t=t, p_adjusted=p_adj, mean_gross=gross,
                     mean_net=net, majority=maj, legs=legs,
                     passed=all(legs.values()))
    return res
