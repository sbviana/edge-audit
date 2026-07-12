"""edge-audit — a validation framework for trading-strategy research.

Extracted from a private intraday-futures research program in which
hundreds of pre-registered hypothesis tests were run against 15 years
of 1-minute data. The hypotheses are not included; the discipline that
evaluated them is.

The framework's job is to make it hard for a backtest to lie to you.
"""

from edge_audit.chain import build_sessions, nonoverlap_chain
from edge_audit.stats import (
    drop_k_mean,
    maxt_adjusted_p,
    session_majority,
    session_t,
    signflip_p,
)
from edge_audit.drift import arm_excess, matched_exposure
from edge_audit.gates import GateSpec, evaluate_cell
from edge_audit.causality import causality_violations, walkforward_splits
from edge_audit.registry import freeze_registration, verify_registration

__version__ = "0.1.0"
__all__ = [
    "build_sessions", "nonoverlap_chain",
    "session_t", "session_majority", "signflip_p", "maxt_adjusted_p",
    "drop_k_mean", "arm_excess", "matched_exposure",
    "GateSpec", "evaluate_cell",
    "causality_violations", "walkforward_splits",
    "freeze_registration", "verify_registration",
]
