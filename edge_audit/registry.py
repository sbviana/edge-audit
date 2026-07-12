"""Pre-registration: freeze the hypothesis before the data can answer.

Evidence is consumed on first sight. Once you have seen a window's
results, no protocol — holdout relabeling, re-running "blind", sliding
the window — can extract honest confirmation from it again, because
hypothesis SELECTION reaches through any amnesia via the choice of
what was tested. The only defenses are (a) freezing the full spec
before results exist and (b) sealing data before anyone hunts on it.

This module gives both a mechanical form:

  * freeze_registration(spec, log_path) appends the spec as canonical
    JSON with a SHA-256 hash and UTC timestamp to an append-only log.
    Commit that log before running the test; the version-control
    timestamp is the proof.
  * verify_registration(spec, entry) re-hashes and compares — any
    parameter drift between registration and evaluation is detectable.

Companion rules that code cannot enforce (put them in your protocol):
  * ONE look per hypothesis per window. A negative is final for that
    spec on that data; variants are new registrations on new data.
  * Register your PREDICTIONS with probabilities, and score them.
    A researcher who never writes down what they expected cannot
    notice that they are always surprised.
  * On every new data import, SEAL the trailing slice (>= 6 months or
    10%) before any code touches it; open it once, for a named
    surviving hypothesis.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

_FORBIDDEN_KEYS = {"result", "results", "pnl", "sharpe", "tstat", "pvalue"}


def _canonical(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"),
                      default=str)


def freeze_registration(spec: dict, log_path) -> dict:
    """Append a frozen registration entry to an append-only JSONL log.

    Refuses specs that already contain result-like fields — a
    registration written after the answer is a confession, not a
    prediction. Returns the entry (including its hash).
    """
    lower = {k.lower() for k in spec}
    if lower & _FORBIDDEN_KEYS:
        raise ValueError(
            f"spec contains result-like fields {lower & _FORBIDDEN_KEYS}; "
            "register BEFORE computing results")
    canon = _canonical(spec)
    entry = {
        "registered_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sha256": hashlib.sha256(canon.encode()).hexdigest(),
        "spec": spec,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def verify_registration(spec: dict, entry: dict) -> bool:
    """True iff `spec` hashes to the registered entry — i.e. not one
    parameter changed between freeze and evaluation."""
    return hashlib.sha256(_canonical(spec).encode()).hexdigest() \
        == entry["sha256"]
