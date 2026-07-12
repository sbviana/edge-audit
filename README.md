# edge-audit

**A validation framework that makes it hard for a trading backtest to lie to you.**

This library was extracted from a private intraday-futures research program:
15 years of 1-minute index data, ~46 hypothesis families, 200+ pre-registered
experiments, run over several weeks by a human research director working with
AI research agents. The program's most valuable output was not a strategy — it
was the discipline that correctly refused every strategy that wasn't real, and
caught several that had already fooled other tools.

The hypotheses, data, and results are not included. The jury is.

## The problem

A backtest is a machine for generating false confidence. In our program, every
one of the following produced "profitable strategies" that died under honest
evaluation:

1. **Lookahead** — a feature that reads one bar of the future turns noise into
   a money machine, invisibly.
2. **Trade-counted significance** — 3,000 trades drawn from 60 sessions are 60
   observations wearing 50 hats each. Trade-level t-statistics overstate
   evidence by integer factors.
3. **Drift ghosts** — any long-tilted signal evaluated over a rising market
   "works." Most retail momentum results are a long position in disguise.
4. **Multiplicity** — try 50 rules, report the best: at p<0.05 pure noise
   hands you 2–3 "discoveries" per 50 attempts, guaranteed.
5. **Session concentration** — a mean carried by five fat days is a lottery
   ticket, not an edge. The typical session must agree.
6. **Fill-model fiction** — limit fills granted at touch without queue
   position; entries credited with the crossing bar's intra-minute run. If
   execution assumptions are generous, the strategy is the assumptions.
7. **Cost amnesia** — edges that are statistically overwhelming and worth
   less than the round-trip toll are the norm, not the exception.
8. **Selection through amnesia** — re-testing a window "blind" after seeing
   its results confirms nothing: selection of what to test reaches through
   any holdout ritual. Evidence is consumed on first sight.
9. **Verdict shopping** — weakening a gate after seeing a near-miss. The most
   human artifact of all.
10. **Era dependence** — the quiet killer: structures that are hyper-
    significant in one regime (we measured one at t = +15) can be absent in
    the previous decade. In-era certification does not transfer.

`edge-audit` gives each of these a mechanical counter.

## Install

```bash
pip install numpy pandas
pip install -e .          # from a clone
pytest                    # 15 tests, each guarding one artifact
python examples/demo.py   # watch a mined mirage fail and a real edge pass
```

## Sixty-second example

```python
import numpy as np
from edge_audit import (build_sessions, nonoverlap_chain, GateSpec,
                        evaluate_cell, maxt_adjusted_p, arm_excess,
                        freeze_registration)

# 0. FREEZE the spec before results exist; commit the log for a timestamp.
freeze_registration(
    {"hypothesis": "my-signal continuation", "horizon": 15,
     "cells": ["H=15"], "gates": "default"},
    "research/registrations.jsonl")

# 1. Sessions and a non-overlapping event chain with hard session walls.
sess, last = build_sessions(bars["ts"])
trades = nonoverlap_chain(event_idx, sides, bars["close"], last,
                          horizon=15, no_entry_final=15)
tsess = sess[trades["entry_idx"].to_numpy()]

# 2. Multiplicity over EVERYTHING you tried (not just what you liked).
adj = maxt_adjusted_p({"H=15": (trades["ret"], tsess), ...})

# 3. Drift attribution: does each arm beat random entry on its own side?
exc = arm_excess(trades["side"], trades["ret"], all_opportunity_rets)

# 4. The verdict — five legs, all must hold, none may be waived.
spec = GateSpec(n_min=200, t_min=2.5, alpha=0.05, majority_min=0.52,
                cost_per_trade=12.0, point_value=50.0)
print(evaluate_cell(trades["ret"], tsess, spec,
                    adjusted_p=adj["H=15"], arm_excess_result=exc).report())
```

## The protocol (the part that matters more than the code)

The library enforces statistics; the discipline is a workflow:

1. **Register first.** Full spec — signal, horizon, cells, gates, costs —
   frozen and hashed *before* any result is computed
   (`freeze_registration`). Commit it; the VCS timestamp is the proof.
2. **Predict.** Write down what you expect, with probabilities, and score
   yourself afterwards. Misses are data about your judgment.
3. **One look per hypothesis per window.** A negative is final for that spec
   on that data. Variants are new registrations, ideally on new data.
4. **Gates are a conjunction.** Sample floor, session-clustered effect size,
   family-wise multiplicity (maxT), positive economics at the *worst* honest
   cost anchor, session majority, and both-arm drift excess. A cell that
   fails one leg fails.
5. **Benchmark ladder.** Every candidate races the trivial predictors it
   might secretly be: always-long, the overnight gap, its own unconditional
   census. If a benchmark matches it, the benchmark wins.
6. **Seal on import.** Every new dataset vaults its trailing slice before any
   code touches it; the vault opens once, for a named surviving hypothesis.
7. **Near-misses go forward, not backward.** A positive-but-underpowered
   result earns a frozen forward test on data that does not exist yet —
   never a re-tune on data that does.

## What's in the box

| Module | Guards against |
|---|---|
| `stats.session_t`, `session_majority` | trade-counted significance, fat-day means |
| `stats.signflip_p` | parametric assumptions; session-block permutation null |
| `stats.maxt_adjusted_p` | multiplicity, with exact correlation-aware FWER (Westfall–Young) |
| `stats.drop_k_mean` | lottery-ticket distributions |
| `chain.nonoverlap_chain`, `build_sessions` | overlapping events, session-boundary leaks |
| `drift.arm_excess`, `matched_exposure` | drift ghosts, exposure-in-disguise |
| `causality.causality_violations` | lookahead (truncation test: delete the future, demand bit-identical features) |
| `causality.walkforward_splits` | K-fold-on-time-series, missing embargo |
| `gates.evaluate_cell` | verdict shopping (the bar is declared before the data answers) |
| `registry.freeze_registration` | post-hoc registration (refuses result-bearing specs) |

## Provenance and honesty

Everything here was battle-tested in the negative sense: these are the guards
that killed our own favorite ideas, including several the standard toolkit
had already blessed. The single most useful finding of the whole program was
that **our near-misses failed on effect size and consistency, not on the
multiplicity penalty** — rigor did not cost us discoveries; it cost us
mirages.

No investment advice. If your backtest passes all of this, you have earned
the right to a *forward* test — nothing more.

## License

MIT — see [LICENSE](LICENSE).
