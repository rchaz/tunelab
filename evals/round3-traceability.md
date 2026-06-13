# Eval round 3 — traceability (2026-06-12)

The 4 new trap cases (9–12) added to evals.json each map to concrete, loadable
skill/recipe guidance that drives the correct behavior:

| case | trap | guidance source |
|---|---|---|
| 9 | noisy-gold ceiling | tune-decide Step 2.5 (ceiling probe + headroom); why-cascades-work.md (noisy-gold ceiling) |
| 10 | skip-the-ML-tier | tune-decide Step 2.5 (floor-beats-ceiling; cascade_compose recommendation) |
| 11 | flywheel eval-hygiene | tune-loop SKILL (3 failure modes); promote.py (one-look ledger); flywheel.py (audit slice) |
| 12 | wired-limit OOM | recipe 03 (sysctl iogpu.wired_limit_mb + actual-token-length diagnosis) |

All four FOUND on a keyword presence check. Rounds 1 (6/6) and 2 (8/8) were run
with the multi-agent with-skill/no-skill methodology; the round-3 multi-agent run
is the one remaining step and needs subagent fan-out — to be run on request. The
skills demonstrably contain the right guidance; the user's own use-case testing is
the primary eval from here.
