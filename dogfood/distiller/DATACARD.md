# DATACARD — mcp-tool-result-distiller (real agent logs, Path B)

- **Provenance:** 100% real traffic — tool results from the author's own Claude Code transcripts
  (local, never committed; `data/` is gitignored). 1,200 (tool_result blob → consuming assistant
  turn) pairs extracted from coding sessions: test runs, numbered file reads, search results,
  CLI output, subagent reports. Blobs 809–11,955 chars, p50 4,263.
- **Dedupe:** 1,200 → 1,149 at char-4-gram Jaccard 0.80 (51 near-dup blobs dropped). 800 blobs
  selected for teacher compression: 25 spot-check + 775 scale (39 session-native batches).
- **Teacher:** tier = **session-native** (model unpinned — Claude Code session, Claude Fable 5,
  2026-06-11), $0.00 API spend. Prompt frozen 2026-06-11 after two spot-check rounds (round 1
  caught value-reformatting and composed-token defects; see EXPERIMENT-LOG.md); verbatim in
  `data/teacher_prompt_frozen.txt` and quoted in `recipes/03-tool-result-distiller.md`.
- **Verification gate (the load-bearing step):** every teacher output mechanically checked —
  each identifier/number atomically split and verbatim-matched against its source blob, plus a
  0.40 hard compression-ratio cap. **683/800 kept (85.4%)**; 117 dropped. *Known gap: the
  per-reason breakdown of the 117 was not retained.*
- **Survivor stats:** compression ratio p50 0.245 / p90 0.249 / max 0.325. Fields-used ground
  truth (identifiers the downstream turn verbatim-reused): median 2 per pair; 190/683 pairs have
  zero, so field recall measured against it is a **floor** metric — the blinded judge carries
  the faithfulness weight.
- **Splits:** exact-dedupe 683 → 682; seeded (42) 80/10/10 → **train 546 / valid 68 / test 68**.
  Not stratified (no label space — open-text generation). Training legs used length-trimmed
  copies for 16GB memory (train 365 / valid 44 at ≤1,850 tokens); **test is untrimmed**, so any
  model trained on the trimmed split is evaluated on blob lengths it never saw — documented,
  intentional (the long blobs are the use case).
- **Known gaps:** (1) single-user, single-workload traffic — one developer's coding-agent
  sessions; no transfer claim beyond similar agent-harness tool traffic. (2) 117-drop breakdown
  unlogged (above). (3) fields-used is a floor, not a measure of everything the consumer needed.
  (4) trimmed train split never shows the model >~1,850-token blobs.
- **Intended use:** narrow internal tool-result compression model for the tunelab dogfood;
  teacher outputs from a session-native teacher (see the distillation concepts note bundled with
  the tunelab plugin for the provider-ToS discussion — this dataset trains a narrow task model
  on the author's own traffic, not a competing general model). **Training is blocked at 16GB**
  (six Metal OOMs; see EXPERIMENT-LOG.md 2026-06-12) — the pre-registered bar in
  EXPERIMENT-LOG.md remains unconsumed and binds any future run against this data.
