# Stage 3 final run log

The record of the one final run on the test split (task 3.7). Rule, fixed in advance
(`docs/hypotheses_stage3.md`, dated 3.7 note): a test-split measurement may be rerun only for a
technical failure (crash, corrupted or incomplete output file, power loss), never because of its
result. Every rerun is listed below with its reason.

- **Code:** the commit tagged `stage3-final-run`.
- **Command:** `python scripts/run_stage3_final.py`, then `python scripts/18_judge_stage3.py`.
- **Step-by-step log:** `results/final/run_log.txt` (not in git; start, end, exit code and duration
  of every step).

## Runs

| Date | What | Outcome |
|---|---|---|
| 25 Sep 2026, 20:50–00:07 IST | `run_stage3_final.py` at `54026c9` (tag `stage3-final-run`), laptop on mains power | All 13 steps exit 0 (7 clean test runs 2.3–3.3 min each; best INT8 + unrounded, 26 conditions, 110 min; five leave-one-out sweeps 9.8–19.6 min). Every sweep's clean check: largest logit difference 0.00. Completeness check: 105 of 105 files complete. |
| 26 Sep 2026, 00:08 IST | `18_judge_stage3.py`, run once, at `54026c9`, no uncommitted changes | H10 PASS; H11 FAIL (within noise); H12 FAIL; H13 PASS (within noise); H14 PASS; H15 PASS; H16 PASS; H17 NOT RUN (postponed to 3.9). Saved to `results/final/mobilenet_v3_large_stage3_verdicts.json`. |

## Reruns (technical failures only)

None so far.
