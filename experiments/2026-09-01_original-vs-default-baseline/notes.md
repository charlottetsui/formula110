# Re-evaluate the 2026-08-31 checkpoint against a real baseline (2026-09-01)

`scripts/eval_sac.py --checkpoint experiments/2026-08-31_sac-minimum-experiment/checkpoints/policy_final.pt`,
no retraining — same checkpoint, evaluated against two baselines instead of
one.

## Why

The 2026-08-31 minimum experiment only compared against `crash_fast`, which
scores `0.0 m` on every seed (it crashes almost immediately; scored
distance excludes contact time). That comparison is nearly vacuous. Added
`racing.student.api.default_student_controller` — a real center-line-
following heuristic — as a second baseline.

## Result

| Seed | SAC vs crash_fast | SAC vs default_student_controller |
| --- | --- | --- |
| 110 | 62.0 m (won 2/2) | 67.2 m vs 99.3 m (lost 0/2) |
| 42 | 57.3 m (won 2/2) | 42.0 m vs 106.9 m (lost 0/2) |
| 7 | 41.0 m (won 2/2) | 31.3 m vs 92.5 m (lost 0/2) |
| 2024 | 80.5 m (won 2/2) | 41.7 m vs 92.6 m (lost 0/2) |
| 8675309 | 48.6 m (won 2/2) | 16.3 m vs 104.1 m (lost 0/2) |

`crash_fast` numbers reproduce exactly (deterministic re-run, sanity check
that `eval_sac.py` works correctly). Against `default_student_controller`,
the SAC-trained controller **loses every race on every seed** (0/10).

## Decision / next steps

"Beats crash_fast" was not evidence of a competitive controller. Both
baselines are now evaluated automatically by every future run
(`training.evaluation.evaluate_against_baselines`), so this comparison
doesn't need to be manually re-run again. See
`docs/lab_notebook.md`'s 2026-09-01 entry.
