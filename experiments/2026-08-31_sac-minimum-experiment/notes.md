# SAC minimum experiment (2026-08-31)

First end-to-end run of `scripts/train_sac.py`, using the design in
`docs/rl_design.md` §5 and default hyperparameters (6 self-play races,
15s/race, `copies_per_side=1`, hidden size 128, buffer capacity 50k,
warmup 1,000 steps, batch size 256, update every 4 steps).

## What this run was for

Check that the self-play/`TrainableController` plumbing actually works end
to end (observation encoding, reward, replay buffer, SAC updates, saving/
loading a checkpoint, evaluation via `run_headless_head_to_head`) and get a
first read on whether the trained policy beats the `crash_fast` baseline
across the fixed 5-seed evaluation set.

## What happened

- Self-play training: 6 races × 15s collected 10,788 transitions and ran
  2,448 SAC gradient updates in 7.6s wall-clock.
- Evaluation (2 head-to-head races per seed, 20s rounds) against
  `crash_fast`:

  | Seed | SAC-trained scored distance | crash_fast scored distance | SAC race wins |
  | --- | --- | --- | --- |
  | 110 | 62.0 m | 0.0 m | 2/2 |
  | 42 | 57.3 m | 0.0 m | 2/2 |
  | 7 | 41.0 m | 0.0 m | 2/2 |
  | 2024 | 80.5 m | 0.0 m | 2/2 |
  | 8675309 | 48.6 m | 0.0 m | 2/2 |

- `crash_fast` scores `0.0 m` on every seed: full-throttle/no-steer drives
  it into a wall almost immediately, and scored distance excludes contact
  time, so it never accumulates any.
- The SAC-trained controller won every evaluation race on every seed after
  under 8 seconds of training — a low bar to clear (`crash_fast` is a
  worst-case baseline, not a competent one), but it confirms the pipeline
  produces a controller that survives and makes real progress, consistently
  across seeds, not just a lucky one.

## Decision / next steps

Plumbing is validated; the design in `docs/rl_design.md` is viable as a
starting point. This run is not evidence of a *competitive* controller —
next evidence needed is a longer run compared against a smarter baseline
(e.g. `default_student_controller` in `racing.student.api`, which actually
tries to stay on-track) and a check for reward/real-progress divergence
over a longer training horizon. See `docs/lab_notebook.md`'s 2026-08-31
entry for full details.
