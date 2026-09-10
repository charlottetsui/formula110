# Halve WEIGHT_CENTER_OFFSET, on top of the uncapped-speed reward (2026-09-03)

Follow-up to `2026-09-02_uncapped-speed-scaled-risk-seed110` (the current
best/reference checkpoint): removing `MAX_REWARDED_SPEED_MPS` increased
top speed (+10%) without regressing safety, but lap time/count got
slightly *worse*, not better. That run's own notes proposed loosening
`WEIGHT_CENTER_OFFSET` next -- a uniform centerline penalty could be
suppressing any real racing line (wider entry/exit through corners, let
alone drift-style sliding) even where deviating from centerline would
actually be faster.

Single-variable change: `WEIGHT_CENTER_OFFSET` 0.3 -> 0.15 (a deliberate
half-step, not a full return to the old 0.05 that caused the 2026-09-01
off-track regression this weight originally fixed). Everything else held
identical to the reference: seed 110, races=40, round_seconds=120,
buffer_capacity=800000, uncapped `forward_progress_m`,
`WALL_PROXIMITY_SPEED_SCALE_MPS = 10.0`, trained from scratch.

## Process note: a real evaluation mistake, caught and fixed

The first `train_sac.py` invocation omitted `--eval-round-seconds 120`
(the reference run's config records `eval_round_seconds: 120.0`, but
`train_sac.py`'s default is `20.0`) -- so the first evaluation pass ran
20s rounds, not comparable to any prior 120s-round result at all (e.g.
59.1m vs. crash_fast at 20s vs. hundreds of meters at 120s elsewhere).
Caught by checking `config.yaml` against the reference's before reporting
any numbers. Training itself was unaffected (round length is a training
parameter too, and it *was* set correctly there --
`round_seconds: 120.0` in the saved config). Re-evaluated the same saved
checkpoint with `scripts/eval_sac.py --eval-round-seconds 120` (no
retraining needed). The mistaken 20s files are kept as
`config_WRONG_20s.yaml` / `eval_results_WRONG_20s.json` for the record,
not used in any comparison below.

## Result: a clear regression, not the hoped-for trade-off improvement

| | races=40, uncapped-speed reference (`WEIGHT_CENTER_OFFSET = 0.3`) | this run (`WEIGHT_CENTER_OFFSET = 0.15`) |
| --- | --- | --- |
| avg max speed | 17.13 m/s | **8.80 m/s (-49%)** |
| avg damage | 0.003 | 0.006 (still ~perfect) |
| eliminated | 0/20 | 0/20 (unchanged) |
| avg off-track | 0.082s | 0.017s (still near-perfect) |
| avg wall contact | 0.076s | 0.007s (still near-perfect) |
| avg marshal/race | 0.25 | 1.00 (worse, still low) |
| avg laps | 3.90 | **1.00 (-74%)** |
| avg best lap time | 28.46s | **78.80s (+177%)** |
| wins vs. `crash_fast` | 10/10 | 10/10 (unchanged) |
| wins vs. `default_student_controller` | 10/10 | **3/10** |

Safety held essentially flat (still ~zero damage/off-track/wall-contact,
zero eliminations) -- but every performance metric got dramatically
worse, not better. Max speed nearly halved, laps dropped from ~4 to 1,
best lap time nearly tripled, and the checkpoint now loses most races
against the real baseline it was beating 10/10 before (down to 3/10).
This is not the "gains a usable racing line" outcome hypothesized in the
prior run's notes -- it's a much less competent driver that happens to
still avoid walls.

Diagnostic: avg `low_progress_seconds` 6.82s and avg `car_contact_seconds`
3.01s per 120s race (not tracked in the reference's own notes, so no
direct before/after here, but notably nonzero) are consistent with a
policy that wanders or stalls against the opponent car more than it used
to, rather than one that found a faster line through corners.

## Decision and rationale

**Reverted** `WEIGHT_CENTER_OFFSET` back to `0.3` in `src/training/
reward.py` (comment updated to record this attempt and its result, same
pattern as the file's other reverted experiments). Not adopting this
checkpoint. `2026-09-02_uncapped-speed-scaled-risk-seed110` remains the
reference.

**Why the hypothesis was wrong, best guess:** a uniform per-tick
centerline penalty is small relative to `WEIGHT_PROGRESS` at any single
tick, so halving it doesn't obviously "unlock" an intentional wider
racing line -- but it does uniformly weaken the one signal that was
keeping the policy tracking a specific line through corners at all.
Without the old cap, `forward_progress_m` already rewards raw speed
anywhere the car points, including off the racing line; weakening the
centerline term on top of that seems to have made cornering *less*
deliberate, not more aggressive-but-controlled. This looks like the same
family of failure as the steering-smoothness and idle-penalty attempts:
a plausible-sounding loosened constraint that removes a signal the
policy was actually depending on, rather than one that was purely
suppressing something better underneath.

## Next steps

1. **(recommended)** Treat `WEIGHT_CENTER_OFFSET` loosening as a dead
   end at this step size, same as `MAX_REWARDED_SPEED_MPS` tuning was
   before it -- don't retry smaller decrements without a different
   diagnosis first.
2. If drift-style/wider-line cornering is still wanted, the next idea
   should target corners specifically (e.g. a term conditioned on
   `camera.lookahead_offsets_m` curvature, or only relaxing the penalty
   when wall-proximity margin is large) rather than a global weight
   change that also touches straights and safe cruising, which this
   change did.
3. Otherwise, per `docs/rl_design.md` §6's running read: treat
   `2026-09-02_uncapped-speed-scaled-risk-seed110` (races=40, uncapped
   speed, flat `WEIGHT_CENTER_OFFSET = 0.3`) as the practical best result
   and shift to consolidation (broader seed testing, repackaging
   `controllers.race_faster`).
