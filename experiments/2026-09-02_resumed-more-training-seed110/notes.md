# Resume-from: continue training the races=40 checkpoint (2026-09-02)

Tests the `--resume-from` capability added this session as a genuinely
different lever than another from-scratch reward variant: fine-tune the
already-good races=40 checkpoint with 40 more races of the same plain
(non-trajectory) reward, `--warmup-steps 0` (don't discard the resumed
policy's competence for random warmup), otherwise identical config (seed
110, round_seconds=120, buffer_capacity=800000).

## Result: safety identical, speed and laps both got worse

| | races=40 reference (from scratch) | resumed +40 more races |
| --- | --- | --- |
| avg damage | 0.000 | 0.000 (identical) |
| avg off-track / wall contact | 0.00s / 0.00s | 0.00s / 0.00s (identical) |
| avg marshal/race | 0.15 | 0.15 (identical) |
| eliminated | 0/20 | 0/20 (identical) |
| avg laps | 4.10 | **2.00 (halved)** |
| avg best lap time | 24.76s | **46.36s (nearly double)** |
| avg max speed | 15.53 m/s | **13.91 m/s (lower)** |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

Safety is *exactly* preserved (identical marshal rate, zero damage/off-
track/wall-contact both before and after). But continuing to train the
already-good policy for 40 more races made it slower and completed fewer
laps, not faster -- the opposite of what resuming was meant to test.

## Why: this confirms, rather than contradicts, an earlier finding

This is the same pattern found in `docs/rl_design.md`'s causal test 11
(2026-09-01): training races=40 -> 80 *from scratch* under the identical
reward also converged speed *down* (15.4-16.1 -> 13.1-14.7 m/s) and laps
*down* (4-5 -> 2-3), because `MAX_REWARDED_SPEED_MPS = 10.0` gives zero
reward benefit for exceeding 10 m/s -- more gradient updates, whichever
path they arrive by, converge the policy toward the reward's actual
optimum (at/near the cap), not away from it. This resumed run reaches a
similar total gradient-update count (~143,937, vs. races=80's ~276,316)
via a different path (continuing from races=40's weights instead of a
fresh random init) and lands in essentially the same place: 13.91 m/s,
between races=40's 15.53 and races=80's 13.1-14.7. Two independent
training paths (from-scratch-then-more, and resume-then-more) now agree:
under this reward, more optimization converges toward ~13-15 m/s, not
past it -- races=40's ~15.53 m/s looks like a residual of not-yet-fully-
converged training, not a stable point.

## Decision and rationale

Not adopting this checkpoint (worse than the reference on the goal being
optimized). `2026-09-01_more-training2-seed110` (races=40) remains best.

This is the **seventh** consecutive experiment today (cap=12.0, cap=20.0,
more training from scratch [races=80], steering smoothness,
trajectory-bonus buggy, trajectory-bonus fixed, this resumed run) to fail
to beat races=40 on speed, and the second (with causal test 11) to
independently confirm the same underlying mechanism: `MAX_REWARDED_SPEED
_MPS = 10.0` is a real ceiling on what any amount of further optimization
under this reward will converge toward, regardless of method. Raising
that ceiling directly was already tried twice (10.0 -> 12.0, -> 20.0) and
both failed too -- a small raise didn't move the needle, a large raise
broke the speed/control balance.

**Taken together with today's full body of evidence, races=40 looks like
a strong, practical stopping point for this reward structure** -- not
because it's provably optimal, but because seven different attempts to
improve on it (four reward variants, one training-budget scale, one
resumed continuation, and their combinations) have not found anything
better, and several of the mechanisms tried are now understood well
enough to explain *why* they didn't help, not just that they didn't.

## Next steps (proposed)

1. **(recommended)** Treat `2026-09-01_more-training2-seed110` (races=40)
   as the practical best result for this SAC track and shift focus to
   consolidating/deploying it: broader seed testing beyond the fixed 5 to
   build more robustness confidence, repackaging
   `controllers.race_faster` from this checkpoint (it currently packages
   the older races=20 one), and the `controllers.minimum_viable` module
   gap noted in the 2026-09-01 packaging entries.
2. If more speed is still wanted later, the next genuinely new idea would
   need to change the reward's structural ceiling itself (not just its
   value) -- e.g. removing `MAX_REWARDED_SPEED_MPS` entirely in favor of
   a reward that scales with *safe* speed specifically (conditioning the
   progress term on wall-proximity margin, rather than a flat cap), which
   hasn't been tried. This is a bigger design change than anything
   attempted today and would warrant its own dedicated investigation
   rather than another quick variant.
