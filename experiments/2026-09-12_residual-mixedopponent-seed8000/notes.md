# Residual RL, mixed self/expert-opponent training curriculum: the worst result yet (2026-09-12)

Seventh attempt at closing the remaining pace gap to `controllers.leaderboard_expert`'s
raw speed (8.94s solo) -- see `docs/rl_design.md` section 6, causal test 37 and
its six prior failed follow-ups (residual scale x2, a 5-seed sweep,
progress-weight reweight, curvature-aware center-offset,
wall-proximity-speed-scale, damage-weight). Unlike those six, this is a
genuinely different *mechanism*, not another reward-weight tweak: alternates
the self-play *training opponent* race-by-race between another in-training
residual copy (self) and `controllers.leaderboard_expert.Controller` directly
(a real, live, already-competent opponent), rather than changing what the
reward pays for. Motivated by a next-step note on the earlier (non-residual)
`--opponent expert` fine-tune dose sweep
(`experiments/2026-09-10_expert-opponent-seed8000/followup_finetune_sweep.md`):
switching the opponent entirely destabilized training at every dose tried,
but alternating never fully leaves the self-play distribution, so it might
avoid that failure mode while still exposing the policy to the expert's
actual driving/interaction style.

**Implementation:** added `--opponent mixed` to `scripts/train_sac.py`
(alongside a `--mixed-opponent-expert-every` ratio, default 2 = every other
race) and a corresponding loop in `train()`: instead of one
`run_headless_head_to_head` call spanning all races, it calls the function
once per race (`race_count=1`), alternating the incumbent between a fresh
self-play `TrainableController` and `create_expert_controller()`, varying
`random_seed` per race so races aren't identical repeats. Smoke-tested on a
tiny config (4 races, 8s rounds) before committing to the full run --
confirmed the 2-vs-expert/2-self-play split and that the automatic
post-training eval correctly includes `leaderboard_expert` as a third
baseline (mirroring how `--opponent expert` already does this). Matched to
`2026-09-11_residual-expert-base-v2-seed8000`'s config otherwise (seed=8000,
races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000, residual_expert_base=True, residual_action_scale=0.3,
mixed_opponent_expert_every=2).

**Along the way, found and fixed a real bug** in `--opponent self`'s
incumbent construction (it wasn't passing `wall_proximity_speed_scale_mps`/
`damage_weight` through, unlike the challenger) -- doesn't affect this run
(neither override was used here), but retroactively caveats the two prior
wall-scale/damage-weight experiments; see their own notes.md for details.

Training collected fewer transitions than v2 (415,906 vs. 575,840) and
finished faster in wall-clock terms (344.8s vs. 497.1s) despite spinning up
a fresh simulator instance per race instead of once for all 40 -- expected,
since expert-opponent races only have one learning copy pushing to the
buffer (the expert itself doesn't learn), versus two copies in every
self-play race.

## Results

**vs. crash_fast + default_student_controller (20 races, standard protocol):**

| | v2 (reference) | mixed-opponent |
| --- | --- | --- |
| avg damage | 0.0254 | 0.0787 (worse) |
| avg off-track | 0.325s | 3.572s (11x worse) |
| avg wall-contact | 0.140s | 2.422s (17x worse) |
| avg best lap time | 11.37s | 12.60s (slower) |
| fastest lap | 10.33s | 11.62s (slower) |
| eliminated | 0/20 | **1/20 (regressed)** |
| wins | 20/20 | 20/20 |

**vs. leaderboard_expert as a live opponent (10 races, same stress test as
every causal test 37 follow-up):**

| | v2 (reference) | mixed-opponent |
| --- | --- | --- |
| eliminated | 0/10 | **2/10 (regressed)** |
| avg laps | 9.80 | 8.60 (worse) |
| avg best lap time | 11.78s | 12.69s (slower) |
| wins | -- | 1/10 |

## Read

**The worst result of any residual-mode variant tested to date, and the
first to regress general driving competence rather than just the
expert-matchup.** Off-track and wall-contact time exploded even against
`crash_fast`/`default_student_controller` -- opponents with no bearing on
whether the policy has learned anything about racing *against* the expert
specifically -- which means this isn't a targeted "handles the expert
matchup differently" effect, it's a broad degradation in ordinary cornering/
line-following. Likely mechanism: half of training races now involve a much
faster, more aggressive, non-learning opponent sharing the track, changing
the traffic/relative-positioning distribution substantially from pure
self-play (where both cars are comparably-paced and learning together) --
plausibly the policy spent training-time budget adapting to (or being
disrupted by) that opponent's presence rather than refining its own
solo-line competence, and 40 races' worth of training wasn't enough to
recover both. Checked `metrics.csv`: critic loss peaked at 290.8 (roughly
10x the 30.0 spike the curvature-aware center-offset test flagged as
notably elevated) and averaged 26.7 over the last 10 updates, well above
this track's normal bounded range for a converged residual run -- this
*is* training instability, closer to what the non-residual `--opponent
expert` attempts showed (severe, fast collapse) than to a clean,
stable-but-worse convergence. Less severe than that fully-switched case
(this checkpoint still drives competently, just worse), consistent with
"alternating never fully leaves the self-play distribution" softening but
not eliminating the instability.

## Decision and rationale

**Not adopted.** This is the seventh structurally different lever to fail at
closing the pace gap to Lucy's raw expert, and unlike every reward-weight
tweak tried before it, this is a full mechanism change -- yet it produced a
*worse* result on every metric than any single-variable reward tweak tested
so far, including reintroducing eliminations on the standard baselines
(never seen in any residual-mode checkpoint since the recovery-passthrough
fix). `--opponent mixed` and `--mixed-opponent-expert-every` are kept as
tested, documented, opt-in flags (default opponent remains `self`, so
existing behavior is unaffected) rather than reverted code, consistent with
this track's practice of preserving negative results.

## Next steps

1. `2026-09-11_residual-expert-base-v2-seed8000` remains the best
   combined-approach checkpoint, now the strongest result after seven
   independent attempts (six reward-tuning + this one mechanism change) to
   beat it on pace.
2. If revisited: a lower expert-exposure ratio (e.g. every 4th or 5th race
   instead of every other) might dilute the disruption enough to test
   whether *some* exposure helps without this much cost -- not yet tried,
   though after seven failures this is a lower-conviction next step than
   concluding the line.
3. Recommending this line of investigation (closing the pace gap to Lucy's
   raw, safety-unconstrained expert) be treated as concluded rather than
   attempting an eighth variation.
