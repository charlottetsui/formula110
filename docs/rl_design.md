# SAC Controller Design

Scope: this document covers the **SAC (Soft Actor-Critic)** exploration
track only — one of the two approaches required by the COMP 590H project
rubric. It is owned and implemented independently; it does not describe or
depend on the second (PPO) approach, which is a separate, independently
developed track.

## 1. Hypothesis

SAC is an off-policy, maximum-entropy actor-critic method for continuous
action spaces. It's a plausible fit here because:

- The controller's action space (`throttle`, `steer`) is already continuous
  in `[-1.0, 1.0]` — no discretization needed.
- SAC is sample-efficient relative to on-policy methods because it reuses
  past experience from a replay buffer instead of discarding data after
  every update, which matters because each simulated race is real wall-clock
  time (headless races still run full physics ticks).
- The entropy bonus encourages exploration without hand-tuned exploration
  noise schedules, which is useful in a track/reward setting with sparse,
  spiky failure signals (wall contact, elimination).

Risk: SAC assumes a standard Gym-style `step(action) -> (obs, reward, done,
info)` loop. Formula 110 does not expose that (see §3). The design has to
work around that gap before SAC's sample-efficiency advantage can be
realized at all.

## 2. MDP formulation

### 2.1 Observation vector

Built from `RobotSensors` (see [SENSORS.md](../SENSORS.md)) each tick,
normalized to roughly `[-1, 1]`:

| Feature | Source | Notes |
| --- | --- | --- |
| Signed speed | `odometry.speed_mps` | scaled by an assumed max speed |
| Heading error | `camera.heading_error_degrees` | `/180` |
| Center offset | `camera.center_offset_m` | scaled, clipped |
| Lookahead offsets (3) | `camera.lookahead_offsets_m` | scaled, clipped |
| Wall LiDAR (7 beams) | `wall_lidar.distances_m` | `inf` → sensor cap (e.g. 20 m), then scaled to `[0,1]` |
| Yaw rate | `imu.yaw_rate_degrees_per_s` | scaled |
| Wall/robot contact | `contact.wall`, `contact.robot` | binary or scaled duration |
| Damage | `contact.damage` | already `[0, 1]` |

`sensors.lidar` (walls + cars) is deferred to the refinement stage once the
minimum experiment shows the wall-only signal is plumbed correctly —
opponent-aware features only matter once a policy can already hold the
track solo.

### 2.2 Action space

`RobotCommand(throttle, steer)`, both continuous `[-1.0, 1.0]`, output
directly by a `tanh`-squashed Gaussian policy head — the standard SAC action
distribution needs no adaptation here.

### 2.3 Reward (proxy, not official score)

Official lap/race progress is **not exposed** to controllers (see §3).
Reward is therefore a proxy built entirely from public sensor fields,
evaluated every tick as `reward(prev_sensors, action, sensors)`:

```
reward = w_progress * forward_progress_proxy
       - w_center    * abs(center_offset_m)
       - w_wall      * wall_proximity_penalty
       - w_contact   * (contact.wall > 0 or contact.robot > 0)
       - w_damage    * (damage_this_tick - damage_prev_tick)
       - w_reverse    * max(0, -speed_mps)
```

- `forward_progress_proxy` = `speed_mps * cos(heading_error_rad) * dt_s`,
  i.e. speed credited only insofar as it's aligned with the local track
  heading. This rewards net forward motion along the track rather than raw
  speed, without needing the private lap-progress model.
- `wall_proximity_penalty` ramps up as `wall_lidar.front_m` /
  `front_left_m` / `front_right_m` shrink below a threshold, so the policy
  learns to slow or steer away before contact, not just after.
- The `damage_this_tick - damage_prev_tick` term already scales up on its
  own for a hard impact that jumps damage a lot in one tick; there's no
  separate discrete penalty for reaching exactly `1.0` (see §2.4 for why
  that exact value is never observed anyway).

Because this is a *proxy*, every evaluation run additionally reports the
simulator's own public race statistics (`HeadToHeadTeamRaceStats`:
`distances_m`, `lap_counts`, `damages`, `max_speeds_mps`,
`best_lap_times_seconds`, etc. — see
[`head_to_head.py`](../src/racing/race/head_to_head.py)) so proxy-objective
divergence (reward going up while real progress doesn't) is visible.

### 2.4 Episode boundaries

There is no explicit `done` flag. Two public signals stand in for it:

- **New episode marker:** `sensors.tick == 0` on a fresh snapshot means the
  simulator just (re)started tracking this car (SENSORS.md: "Starts at `0`
  for a new car/controller run"). In self-play (§3), every car in every
  race already gets a brand-new `TrainableController` instance via
  `copy_for_car`, so `previous_observation` is already `None` whenever
  `tick == 0` is seen in practice; this signal is kept mainly for a future
  training loop that reuses one controller instance across more than one
  car life.
- **Near-elimination, not exact elimination:** the design originally
  assumed `contact.damage >= 1.0` was observable a tick early via the
  damage delta. It isn't: `run_headless_head_to_head` stops calling a car's
  controller once its `eliminated` flag is set, and that flag is set from
  damage applied *after* the tick whose sensors the controller last saw
  (`apply_wall_impact_damage` runs after the per-tick control loop, before
  the *next* tick's sensors would reflect it — see
  [`head_to_head.py`](../src/racing/race/head_to_head.py)). The exact
  terminal `damage == 1.0` reading is therefore never delivered to
  `__call__`. `src/training/reward.py` instead treats
  `contact.damage >= NEAR_ELIMINATION_DAMAGE` (`0.9`) as terminal — an
  approximation, not the literal last tick, but close enough that the
  bootstrap term it zeroes out (`(1 - done) * target_Q`) is negligible
  either way. Documented here as a correction to the original assumption;
  see `docs/lab_notebook.md`'s 2026-08-31 entry for how this was found.

A round timeout (`--round-seconds`, default 30s) also ends an episode
without a discrete signal in-band; because each race's controller instances
are freshly created and simply stop being called when the round ends, the
final in-progress transition of a race is never explicitly closed with
`done=True` (it stays `done=False`, i.e. treated as if the episode
continued). This is a known, accepted simplification for the minimum
experiment — see the refinement plan (§6) for revisiting it if it turns out
to bias value estimates near the round boundary.

## 3. Self-play training architecture

**Constraint:** the simulator has no public single-car headless step API.
The only public headless entry point is
[`run_headless_head_to_head`](../src/racing/race/head_to_head.py), which
runs whole races between a `challenger_controller` and an
`incumbent_controller` (each ≥1 car copy) and returns aggregate stats only
after the race(s) finish. There's no external `env.step()` to call once per
tick, and official lap progress / physics-world / other-controller internals
are all private.

**Resolution: the controller is its own RL agent, trained via self-play.**

- A `TrainableController` (in `src/training/`, imported by a thin wrapper
  under `src/controllers/` per the packaging contract) implements
  `RobotController.__call__(sensors) -> RobotCommand`. Internally, each call:
  1. Builds the observation vector from `sensors` (§2.1).
  2. If a previous observation/action is cached, computes the reward (§2.3)
     transitioning from that previous obs to this one, and pushes
     `(obs_prev, action_prev, reward, obs, done)` into a **shared replay
     buffer**.
  3. Samples the action for `obs` from the current SAC policy (or a warmup
     random policy early in training).
  4. Periodically (every N env ticks) draws a minibatch from the shared
     buffer and runs one SAC gradient step (actor, twin critics, entropy
     temperature).
  5. Caches `obs`, the chosen action, and `sensors.tick`/damage for the next
     call, and returns the resulting `RobotCommand`.
- **Self-play**: `run_headless_head_to_head` is called with
  `challenger_controller` and `incumbent_controller` both pointing at
  `TrainableController` instances that share the same policy weights and
  replay buffer. `copies_per_side > 1` runs several in-training cars per
  race concurrently, each an independent data-collection stream feeding the
  one shared buffer — the racing-against-itself traffic also forces the
  policy to cope with opponents (car `lidar`, `camera.competitors`) instead
  of learning a solo-track policy that falls apart in traffic.
- `RobotController.copy_for_car()` (used by `controller_for_copy` in
  `head_to_head.py`) is implemented to return a fresh `TrainableController`
  wrapping the *same* shared buffer/agent object, so per-car episode state
  (cached previous obs, tick tracking) stays independent while learning
  stays shared.
- The training loop itself is just a Python script that repeatedly calls
  `run_headless_head_to_head(...)` over many races/seeds, letting the
  gradient steps happen inside the controller callbacks rather than in an
  external `for step in range(...)` loop. This uses only the public API —
  no private/underscore-prefixed simulator internals.
- `sensor_sample_callback` (a `run_headless_head_to_head` parameter) is used
  for **evaluation/logging only** — e.g. dumping per-tick sensor snapshots
  to `experiments/<run>/` for offline analysis — not for training, since the
  controller already receives every sensor snapshot directly.

This design trades off a standard Gym-style training loop for one that
respects the simulator's actual public surface. The main risk it accepts:
gradient steps happen inside the hot control-tick path, so update frequency
and batch size must stay cheap enough not to distort tick timing in a
headless race (see 512 MiB / CPU-only rules in
[README.md](../README.md#cpu-and-memory-boundary), which also apply during
training since the same controller object races).

## 4. Dependencies

Not yet in `pyproject.toml`; add when training code lands:

```bash
uv add torch numpy
uv sync --managed-python
```

No `gymnasium` / `stable-baselines3` dependency — §3 explains why a
standard `Env` wrapper doesn't fit the public API, so SAC (actor network,
twin Q critics, target networks, entropy temperature, replay buffer) is
implemented directly in `src/training/`. Commit `pyproject.toml` and
`uv.lock` together whenever dependencies change.

## 5. Minimum experiment (exploration stage)

Smallest implementation that produces evidence the approach is viable:

- Observation subset from §2.1, reward from §2.3, both wired up.
- A small replay buffer (e.g. 50k transitions), simple 2-layer MLP actor +
  critics, a modest number of self-play races (enough ticks to fill the
  buffer once and take a few hundred gradient steps) — enough to check the
  plumbing works end-to-end, not to produce a competitive controller yet.
- Baseline for comparison: **two** baselines, evaluated automatically by
  `training.evaluation.evaluate_against_baselines` (used by both
  `scripts/train_sac.py` and `scripts/eval_sac.py`) —
  `src/controllers/crash_fast.py` (always full throttle, no steering) and
  `racing.student.api.default_student_controller` (a real center-line-
  following heuristic). `crash_fast` alone is a near-vacuous floor (it
  scores `0.0 m` on every seed, since it crashes almost immediately and
  scored distance excludes contact time) — see `docs/lab_notebook.md`'s
  2026-09-01 entry for why a second, real baseline was added after the
  first minimum experiment made `crash_fast`-only comparisons look better
  than they were.

**Evaluation:**

- Fixed seed set for reproducibility across runs: `110, 42, 7, 2024, 8675309`
  (same seeds used by both single-car and h2h per the README's deterministic
  spawn contract).
- Metrics, read from `HeadToHeadTeamRaceStats` / `HeadToHeadResult`, not the
  training-time proxy reward: scored distance, lap count, elimination rate,
  off-track seconds, marshal count/penalty, max speed, variance across the
  5 seeds.
- Compare against **both** baselines: beating `crash_fast` only shows the
  controller survives longer than a controller that doesn't try; beating
  `default_student_controller` is the bar that actually means something.
- Record wall-clock training time — SAC's sample efficiency claim (§1) is
  only worth something if training time is competitive with the alternative
  approach's dev/train time.

Results, logs, and configs for this and later runs live under
`experiments/` (see `experiments/README.md` for the per-run convention).

## 6. Refinement plan (post-selection)

Candidates for the next round of experiments once SAC is selected as the
primary approach, roughly in order of expected leverage:

0. **Reward risk-asymmetry / seed instability (new top priority,
   2026-09-01)** — the repeated-seed check below turned up something more
   important than a noisy pace number: at the scaled training budget, two
   different training seeds converged to **qualitatively different
   behaviors**, not two samples of similar competence. Seed `110`:
   damage/off-track/wall-contact all nonzero, real progress (avg 100.7m
   scored distance vs. `crash_fast`, 10/10 race wins). Seed `909`: **zero**
   damage, **zero** off-track time, **zero** wall contact in every one of
   10 evaluation races, but 27% of each race spent essentially stationary
   or crawling (avg 2.2m scored distance, only 5/10 wins, lost one race
   outright). The seed-909 policy appears to have found a "do nothing"
   local optimum: standing still/crawling never touches
   `WEIGHT_DAMAGE = 5.0`, `WEIGHT_CONTACT`, or (if it stays near spawn)
   `WEIGHT_CENTER_OFFSET`, and nothing in the reward directly penalizes
   near-zero forward speed (`WEIGHT_REVERSE` only fires on *negative*
   speed) — so it can be a locally rational strategy to just not drive.
   See `experiments/2026-09-01_scaled-training-budget-seed909/notes.md`
   and `docs/lab_notebook.md`'s 2026-09-01 entry for the full comparison.

   **Causal test 1 (run):** added `WEIGHT_IDLE = 0.2` (penalizes
   `abs(speed_mps) < 0.5`), re-ran with seed `909` held fixed. Fixed the
   freeze (5/10 -> 10/10 race wins, 2.2m -> 31.8m avg distance) but
   overcorrected: `damage == 1.0` (full elimination) in **all 10**
   evaluation races, max speed up 2-4x (to 11-19 m/s). See
   `experiments/2026-09-01_idle-penalty-seed909/notes.md`. Hypothesis: the
   simulator stops calling an eliminated car's controller, so dying early
   *ends* the idle penalty's per-tick accrual for the rest of the round —
   for a long enough round, "sprint and crash early" can look cheaper than
   "survive idly."

   **Causal test 2 (run):** added `WEIGHT_TERMINAL_PENALTY = 10.0` (a
   one-time penalty on the tick that crosses `NEAR_ELIMINATION_DAMAGE`, on
   top of the existing delta-based `WEIGHT_DAMAGE`), re-ran with seed `909`
   again held fixed. **Zero measurable effect** — `eval_results.json` and
   `metrics.csv` came back byte-identical to causal test 1's. A diagnostic
   run (`sensor_sample_callback` logging max damage during training)
   confirmed why: damage never exceeded 0.33 in any 60s *training* round,
   so `is_terminal()` was never true during training — the penalty had
   zero opportunities to apply. The elimination we see happens in the
   120s *evaluation* round, in territory the policy has never once
   experienced in training. See
   `experiments/2026-09-01_terminal-penalty-seed909/notes.md`.

   **Causal test 3 (run, hypothesis rejected):** re-ran with seed `909`
   held fixed, same reward (idle + terminal penalties), training round
   60s → 120s (matching eval length), so the terminal penalty would get a
   chance to apply. Elimination rate was **unchanged** (10/10, same as
   before) and average max speed got **worse**, jumping to 40.5 m/s (up
   from 15.9 m/s at 60s training, and ~6x the seed-110 reference's
   ~6.9 m/s) — very low off-track/wall-contact/low-progress times suggest
   it now crashes within the first couple of seconds, likely flooring the
   throttle in a straight line into the nearest wall. Verified this isn't
   a simulator artifact (checked `reset_robot_vehicle` zeroes velocity
   correctly on marshal reset). See
   `experiments/2026-09-01_longer-training-round-seed909/notes.md`.

   **Current read:** "extend training round length" was the wrong lever,
   or insufficient on its own — it made the safety problem worse, not
   better.

   **Causal test 4 (run, hypothesis rejected):** tested the train/eval
   behavior-mismatch hypothesis directly — added a `deterministic`
   parameter to `TrainableController`/`evaluate_against_baselines`
   (previously coupled to `training`) so the same checkpoint could be
   evaluated with sampled (stochastic, like training) actions instead of
   only the mean action. Loaded the causal-test-3 checkpoint (40.5 m/s,
   100% elimination) and compared: deterministic avg max speed 40.46 m/s
   vs. stochastic 40.02 m/s; damage 1.000 vs. 1.000; elimination rate 100%
   vs. 100%. **Essentially identical — hypothesis rejected.** The danger
   is not an eval-time artifact; the policy learned this behavior
   substantively. (This also corrected a misreading: self-play's printed
   training total, 913.4m over 10 races = 91.3m/race, matches the eval
   averages almost exactly — it was never evidence of safe training
   behavior, it just wasn't divided by race count at the time.) See
   `experiments/2026-09-01_stochastic-vs-deterministic-diagnosis/notes.md`.

   **Causal test 5 (run, hypothesis rejected):** added
   `MAX_REWARDED_SPEED_MPS = 10.0`, capping the speed used in
   `forward_progress_m` (chosen above the zero-elimination seed-110
   checkpoint's ~6.9 m/s and the heuristic baseline's ~5 m/s, well below
   the 15-40+ m/s crash regime). Re-ran with seed `909` held fixed, same
   120s training round as causal test 3. **Essentially no effect**: avg
   max speed 40.5 → 38.8 m/s, elimination rate unchanged at 10/10. See
   `experiments/2026-09-01_speed-cap-seed909/notes.md`.

   **Why, quantitatively:** at the capped speed, progress reward is
   `1.0 * 10.0 / 60 ≈ 0.167` per tick. Sustained for just 2-3 seconds
   (120-180 ticks) — roughly what it takes to cover the observed ~85-92m
   at ~38 m/s before crashing — that's already 20-30 cumulative reward,
   2-3x `WEIGHT_TERMINAL_PENALTY = 10.0`. The cap removed the reward for
   exceeding 10 m/s, but did nothing to change the more basic fact that a
   short, capped-speed burst is already profitable enough to make dying
   net-positive for the episode. The lever that matters is the *size* of
   `WEIGHT_TERMINAL_PENALTY` relative to achievable per-episode cumulative
   reward, not the shape of the progress term.

   **Causal test 6 (run, no meaningful effect):** raised
   `WEIGHT_TERMINAL_PENALTY` 10.0 → 100.0 (needs ~10s of at-cap driving to
   break even, vs. ~2s before) and, as a direct wall-avoidance
   strengthening pass, `WEIGHT_WALL_PROXIMITY` 0.5 → 1.0 and
   `WALL_WARNING_DISTANCE_M` 3.0 → 6.0 (3.0m gives almost no reaction time
   at 15-40+ m/s). Re-ran with seed `909` held fixed, same 120s training
   round. **Essentially no change**: avg max speed 38.8 → 39.0 m/s
   (unchanged), elimination rate 10/10 → 9/10 (marginal — the one survivor
   took 0.469 damage and then spent 89% of the race stuck, not a genuine
   wall-avoidance success). See
   `experiments/2026-09-01_terminal100-walldist6-seed909/notes.md`.

   **Pattern across three consecutive reward-tuning attempts on seed
   909** (speed cap, then this): average max speed has stayed pinned at
   38-40 m/s regardless of what the reward does, even though each fix was
   confirmed active during training (unlike the terminal-penalty test that
   never fired). This suggests seed 909's policy may be stuck in a
   resistant local optimum — a simple "floor it straight" behavior that's
   easy to represent and got reinforced early — that isn't responding to
   reward-shape changes alone at this training budget (~17-21k updates).
   Continuing to iterate using only seed 909 risks overfitting conclusions
   to one seed's particular stuck optimum. **Proposed next step: test the
   current, now substantially revised reward on seed 110** (the only
   checkpoint so far with zero eliminations) to see whether today's
   changes help, hurt, or don't matter there.

   **Causal test 7 (run) — a materially different, more encouraging
   result:** trained seed `110` from scratch with the full current reward
   (idle penalty, `WEIGHT_TERMINAL_PENALTY = 100.0`, speed cap, wall-
   avoidance changes), same 120s-round config as the seed-909 tests. Not a
   repeat of seed 909's near-total failure: best lap times of 15.8-22.3s
   (~9-11.6 m/s average pace, right around `MAX_REWARDED_SPEED_MPS`), 8-9
   laps completed in successful races, max speed down to 25-35 m/s (vs.
   38-47 m/s on every seed-909 test today), and **the first-ever win
   against `default_student_controller`** (6/10 races, avg scored distance
   841.8m vs. the baseline's own ~600-745m). But reliability is still the
   open problem: 6/10 races end in full elimination (some after productive
   laps, some almost immediately). See
   `experiments/2026-09-01_full-reward-seed110/notes.md`.

   **Read:** the same reward that left seed 909 pinned at ~90-100%
   elimination across three attempts produces a much better outcome
   starting from seed 110 — supports the "seed 909 stuck in a resistant
   local optimum" hypothesis over "the reward doesn't work." The problem
   has shifted from *no speed / no wall-avoidance at all* to
   *inconsistent — crashes in ~40% of races*, which is a different (and
   more tractable) problem than where this causal-test chain started.

   **Causal test 8 (run) — reliability problem resolved:** same seed
   `110`, same reward, same 120s round length as causal test 7 — only
   `--races 10 → 20` (and buffer capacity raised to match). Result:
   **zero eliminations across all 20 evaluation races** (5 seeds × 2
   races × 2 baselines), every single race completing exactly 4 laps,
   max speed settled to a controlled 17.8-21.2 m/s (down from 25-47 m/s),
   best lap times 21.2-27.9s (~6.5-8.6 m/s average lap pace — genuinely
   competent cornering, not just a straightaway sprint), and **20/20 race
   wins against both baselines**. Checked the training/eval seed overlap
   directly: the 4 genuinely held-out seeds (never used in training)
   show the identical pattern — this is not an artifact of testing on the
   training seed. See
   `experiments/2026-09-01_more-training-seed110/notes.md`.

   **Read:** more training on the same already-productive seed+reward
   combination (not a different seed, not a different reward) was enough
   to close the reliability gap — 2x the races (10 → 20, ~29k → ~60k
   gradient updates) took elimination rate from 60% to 0% and converged
   max speed to a controlled, consistent pace. This is the strongest
   result of the day by every metric that matters (reliability,
   consistency across seeds, and competitiveness against the strong
   baseline) and is now the reference checkpoint for further work.

   **Causal test 9 (run) — still improving, not plateaued:** same seed,
   reward, and round length again; `--races 20 → 40` (~60k → ~132k
   gradient updates). Clean, monotonic trend across all three
   training-budget levels tested today:

   | training budget | avg damage | avg off-track | avg wall contact | avg max speed | eliminated |
   | --- | --- | --- | --- | --- | --- |
   | races=10 | 0.746 | 1.88s | 0.92s | 30.2 m/s | 6/10 |
   | races=20 | 0.062 | 0.84s | 0.12s | 18.4 m/s | 0/10 |
   | races=40 | **0.000** | **0.00s** | **0.00s** | 15.5 m/s | 0/20 |

   At races=40: zero damage, zero off-track time, and zero wall contact in
   **every single one of 20 evaluation races** (all 5 seeds, both
   baselines) — not just avoiding elimination but not touching a wall or
   leaving the track at all. Max speed kept dropping and converged very
   tightly (15.4-16.1 m/s). Still 20/20 race wins, now with a larger
   margin (~1740-1820m vs. ~1550-1650m average scored distance). See
   `experiments/2026-09-01_more-training2-seed110/notes.md`.

   **Read:** every tracked metric moved monotonically in the same
   direction across all three training-budget levels — a genuine learning
   curve, not noise. Safety-relevant incidents are now at floor (exactly
   zero across every evaluated race), so further training from here would
   likely show up as speed/lap-count gains rather than more safety
   headroom. Reasonable point to pause the training-budget scaling and
   consider other directions.

   **Causal test 10 (run, regression, reverted):** attempted to optimize
   for speed by raising `MAX_REWARDED_SPEED_MPS` 10.0 → 20.0 (the
   races=40 reference already exceeded the old cap on max speed, 15.4-16.1
   m/s, while its average lap pace stayed well under even 10.0 — the cap
   wasn't blocking top speed, just not crediting sustained higher speed).
   Trained from scratch, same seed/races/round-length as the reference,
   every safety-side weight left unchanged. **Clear regression**: max
   speed roughly doubled (34-38 m/s) but laps completed dropped (4.1 avg
   → 0-2), lap times got *slower* despite the higher top speed (21-28s →
   36-96s), damage came back (0.000 → 0.14-0.66), marshal recoveries
   spiked to as high as 21/race (from ~0), and it lost races against
   `default_student_controller` for the first time since the races=40
   breakthrough (5/10, down from 10/10). Reverted `MAX_REWARDED_SPEED_MPS`
   back to `10.0`. See
   `experiments/2026-09-01_speedcap20-seed110/notes.md`.

   **Read:** the speed cap was never the bottleneck on top speed (already
   exceeded), so raising it didn't unlock more speed — it just shifted the
   reward's relative balance toward raw speed at the expense of cornering
   control, reopening the same speed-vs-control trade-off seen in the
   seed-909 causal chain, from a different starting point. **Reward
   magnitude is not the right lever for improving lap times from here;**
   more training on the existing reward is the lever that has worked
   cleanly all day (see causal tests 8-9).

   **Causal test 11 (run) — more training converges toward the cap, not
   past it:** same seed/reward/round-length again, `--races 40 → 80`
   (~132k → ~276k gradient updates). Safety metrics stayed exactly at
   floor (0.000 damage, 0.00s off-track, 0.00s wall contact, still
   perfect across all 20 races) — but max speed and lap count both
   *decreased*: 15.4-16.1 → 13.1-14.7 m/s, 4-5 → 2-3 laps, best lap times
   21.2-27.9s → 34.3-42.1s (slower). See
   `experiments/2026-09-01_more-training3-seed110/notes.md`.

   **Read:** `MAX_REWARDED_SPEED_MPS = 10.0` gives zero reward benefit for
   exceeding 10 m/s, only unrewarded risk — so continued training pushes
   the policy toward the actual reward-maximizing speed (at/near the cap),
   not past it. races=40's ~15-16 m/s was a residual of less-refined
   training, not a reward-seeking choice; races=80 converged it back down.
   **More training on the unchanged reward has hit its ceiling for the
   speed goal specifically** — it will keep converging toward the cap,
   not exceed it, however much further it's trained. Combined with causal
   test 10 (doubling the cap outright regressed badly), the most promising
   untried lever is a **smaller** cap increase (e.g. 10.0 → 12-13.0,
   close to races=40's own organic ceiling) rather than either "leave it"
   or "double it." races=40 remains the best checkpoint for the speed
   goal specifically (races=80 is arguably safer/more consistent, but
   that's not what's being optimized for).

   Paused here (eleven experiments deep) — see `docs/lab_notebook.md`'s
   2026-09-01 entry for next-step options. **Current best checkpoint for
   speed+safety: `2026-09-01_more-training2-seed110`** (races=40) — 0/20
   eliminations, zero damage/off-track/wall-contact in every evaluated
   race, ~15.5 m/s max speed, 4-5 laps and 20/20 race wins across both
   training and held-out seeds. (Note: `controllers.sac_candidate`
   auto-selects the *newest* checkpoint by file time, which after this
   entry is the races=80 one — point `FORMULA110_SAC_CHECKPOINT` at
   `2026-09-01_more-training2-seed110` explicitly if watching it live.)
1. **Training budget** — scale up races/round length/gradient updates.
   First attempt (2026-09-01: races 6→10, round length 15s→60s,
   ~2,400→17,751 gradient updates, same reward/hyperparameters/seed as the
   reward-reweight run) raised raw distance from ~13-30% of a lap to
   ~75-95%, produced the first-ever completed lap in evaluation (seed
   2024, 98.6s lap time), and improved off-track fraction (~18% -> ~12%)
   with damage/marshal rate roughly flat — but only for that one training
   seed. The repeated-seed check (item 0 above) shows training budget
   alone does not reliably produce a competent driving policy: it can
   also produce the frozen/"do nothing" failure mode, or (once that's
   patched) a sprint-to-death mode never trained against. Read as evidence
   that **training budget and reward risk-asymmetry are the same problem
   from two angles**, not two independent, separately-fixable issues.
2. **Reward shaping** — tune `w_progress`/`w_center`/`w_wall` weights;
   check whether the proxy reward and real scored distance move together
   across training (the divergence check from §2.3). A first attempt
   (raising `w_center` 0.05 → 0.3, everything else held fixed, same
   training seed/hyperparameters as the 2026-08-31 baseline, before the
   training-budget scale-up) produced no measurable change — see
   `docs/lab_notebook.md`'s 2026-09-01 entry. Superseded in priority by
   item 0's more specific, causally-testable hypothesis (the damage-weight
   risk asymmetry), but broader reward tuning (progress/center/wall
   weights together) remains worth revisiting once item 0 is resolved.
3. **Robustness across seeds** — widen the training seed distribution
   (rather than a fixed handful) so the policy doesn't overfit to specific
   spawn points; evaluate on held-out seeds never used in training. Note
   this is about the *spawn-point* seed distribution used during training,
   distinct from item 0's finding about training-*run* (network/
   exploration) seed sensitivity — both matter, but item 0 is the more
   urgent one since it produces qualitatively broken policies, not just
   overfit ones.
4. **Opponent traffic** — increase `copies_per_side` so self-play produces
   denser traffic, and start using `sensors.lidar` /
   `camera.competitors` in the observation once solo-track driving is
   solid.
5. **Network/optimizer tuning** — hidden sizes, learning rates, target
   network update rate (`tau`), entropy temperature (fixed vs. learned).
6. **Replay buffer** — size, and whether uniform sampling is good enough or
   prioritized replay is worth the complexity.
7. **Reduce hesitation** — penalize small/oscillating steering deltas if
   the trained policy shows jittery control in watched races.
8. **Packaging for the leaderboard (done, 2026-09-01)** — exported the
   `2026-09-01_more-training-seed110` checkpoint (causal test 8, current
   best: 0/20 eliminations, 20/20 race wins) as a self-contained
   `src/controllers/race_faster.py`, per the
   [packaging contract](../README.md#packaging-a-controller) and the
   Gradescope "improved module" convention in
   [`autograder/README.md`](../autograder/README.md). Unlike
   `sac_candidate.py` (a local-dev viewer that imports `training/`), this
   module inlines the actor network and observation encoding so it has no
   dependency outside `src/controllers/`, loading only the policy weights
   (trimmed from the full training checkpoint, 396KB -> 81KB, dropping
   critics/optimizer state) from a bundled `checkpoints/` file. CPU-only,
   `eval()` + `torch.inference_mode()`, always deterministic. Verified:
   output matches the full training checkpoint's `agent.act(...,
   deterministic=True)` bit-for-bit on 5 random observations, passes `ruff
   check`/`ruff format --check`/`pyright` (project's strict mode) with zero
   findings, and drives correctly in a real (non-synthetic)
   `racing h2h` race (seed 110, 30s: 0 damage-eliminations, ~18.3 m/s max
   speed, 1 lap, 210.6m vs. `crash_fast`'s 16.4m — consistent with the
   checkpoint's known behavior). Exported via
   `scripts/export_student_controllers.py --all-controllers` (needed
   because the checkpoint is a non-Python file) to
   `artifacts/formula110-student-controllers.zip`.

   **Scope note:** the Gradescope rubric
   (`autograder/README.md`) actually names two required modules —
   `controllers.minimum_viable` (a safe heuristic, judged on zero
   damage/wall-contact) and `controllers.race_faster` (the "improved"
   module this checkpoint fills, judged on survival + distance vs. the
   minimum module). No `minimum_viable` module exists yet; per user
   direction this submission covers only the SAC/`race_faster` half by
   design — writing a hand-tuned minimum-viable controller was judged out
   of scope for this track and deferred rather than attempted here.

   **Superseded (2026-09-01, causal test 9):** `race_faster.py` currently
   packages the races=20 checkpoint. `2026-09-01_more-training2-seed110`
   (races=40) is strictly better on every tracked metric — zero
   damage/off-track/wall-contact across all 20 evaluated races, vs. this
   one's small-but-nonzero damage (avg 0.062) and off-track/wall-contact
   time. Not yet repackaged pending direction — see
   `docs/lab_notebook.md`'s 2026-09-01 (continued, 13) entry.

Each refinement experiment should change a limited number of variables at
once and be compared against both the fixed baseline (`crash_fast`) and the
best prior SAC checkpoint, per `experiments/README.md`.
