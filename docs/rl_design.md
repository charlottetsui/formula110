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
| Obstacle LiDAR (7 beams) | `lidar.distances_m` | walls + other robots; same encoding as wall LiDAR |
| Yaw rate | `imu.yaw_rate_degrees_per_s` | scaled |
| Wall/robot contact | `contact.wall`, `contact.robot` | binary or scaled duration |
| Damage | `contact.damage` | already `[0, 1]` |

**Obstacle LiDAR added 2026-09-07** (`OBSERVATION_DIM` 17 → 24): originally
deferred here as "opponent-aware features only matter once a policy can
already hold the track solo" — solo driving reached that bar on 2026-09-01
(zero damage/off-track/wall-contact at the `2026-09-01_more-training2-
seed110` checkpoint and every checkpoint since), but the deferred feature
was never revisited until a per-tick diagnostic on 2026-09-07 found the
policy was colliding blind with a stationary opponent car once per lap —
`camera.competitors` and `sensors.lidar` were the only public signals that
see other cars at all, and neither was in the observation vector. See
`docs/lab_notebook.md`'s 2026-09-07 entry for the diagnostic and the
before/after result. **This changes the network's input dimension — a
checkpoint saved before this change (`observation_dim=17`) cannot be loaded
by code built after it, and vice versa; there is no migration path, only
retraining.** `src/controllers/race_faster.py` is unaffected since it
inlines its own frozen copy of the encoding rather than importing
`training.observation`.

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
- **Fixed-opponent alternative (added 2026-09-10, tested, not adopted):**
  `scripts/train_sac.py --opponent expert` swaps the incumbent for a
  frozen `controllers.leaderboard_expert.Controller` instead of a second
  `TrainableController`, so only the challenger learns. Motivated by
  wanting a genuinely different sparring partner than a mirror of itself
  (see §6 causal tests 21–23's opponent-collision failure mode); found to
  cause severe training instability and a 100% elimination rate when used
  for a full from-scratch run — see §6 causal test 35. Default remains
  `self` (the architecture described above); kept in code as a tested
  mechanism, disabled by default, per this file's convention.

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

**n-step returns (added 2026-09-07):** the critic target uses n-step
returns instead of plain 1-step TD — each pushed transition sums `n_step`
real, consecutive ticks of reward (discounted) before bootstrapping from
the critic, rather than bootstrapping after a single tick. Configurable
via `--n-step` on `scripts/train_sac.py` (default 1, reproducing the
original 1-step behavior exactly). Motivation: several 2026-09-01/09-02
reward-tuning attempts (`docs/lab_notebook.md`) found that a delayed
penalty (e.g. `WEIGHT_TERMINAL_PENALTY` for a crash several ticks after a
risky action) only reaches the states responsible for it after many
sequential 1-step Bellman backups — n-step returns shorten that path by
including several ticks of real, observed consequence directly in a
single target. Implemented as a per-car sliding window in
`TrainableController` (`src/training/controller.py`); `ReplayBuffer`/
`ReplayBatch` carry a per-transition `discount = gamma**actual_n` (not a
shared scalar) since a window truncated by early termination bootstraps
over fewer than `n_step` ticks. See `docs/lab_notebook.md`'s 2026-09-07
entry and `experiments/2026-09-07_nstep3-seed110/notes.md` for the first
result (n=3 beat the prior reference on safety and lap time
simultaneously, at a lower top speed).

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

   **Causal test 12 (run, 2026-09-02) — small cap increase, no
   improvement:** tried `MAX_REWARDED_SPEED_MPS` 10.0 → 12.0 (a
   deliberately small, 20% step vs. the previous 2x jump), same seed
   (110), races=40, round length as the reference, trained from scratch.
   Result: essentially a tie on safety (still 0.000 damage, 0.00s
   off-track/wall-contact, 20/20 wins) and slightly *more* consistent lap
   completion (4 laps in every race vs. the reference's mix of 4-5), but
   average best-lap time was **slower**, not faster (24.76s → 28.20s), and
   max speed was not meaningfully different (15.53 → 15.88 m/s). See
   `experiments/2026-09-02_speedcap12-seed110/notes.md`.

   **Read:** three points on the `MAX_REWARDED_SPEED_MPS` axis have now
   been tried from this seed/config — 10.0 (reference, still the best),
   12.0 (this test, a tie or mild regression on lap time), 20.0
   (2026-09-01, a severe regression). None beat the original 10.0.
   Reverted back to `10.0`. Treating `MAX_REWARDED_SPEED_MPS` tuning as a
   dead end for the speed goal rather than continuing to search this
   axis — the lever that has actually worked today is training budget
   (causal tests 7-8), and the lever most likely to help further is a
   different one entirely (reducing hesitation/oscillation, or
   fine-tuning from an existing checkpoint rather than retraining from
   scratch each time, which `scripts/train_sac.py` doesn't yet support).

   **Causal test 13 (run, 2026-09-02) — steering-smoothness penalty, the
   clearest regression yet:** added `WEIGHT_STEERING_SMOOTHNESS = 0.1`,
   penalizing tick-to-tick change in `imu.yaw_rate_degrees_per_s` as a
   proxy for jerky steering (`step_reward` has no direct access to the
   steer action itself). Same seed/races=40/round-length as the
   reference, trained from scratch. Result: laps completed **halved**
   (4.10 avg → exactly 2.00, every single race), best lap time **nearly
   doubled** (24.76s → 45.48s), max speed dropped 31% (15.53 → 10.75 m/s)
   — safety unaffected (still perfect). Reverted
   `WEIGHT_STEERING_SMOOTHNESS` to `0.0` (mechanism kept in code,
   disabled by weight). See
   `experiments/2026-09-02_steering-smoothness-seed110/notes.md`.

   **Read:** raw yaw-rate *change* can't distinguish wasteful oscillation
   from a legitimate, necessary steering input for cornering — both
   involve yaw rate changing quickly — so the penalty suppressed real
   cornering, not just hesitation. This is the **fourth** consecutive
   reward-tuning attempt aimed at the speed goal (cap=12.0, cap=20.0,
   more training at cap=10.0, this term) to fail to beat the plain
   races=40 reference. That consistency is itself the finding:
   `2026-09-01_more-training2-seed110` sits in a fairly strong local
   optimum for this reward structure that isolated tweaks — each tested
   as a fresh from-scratch run — haven't improved on. **Recommending a
   pause on further reward-tuning attempts at pure speed optimization**
   until either a genuinely different mechanism is available (e.g.
   fine-tuning from the existing checkpoint instead of retraining from
   scratch) or the team decides the current speed is good enough and
   shifts focus elsewhere.

   **Causal test 14 (run, 2026-09-02) — a new mechanism, and its worst
   outcome yet:** rather than another global-constant tweak, added
   `src/training/trajectory.py` (`BestTrajectoryTracker`): reward relative
   to the best-known distance-at-tick the shared training run has ever
   reached, a location/time-specific "beat your own record" curriculum
   instead of a fixed constant applied everywhere. Wired in via a new,
   default-off `--trajectory-bonus` flag on `scripts/train_sac.py`. Same
   seed/races=40/round-length as the reference. **Result: the worst
   outcome of the day** — self-play's own training distance collapsed to
   3.0m/0.0m over 40 whole races (every prior run: hundreds to tens of
   thousands of meters), 0 laps completed in all 20 evaluation races,
   marshal recoveries up to 56/race (previous worst: 21), up to 93% of a
   race spent stuck. Unlike every prior regression today (each a
   coherent single strategy — uniformly faster-and-crashier or uniformly
   slower-and-cautious), this was incoherent, unstable training. See
   `experiments/2026-09-02_trajectory-bonus-seed110/notes.md`.

   **Root cause found, not just observed:** self-play runs two copies of
   the same policy in one race, controlled sequentially within each
   physics tick (`_run_headless_student_runtime_step` in
   `src/racing/race/head_to_head.py`), sharing one
   `BestTrajectoryTracker`. Whichever copy is processed first in a tick
   writes that tick's "record" *before* the second copy's bonus is
   computed from it — so a copy gets compared against a "best" its own
   rival just set in the *same race, same instant*, not a genuinely
   separate historical best. Since one grid position starts ahead of the
   other, this is a systematic, adversarial corruption between the two
   self-play copies, not the intended self-improving curriculum. A real
   design bug in the tracker's update timing, not a bad weight — the
   underlying idea (reward relative to your own best pace) remains
   well-motivated and worth revisiting once the timing issue is fixed
   (e.g. only write records from fully-completed episodes, not
   continuously while other copies are still racing and reading them).

   `--trajectory-bonus` defaults to off, so this doesn't affect any
   existing default behavior. Not adopting this checkpoint.

   **Causal test 15 (run, 2026-09-02) — bug fixed, still doesn't beat the
   reference:** decoupled reads from writes in `BestTrajectoryTracker`
   (`src/training/trajectory.py`): each episode now takes a frozen
   `snapshot()` at its own first tick and reads bonuses from that
   snapshot for its whole run, while `update()` still writes to the live
   tracker for *future* episodes. Re-ran the identical config. **The bug
   is fixed** — marshal recoveries dropped from 22.25/race back to
   0.50/race (near the reference's 0.15), off-track/wall-contact time
   back near zero, self-play's own training distance back to a normal
   order of magnitude (23,603m/25,141m over 40 races, vs. the buggy run's
   3.0m/0.0m). **But it still doesn't beat the plain reference**: 2.70
   avg laps (vs. 4.10), 32.73s avg best lap (vs. 24.76s), 2/20 eliminated
   (vs. 0/20). Max speed marginally higher (17.33 vs. 15.53 m/s) but
   doesn't translate to better lap times or reliability. See
   `experiments/2026-09-02_trajectory-bonus-fixed-seed110/notes.md`.

   **Read:** this is the sixth consecutive reward-tuning/mechanism
   attempt today that failed to beat races=40, though the first that's a
   genuine design fix rather than a hyperparameter guess. With n=1 per
   condition (every experiment today is one from-scratch training run),
   this can't yet distinguish "the idea doesn't help" from "this run had
   worse luck than the reference run." Rather than a seventh from-scratch
   attempt, shifting to a genuinely different mechanism: `--resume-from`
   (added to `scripts/train_sac.py` this session) to continue training
   the already-good races=40 policy directly, instead of re-deriving a
   new one from random init under each reward variant.

   **Causal test 16 (run, 2026-09-02) — resuming confirms, doesn't
   contradict, causal test 11:** used `--resume-from` to continue
   training the races=40 checkpoint for 40 more races (plain reward,
   `--warmup-steps 0`), rather than another from-scratch reward variant.
   Safety **exactly preserved** (identical 0.000 damage, 0.00s off-track/
   wall-contact, 0.15 marshal/race, both before and after). But laps
   *halved* (4.10 → 2.00) and best lap time nearly *doubled* (24.76s →
   46.36s), landing at 13.91 m/s — between races=40's 15.53 m/s and
   races=80's 13.1-14.7 m/s (causal test 11, same reward, same total
   gradient-update count reached via a fresh-from-scratch run instead of
   resuming). **Two independent training paths now agree**: under
   `MAX_REWARDED_SPEED_MPS = 10.0`, more optimization — however it's
   reached — converges the policy toward ~13-15 m/s, not past it.
   races=40's ~15.53 m/s looks like a residual of not-yet-fully-converged
   training, not a stable point the reward actually rewards. See
   `experiments/2026-09-02_resumed-more-training-seed110/notes.md`.

   **Overall read, seven experiments into the speed-optimization
   question:** cap=12.0, cap=20.0, more training from scratch (races=80),
   steering smoothness, trajectory-bonus (buggy and fixed), and this
   resumed run have *all* failed to beat races=40 on speed — either by
   regressing safety (cap=20.0, trajectory-bonus buggy) or by regressing
   speed while preserving safety (cap=12.0, races=80, trajectory-bonus
   fixed, this resumed run). The mechanism is now understood, not just
   observed: `MAX_REWARDED_SPEED_MPS` sets a hard ceiling on what any
   amount of further optimization converges toward, and directly raising
   that ceiling (tried twice) either does nothing (small raise) or breaks
   the speed/control balance (large raise). **Recommending races=40 as
   the practical stopping point for this reward structure** — not
   provably optimal, but the strongest result found after seven
   independent attempts to beat it, several of which are now understood
   well enough to explain why they didn't help.

   Paused here (sixteen experiments deep) at the end of 2026-09-01/02's
   speed-optimization arc. **Superseded 2026-09-02** — see causal test 17
   below, which removes `MAX_REWARDED_SPEED_MPS` structurally rather than
   continuing to search its value.

   **Causal test 17 (run, 2026-09-02) — structural change, not another
   value on the same axis:** directed to increase throttle/speed
   specifically (the checkpoint above was judged "too safe"), and to
   explore drift-style cornering. A hand-coded drift controller isn't
   compatible with the self-play/SAC architecture (rule-based control
   fighting the learned policy, out of scope) or verifiable without
   testing whether the physics model even rewards it — instead, removed
   `MAX_REWARDED_SPEED_MPS` entirely (`forward_progress_m` is now
   uncapped) and added `WALL_PROXIMITY_SPEED_SCALE_MPS = 10.0`: the
   wall-proximity penalty now scales with current speed (2x at 10 m/s, 3x
   at 20 m/s, ...) instead of being speed-blind. Same intent as every
   prior wall-proximity change, sharpened: price risk by how dangerous
   the *current situation* is, not by a flat speed ceiling that treats
   "near a wall at 1 m/s" the same as "near a wall at 35 m/s." Same
   seed/races=40/round-length as every comparison this week.

   **Result — genuinely different from all seven prior speed attempts**:
   avg max speed 15.53 → **17.13 m/s (+10%)**, with safety essentially
   preserved (0/20 eliminated both before and after; damage/off-track/
   wall-contact still near-zero, not literally 0.000 anymore but close).
   This is the first speed-focused change all day to increase speed
   *without* regressing safety — every earlier attempt did one or the
   other, never both. The catch: avg laps 4.10 → 3.90 and avg best lap
   time 24.76s → 28.46s (slower) — the extra top speed didn't translate
   into a better overall race, for reasons not yet diagnosed (can't tell
   from headless stats alone whether it's cornering differently or just
   faster on straights without converting that into pace). See
   `experiments/2026-09-02_uncapped-speed-scaled-risk-seed110/notes.md`.

   **Why this one differs from the seven that failed:** those all changed
   a *value* on an axis with a real ceiling (`MAX_REWARDED_SPEED_MPS`) —
   more training, resuming, or raising the number all converged back
   toward (or collapsed around) that same ceiling. This change removes
   the ceiling's existence, not its value, so "more training converges
   speed down" (causal tests 11, 16) has no obvious reason to apply here
   — worth testing directly rather than assuming it still holds.

   Provisionally adopting this as the new reference point, flagged
   explicitly as an unresolved trade-off (speed up, lap time down) rather
   than an unambiguous win. **Current checkpoint:
   `2026-09-02_uncapped-speed-scaled-risk-seed110`** — 0/20 eliminations,
   ~17.1 m/s avg max speed (up from ~15.5), 3.90 avg laps, 20/20 race
   wins. (`controllers.sac_candidate` auto-selects the newest checkpoint
   by file time — this one, as of this entry.)

   **Causal test 18 (run, 2026-09-03) — loosening `WEIGHT_CENTER_OFFSET`,
   a clear regression, reverted:** directed to plan a full restart-and-
   retrain aimed at speed and safety, exploring drifting. Reviewed the
   full causal-test history above and recommended *not* a true
   from-scratch restart — instead, keep seed `110` (the only seed that
   avoided the `909`-style dead-end local optimum) and the current
   uncapped-speed/speed-scaled-wall-risk reward (the only prior attempt
   that gained speed without losing safety), and test the one lever
   flagged but not yet tried: `WEIGHT_CENTER_OFFSET` 0.3 → 0.15 (a
   deliberate half-step, not a full return to the old 0.05 that caused
   the 2026-09-01 off-track regression this weight originally fixed).
   Same seed/races=40/round-length as every comparison this week, trained
   from scratch.

   **Process note:** the first evaluation pass on this run accidentally
   used `eval_sac`'s 20s default round length instead of the reference's
   120s (a missing `--eval-round-seconds 120` flag on the training
   invocation) — caught by diffing the saved `config.yaml` against the
   reference before reporting any numbers, and fixed by re-evaluating the
   same saved checkpoint via `scripts/eval_sac.py --eval-round-seconds
   120` (no retraining needed, since training itself used the correct
   120s round length throughout). The mistaken 20s files are kept as
   `config_WRONG_20s.yaml` / `eval_results_WRONG_20s.json` in the
   experiment directory, unused in any comparison.

   **Result — a clear regression, not the hoped-for trade-off:** avg max
   speed **17.13 → 8.80 m/s (-49%)**, avg laps **3.90 → 1.00 (-74%)**, avg
   best lap time **28.46s → 78.80s (+177%)**, wins vs.
   `default_student_controller` **10/10 → 3/10**. Safety stayed
   essentially flat (damage 0.003 → 0.006, still ~0 off-track/wall-
   contact, 0/20 eliminated both before and after) — so this wasn't a
   safety-for-speed trade-off, it was a straightforward loss of
   competence with safety incidentally preserved. See
   `experiments/2026-09-03_center-offset-half-seed110/notes.md`.

   **Read:** the hypothesis that a uniform centerline penalty was
   suppressing a faster racing line was wrong, or at least wrong at this
   step size — halving the weight didn't unlock deliberate wider cornering,
   it weakened the signal keeping cornering deliberate at all, in a reward
   that (post-cap-removal) already rewards raw speed off the racing line
   too. Same failure family as the steering-smoothness and idle-penalty
   attempts: a plausible-sounding loosened constraint removing a signal
   the policy actually depended on. Reverted `WEIGHT_CENTER_OFFSET` to
   `0.3` in `src/training/reward.py` (comment records the attempt and
   result, per the file's convention for reverted experiments). **Current
   best checkpoint is unchanged: `2026-09-02_uncapped-speed-scaled-risk-
   seed110`.** If wider/drift-style cornering is revisited, the next idea
   should target corners specifically (e.g. conditioned on
   `camera.lookahead_offsets_m` curvature or wall-proximity margin) rather
   than a global weight change that also touches straights and safe
   cruising.

   **Causal test 19 (run, 2026-09-07) — a different mechanism (n-step
   returns), and the first change to improve safety and lap time at
   once:** all eighteen prior causal tests changed the reward function or
   trained longer; this instead changed the SAC algorithm itself.
   Implemented n-step returns (§4) so the critic target sums `n` real
   ticks of reward before bootstrapping instead of just one — motivated
   by the observation, running through causal tests 5-8 in particular,
   that a delayed penalty (`WEIGHT_TERMINAL_PENALTY`) only reaches the
   states responsible for a crash after many sequential 1-step Bellman
   backups, which is why raising that penalty's *magnitude* kept being
   the only lever that moved anything. Tested `n=3` against the current
   reference (`2026-09-02_uncapped-speed-scaled-risk-seed110`), same
   seed/races=40/round-length/reward, trained from scratch.

   **Result:** avg damage/off-track/wall-contact all dropped to exact
   zero (matching the best safety result seen anywhere in this project),
   while avg best lap time *improved* 28.46s → 25.06s (-12%) and avg laps
   3.90 → 4.00 — despite avg max speed *dropping* 17.13 → 14.73 m/s
   (-14%). Every prior speed-focused experiment (causal tests 9-18) was a
   trade-off in one direction or another; this is the first to improve
   safety and lap time simultaneously, at a lower top speed, on the same
   seed and reward. See
   `experiments/2026-09-07_nstep3-seed110/notes.md`.

   **Read:** consistent with the n-step motivation — a lower top speed
   but faster lap time suggests less time lost to hesitation, near-miss
   recovery, or an indirect racing line, rather than a pure straight-line
   speed change. Not confirmed mechanistically (would need a live watch
   or per-tick sensor logging to see whether cornering technique actually
   changed). **Adopted `2026-09-07_nstep3-seed110` as the new best/
   reference checkpoint** — it dominates the uncapped-speed reference on
   every tracked metric except top speed, where the net effect (lap time)
   is still better.

   **Causal test 20 (run, 2026-09-07) — n=5 regression, reverted:** tested
   the next point on the n-step axis (n=3 → n=5), same seed/races/round-
   length/reward as causal test 19. **Clear regression, not further
   improvement**: avg low-progress time (this project's headless proxy for
   hesitation) rose 3.18s → 5.07s — worse than even the 1-step reference's
   2.89s — avg laps dropped 4.00 → 1.50, avg best lap time nearly doubled
   (25.06s → 59.37s), avg scored distance collapsed (869.8m → 363.5m).
   Safety stayed at floor (0/20 eliminated, zero damage/off-track/wall-
   contact both before and after) and training budget was essentially
   identical (571,812 vs. 570,114 transitions), so this isn't a training-
   time confound — it matches the off-policy-staleness/variance tradeoff
   discussed before running it: a longer real-reward window increasingly
   reflects a policy several updates out of date. **Not adopted; n=3
   remains the reference.** See
   `experiments/2026-09-07_nstep5-seed110/`.

   **Root-cause diagnosis (2026-09-07) — "hesitation" is mostly blind
   car-collision, not steering indecision:** before testing another
   reward-side hesitation fix, wrote a one-off diagnostic
   (`sensor_sample_callback` logging `contact.robot`/speed/position per
   tick across a real evaluation race) rather than guessing. Found
   `wall_contact` was exactly 0.00s across every evaluated race, while
   `car_contact` repeated in windows spaced ~180-190m apart — matching the
   track's ~183m lap length — meaning the policy collides with the
   (stationary, in the `crash_fast` case) opponent car at roughly the same
   point on the track once per lap, every lap. Root cause: `training.
   observation`'s 17-dim vector had zero information about other cars —
   `camera.competitors`/`sensors.lidar` were both deferred by the original
   §2.1 design pending solo-driving competence, which has been solid since
   2026-09-01 and was never revisited. A previously-implemented, untested
   hesitation fix (`WEIGHT_STEERING_REVERSAL`, targeting steering-direction
   flips) was disabled (weight 0.0, mechanism kept — see
   `src/training/reward.py`) rather than tested alongside this, to avoid
   confounding the real fix with an unconfirmed one.

   **Causal test 21 (run, 2026-09-07) — obstacle LiDAR added to the
   observation, net improvement with one flagged outlier:** added
   `sensors.lidar` (7 beams, same encoding as `wall_lidar`) to the
   observation vector (§2.1), `OBSERVATION_DIM` 17 → 24. Same seed/races/
   round-length/reward/n_step=3 as causal test 19 — the observation change
   is the only new variable.

   | | n-step=3 (no obstacle lidar) | + obstacle lidar |
   | --- | --- | --- |
   | avg car-contact | 1.595s | **0.952s (-40%)** |
   | avg marshal/race | 0.25 | **0.10** |
   | avg max speed | 14.73 m/s | 23.24 m/s |
   | avg best lap time | 25.06s | 24.79s |
   | avg low-progress (all 20 races) | 3.18s | 3.61s (worse, headline) |
   | avg low-progress (19/20, outlier excluded) | 3.18s | **2.63s (better)** |
   | avg damage | 0.0000 | 0.0027 |
   | eliminated | 0/20 | 0/20 |

   The one outlier (`seed=2024 vs default_student_controller race=2`) had
   low-progress=22.17s vs. 1.65-4.40s everywhere else in the run, plus the
   only nonzero wall-contact (0.18s) and real damage (0.054) in the whole
   run — diagnosed directly (same per-tick logging approach) rather than
   discarded as noise: ~54 of 120s spent below 1.2 m/s, with a contact
   window around tick 6007-6221 showing *negative* speed (reversing,
   -3.33 to -0.04 m/s) while off-center (-0.5 to -0.67m) — consistent with
   the car getting physically wedged against the opponent near the track
   edge rather than a brief bump-and-recover. One marshal recovery fired,
   but only after most of the time was already lost.

   **Read:** the fix works as intended on 19/20 races — car-contact time
   dropped substantially and consistently, marshal interventions dropped,
   and lap time held or improved, all while top speed rose sharply (likely
   because the policy no longer needs to drive as cautiously around a risk
   it couldn't previously perceive at all). But it introduces (or exposes)
   a rarer, more severe failure mode: getting stuck against a moving
   opponent rather than a stationary wall is harder to recover from, and
   a single marshal reset doesn't reliably resolve it quickly. This is a
   genuine, not-yet-solved trade-off, not a clean win — reporting it
   honestly rather than only the flattering 19/20 subset.

   **Decision and rationale:** Adopting
   `2026-09-07_obstacle-lidar-nstep3-seed110` as the new best/reference
   checkpoint — the aggregate improvement (contact, marshal, lap time) is
   real and consistent across seeds, and the outlier is high-cost but rare
   (1/20 races) and non-fatal (no elimination). Flagging the stuck-against-
   opponent failure mode as an open item rather than a solved one.

   **Next steps:**
   1. Investigate whether the stuck-against-opponent case is specific to
      one spawn/track geometry or a general risk — a small seed sweep at
      this checkpoint (not yet done for any checkpoint on this track)
      would help distinguish the two.
   2. Consider a reward-side complement now that the observation can
      support it: a proximity-based penalty for closing distance on a
      competitor (mirroring `WEIGHT_WALL_PROXIMITY`'s speed-scaled
      approach), so the policy is taught to avoid getting close in the
      first place, not just given the sensory input to react once close.
   3. Re-test the disabled `WEIGHT_STEERING_REVERSAL` mechanism on its own,
      now that the dominant hesitation cause (blind collision) has a
      different fix — it may still matter for any residual steering
      wobble, now measurable in isolation from the collision effect.
   4. Still open, unchanged from every prior entry: a genuine multi-
      training-seed sweep, and repackaging `controllers.race_faster`
      (still ships the races=20 checkpoint from 2026-09-01, now many
      generations behind).

   **Causal test 22 (run, 2026-09-07/09-08) — competitor-proximity reward,
   a regression that made the target problem worse:** implemented next
   step 2 above: `WEIGHT_ROBOT_PROXIMITY`, the direct structural analog of
   `WEIGHT_WALL_PROXIMITY` but for the nearest `camera.competitors`
   reading, no angle restriction, same speed-scaled multiplier as wall
   proximity. Same seed/races/round-length/n_step/observation as causal
   test 21 — the reward term is the only new variable.

   **Result:** avg car-contact improved only slightly (0.952s → 0.825s)
   while avg laps dropped 4.00 → 2.90, avg best lap time rose 24.79s →
   35.31s (+42%), avg marshal/race rose 5.5x (0.10 → 0.55), and max speed
   converged to a suspiciously uniform ~17.3-17.7 m/s across nearly every
   race (previously 14.7-23+ m/s, situation-dependent) — a broad, uniform
   slowdown, not a targeted fix. Worse: the specific problem this was
   built to solve got *worse*, not better — `seed=110 vs crash_fast`
   (both races, reproducibly) rose to 6.3-6.8s of car-contact with 2
   marshal recoveries and 2 laps completed, exceeding the prior worst
   case (2.68s) this term was meant to prevent. See
   `experiments/2026-09-07_robot-proximity-obstacle-lidar-nstep3-seed110/notes.md`.

   **Read:** with no angle restriction, the penalty fires for any nearby
   competitor regardless of whether it's actually in the way, teaching
   generalized caution around any competitor rather than specifically
   avoiding collisions — for a stationary mid-track blocker, that
   generalized caution plausibly makes committing to a clean pass harder,
   not easier. Same failure family as other plausible-sounding caution
   terms on this track that backfired into broad overcaution rather than
   a targeted fix (`WEIGHT_STEERING_SMOOTHNESS`, the idle-penalty
   overcorrection, the `WEIGHT_CENTER_OFFSET` halving attempt).

   **Decision and rationale:** Not adopted. Disabled
   `WEIGHT_ROBOT_PROXIMITY` (weight 0.0, mechanism kept — same convention
   as the file's other two reverted terms).
   `2026-09-07_obstacle-lidar-nstep3-seed110` remains the reference
   checkpoint.

   **Next steps:**
   1. An angle-restricted variant (mirroring
      `WALL_WARNING_BEAM_ANGLES_DEGREES`'s front-only beams) is untested
      and may avoid this failure mode.
   2. A closing-speed-scaled variant (`closing_speed_mps` from
      `CameraCompetitorReading` instead of own absolute speed) is also
      untested — would only penalize proximity while actually gaining on
      the competitor.
   3. Two consecutive hesitation-specific reward fixes (steering-reversal,
      then this) have not been the answer — consider treating the
      obstacle-lidar observation change alone (real improvement on 19/20
      races) as sufficient for now, and revisit competitor-avoidance
      reward shaping later with more diagnostic detail on the
      seed=110/crash_fast case specifically.
   4. Still open: a genuine multi-training-seed sweep, and repackaging
      `controllers.race_faster`.

   **Causal test 23 (run, 2026-09-08) — angle-restricted variant, a
   collapse, not a fix:** implemented next step 1 above:
   `ROBOT_WARNING_ANGLE_DEGREES = 45.0`, restricting the proximity penalty
   to competitors within 45° of straight ahead (mirroring
   `WALL_WARNING_BEAM_ANGLES_DEGREES`'s front-only beams). Same seed/
   races/round-length/n_step/observation as causal test 22.

   **Result: much worse, not better.** 17 of 20 evaluation races
   completed exactly zero laps, uniformly across every seed and both
   baselines (not one outlier) — avg low-progress time exploded to
   46.2s/race (38.5% of the round; one race hit 119.58s, essentially the
   entire round), avg marshal/race rose to 6.35, avg max speed dropped to
   8.30 m/s. Car-contact time was actually the lowest of any
   configuration tested (0.662s) — the car genuinely avoided the
   opponent, but only by nearly never moving. See
   `experiments/2026-09-08_robot-proximity-angle-restricted-seed110/notes.md`.

   **Read:** avoiding a wall requires active steering — there's no way to
   "just not approach" a wall on a fixed track — so
   `WEIGHT_WALL_PROXIMITY` teaches real avoidance. Avoiding an *ahead*
   competitor is trivially satisfiable by never closing distance at all;
   crawling at 6-10 m/s clears `WEIGHT_IDLE`'s 0.5 m/s threshold while
   permanently avoiding the front-cone penalty, with no need to ever
   commit to a pass. The unrestricted version (causal test 22: broad,
   situation-blind caution, still moving) and this version (a clean
   escape hatch: just never approach) are different failure modes, not
   two points on a spectrum from bad to good.

   **Decision and rationale:** Not adopted. Disabled
   `WEIGHT_ROBOT_PROXIMITY` again (weight 0.0; both the base mechanism
   and the angle restriction kept in code and unit-tested directly, since
   the angle-filtering logic itself is verified correct — it's the
   policy's response to the resulting incentive that failed, not a
   reward-code bug). Not tuning this mechanism further without a
   different underlying idea, given two consecutive attempts have each
   failed in a distinct way. **`2026-09-07_obstacle-lidar-nstep3-seed110`
   (obstacle lidar in the observation, no robot-proximity term) is the
   config locked in going forward.**

   **Next steps:**
   1. If revisited later: a closing-speed-scaled variant
      (`closing_speed_mps` instead of own speed) is untested, though a
      similar "never gain on them" escape hatch could plausibly still
      emerge.
   2. Proceed to the multi-training-seed sweep (still the standing gap
      across this entire track — every result so far comes from a single
      training seed, 110) using the locked-in config above.

   **Causal test 24 (run, 2026-09-08) — the first genuine multi-training-
   seed sweep, and the best news on this track to date:** every result
   since 2026-08-31 came from training seed 110. The one prior seed
   comparison (110 vs. 909, 2026-09-01, under a much cruder reward/
   observation) found a catastrophic qualitative difference — seed 909
   converged to a "do nothing" freeze. That finding was never rechecked
   under the current stack (idle penalty, terminal penalty, n-step
   returns, obstacle-lidar observation). Trained 4 new seeds — 909 (the
   historically unstable one) plus 3 fresh seeds never used on this track
   (1000, 2000, 3000), none overlapping the fixed evaluation seed set —
   with the locked-in config from causal test 23 (n_step=3, obstacle
   lidar, no robot-proximity term), identical races/round-length/buffer
   to every seed-110 comparison this week.

   | seed | avg damage | off-track | wall-contact | avg laps | avg lap time | avg max speed | eliminated | wins |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | 110 (original reference) | 0.0027 | 0.03s | 0.01s | 4.00 | 24.79s | 23.2 m/s | 0/20 | 20/20 |
   | 909 (2026-09-01's unstable seed) | 0.0000 | 0.00s | 0.00s | 5.00 | 22.50s | 17.4 m/s | 0/20 | 20/20 |
   | 1000 (fresh) | 0.0013 | 0.02s | 0.00s | **6.50** | **17.19s** | 31.0 m/s | 0/20 | 20/20 |
   | 2000 (fresh) | 0.0000 | 0.00s | 0.00s | 2.95 | 34.00s | 16.3 m/s | 0/20 | 20/20 |
   | 3000 (fresh) | 0.0127 | 0.22s | 0.15s | 6.20 | 17.26s | 33.7 m/s | 0/20 | 20/20 |

   **Result:** every one of the 5 seeds produces a safe, competent policy
   — 0/20 eliminated and 20/20 wins against both baselines across the
   board, with damage/off-track/wall-contact small-to-zero everywhere.
   **The 2026-09-01 seed-909 catastrophic freeze does not recur** — under
   the current reward/observation stack, seed 909 is now one of the
   better performers (5.00 avg laps, 22.50s avg lap time), not a broken
   outlier. This is strong evidence that the accumulated fixes since then
   (idle penalty, terminal penalty at the right magnitude, obstacle-lidar
   observation) actually resolved the underlying instability rather than
   papering over it for one lucky seed.

   There is, however, real and worth-reporting variance in *pace*: lap
   time ranges 17.19s–34.00s (~2x) and laps completed ranges 2.95–6.50
   across the 5 seeds. Seed 1000 is the standout — fastest lap time
   (tied with 3000) and most laps, with the best safety margin of the
   three fast/mid seeds. Seed 2000 is the weakest — safe, but meaningfully
   slower and less complete than every other seed. **Seed 110 — the only
   seed used for every prior checkpoint decision on this track — turns
   out to be middle-of-the-pack, not representative of the best available
   outcome.**

   **Decision and rationale:** Treating seed-level robustness (does
   training produce a *safe* policy) as resolved for the current reward/
   observation config — 5/5 seeds safe is real evidence, not proof for
   all possible seeds, but a large improvement over the single-seed
   evidence this track has relied on throughout. Treating seed-level
   *pace* as a separate, still-open question — since it varies
   meaningfully, the choice of which checkpoint to submit or present
   should not default to "whichever seed happened to be used first."
   **Recommending `2026-09-08_seed-sweep-1000` as the new best checkpoint**
   for speed-sensitive purposes (e.g. repackaging `controllers.race_faster`,
   still open) given its combination of fastest pace and strong safety;
   `2026-09-07_obstacle-lidar-nstep3-seed110` and the other sweep seeds
   remain valid, safe alternatives if reproducing exact prior comparisons
   matters more than pace.

   **Next steps:**
   1. Repackage `controllers.race_faster` from `seed-sweep-1000` — this
      is now clearly motivated, not just "the current best exists,
      update the export." (Done same day — see item 8 below.)
   2. A larger sweep (10+ seeds) would tighten the pace-variance estimate,
      but 5/5 safe is already a meaningful improvement in confidence over
      the single-seed evidence used throughout this track; not urgent.
   3. Investigate *why* pace varies so much across seeds (2000 vs.
      1000/3000) — e.g. whether it's initialization variance in the
      actor network, or genuinely different local optima in driving
      style — a live watch of seed 2000 vs. seed 1000 could show whether
      the difference is visible qualitatively (hesitant cornering vs.
      committed cornering) or not.
   4. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 25 (run, 2026-09-08) — a shared training-dynamics
   attractor, then a foundational bug that reframes it:** per next step 3
   above, ran a per-tick diagnostic comparing seed 1000 (fast) and seed
   2000 (slow) head-to-head, aligned by track position
   (`distance_m mod ~183m`). Finding: seed 2000 wasn't hesitating at
   specific corners — it was uniformly slower across nearly every point
   on the track (never exceeded 20 m/s at all, vs. seed 1000's 7.5% of
   ticks above 20 m/s), consistent with a track-wide risk-tolerance
   difference rather than a localized issue.

   Tested whether more training resolves it via `--resume-from`: resuming
   seed 2000 (+40, then +80 more races) improved it substantially (34.00s
   → 26.71s → 26.10s avg lap time). But resuming seed 1000 — already the
   *fastest* checkpoint — for 40 more races made it **worse**, not better
   (17.19s → 25.08s, max speed 31.02 → 18.65 m/s). Both seeds converged
   toward the same ~25-26s/~19-23 m/s regime regardless of which side
   they started on, suggesting a shared training-dynamics attractor and
   that seed 1000's original 17.19s result was a fast transient, not a
   stable achievement.

   Tested the most directly implicated lever: raised
   `WALL_PROXIMITY_SPEED_SCALE_MPS` 10.0 → 15.0 (chosen since it was set
   "to match the old `MAX_REWARDED_SPEED_MPS` cruising target," making it
   the most plausible determinant of where the attractor sits), trained
   fresh on seed 1000 (races=40, matching the reference config exactly).
   **Result: worse, not better** — 30.31s avg lap time, 20.39 m/s avg max
   speed (both worse than the scale=10.0 reference), landing in the same
   ~25-30s range rather than a faster one. Reverted to 10.0.

   **Then a foundational bug was found that reframes all of the above:**
   `scripts/train_sac.py`'s `train()` never passed `seed=args.seed` to
   `SACAgent(...)`, which defaults to `seed=0`. **Every training run on
   this track so far — every "training seed" experiment, including the
   full 5-seed sweep (causal test 24) and every resume/reward test
   above — started from bit-for-bit identical network initialization.**
   `--seed` only ever varied self-play spawn positions and the
   replay-buffer/warmup sampling order, never the actor/critic network
   weights. Verified directly: `SACAgent(..., seed=1000)` and
   `SACAgent(..., seed=2000)` produced identical initial policy weights
   before the fix, different (and individually reproducible) weights
   after. Fixed by passing `seed=args.seed` through.

   **Read:** the apparent shared attractor may be partly or entirely an
   artifact of every run starting from the same point in weight-space —
   since all runs begin identically and are only perturbed by spawn/
   sampling order, it's unsurprising they'd tend to traverse a similar
   trajectory shape (an early higher-variance phase settling into a more
   conservative one) even without a "true" attractor in the reward
   landscape itself. This doesn't invalidate the raw observations (the
   lap-time numbers are real), but it means the *seed-sensitivity* and
   *shared-equilibrium* framing from causal test 24 and this entry
   describe a narrower phenomenon (sensitivity to training trajectory
   from one fixed starting point) than originally stated (network-
   initialization sensitivity). A genuine network-initialization sweep
   has not yet been run on this track.

   **Decision and rationale:** Fix adopted
   (`scripts/train_sac.py`), reward change not adopted
   (`WALL_PROXIMITY_SPEED_SCALE_MPS` reverted to 10.0 — a null/negative
   result, now additionally confounded by the bug above). Not re-running
   the 5-seed sweep or the reward-magnitude test automatically — flagging
   this for direction, since it means re-spending the same order of
   compute (5+ training runs) to get a *more meaningful* seed sweep than
   the one already spent on causal test 24.

   **Next steps:**
   1. Re-run the multi-seed sweep with the fix in place, to separate
      genuine network-initialization sensitivity from trajectory
      sensitivity from a fixed start — this is now the more scientifically
      meaningful version of causal test 24's question.
   2. Re-test `WALL_PROXIMITY_SPEED_SCALE_MPS` (and any other
      seed-dependent conclusion drawn before this fix) once genuine
      initialization diversity is available, since the null result above
      may not hold under real diversity.
   3. `2026-09-08_seed-sweep-1000` (packaged in `race_faster.py`) remains
      the best checkpoint found so far by direct evaluation, regardless
      of this bug — the fix changes what future training runs will
      explore, not what this specific checkpoint already does.

   **Causal test 26 (run, 2026-09-08) — the seed sweep re-run with
   genuine network-initialization diversity:** re-ran the identical
   5-seed sweep (110, 909, 1000, 2000, 3000; same n_step=3/obstacle-
   lidar/no-robot-proximity config, races=40, round_seconds=120,
   buffer_capacity=800000) with the `seed=args.seed` fix in place.

   | seed | v1 (shared init=0, buggy) lap time | v2 (genuine init) lap time | v2 max speed | v2 laps | v2 damage |
   | --- | --- | --- | --- | --- | --- |
   | 110 | 24.79s | **17.96s** | 30.6 m/s | 6.05 | 0.0045 |
   | 909 | 22.50s | 26.60s | 11.0 m/s | 4.00 | 0.0000 |
   | 1000 | 17.19s (was fastest) | **48.86s (now slowest)** | 8.9 m/s | 2.00 | 0.0000 |
   | 2000 | 34.00s | 41.37s | 9.9 m/s | 2.10 | 0.0000 |
   | 3000 | 17.26s | 22.73s | 19.8 m/s | 4.95 | 0.0000 |

   **Result:** the seed-to-outcome mapping is not preserved across the
   fix — seed 1000, the winner of the buggy sweep and the checkpoint
   currently packaged in `race_faster.py`, is now the *worst* performer
   of the five when actually given its own real initialization. Seed
   110, previously middling, is now the best. The pace range widened
   (17.96s–48.86s, ~2.7x) relative to the buggy sweep's already-wide
   17.19s–34.00s (~2x) — genuine initialization diversity carries *more*
   variance than the trajectory-only variance measured before the fix,
   not less. Safety held up across all five regardless: 0/20 eliminated,
   near-zero damage/off-track/wall-contact everywhere — the "no
   catastrophic freeze" finding from causal test 24 is reconfirmed, now
   on a methodologically sound footing rather than a confounded one.

   **Read:** this settles the ambiguity from causal test 25 the direct
   way rather than by inference — genuine network-initialization
   sensitivity is real, present, and larger in magnitude than the
   spawn/sampling-order sensitivity alone. The "seed lottery" framing
   from causal test 24 was directionally right (pace varies a lot by
   what training happens to hit) but was drawing that conclusion from a
   narrower and less representative source of variance than actually
   exists.

   **Decision and rationale:** `2026-09-08_seed-sweep-1000`
   (`17.19s`, `6.50` laps, `0.0013` damage) remains the best directly-
   evaluated checkpoint and stays packaged in `race_faster.py` — the new
   sweep's best result (v2 seed 110: `17.96s`, `6.05` laps, `0.0045`
   damage) is close but does not beat it on either speed or safety. That
   checkpoint's validity was never in question; only whether asking for
   "seed 1000" again would reproduce it, which it no longer does. Given
   the now-confirmed wide variance, best-of-N seed sampling is itself a
   legitimate, cheap (~7-8 min/seed) strategy for finding a better
   checkpoint, distinct from tuning the reward.

   **Next steps:**
   1. Sample more genuine-init seeds if further pace improvement is
      wanted — this sweep suggests the tail of the distribution (a seed
      meaningfully better than 17.19s) hasn't necessarily been found yet
      with only 5 draws under real diversity.
   2. Re-test `WALL_PROXIMITY_SPEED_SCALE_MPS` under genuine init
      diversity (multiple seeds, not one) before drawing any conclusion
      about that constant specifically — the earlier null result
      (causal test 25) was a single seed under the old bug.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 27 (run, 2026-09-08) — best-of-N seed sampling finds a
   new pace record, with one flagged near-miss:** sampled 5 more fresh
   genuine-init seeds (4000, 5000, 6000, 7000, 8000), same locked-in
   config as causal test 26. Four (4000/5000/6000/7000) landed in the
   already-seen 26-30s range; seed 8000 broke the record:

   | | previous record (v1 seed 1000) | seed 8000 |
   | --- | --- | --- |
   | avg best lap time | 17.19s | **14.96s** |
   | avg laps | 6.50 | **7.30** |
   | avg scored distance | 1267.2m | **1444.6m** |
   | avg max speed | 31.0 m/s | 26.7 m/s |
   | avg damage | 0.0013 | 0.0298 |
   | avg off-track | 0.02s | 0.15s |
   | avg wall-contact | 0.00s | 0.11s |
   | eliminated | 0/20 | 0/20 |

   The aggregate damage/off-track numbers look worse than the record, but
   per-race detail shows this is one outlier, not a general regression:
   18 of 20 races are exceptionally clean (0.0000 damage, 7-8 laps every
   time — faster *and* more consistent than the record on races where
   nothing goes wrong). One race
   (`seed=8675309 vs default_student_controller race=1`) took 0.5942
   damage — a serious near-crash, close to but short of the
   `NEAR_ELIMINATION_DAMAGE = 0.9` terminal threshold.

   Diagnosed the outlier directly (per-tick logging, same method used for
   every prior flagged outlier this session): over ~0.5s the car
   accelerated hard (8.8 → 14.1 m/s) while drifting off-center (-0.20m →
   -1.96m) as wall clearance shrank (3.90m → 1.30m) — committing to a
   fast line through what looks like a tightening corner, then taking one
   hard wall impact (damage 0.0 → 0.548 in a single tick) rather than a
   repeated or systemic pattern. `contact.robot` was zero throughout,
   ruling out the opponent-collision failure mode from earlier sessions.
   See `experiments/2026-09-08_seed-sweep-v2-8000/notes.md`.

   **Read:** a genuine, understood speed-vs-cornering-margin trade-off,
   not a bug — a faster policy occasionally commits to a corner entry it
   can't quite hold, in 1 of 20 evaluated races, never resulting in
   elimination or a lost race. Sharper trade-off than prior "adopt
   despite an outlier" decisions this session (a near-crash is more
   severe than a stuck-and-slow race), so flagged for direction rather
   than adopted automatically.

   **Decision and rationale:** Not yet adopted — awaiting direction,
   since this genuinely trades a small, bounded, understood risk for a
   substantial (13%) pace gain, unlike every earlier adoption this
   session where the improvement was closer to unambiguous.
   `2026-09-08_seed-sweep-1000` remains packaged in `race_faster.py`
   pending that decision.

   **Next steps:**
   1. If adopted: repackage `controllers.race_faster` from
      `seed-sweep-v2-8000`. (Done — see item 8 below.)
   2. Continue sampling more genuine-init seeds if a cleaner-margin
      checkpoint (record pace without the near-miss) is preferred over
      accepting this trade-off.
   3. A corner-aware wall-proximity term (scaling with
      `camera.lookahead_offsets_m` curvature, giving reaction time before
      entering a corner rather than only reacting to the current tick's
      distance) is a candidate fix for this specific failure mode if it
      recurs across future fast checkpoints — not yet tested.

   **Causal test 28 (run, 2026-09-08) — recalibrating wall-warning
   distance for the new speed regime fixes the near-miss, but at a steep
   pace cost:** adopted per direction; then, per the follow-up direction
   to "improve safety from there," traced the near-miss to a stale
   calibration rather than a new mechanism: `WALL_WARNING_DISTANCE_M =
   6.0` was explicitly tuned (2026-09-01) for "~0.6s at 10 m/s" reaction
   time — this checkpoint cruises at ~27 m/s, where 6.0m gives only
   ~0.22s. Doubled to 12.0 (matching the relative size of the 3.0→6.0
   step that worked originally) and trained fresh on seed 8000 — the
   exact seed that produced the near-miss.

   **Result:** the near-miss was essentially eliminated (avg damage
   0.0298 → 0.0000, off-track/wall-contact both near zero), but avg laps
   dropped 7.30 → 3.35, avg best lap time rose 14.96s → 31.46s, avg max
   speed dropped 26.65 → 18.74 m/s — the same "safety fix overcorrects
   into a much slower policy" pattern seen twice already this session
   with `WEIGHT_ROBOT_PROXIMITY` (causal tests 22-23). Reverted to 6.0.
   See `experiments/2026-09-08_wallwarn12-seed8000/notes.md`.

   **Read:** three consecutive safety-motivated reward changes this
   session (both `WEIGHT_ROBOT_PROXIMITY` attempts, now this one) have
   each produced an all-or-nothing trade — a clear pace regression or a
   near-total collapse — rather than a middle ground trading a little
   pace for a lot of safety. Training appears to converge to
   qualitatively different equilibria (aggressive-and-risky vs.
   cautious-and-slow) rather than smoothly interpolating as reward
   constants are tuned, at least along every axis tried so far.

   **Decision and rationale:** Not adopted; `WALL_WARNING_DISTANCE_M`
   reverted to 6.0. `2026-09-08_seed-sweep-v2-8000` remains packaged,
   near-miss and all. Given the pattern across three attempts, further
   reward-magnitude search for a middle ground on this axis is
   deprioritized in favor of two qualitatively different levers.

   **Next steps:**
   1. Keep sampling genuine-init seeds (the strategy that already found
      `seed-sweep-v2-8000`) looking specifically for one matching or
      beating its pace *without* a near-miss in its own evaluation.
   2. A controller-level (not reward-level) hard safety backstop —
      overriding throttle when a forward wall reading is both very close
      and speed is high, regardless of the learned policy's output — was
      proposed early in this track's refinement plan (§6 item 7's
      predecessor discussions) and remains untested. It wouldn't have the
      "policy learns to avoid the situation entirely" side effect reward
      shaping keeps producing, since it changes what happens at inference
      time, not what's being optimized during training.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 29 (run, 2026-09-08) — a second seed batch finds a
   near-record, dramatically safer alternative:** sampled 5 more fresh
   genuine-init seeds (9000, 10000, 11000, 12000, 13000), same locked-in
   config as causal test 27 (with `WALL_WARNING_DISTANCE_M` reverted to
   6.0 per causal test 28). None beat seed 8000's 14.96s lap time
   outright — 9000/12000/13000 were perfectly clean (0.0000 damage) but
   slower (20.99-29.71s); 11000 was mostly clean (max damage 0.0447) and
   still slower (24.86s). Seed 10000 came closest to the pace target with
   a dramatically better safety margin:

   | | champion (seed 8000) | seed 10000 |
   | --- | --- | --- |
   | avg best lap time | 14.96s | 16.70s (+12%) |
   | avg laps | 7.30 | 6.80 |
   | avg max speed | 26.7 m/s | 35.4 m/s (higher) |
   | avg damage | 0.0298 | 0.0008 |
   | max damage (worst race) | 0.5942 | **0.0154 (38x smaller)** |

   18 of seed 10000's 20 races have exactly 0.0000 damage, 6-7 laps
   every time, and a remarkably consistent ~35-36 m/s max speed across
   literally every race — tighter variance than the champion's. The only
   two blips are trivial: a 2.10s off-track excursion with zero
   damage/wall-contact, and one 0.0154-damage graze. Notably, seed
   10000's raw top speed (35+ m/s) exceeds the champion's (26.7 m/s) —
   consistent with the long-standing finding that top speed and lap pace
   are not the same thing; it likely corners more conservatively despite
   faster straights. See `experiments/2026-09-08_seed-sweep-v2-10000/notes.md`.

   **Decision and rationale:** Not automatically adopted — presenting as
   a genuine alternative for direction, the same way seed 8000 itself was
   presented rather than swapped in unilaterally. The trade (12% slower
   lap time for a ~38x smaller worst-case damage event) is a judgment
   call, not an unambiguous improvement in either direction.

   **Next steps:**
   1. Awaiting direction: adopt seed 10000 in place of seed 8000, keep
      seed 8000, or continue sampling for something that beats 14.96s
      outright without a near-miss.
   2. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 30 (run, 2026-09-08) — a third seed batch finds nothing
   better; yield is dropping:** sampled 5 more fresh genuine-init seeds
   (14000-18000), same locked-in config. All 5 were perfectly clean
   (0.0000 damage) but slower than both existing candidates: 20.25s
   (seed 18000, the best of this batch) to 45.36s. None approached
   8000's 14.96s or 10000's 16.70s.

   Across all 15 fresh genuine-init seeds sampled today (4000-18000,
   plus the original 5-seed sweep 110-3000, 20 total), only two have
   landed in the 15-17s range: seed 8000 (fast, one real near-miss) and
   seed 10000 (nearly as fast, no near-miss). The other 18 cluster in a
   20-45s range. This suggests the "very fast" outcome is uncommon
   (roughly 2/20 ≈ 10% hit rate so far for anything near record pace),
   and the marginal yield of further blind sampling is dropping — three
   consecutive batches of 5 have found one improvement (8000), one
   good-but-not-better alternative (10000), and zero in the most recent
   batch.

   **Decision and rationale:** Reporting the honest search-yield picture
   rather than continuing to sample by default — with diminishing returns
   apparent, whether to keep spending compute on blind sampling
   (~7-8 min/seed) vs. settling on 8000 or 10000 vs. trying a
   qualitatively different lever (e.g. the untested controller-level
   safety backstop, or biasing the search — training longer per seed, or
   resuming/fine-tuning from 10000 specifically to try to close its
   pace gap to 8000) is a decision worth making deliberately rather than
   by default momentum.

   **Next steps:**
   1. Awaiting direction on how to proceed given the dropping yield.
   2. If continuing to sample, consider whether a larger batch is more
      efficient than repeated batches of 5 given the apparent ~10% hit
      rate for near-record pace.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 31 (run, 2026-09-08) — a batch of 10 confirms the
   plateau, still no strict improvement:** doubled the batch size (10
   fresh seeds, 19000-28000, vs. 5 in each prior batch) to test whether a
   bigger single draw would be more efficient than repeated small ones.
   Still no strict improvement over either candidate. Closest results:
   seed 25000 (18.50s avg lap time, 5.75 laps, 0.0179 max damage — still
   slower than both 8000's 14.96s and 10000's 16.70s), seed 24000
   (19.49s, perfectly clean), seed 21000 (20.04s, near-clean). The other
   7 landed at 22.57-36.15s.

   Across all **30 genuine-init seeds sampled today** (the original
   5-seed sweep plus four further batches), only 2 have landed in the
   15-17s range — seed 8000 and seed 10000 — a ~6.7% hit rate, consistent
   with (not better or worse than) the ~10% estimate from a smaller
   sample. Increasing batch size did not reveal a hidden pace tier
   between the "record" pair and the "everything else" cluster.

   **Decision and rationale:** With 30 samples now taken and the hit rate
   stable, continuing pure random seed sampling is unlikely to reliably
   turn up a strict improvement without substantially more compute per
   additional attempt at the current odds. Not launching further batches
   automatically — this is a natural point to either accept one of the
   two existing candidates, or switch to a qualitatively different lever
   (fine-tuning/resuming from seed 10000 specifically, or the untested
   controller-level safety backstop) rather than continuing to search
   blindly.

   **Next steps:**
   1. Awaiting direction: settle on 8000 or 10000, keep sampling anyway
      (now at known, stable odds), or pursue a different lever. (Resolved
      same day — see causal test 32 below.)
   2. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 32 (run, 2026-09-08) — fine-tuning seed 8000 finds a
   dominant checkpoint, and a non-monotonic training curve:** directed to
   continue optimizing seed 8000 specifically rather than keep sampling
   blindly. Ran two `--resume-from` experiments in parallel from
   `2026-09-08_seed-sweep-v2-8000/checkpoints/policy_final.pt`
   (`--warmup-steps 0`, same seed/round-length/reward/observation): +10
   races and +40 races, testing whether a smaller nudge might catch a
   partial safety improvement before the "resuming converges toward a
   shared, more conservative equilibrium" pattern (seen with seeds 1000
   and 2000 earlier this session) took hold.

   | | original (races=40) | +10 resumed | +40 resumed |
   | --- | --- | --- | --- |
   | avg best lap time | 14.96s | **15.54s** | 16.36s |
   | avg laps | 7.30 | 6.75 | 7.05 |
   | avg damage | 0.0298 | **0.0004** | 0.0138 |
   | max damage (worst race) | 0.5942 | **0.0047 (126x smaller)** | 0.2764 |
   | avg max speed | 26.7 m/s | 30.3 m/s | 34.4 m/s |

   **Result:** +10 races finds a clear sweet spot — the near-miss is
   essentially eliminated (18 of 20 races exactly 0.0000 damage, the
   other two trivial grazes) at a small pace cost (+3.9% lap time).
   +40 races overshoots it: the near-miss partially reappears (0.2764,
   worse than +10's 0.0047 though better than the original's 0.5942) and
   pace gets slower too — non-monotonic, not a smooth trade-off curve.
   The +10-resumed checkpoint also **strictly beats seed 10000** (the
   other safety candidate from causal test 29) on lap time (15.54s vs.
   16.70s), avg damage (0.0004 vs. 0.0008), and max damage (0.0047 vs.
   0.0154), with laps essentially tied. See
   `experiments/2026-09-08_seed8000-resumed-short/notes.md`.

   **Read:** fine-tuning duration is itself a sensitive hyperparameter,
   not a dial that trades pace for safety smoothly — this echoes causal
   test 11's races=40→80 regression from much earlier in the project
   (more training from scratch, not resumed, also regressed
   non-monotonically), now shown to hold for resume-based fine-tuning of
   an already-good checkpoint too.

   **Decision and rationale:** Adopting
   `2026-09-08_seed8000-resumed-short` as the new best/reference
   checkpoint — it dominates both prior candidates (original seed 8000,
   seed 10000) rather than requiring a judgment-call trade-off. Not
   adopting the +40 resume (worse than +10 on both axes).

   **Next steps:**
   1. Repackage `controllers.race_faster` from
      `seed8000-resumed-short`. (Done — see item 8 below.)
   2. A finer search around the sweet spot (e.g. +5, +15, +20 races) could
      find an even better point, though the non-monotonicity means
      "closer to 10 is better" isn't guaranteed to hold smoothly.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 33 (run, 2026-09-08) — increased self-play traffic, a
   fourth consecutive hesitation-reduction attempt to regress instead:**
   with the current checkpoint's hesitation level already modest (2.8%
   of race time, no severe outlier) and three reward-shaping attempts
   already failed (`WEIGHT_STEERING_REVERSAL` untested,
   `WEIGHT_ROBOT_PROXIMITY` x2 both regressed), tried a genuinely
   different, training-procedure lever instead: `--copies-per-side 2`
   (doubling self-play traffic), trained fresh on seed 8000, otherwise
   identical config.

   **Result: a clear, uniform regression.** Every one of 20 evaluation
   races landed at exactly 3 laps and ~12 m/s max speed — not an outlier,
   a genuinely different, much more conservative policy. Avg car-contact
   time improved only modestly (1.70s → 1.38s, ~19%) while avg best lap
   time more than doubled (15.54s → 32.95s) and avg laps dropped by more
   than half (6.75 → 3.00). Note: doubling `copies_per_side` also roughly
   doubles transitions collected per race, so this run reached 287,110
   gradient updates vs. the usual ~143,000 at races=40 — a confound that
   means this result can't cleanly separate "more opponent traffic helps
   avoidance" from "roughly double the effective training budget," which
   has independently been shown to push toward more conservative
   equilibria elsewhere this session (causal test 11, the seed 1000/2000
   resume experiments). See `experiments/2026-09-08_copies2-seed8000/notes.md`.

   **Decision and rationale:** Not adopted.
   `2026-09-08_seed8000-resumed-short` remains the reference checkpoint.
   This is the fourth consecutive hesitation-reduction attempt this
   session to regress pace instead of achieving a clean improvement —
   the evidence increasingly suggests the current checkpoint's hesitation
   level may be close to a practical floor for this reward/observation/
   training setup rather than an easily-closable gap.

   **Next steps:**
   1. If self-play traffic is revisited, control for the gradient-update
      confound by reducing `--races` proportionally when increasing
      `--copies-per-side` for a cleaner comparison. (Done — see causal
      test 34 below.)
   2. Otherwise, treat the current checkpoint's hesitation level as
      acceptable and consider this thread closed for now.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 34 (run, 2026-09-08) — the confound removed, and the
   verdict is decisive:** re-tested self-play traffic density the
   controlled way — fine-tuned the *already-good* current checkpoint via
   `--resume-from` with `--copies-per-side 2 --races 10` (a small dose,
   matching the successful "+10 races" pattern from causal test 32),
   rather than retraining from scratch, specifically to avoid conflating
   "denser traffic" with "more total training."

   **Result: car-contact time is completely unchanged** (1.70s → 1.72s)
   while avg best lap time still regressed substantially (15.54s →
   24.37s) and avg laps dropped (6.75 → 4.50). Unlike the from-scratch
   version (causal test 33, confounded, showed a modest ~19% car-contact
   improvement), this controlled version shows **zero measurable effect**
   on the metric it was specifically testing, while still costing pace.
   See `experiments/2026-09-08_copies2-finetune-seed8000/notes.md`.

   **Decision and rationale:** Not adopted. With the confound removed,
   the result is unambiguous rather than merely suggestive: self-play
   traffic density is not a productive lever for hesitation reduction on
   this checkpoint, at any dose or training procedure tested.
   Recommending this thread be closed — across five independently-
   designed experiments this session (three reward-shaping terms, two
   self-play-traffic configurations), none has cleanly improved
   hesitation without a pace cost, and this last one specifically found
   no improvement at all even controlling for every confound identified
   along the way. The current checkpoint's hesitation level (2.8% of
   race time, no severe outlier) is treated as a practical floor for this
   reward/observation/architecture combination.

   **Next steps:**
   1. The two remaining genuinely untried levers, if this is revisited:
      a closing-speed-scaled competitor-proximity term (lower confidence
      — may share the "maintain constant distance forever" escape hatch
      seen in the two failed distance-based variants), or a
      controller-level deterministic safety backstop (structurally
      different — an inference-time override rather than a training-time
      incentive, immune to the failure mode every reward-based attempt
      has hit).
   2. Otherwise, accept the current checkpoint's hesitation level as
      final for this track.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 35 (run, 2026-09-10) — training against a fixed
   opponent instead of self-play, a severe regression:** part of a
   combined-approach exploration with the imitation-learning track
   (`lucy-il`, merged into `combined-approach` on 2026-09-08). Added a
   `--opponent {self,expert}` flag to `scripts/train_sac.py` (default
   `self`, unchanged behavior) so the incumbent can be a frozen
   `controllers.leaderboard_expert.Controller` instead of a second
   `TrainableController` sharing the policy/buffer — motivated by
   `leaderboard_expert.py` already encoding, as hand-written deterministic
   rules, exactly the hazard behaviors this section's reward-shaping
   attempts (causal tests 21–23, 33–34) repeatedly failed or regressed on
   (competitor-proximity speed capping, speed-scaled wall-braking
   horizon, stuck recovery). Directed to always train against the expert
   and look for improvement. Same seed/races/round-length/n-step/
   hidden-size as the reference (`2026-09-08_seed-sweep-v2-8000`) —
   `--opponent` is the only new variable.

   **Result: catastrophic, not an improvement.** 100% elimination across
   all 30 evaluated races (0/20 for the reference), avg laps 0.63 (vs.
   7.30), 2/10 wins vs. `default_student_controller` (down from 20/20),
   0/10 vs. the expert itself. Avg opponent-collision time was actually
   *lower* than the reference (0.27s vs. 1.21s) — not a recurrence of the
   causal-test-21–23 failure mode — the car instead crashes into walls at
   much higher speed (avg max speed 37.3 vs. 26.7 m/s). `metrics.csv`
   shows real training instability: critic loss swings upward over the
   run rather than settling, and the entropy temperature collapses
   (~1.0 → ~0.03) very early, before the critic had anything reliable to
   be confident about. Total transitions collected (111,472) were far
   below the theoretical maximum for 40 uninterrupted races — the
   challenger was also being eliminated frequently *during* training, not
   just in evaluation. See
   `experiments/2026-09-10_expert-opponent-seed8000/notes.md` for the
   full diagnosis.

   **Read:** self-play's incumbent co-evolves with the challenger, so
   opponent difficulty always roughly matches current skill. A fixed,
   already-competent expert opponent from tick zero instead exposes an
   unskilled early-training policy to a distribution dominated by
   "recovering from/chasing a much faster car," plausibly biasing the
   whole run toward reckless, low-exploration behavior rather than the
   calm solo-driving experience self-play provides. This is a
   training-distribution-shift problem, not evidence the underlying
   "practice against a genuinely different opponent" idea is wrong in
   general — see next steps.

   **Decision and rationale:** Not adopted. `--opponent` defaults to
   `self`; `2026-09-08_seed8000-resumed-short` remains the reference
   checkpoint and `race_faster.py` is unchanged. Rejects the literal
   "always train against the expert from scratch" version of this
   combined-approach direction as tested. The `--opponent` flag is kept
   in code (default off), per this track's convention of preserving
   tested-but-rejected mechanisms rather than deleting them.

   **Next steps:**
   1. A curriculum variant is untested and more consistent with what
      actually differs here: `--resume-from` an already-good self-play
      checkpoint and fine-tune against the expert for a small dose of
      races (mirroring causal test 32's successful small-dose
      fine-tuning pattern), rather than training against it from random
      init for the full run.
   2. A mixed-opponent variant (alternating self-play and expert-opponent
      races within one run) is also untested and would avoid committing
      the entire training trajectory to the harder distribution before
      the policy has any baseline competence.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 35 follow-up (run, 2026-09-10) — fine-tuning instead of
   retraining from scratch, a real but insufficient improvement:** per
   next step 1 above, tested `--resume-from` the reference checkpoint and
   continuing training against the expert at 2 and 10 races. **Both
   collapsed within a handful of races** (27–29/30 eliminated) — the
   checkpoint's own `log_alpha` had already converged to ~0.035 (almost no
   exploration noise) from its self-play training, plausibly leaving it
   unable to recover once the unfamiliar opponent distribution started
   producing mistakes. Added `SACAgent.load_policy_only`
   (`src/training/sac.py`) and `--resume-policy-only`
   (`scripts/train_sac.py`) to test that hypothesis directly: load only
   the actor's weights, leaving critics and `log_alpha` fresh (full
   exploration budget restored). Ran at doses 2, 10, 20, and 40 races.

   | variant | avg damage | eliminated /30 | avg laps | avg lap time | avg max speed | avg car-contact |
   | --- | --- | --- | --- | --- | --- | --- |
   | reference | 0.0004 | 0/20 | 6.75 | 15.54s | 30.33 m/s | 1.702s |
   | full-resume, dose=2 | 0.9667 | 29 | 1.30 | 18.36s | 37.94 m/s | 0.456s |
   | full-resume, dose=10 | 0.9948 | 27 | 0.20 | 15.79s | 38.71 m/s | 0.126s |
   | policy-only, dose=2 | 0.4988 | 14 | 0.40 | 64.48s | 25.96 m/s | 1.777s |
   | policy-only, dose=10 | 0.1391 | 1 | 2.00 | 48.41s | 16.07 m/s | 3.906s |
   | policy-only, dose=20 | 0.3540 | 1 | 3.07 | 32.07s | 22.29 m/s | 3.265s |
   | policy-only, dose=40 | 0.1545 | 2 | 3.87 | 27.22s | 20.41 m/s | 4.326s |

   **Result:** policy-only resume confirmed the hypothesis — a
   qualitatively more stable regime at every dose (1–14/30 eliminated vs.
   27–29 for full resume), with a real, mostly-monotonic improving trend
   from dose 2→40 (damage, laps, and lap time all improve; by dose=20 it
   already sweeps 20/20 against both standard baselines, matching the
   reference's win record there). **But even at dose=40, no variant closes
   the gap to the reference on any metric, and the best-performing
   variant's car-contact time (4.326s) is *higher* than the reference's
   (1.702s)** — worse, not better, on the exact metric this whole
   direction was meant to improve. See
   `experiments/2026-09-10_expert-opponent-seed8000/followup_finetune_sweep.md`
   for the full write-up.

   **Decision and rationale:** Not adopted at any tested dose or resume
   mode. `2026-09-08_seed8000-resumed-short` remains the reference
   checkpoint; `race_faster.py` unchanged. Recommending a pause on
   training/fine-tuning against a fixed expert opponent (either resume
   mode) — the improving trend is real but has not yet helped
   opponent-avoidance at any point tested, and closing the remaining gap
   looks like it would need substantially more compute for an uncertain
   payoff. `--opponent expert`, `--resume-policy-only`, and
   `SACAgent.load_policy_only` are kept in code (all default off/opt-in)
   as tested, working, documented mechanisms.

   **Next steps:**
   1. A mixed-opponent curriculum (alternating self-play and
      expert-opponent races within one run, rather than switching to the
      expert entirely) is structurally different from every variant
      tried here and remains untested.
   2. Otherwise, treat the reference checkpoint as the best available for
      this direction and redirect combined-approach effort elsewhere —
      e.g. the shielded/safety-override idea (borrowing
      `leaderboard_expert`'s hazard rules as an inference-time override
      rather than a training signal), which carries none of this
      training-instability risk since it never retrains the policy.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 36 (run, 2026-09-10) — expert-match reward bonus,
   restricted to hazard states: the first genuine combined-approach win:**
   every prior combination attempt (causal test 35 and its follow-up)
   changed self-play's *opponent* to the expert and failed at every dose
   and resume strategy — the training *distribution* itself is what broke.
   This instead adds `WEIGHT_EXPERT_MATCH` to `training/reward.py`: a
   bonus for matching what `controllers.leaderboard_expert.Controller`
   would have done, restricted to ticks the existing wall/robot-proximity
   checks already judge hazardous. A private, non-controlling shadow
   expert instance inside `TrainableController` (`expert_match` flag,
   `--expert-match-bonus` on `scripts/train_sac.py`) computes the
   comparison; self-play's actual opponent (`--opponent self`, unchanged)
   never sees the expert. Matched the original from-scratch config of the
   current checkpoint's lineage (`2026-09-08_seed-sweep-v2-8000`) exactly —
   `--expert-match-bonus` is the only new variable.

   | | v2-8000 (from-scratch reference) | expert-match-bonus |
   | --- | --- | --- |
   | avg damage | 0.0298 | **0.0129 (-57%)** |
   | avg off-track | 0.152s | 0.372s (higher) |
   | avg wall-contact | 0.107s | 0.236s (higher) |
   | avg car-contact (target metric) | 1.212s | **1.136s (-6%)** |
   | avg laps | 7.30 | 7.30 |
   | avg best lap time | 14.96s | 14.98s |
   | avg max speed | 26.65 m/s | **35.11 m/s (+32%)** |
   | eliminated | 0/20 | 0/20 |
   | wins vs. both baselines | 20/20 | 20/20 |

   **Result:** the target metric (opponent-collision time) improved for
   the first time in any combined-approach experiment, alongside real
   damage and speed gains, at essentially no lap-time cost. Off-track/
   wall-contact time both rose modestly but stayed small in absolute
   terms; `metrics.csv` showed normal, stable training throughout (no sign
   of the divergence in every expert-opponent experiment), and per-race
   detail confirmed the increase was spread across several races/seeds,
   not one outlier. Also beats the currently-packaged reference
   (`2026-09-08_seed8000-resumed-short`) on car-contact, lap time, and max
   speed, at a still-negligible damage cost. See
   `experiments/2026-09-10_expert-match-bonus-seed8000/notes.md`.

   **Decision and rationale:** Not unilaterally adopted — presented as a
   genuine candidate for direction, consistent with how prior judgment-call
   improvements on this track (e.g. causal test 27) were handled. This is
   a single run/seed; per this track's established caution about n=1
   results, it shows the mechanism *can* help, not that it reliably will.

   **Next steps:**
   1. Await direction on whether to adopt this checkpoint or repackage
      `controllers.race_faster` from it.
   2. A seed sweep would establish robustness versus one favorable draw.
   3. The off-track/wall-contact increase is small but unexplained — a
      per-tick diagnostic would help if pursued further.
   4. `WEIGHT_EXPERT_MATCH = 0.5` was chosen by analogy to
      `WEIGHT_WALL_PROXIMITY`'s scale, not tuned by a dedicated sweep.
   5. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37 (run, 2026-09-11) — residual RL fixes the shield's
   discontinuity problem, the strongest combined-approach result yet:**
   a separate inference-time combination (`controllers.hybrid_controller`,
   a hard switch to `leaderboard_expert` during wall/robot-proximity
   hazards) reduced average contact time but traded it for a much higher
   crash rate against the expert (0/10 → 6/10 eliminated), diagnosed as
   likely coming from the discontinuity of switching instantly between
   two unrelated, independently-tuned control laws. This tests the fix:
   make the expert's influence *continuous* instead of switched. Added a
   `residual_base` mode to `TrainableController`
   (`src/training/controller.py`) — the expert's command is the base
   action every tick, and the SAC policy learns a bounded correction on
   top of it (`RESIDUAL_ACTION_SCALE = 0.3`, clamped to `[-1, 1]`); the
   *raw* policy output, not the blended command, is what's pushed to the
   replay buffer, since that's the actual action space the policy
   controls. `--residual-expert-base` on `scripts/train_sac.py`; stays
   inside ordinary self-play (`--opponent self`, unchanged) — no
   training-distribution risk like causal test 35. (This also surfaced
   and fixed a real correctness gap: `training.evaluation
   .evaluate_against_baselines` didn't know about `residual_base` and
   would have silently evaluated a residual checkpoint's correction
   output as an absolute command — now threaded through there and
   `scripts/eval_sac.py` too.) Matched the original from-scratch
   reference config exactly (`2026-09-08_seed-sweep-v2-8000`) —
   `--residual-expert-base` is the only new variable.

   | | v2-8000 (matched reference) | residual RL |
   | --- | --- | --- |
   | avg damage | 0.0298 | 0.0662 |
   | avg car-contact | 1.212s | 1.669s |
   | avg laps | 7.30 | **10.10** |
   | avg best lap time | 14.96s | **11.25s (-25%)** |
   | eliminated | 0/20 | 0/20 |
   | wins | 20/20 | 20/20 |

   The more important comparison — vs. `leaderboard_expert` as a live
   opponent, the same stress test that exposed the shield's failure:

   | | pure SAC | hybrid shield (hard switch) | residual RL |
   | --- | --- | --- | --- |
   | eliminated | 0/10 | **6/10** | **1/10** |
   | avg damage | 0.1471 | 0.6055 | 0.2278 |
   | avg laps | 6.50 | 4.70 | **9.50** |
   | avg best lap time | 16.51s | 14.99s | **12.08s** |

   **Result:** recovers almost all of plain SAC's safety against the
   expert (1/10 vs. 0/10 eliminated, vs. the shield's 6/10) while being
   the fastest and most complete of all three variants in every matchup
   tested. `metrics.csv` showed normal, stable training throughout
   (critic loss bounded, entropy settling smoothly 1.0 → ~0.035) — unlike
   every train-against-a-fixed-opponent experiment (causal test 35),
   confirming that changing the action *composition* rather than the
   training *opponent* avoids that instability entirely. See
   `experiments/2026-09-11_residual-expert-base-seed8000/notes.md`.

   **Decision and rationale:** Not unilaterally adopted or repackaged
   into `race_faster.py` — presented as the strongest combined-approach
   candidate found this session. Not a perfect result: damage/off-track/
   wall-contact all rose somewhat against the standard baselines (though
   eliminations there stayed at zero), and the one elimination against
   the expert (seed 2024, race 1 — a real, not-yet-diagnosed outlier)
   means this is not as unconditionally safe as plain SAC alone. A
   genuine trade — meaningfully faster, slightly less safe — not a strict
   improvement on every axis, but a much more favorable trade than the
   shield's.

   **Next steps:**
   1. Await direction on whether to adopt this (e.g. package a
      self-contained `controllers.*` module analogous to `race_faster.py`,
      composing the frozen residual actor with the expert the same way
      at inference time).
   2. Diagnose the one elimination against the expert with a per-tick
      trace.
   3. A seed sweep would establish robustness versus one favorable draw.
   4. `RESIDUAL_ACTION_SCALE = 0.3` was chosen without a dedicated sweep.
   5. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37 follow-up (run, 2026-09-11) — recovery-passthrough
   fix resolves the one crash entirely, no trade-off:** a per-tick trace
   of the one elimination above (seed 2024 vs. `leaderboard_expert`)
   found a repeated stuck-against-the-same-wall-spot loop, not a single
   high-speed impact — the car hit the identical wall location 8 times
   over ~8 seconds, backing off via the expert's stuck-recovery maneuver
   each time and driving straight back into it. Root cause: the SAC
   correction was still being applied *during* the expert's deliberate
   recovery maneuver, diluting a precise fixed escape trajectory. Fixed
   in `src/training/controller.py`: detect a recovery command (checking
   `leaderboard_expert.Controller`'s `_recovery_ticks_remaining` before
   and after calling it) and pass it through unmodified, with zero
   residual applied. Confirmed the fix needs a retrain, not just an
   inference patch — applied to v1's existing weights alone, the
   identical crash reproduced. Retrained fresh, otherwise identical
   config.

   | | pure SAC | hybrid shield | residual v1 | residual v2 (fix) |
   | --- | --- | --- | --- | --- |
   | eliminated (vs. expert) | 0/10 | 6/10 | 1/10 | **0/10** |
   | avg best lap time (vs. expert) | 16.51s | 14.99s | 12.08s | **11.78s** |
   | avg laps (vs. expert) | 6.50 | 4.70 | 9.50 | **9.80** |
   | avg damage (standard baselines) | — | — | 0.0662 | **0.0254** |

   **Result:** the fix fully resolved the crash (seed 2024 now completes
   both races cleanly) and, unlike almost every other safety fix
   attempted on this track, **improved every safety metric with no pace
   cost** — v2 beats v1 on damage/off-track/wall-contact/car-contact on
   the standard baselines *and* is faster with more laps against the
   expert. v2 now matches pure SAC's perfect elimination record while
   being the fastest and most complete variant tested in every matchup.
   See `experiments/2026-09-11_residual-expert-base-v2-seed8000/notes.md`.

   **Remaining gap:** Lucy's raw expert alone, same protocol, still laps
   faster in isolation (8.94s avg, 0.047 avg damage, 0/10 eliminated) —
   not yet closed, though this combined controller now unambiguously
   surpasses plain SAC alone and the hybrid shield on every metric
   tracked.

   **Decision and rationale:** Not unilaterally adopted or repackaged
   into `race_faster.py` — presented as the strongest, most complete
   combined-approach candidate found this session, a materially stronger
   case than either the shield or v1 since it dominates rather than
   trades off against plain SAC.

   **Next steps:**
   1. Await direction on adoption / packaging as a self-contained
      `controllers.*` module.
   2. Closing the remaining pace gap to Lucy's raw expert would likely
      need more training, a larger `RESIDUAL_ACTION_SCALE`, or may partly
      reflect her expert's greater risk tolerance.
   3. A seed sweep would establish robustness versus one favorable draw.
   4. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, second follow-up (run, 2026-09-11) — widening the
   residual scale is a clean regression, not a speed gain:** tested the
   most direct lever for closing the remaining pace gap to Lucy's raw
   expert (8.94s solo, vs. v2's 11.37-11.78s): made
   `RESIDUAL_ACTION_SCALE` configurable (`--residual-action-scale` on
   `scripts/train_sac.py`) and trained fresh at `0.45` (up from `0.3`),
   config otherwise identical to v2.

   | | v2 (scale=0.3) | scale=0.45 |
   | --- | --- | --- |
   | avg off-track (standard) | 0.325s | 5.737s (17.6x worse) |
   | avg wall-contact (standard) | 0.140s | 3.305s (23.6x worse) |
   | avg best lap time (standard) | 11.37s | 11.83s (slower, not faster) |
   | wins (standard) | 20/20 | 19/20 |
   | avg off-track (vs. expert) | 1.320s | 9.92s (7.5x worse) |
   | avg best lap time (vs. expert) | 11.78s | 12.47s (slower) |

   **Result:** a clean, consistent regression across both evaluation
   protocols, not an unstable-training artifact (`metrics.csv` showed
   normal, bounded critic loss and smooth entropy settling). More freedom
   to deviate from the expert's line did not translate into a faster
   line — lap time got marginally *slower* in both matchups — while
   off-track/wall-contact time exploded 7-24x. See
   `experiments/2026-09-11_residual-scale045-seed8000/notes.md`.

   **Decision and rationale:** Not adopted; reverted.
   `2026-09-11_residual-expert-base-v2-seed8000` (`residual_scale=0.3`,
   the default) remains the best combined-approach checkpoint.
   `--residual-action-scale` kept as a configurable, tested parameter
   (default unchanged). This axis is treated as exhausted after one clear
   negative result, consistent with this track's practice of not
   continuing to search an axis without a different underlying idea. The
   remaining pace gap to Lucy's raw expert may be partly structural: her
   expert explicitly accepts more risk to maximize speed, while the RL
   reward function balances speed against safety by design.

   **Next steps:**
   1. This axis (residual scale) is exhausted for now — a different
      lever (more training, or residual-mode-specific reward tuning)
      would be needed, not more of this same axis.
   2. Otherwise, treat v2 as the practical best combined-approach result:
      it already unambiguously surpasses plain SAC alone and the hybrid
      shield on every metric tracked, even without closing the gap to
      Lucy's raw, safety-unconstrained pace.
   3. A seed sweep would establish robustness versus one favorable draw.
   4. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, third follow-up (run, 2026-09-12) — narrowing the
   scale is also worse, the axis is now closed:** tested the opposite
   direction from the failed `scale=0.45` run: `residual_scale=0.15`
   (down from v2's `0.3`), on the hypothesis that a narrower correction
   would reduce erratic driving with little speed cost. Same matched
   config otherwise.

   | | v2 (scale=0.3) | scale=0.15 | scale=0.45 |
   | --- | --- | --- | --- |
   | avg damage (standard) | 0.0254 | 0.0100 (better) | 0.1253 |
   | avg best lap time (standard) | 11.37s | 12.08s (worse) | 11.83s |
   | eliminated (vs. expert) | 0/10 | **1/10 (regressed)** | 0/10 |
   | avg car-contact (vs. expert) | 4.623s | 5.698s (worse) | — |

   **Result:** not simply safer — average damage improved on the standard
   baselines, but every other metric got worse, and a real elimination
   reappeared against the expert (one that v2 had eliminated entirely). A
   narrower correction has less power to react when a real course change
   is genuinely needed. Combined with `scale=0.45`, three points on this
   axis are now tested (0.15, 0.3, 0.45) and `0.3` (v2) wins outright or
   ties on nearly every metric in both directions — a genuine local
   optimum, not an arbitrary first guess. See
   `experiments/2026-09-12_residual-scale015-seed8000/notes.md`.

   **Decision and rationale:** Not adopted. The residual-scale axis is
   now treated as closed. Further concrete improvement should come from a
   different lever — a seed sweep (genuine network-initialization
   diversity) is this project's own best-precedented way of finding real
   gains (see causal tests 24-32 on the plain-SAC side of this section),
   and has not yet been tried for residual mode.

   **Next steps:**
   1. Seed sweep at the locked-in v2 config (residual_base, scale=0.3,
      recovery-passthrough fix) — not yet done.
   2. Otherwise, treat v2 as the practical best combined-approach result.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, seed sweep (run, 2026-09-12) — two draws, no
   improvement over v2:** per next step 1 above, sampled two fresh
   network initializations (seeds 9000, 10000) at the locked-in config,
   otherwise identical to v2.

   | | v2 (seed=8000) | seed=9000 | seed=10000 |
   | --- | --- | --- | --- |
   | eliminated (standard) | 0/20 | **1/20** | 0/20 |
   | avg laps (standard) | 10.15 | 9.45 | 9.25 |
   | avg best lap time (standard) | 11.37s | 11.64s | 12.31s |
   | eliminated (vs. expert) | 0/10 | — | 0/10 |
   | avg best lap time (vs. expert) | 11.78s | — | 12.55s |

   Seed 9000 was clearly worse (a reintroduced elimination plus worse
   damage/laps/pace). Seed 10000 was mixed — matched v2's elimination
   record and improved car-contact time, but lost on off-track/
   wall-contact/laps/pace in both protocols. Neither beat v2 overall;
   `metrics.csv` showed stable training for both. See
   `experiments/2026-09-12_residual-seedsweep-9000/notes.md` and
   `experiments/2026-09-12_residual-seedsweep-10000/notes.md`.

   **Decision and rationale:** Neither adopted; v2 remains the best
   checkpoint after two sampled seeds. Consistent with (not better or
   worse than) the plain SAC track's own observed per-seed hit rate for a
   strict improvement (roughly 10-20%) — two failed draws doesn't rule
   out the approach, but is a real data point.

   **Next steps:**
   1. Awaiting direction: continue sampling (a larger batch), or
      conclude the seed sweep and settle on v2 as the final result.
   2. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, seed sweep continued (run, 2026-09-12) — seed 12000
   is a safety standout and the first checkpoint to beat the expert
   outright, but the pace gap holds across 5 seeds:** sampled two more
   seeds (11000, 12000) at the locked-in config, now tracking each
   checkpoint's *fastest individual lap*, not just its average, since a
   single fast lap directly bears on whether Lucy's pace is reachable.

   | seed | avg damage (standard) | avg lap time (standard) | fastest lap | eliminated (vs. expert) | wins (vs. expert) |
   | --- | --- | --- | --- | --- | --- |
   | 8000 (v2) | 0.0254 | 11.37s | 10.33s | 0/10 | 0/10 |
   | 9000 | 0.0682 | 11.64s | — | — | — |
   | 10000 | 0.0319 | 12.31s | — | 0/10 | 0/10 |
   | 11000 | 0.0216 | 11.58s | 10.45s | 0/10 | 0/10 |
   | 12000 | **0.0020** | 11.57s | **10.23s** | 0/10 | **1/10** |

   **Result:** seed 12000 is an order of magnitude safer than v2 in both
   test protocols (12.7x lower damage on standard baselines, 11.6x lower
   vs. the expert) while essentially tying v2's pace and setting a new
   fastest-lap record — and it's the first checkpoint of any kind on this
   track (plain SAC, hybrid shield, or any residual variant) to win a
   race outright against `leaderboard_expert`. But its *average* pace
   against the expert was slower than v2's (13.25s vs. 11.78s), so it
   doesn't advance the pace goal specifically. **Across all 5 sampled
   seeds, no average lap time has come within 25% of Lucy's raw 8.94s,
   and the single fastest individual lap found (seed 12000's 10.23s) is
   still ~14% off her average** — consistent across genuinely different
   network initializations, which is evidence (not proof) that the gap
   reflects the reward function's structural speed/safety balance rather
   than initialization luck. See
   `experiments/2026-09-12_residual-seedsweep-12000/notes.md`.

   **Decision and rationale:** Neither seed unilaterally adopted. Seed
   12000 flagged as a genuinely strong alternative to v2 if safety is
   weighted heavily; v2 remains the reference pending direction on which
   axis to prioritize.

   **Next steps:**
   1. Awaiting direction: adopt seed 12000, keep sampling, or try a
      different lever (e.g. residual-mode-specific reward tuning) if
      closing the pace gap specifically remains the priority.
   2. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, reward-weight lever (run, 2026-09-12) — a third
   consecutive clean failure, this time backfiring on speed itself:**
   tried a structurally different lever from the residual-scale sweep and
   the seed sweep: `--progress-weight`, a training-only override of
   `WEIGHT_PROGRESS` (threaded through `step_reward` and
   `TrainableController`), on the hypothesis that the fixed caution terms
   (tuned entirely for plain self-play) might be more conservative than
   residual mode needs, since the base action already comes from a
   competent expert. Tested `progress_weight=1.5`, config otherwise
   identical to v2.

   | | v2 (progress_weight=1.0) | progress_weight=1.5 |
   | --- | --- | --- |
   | avg damage (standard) | 0.0254 | 0.0569 (worse) |
   | avg best lap time (standard) | 11.37s | **12.03s (slower)** |
   | fastest individual lap (standard) | 10.33s | **11.48s (slower)** |
   | avg best lap time (vs. expert) | 11.78s | 12.24s (slower) |
   | eliminated (both protocols) | 0/20, 0/10 | 0/20, 0/10 |

   **Result:** the opposite of the intended effect, on every axis
   including the one this was meant to improve — weighting progress more
   heavily made the policy slower, not faster, while also less safe on
   damage/off-track/wall-contact. `metrics.csv` showed no training
   instability. See
   `experiments/2026-09-12_residual-progressweight15-seed8000/notes.md`.

   **Decision and rationale:** Not adopted. This is the third
   consecutive, structurally different lever aimed at closing the pace
   gap (residual scale, network initialization, now reward weighting) to
   fail cleanly, each pointing the same direction. Recommending against
   further from-scratch runs purely aimed at beating Lucy's raw pace via
   scale/seed/reward-weight tuning — the evidence increasingly reads as a
   real structural trade-off rather than a nearby local optimum.

   **Next steps:**
   1. If pace remains the priority, a genuinely different mechanism would
      be needed (e.g. reward shaping targeting cornering technique
      specifically), or accept a higher damage/elimination rate as the
      deliberate cost of matching Lucy's pace.
   2. Otherwise, treat this line of investigation as concluded: v2
      (pace-balanced) and seed=12000 (safety-focused) are the two
      combined-approach checkpoints worth keeping.
   3. Still open: the `controllers.minimum_viable` module gap.

   **Causal test 37, cornering-specific reward shape (run, 2026-09-12) —
   a fourth consecutive clean failure:** the "more invasive" direction
   after three prior levers failed. Gated `WEIGHT_CENTER_OFFSET` by
   upcoming bend sharpness (`_bend_score`, the same formula
   `leaderboard_expert.py` itself uses) instead of applying it uniformly
   — full strength approaching a corner, reduced on straights
   (`--curvature-aware-center-offset`). Unlike a 2026-09-03 attempt on
   the plain-SAC track that halved this weight *uniformly* (including in
   corners) and regressed badly, this only loosens the penalty where the
   track is straight. Same matched config as v2 otherwise.

   | | v2 (uniform penalty) | curvature-aware |
   | --- | --- | --- |
   | eliminated (standard) | 0/20 | **1/20 (reintroduced)** |
   | avg best lap time (standard) | 11.37s | 11.56s (slower) |
   | fastest individual lap (standard) | 10.33s | 11.20s (slower) |
   | eliminated (vs. expert) | 0/10 | **1/10 (reintroduced)** |
   | avg best lap time (vs. expert) | 11.78s | 11.98s (slower) |

   **Result:** lost safety in both protocols without gaining speed on
   average — not even a trade-off. `metrics.csv` showed an early
   critic-loss spike (30.0, well above every other residual run's
   typical range) that settled by the end, not a clean run throughout.
   The specific hypothesis (that the 2026-09-03 failure's *uniform*
   nature was why it regressed) wasn't confirmed — a curvature-gated
   version regressed too, just less severely, suggesting this penalty's
   shape isn't the actual pace bottleneck. See
   `experiments/2026-09-12_residual-curvature-aware-seed8000/notes.md`.

   **Decision and rationale:** Not adopted; reverted. **Four
   structurally different levers have now failed to close the pace gap**
   (residual scale, network initialization, global progress-weight, and
   now a targeted cornering-specific reward shape) — a strong, consistent
   pattern, not one unlucky axis. Recommending this line of investigation
   be concluded rather than attempting a fifth variation.

   **Next steps:**
   1. Recommending against further from-scratch experiments purely aimed
      at beating Lucy's raw pace via training-side tuning.
   2. v2 and seed=12000 remain the two combined-approach checkpoints
      worth keeping.
   3. If pace is still a priority, it likely needs a fundamentally
      different approach outside this reward-tuning family, or accepting
      that matching Lucy's raw pace requires giving up the safety balance
      this whole effort was built around.
   4. Still open: the `controllers.minimum_viable` module gap.

   **Line of investigation concluded (2026-09-12), then reopened same
   day:** per direction earlier in this session, closing the remaining
   pace gap to Lucy's raw expert (8.94s solo) was paused after four
   structurally different levers (residual scale in both directions, a
   5-seed sweep, reward reweighting, cornering-specific reward shaping)
   all failed cleanly above. Reopened later the same session per direct
   request to try two more specific levers: raising the effective "speed
   ceiling" and accepting more risk on purpose. See the two follow-ups
   immediately below.

   **Causal test 37, wall-proximity-speed-scale lever (run, 2026-09-12) —
   a fifth consecutive failure:** `MAX_REWARDED_SPEED_MPS` no longer exists
   (removed structurally in causal test 17, 2026-09-02) so there is no
   literal "speed ceiling" left to raise — the closest surviving analog is
   `WALL_PROXIMITY_SPEED_SCALE_MPS`, which controls how fast the
   wall-proximity penalty grows with speed. Added an opt-in
   `wall_proximity_speed_scale_mps` override to `step_reward`/
   `TrainableController`/`scripts/train_sac.py --wall-proximity-speed-scale`
   (default unchanged, so plain self-play is unaffected) and trained at
   `20.0` (2x default), matched to v2's exact config otherwise.

   | | v2 (reference) | wall-scale=20 |
   | --- | --- | --- |
   | avg damage (standard) | 0.0254 | 0.0505 (worse) |
   | avg best lap time (standard) | 11.37s | 11.88s (slower) |
   | fastest lap (standard) | 10.33s | 11.40s (slower) |
   | eliminated (vs. expert) | 0/10 | 0/10 |
   | avg best lap time (vs. expert) | 11.78s | 13.37s (slower) |

   **Result:** every scored metric moved the wrong direction except lap
   count against the expert (roughly flat). Tolerating more risk at speed
   did not translate into a faster line in either matchup —
   `metrics.csv` showed normal, stable training throughout, so this isn't
   an instability artifact. See
   `experiments/2026-09-12_residual-wallscale20-seed8000/notes.md`.

   **Causal test 37, damage-weight lever (run, 2026-09-12) — a sixth
   consecutive failure, and the first to reintroduce eliminations:**
   `WEIGHT_DAMAGE` has been held at `5.0` unchanged since this reward's
   inception and was never itself the variable in any prior causal test —
   the most direct "accept more risk on purpose" lever available. Added an
   opt-in `damage_weight` override the same way, trained at `2.5` (half
   default), matched to v2's exact config otherwise.

   | | v2 (reference) | damage-weight=2.5 |
   | --- | --- | --- |
   | avg damage (standard) | 0.0254 | 0.0606 (worse) |
   | avg best lap time (standard) | 11.37s | 11.66s (slower) |
   | fastest lap (standard) | 10.33s | 11.12s (slower) |
   | eliminated (vs. expert) | 0/10 | **2/10 (regressed)** |
   | avg best lap time (vs. expert) | 11.78s | 12.32s (slower) |

   **Result:** a clean regression, not a trade-off — pace got worse in
   both matchups *and* safety visibly degraded (two real eliminations
   against the expert, where v2 and the wall-scale variant above both hold
   0/10). Same failure family as prior "loosen a caution term hoping to
   unlock speed" attempts (2026-09-03's `WEIGHT_CENTER_OFFSET` halving,
   this session's curvature-aware center-offset test) — the caution term
   being loosened was load-bearing for competent driving, not merely
   capping top speed. See
   `experiments/2026-09-12_residual-damageweight25-seed8000/notes.md`.

   **Decision and rationale:** Neither adopted. Both overrides
   (`wall_proximity_speed_scale_mps`, `damage_weight`) are kept as tested,
   documented, opt-in parameters (defaults unchanged) rather than reverted
   code, consistent with this track's practice of preserving negative
   results. **Six structurally different levers have now failed to close
   the pace gap** (residual scale x2, network-initialization seed sweep,
   progress-weight reweight, curvature-aware center-offset,
   wall-proximity-speed-scale, damage-weight) — a consistent pattern
   across every category of lever this reward structure offers (scale,
   seed, and every weight/shape term touched so far), not one unlucky
   axis.

   **Caveat found later the same session:** `--opponent self`'s incumbent
   construction wasn't passing `wall_proximity_speed_scale_mps`/
   `damage_weight` through (only the challenger got them), so roughly half
   of both runs above' transitions were actually computed under the
   *default* weights, not the intended overrides. Fixed in
   `scripts/train_sac.py`; neither run was retrained under the fix. Treat
   both results above as diluted (~half-strength) tests, not clean ones --
   see each experiment's own `notes.md` for detail. The regressions
   observed are, if anything, a lower bound on the downside of applying
   either lever at full strength.

   **Causal test 37, mixed-opponent curriculum (run, 2026-09-12) — a
   genuinely different mechanism, and the worst result yet:** per the
   next-step note on the (non-residual) `--opponent expert` fine-tune dose
   sweep (`experiments/2026-09-10_expert-opponent-seed8000/
   followup_finetune_sweep.md`), tried the one lever flagged as
   structurally different from every reward-tuning attempt above:
   alternating the self-play training opponent race-by-race between
   another in-training residual copy and `controllers.leaderboard_expert`
   directly, rather than changing what the reward pays for. Added
   `--opponent mixed` / `--mixed-opponent-expert-every` (default 2, i.e.
   every other race) to `scripts/train_sac.py`; smoke-tested on a tiny
   config first, then trained matched to v2's config otherwise.

   | | v2 (reference) | mixed-opponent |
   | --- | --- | --- |
   | avg damage (standard) | 0.0254 | 0.0787 (worse) |
   | avg off-track (standard) | 0.325s | 3.572s (11x worse) |
   | avg wall-contact (standard) | 0.140s | 2.422s (17x worse) |
   | avg best lap time (standard) | 11.37s | 12.60s (slower) |
   | eliminated (standard) | 0/20 | **1/20 (regressed)** |
   | eliminated (vs. expert) | 0/10 | **2/10 (regressed)** |
   | avg best lap time (vs. expert) | 11.78s | 12.69s (slower) |

   **Result:** the worst outcome of any residual-mode variant tested to
   date, and the first to regress general driving competence rather than
   just the expert matchup specifically — off-track/wall-contact time
   exploded even against `crash_fast`/`default_student_controller`, which
   have no bearing on expert-matchup skill. `metrics.csv` showed real
   instability (critic loss peaked at 290.8, ~10x the level the
   curvature-aware test flagged as notably elevated), closer to the
   severe, fast collapse the fully-switched (non-residual) `--opponent
   expert` attempts showed than to a stable-but-worse convergence — milder
   than that full switch, but not the clean stability hoped for from never
   fully leaving the self-play distribution. See
   `experiments/2026-09-12_residual-mixedopponent-seed8000/notes.md`.

   **Decision and rationale:** Not adopted. This is the seventh
   structurally different lever — and the first full mechanism change
   rather than a reward-weight tweak — to fail at closing the pace gap to
   Lucy's raw expert, producing a worse result than any single-variable
   reward tweak tried before it. `--opponent mixed` is kept as a tested,
   documented, opt-in flag (default `self`, so existing behavior is
   unaffected) rather than reverted code.

   **Line of investigation concluded again (2026-09-12).**
   `2026-09-11_residual-expert-base-v2-seed8000` is adopted as the final
   combined-approach result, packaged as `src/controllers/combined_candidate.py`
   (see `docs/lab_notebook.md`'s 2026-09-11 entry for the packaging work),
   now the strongest result after seven independent attempts (six
   reward-tuning levers plus this one mechanism change) to beat it on
   pace. No further attempts at this specific goal are planned. The
   seed=12000 safety-focused alternative remains available if priorities
   change, but is not being pursued by default.
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
   this is about the *spawn-point* seed distribution used during training
   (a single run only ever spawns from one seed's fixed starting
   positions), distinct from item 0's finding about training-*run*
   (network/exploration) seed sensitivity. **The training-run sensitivity
   question is now resolved (causal test 24, 2026-09-08): a 5-seed sweep
   found every seed produces a safe policy under the current stack, with
   the 2026-09-01 catastrophic-freeze instability no longer present** —
   real pace variance remains across seeds (~2x lap time), but not the
   qualitative brokenness this item originally worried about. This
   spawn-point-diversity item remains open and untouched by that result.
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

   **Repackaged (2026-09-08, causal test 24's checkpoint):** exported
   `2026-09-08_seed-sweep-1000` — the best-pace checkpoint of the first
   genuine multi-training-seed sweep (§6 causal test 24): 0/20
   eliminations, 20/20 wins against both baselines, 6.50 avg laps, 17.19s
   avg best lap time, ~31 m/s avg max speed. This is a bigger update than
   a routine checkpoint swap: the observation encoding itself changed
   shape (17 → 24 dims) on 2026-09-07 when obstacle LiDAR was added to
   `training.observation` (§2.1), so `race_faster.py`'s inlined encoding
   had to be updated to match, not just the checkpoint file — a mismatch
   here would either fail to load (wrong tensor shape) or silently
   misinterpret the observation vector. Re-verified the full chain before
   treating this as done: bit-for-bit match against
   `SACAgent.act(..., deterministic=True)` on 20 random 24-dim
   observations (`max abs diff = 0.0`), `ruff check`/`ruff format
   --check`/`pyright` all clean, and a real `racing h2h` race (seed 110,
   30s, vs. `crash_fast`): 0 damage, 1 lap in 16.88s, 0 off-track/
   wall-contact, 291.4m margin — consistent with the checkpoint's
   documented headless-eval behavior. Exported via
   `scripts/export_student_controllers.py --all-controllers` to
   `artifacts/formula110-student-controllers.zip` (not yet re-uploaded to
   Gradescope — packaging only, per CLAUDE.md's distinction between
   building the export and the separate submission-manifest step for an
   actual upload). Superseded chain: `2026-09-01_more-training-seed110`
   (races=20, original) → `2026-09-08_seed-sweep-1000` (this one,
   spanning both the training-budget scale-up and the observation-
   dimension change in between).

   **Repackaged again (2026-09-08, causal test 27's checkpoint):**
   exported `2026-09-08_seed-sweep-v2-8000` — the best-of-N pace record
   found via genuine-network-initialization seed sampling (§6 causal
   test 27): 0/20 eliminations, 20/20 wins, 7.30 avg laps, 14.96s avg
   best lap time (-13% vs. the previous package), ~27 m/s avg max speed.
   Adopted with a documented, diagnosed near-miss (1/20 races, 0.5942
   damage from a hard corner-entry wall impact, no elimination) accepted
   explicitly rather than overlooked — see the module's own docstring
   for the risk note. Observation encoding unchanged from the previous
   packaging (both checkpoints trained under the same 24-dim
   obstacle-LiDAR observation), so only the checkpoint and docstring
   changed this time. Re-verified: bit-for-bit match against
   `SACAgent.act(..., deterministic=True)` on 20 random observations
   (`max abs diff = 0.0`), `ruff check`/`ruff format --check`/`pyright`
   all clean, and a real `racing h2h` race (seed 110, 30s, vs.
   `crash_fast`): 0 damage, 1 lap in 15.75s (faster than the previous
   package's 16.88s), 0 off-track/wall-contact. Exported via
   `scripts/export_student_controllers.py --all-controllers`. Directed
   next step: improve safety from this faster baseline (e.g. the
   corner-aware wall-proximity idea noted in causal test 27), not yet
   started.

Each refinement experiment should change a limited number of variables at
once and be compared against both the fixed baseline (`crash_fast`) and the
best prior SAC checkpoint, per `experiments/README.md`.
