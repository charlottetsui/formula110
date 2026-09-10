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
