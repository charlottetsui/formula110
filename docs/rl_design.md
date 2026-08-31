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
- Large one-time penalty when `contact.damage` reaches `1.0` (elimination).

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
  for a new car/controller run"). The controller uses this to close out the
  previous transition with `done=True` and reset its internal previous-obs
  state.
- **Elimination:** `contact.damage >= 1.0`. Detected a tick early via the
  damage delta so the terminal transition carries the penalty.

A round timeout (`--round-seconds`, default 30s) also ends an episode
without a discrete signal in-band; the controller times this out using its
own tick counter against the configured `fixed_delta_seconds`.

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
- Baseline for comparison: `src/controllers/crash_fast.py` (always full
  throttle, no steering) and a simple hand-written wall-avoidance
  controller if one exists by then.

**Evaluation:**

- Fixed seed set for reproducibility across runs: `110, 42, 7, 2024, 8675309`
  (same seeds used by both single-car and h2h per the README's deterministic
  spawn contract).
- Metrics, read from `HeadToHeadTeamRaceStats` / `HeadToHeadResult`, not the
  training-time proxy reward: scored distance, lap count, elimination rate,
  time-to-first-wall-contact, max speed, variance across the 5 seeds.
- Compare: does the SAC-trained controller beat `crash_fast` on scored
  distance and survive longer before elimination, and is that consistent
  across all 5 seeds (not one lucky start)?
- Record wall-clock training time — SAC's sample efficiency claim (§1) is
  only worth something if training time is competitive with the alternative
  approach's dev/train time.

Results, logs, and configs for this and later runs live under
`experiments/` (see `experiments/README.md` for the per-run convention).

## 6. Refinement plan (post-selection)

Candidates for the next round of experiments once SAC is selected as the
primary approach, roughly in order of expected leverage:

1. **Reward shaping** — tune `w_progress`/`w_center`/`w_wall` weights;
   check whether the proxy reward and real scored distance move together
   across training (the divergence check from §2.3).
2. **Robustness across seeds** — widen the training seed distribution
   (rather than a fixed handful) so the policy doesn't overfit to specific
   spawn points; evaluate on held-out seeds never used in training.
3. **Opponent traffic** — increase `copies_per_side` so self-play produces
   denser traffic, and start using `sensors.lidar` /
   `camera.competitors` in the observation once solo-track driving is
   solid.
4. **Network/optimizer tuning** — hidden sizes, learning rates, target
   network update rate (`tau`), entropy temperature (fixed vs. learned).
5. **Replay buffer** — size, and whether uniform sampling is good enough or
   prioritized replay is worth the complexity.
6. **Reduce hesitation** — penalize small/oscillating steering deltas if
   the trained policy shows jittery control in watched races.
7. **Packaging for the leaderboard** — export the trained policy as a
   frozen-weights `Controller` under `src/controllers/` per the
   [packaging contract](../README.md#packaging-a-controller): CPU-only,
   evaluation/inference mode, well under 512 MiB, no training-only
   dependencies imported at inference time.

Each refinement experiment should change a limited number of variables at
once and be compared against both the fixed baseline (`crash_fast`) and the
best prior SAC checkpoint, per `experiments/README.md`.
