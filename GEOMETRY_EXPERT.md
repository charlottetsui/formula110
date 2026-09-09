# Leaderboard Expert and Imitation Learning

The current expert is
[`src/controllers/leaderboard_expert.py`](src/controllers/leaderboard_expert.py).
It uses public sensors to carry speed through corners, brake when needed, and
resume acceleration without unnecessarily stopping. It is the strongest
measured expert in this project's recent solo trials, and a starting point for
imitation learning. It has not been established as an optimal controller or
validated across every recovery situation.

The superseded expert implementations and their dedicated tests have been
removed. `crash_fast.py` remains the minimal student starter template. This
guide keeps its existing filename so links to it continue to work.

## Run and test the expert

Follow [GETTING_STARTED.md](GETTING_STARTED.md) to install the project, then run:

```bash
uv run racing --seed 110 --student-module controllers.leaderboard_expert
```

Try other deterministic starting positions:

```bash
uv run racing --seed 2026 --student-module controllers.leaderboard_expert
uv run racing --seed 7 --student-module controllers.leaderboard_expert
uv run racing --seed 42 --student-module controllers.leaderboard_expert
```

These seeds vary starts on the existing track; they do not generate different
track layouts. Reuse the same seeds and simulator settings for fair comparisons.

Run the controller and actuator regression tests with:

```bash
uv run pytest tests/test_leaderboard_expert.py tests/test_physics.py
```

The tests cover acceleration, corner braking, low-speed behavior, recovery,
command bounds, and release of the simulator's braking state while still moving.

## Observations and controller state

At the usual 60 Hz simulation rate, the controller receives a `RobotSensors`
snapshot and returns `RobotCommand(throttle, steer)`. Both fields are bounded
to `[-1.0, 1.0]`. Negative steering turns left; positive steering turns right.
Signed throttle requests reverse or forward drive, with braking when the
request opposes current motion. Exactly zero throttle coasts.

| Public observation | Purpose |
| --- | --- |
| Camera heading error and lookahead offsets | Anticipate turn direction and severity |
| Camera center offset | Correct large displacement from the centerline |
| Camera visibility | Select fallback steering and speed control |
| IMU yaw rate | Dampen excessive rotation |
| Signed odometry speed | Choose acceleration, braking, or recovery |
| Wall LiDAR | Avoid boundaries and anticipate a wall ahead |
| Full LiDAR | Detect nearer obstacles beyond wall-only information |
| Wall contact | Help identify a low-speed obstruction |

Distance calculations cap LiDAR readings at 80 m and treat non-finite readings
as that cap. The controller does not access the private physics world, official
lap progress, or a global track map.

`create_controller()` gives each car a fresh `Controller` instance. It remembers
previous steering, previous emitted throttle, recovery ticks remaining, and
recovery steering. This history matters when learning to imitate its actions.

## Steering and corner speed

Steering combines heading error, near/middle/far lookahead, yaw damping, and
repulsion from close side and diagonal walls. Center correction begins only
beyond 2 m of displacement. Normal steering is bounded to `[-1.0, 1.0]`, with
a maximum change of 0.14 per tick to limit rapid left/right oscillation.

The bend score is the absolute far-to-near lookahead difference plus 0.45 times
the absolute middle-to-near difference. The first matching phase supplies the
normal target speed:

| Phase | Bend score | Absolute heading error | Target speed |
| --- | ---: | ---: | ---: |
| Clear straight | < 1.2 | < 9 degrees | 38 m/s |
| Gentle bend | < 3.2 | < 18 degrees | 36 m/s |
| Medium bend | < 6.5 | < 32 degrees | 31 m/s |
| Severe bend | Otherwise | Otherwise | 23 m/s |

A front wall within `max(3.5, speed_mps * 0.42)` meters lowers the target using
front clearance, bounded between 6 and 18 m/s. An obstacle within 5 m that is
more than 0.25 m nearer than the wall reading caps the target at 12 m/s.

The controller requests full acceleration when more than 1 m/s below target,
full braking when more than 1 m/s above target, and proportional throttle
between those thresholds. These are heuristic targets, not a proven optimal
racing line or a dedicated drift planner.

## Brake release and recovery

Negative throttle arms the simulator's brake-before-reverse state. Switching
directly to positive throttle can leave braking active until the car nearly
stops. Raising the target speed alone does not clear this state.

The expert therefore inserts one tick of **exactly zero throttle** whenever
its requested command changes from negative to positive. This clears the
pending direction change. The next tick can accelerate while the car is still
moving, with steering preserved during the coast tick.

The 6 m/s rolling target also limits normal wall-related slowing. If camera
visibility is lost, the expert steers toward the more open side, requests
gentle braking at or above 6 m/s, and requests acceleration below that speed.
The same brake-release wrapper applies to this fallback.

Reverse recovery starts only when absolute speed is below 2 m/s and either
wall contact is present or the front wall is less than 0.35 m away. It lasts
28 ticks, about 0.47 seconds at 60 Hz, with steering chosen from available
left/right space. Contact during an active recovery does not reset its timer.
Another recovery can start after the timer expires if the obstruction remains.

These rules address unnecessary braking stops; they do not guarantee a
minimum physical speed after collisions, loss of traction, or every bad pose.

## Latest measured results

The following local measurements used the autograder worker's solo trial loop
with an in-process controller, 30 simulated seconds at 60 Hz, and no marshal
recovery. They include the neutral-tick brake-release fix. They are local
physics results, not a new Gradescope submission or isolated-worker validation.

| Metric | Seed 110 | Seed 2026 |
| --- | ---: | ---: |
| Forward progress | 656.613 m | 680.625 m |
| Partial laps | 3.5868 | 3.7179 |
| Completed laps | 3 | 3 |
| First lap | 8.850 s | 8.817 s |
| Best completed lap | 8.000 s | 8.000 s |
| Top speed | 37.588 m/s | 37.589 m/s |
| Damage | 0% | 0% |
| Wall-contact time | 0 s | 0 s |
| Near-stop episodes after startup, without wall contact | 0 | 0 |

The diagnostic counted near-stop episodes as transitions into an absolute
speed below 0.5 m/s with no wall contact, after tick 120. Before the brake-release
fix, the matched trials each had nine such episodes and traveled 502.487 m and
521.487 m. Progress increased by about 31% on each seed after the fix.

These two starts establish a useful baseline, not universal reliability.
Earlier speed-tuning conclusions made before the braking fix should be
retested before choosing new targets.

## Continue with imitation learning

The combined branch now includes an automated expert collector, a small
history-aware cloning model, and autonomous evaluation. See
[Model Training](MODEL_TRAINING.md) for commands, the first trained artifact,
measured comparisons, and remaining work. The sections below explain the
original plan; the implemented first model uses bounded frame history instead
of recurrence, and correction/DAgger collection is still proposed work.

### 1. Freeze and evaluate the teacher

Save the expert source revision and simulator configuration with each dataset.
Test longer runs, held-out starting seeds, and deliberately perturbed positions,
headings, and corner-entry speeds. Add traffic scenarios if the learner will
race against other cars. Developing these perturbation scenarios is part of
the evaluation work; the seed option alone does not supply them.

Measure progress, lap times, damage, contact time, survival, and near-stop
episodes after startup. Prioritize useful recovery behavior before further
increasing top speed: a learner will visit states that clean expert laps miss.

### 2. Record expert trajectories

Build an automated collector around the headless simulation loop. Record the
observation before applying its corresponding action, preserving every 60 Hz
tick, including neutral brake-release ticks:

```text
episode ID, seed, tick, public sensors,
previous applied action, expert action, applied action
```

Keep episode outcomes and version metadata alongside these records. Separate
expert labels from applied actions so the format can later support learner
rollouts and intervention. They coincide during ordinary expert-only collection.

Reuse `robot_sensors_to_dict` and `robot_command_to_dict` from
[`src/racing/game/recording.py`](src/racing/game/recording.py), but give automated
records their own record type. The existing `--record-human` flag is restricted
to manual driving and cannot be combined with `--student-module`.

### 3. Train a history-aware behavioral clone

A model given only the current sensor snapshot cannot always reproduce this
stateful teacher. Start with a small recurrent model receiving relevant public
sensors and the previous applied action. Reset its memory at episode boundaries.
Split complete episodes and scenario groups into training, validation, and test
sets; do not randomly split neighboring ticks.

Use identical observation preprocessing in training and deployment. JSON
recordings encode LiDAR no-hit values as `null`; convert these to finite capped
distances and validity indicators before model input. Fit any normalization
statistics using training data only.

Train steering and throttle prediction on sequences that include corners,
brake releases, and recoveries, rather than allowing full-throttle straights
to dominate sampling. Keep the exact neutral-tick release rule in a deterministic
output wrapper: a predicted throttle of `0.001` is not equivalent to `0.0` for
the simulator's pending braking state. Log requested and applied commands, and
evaluate the model together with this wrapper.

### 4. Evaluate autonomous driving and collect corrections

Compare the learner and teacher on the same held-out scenarios. Prediction
error on recorded actions is only part of validation; let the learner drive
and measure its resulting race performance.

Once basic cloning works, use DAgger: run the learner, query the expert on
the states the learner encounters, aggregate those labeled examples with the
dataset, and retrain. This addresses the changing observation distribution
caused by the learner's own actions. See the
[original DAgger paper](https://proceedings.mlr.press/v15/ross11a.html).

This expert requires special care when queried during learner rollouts. Its
previous-action state must reflect actual applied actions rather than expert
suggestions that were never executed. Define and test how recovery state is
maintained as well; calling the existing controller as an independent shadow
driver is not sufficient to guarantee consistent labels.

The next practical milestone is an automated collector, a small cloning model,
and repeatable autonomous evaluation. Use the observed failures to decide
whether to improve the teacher, dataset coverage, or learner architecture.
