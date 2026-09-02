# Building the Geometry Expert

This guide explains the first rule-based expert controller in this project. It
assumes you have not built a racing controller before.

The controller is in
[`src/controllers/geometry_expert.py`](src/controllers/geometry_expert.py). Its
goal is reliability rather than maximum racing speed. Later, its decisions can
be recorded as demonstration labels for an imitation-learning model.

A faster, separately tunable version is in
[`src/controllers/geometry_expert_fast.py`](src/controllers/geometry_expert_fast.py).
It retains the original controller as a reliability baseline instead of
replacing it.

## 1. Understand the controller loop

The simulator calls the controller 60 times per simulated second. On every
call, it supplies a `RobotSensors` snapshot. The controller must return exactly
one `RobotCommand` containing:

- `throttle`: `-1.0` for reverse through `1.0` for forward
- `steer`: `-1.0` for full left through `1.0` for full right

The expert uses a `Controller` class and `create_controller()` rather than a
single `control()` function. This gives every car fresh recovery state at the
start of a race, as required by the packaging guidance in `README.md`.

## 2. Choose public observations

The expert deliberately reads only documented public sensors:

| Sensor | How the expert uses it |
| --- | --- |
| `camera.center_offset_m` | Steers toward the track center |
| `camera.heading_error_degrees` | Turns to match the track direction |
| `camera.lookahead_offsets_m` | Anticipates the next bend |
| `odometry.speed_mps` | Accelerates or brakes toward a target speed |
| `wall_lidar` | Keeps space from walls and detects a blocked path |
| `contact.wall` | Starts recovery after touching a barrier |

This matters for future imitation learning: the student model can receive the
same observations that caused the expert's command. The expert does not import
the private physics world, official lap progress, or exact global track map.

## 3. Compute steering

The normal steering command combines four small corrections:

1. **Center correction:** move toward the centerline.
2. **Heading correction:** point in the local direction of the track.
3. **Lookahead correction:** begin turning toward upcoming centerline points.
4. **Wall correction:** move away from a close side wall.

These signals share the same sign convention: negative means left and positive
means right. Their sum is clamped to `[-0.9, 0.9]`, leaving the controller
responsive without using full steering during ordinary driving.

The numerical multipliers in the source are tuning parameters, not universal
constants. They express how strongly each observation influences steering.

## 4. Select a safe speed

The expert starts with a maximum target speed of 4.4 m/s. It lowers that target
when:

- the car has a large heading error;
- the car is far from the centerline;
- the near and far lookahead offsets indicate a bend; or
- the front wall reading is becoming short.

It then compares target speed with measured speed. A positive error produces
forward throttle. A sufficiently negative error produces negative throttle,
which brakes a forward-moving car before requesting reverse, as described in
the project README.

This target-speed design is easier to understand and tune than assigning a
fixed throttle to every situation.

## 5. Recover from walls

A one-tick reverse command is often too short to free a stuck car. The expert
therefore enters recovery for 42 ticks (about 0.7 simulated seconds) when it
touches a wall or sees a wall less than 0.65 m ahead.

At recovery start, it compares open space on the left and right, chooses a turn
direction, and keeps a moderate reverse command for the entire recovery. The
state is bounded: it cannot grow indefinitely, and a new controller starts with
no old race history.

This behavior is particularly useful for a future demonstration dataset because
it supplies examples of mistakes and recoveries, not only ideal centerline
driving.

## 6. Handle LiDAR no-hit readings

The sensor reference documents `math.inf` as a valid LiDAR no-hit value. The
expert converts it to a finite 30 m cap before doing arithmetic. This prevents
expressions such as infinity minus infinity and follows the same preprocessing
principle that a future ML model should use.

## 7. Run the expert

Install the project as described in `GETTING_STARTED.md`, then run:

```bash
uv run racing \
  --seed 110 \
  --student-module controllers.geometry_expert
```

Try several seeds because each seed selects a deterministic starting position:

```bash
uv run racing --seed 7 --student-module controllers.geometry_expert
uv run racing --seed 42 --student-module controllers.geometry_expert
uv run racing --seed 110 --student-module controllers.geometry_expert
```

Reusing a seed makes before-and-after tuning comparisons fair.

To compare it with another automated controller over several headless races:

```bash
uv run racing h2h \
  --challenger-module controllers.geometry_expert \
  --incumbent-module controllers.crash_fast \
  --seed 110 \
  --races 7 \
  --round-seconds 30
```

Head-to-head mode includes other cars, so it tests more than basic solo track
following. Watch a race when diagnosing behavior; use repeated headless races
when collecting comparative evidence.

## 8. Test and tune responsibly

The focused tests in `tests/test_geometry_expert.py` check that the controller:

- turns toward right-hand track geometry;
- requests less throttle for a sharp bend;
- continues recovery after contact ends; and
- always returns commands within the documented ranges.

Run them with:

```bash
uv run pytest tests/test_geometry_expert.py
```

When tuning, change one group of constants at a time and rerun the same seeds.
Track at least survival time, distance traveled, damage, wall contacts, and
whether the car becomes stuck. A future "expert" should first be consistent and
recoverable; speed can be improved after those properties are dependable.

## 9. Prepare for imitation learning later

The next project stage can record pairs of:

```text
(public RobotSensors snapshot, expert RobotCommand)
```

across many seeded starts. Keep complete runs together when splitting training
and evaluation data. Do not randomly split adjacent 60 Hz ticks, because nearby
ticks are almost duplicates and would make test results look better than real
generalization.

## 10. Use the faster expert

Run the faster variant with:

```bash
uv run racing \
  --seed 110 \
  --student-module controllers.geometry_expert_fast
```

The faster controller preserves the same public-sensor and stateful-recovery
design, but prioritizes leaderboard progress in its normal driving behavior:

- maximum target speed rises from 4.4 m/s to 9.6 m/s;
- it uses the full `1.0` forward command on a clear straight;
- it can use strong negative throttle to brake late for a corner;
- center offsets smaller than 1.15 m do not reduce speed by themselves;
- lookahead bend, heading error, steering demand, and front-wall distance cause
  anticipatory braking; and
- steering and side-wall corrections are slightly stronger at higher speed.

These changes are intended to increase speed without deleting the dependable
expert. Keep both versions so every future tuning change can be evaluated
against the same conservative baseline.

In a deterministic 15-race, 45-second comparison against the reliable expert,
the fast expert traveled 2,905.5 m compared with 1,317.9 m—about 120% farther.
It completed one lap in every race, reached 9.4 m/s, and recorded 0% average
damage, no eliminations, and zero wall-contact time. Both controllers received
five marshal resets after symmetric car-contact events.

An additional 10-race evaluation with seed `2026` produced 10 lap completions,
a 42.75-second best lap, 0% damage, no wall contact, no eliminations, and one
marshal reset. These results demonstrate a substantial improvement for the
tested starts, but they are not proof for every possible state. Continue
evaluating new seed sets before treating the controller as a final
demonstration source or leaderboard submission.

## 11. Use the boundary-prioritized expert

For a leaderboard-oriented experiment that gives up strict centerline tracking,
run:

```bash
uv run racing \
  --seed 110 \
  --student-module controllers.boundary_expert_fast
```

This is a separate controller in
[`src/controllers/boundary_expert_fast.py`](src/controllers/boundary_expert_fast.py),
so the two earlier experts remain available as baselines.

The boundary expert uses camera heading and lookahead offsets to determine where
the track turns, but ignores the first 1.7 m of center offset. Its hard driving
constraints instead come from:

- side and diagonal `wall_lidar` beams, which produce progressively stronger
  steering only near a boundary;
- the front wall beam, which causes late braking only when clearance becomes
  genuinely short;
- full LiDAR, which distinguishes cars and blockers from the wall-only reading;
  and
- contact-triggered bounded recovery, retained as a last resort.

It targets 10.7 m/s and keeps at least a 6.6 m/s target through severe geometry
unless the front clearance requires braking. This is intentionally a more
aggressive expert: progress and lap time take priority over holding a neat
centerline.

In a matched 15-race comparison, it produced a 21.87-second best lap versus
41.93 seconds for `geometry_expert_fast`, completed 25 laps versus 15, and
traveled 5,523.6 m versus 2,924.3 m. That is approximately 1.9 times the lap
throughput and distance, while both controllers recorded 0% damage, no wall
contact, and no eliminations in the test.

On a separate 10-race sequence using seed `2026`, the boundary expert completed
20 laps, recorded a 21.65-second best lap, reached 10.6 m/s, and had zero
damage, wall contact, eliminations, or marshal resets. The vehicle's observed
top speed is around 10–11 m/s, so a literal two- or three-times increase over
the previous 9.4 m/s top speed is not physically available. The roughly 1.9x
gain comes from maintaining high speed through the lap instead.

These measurements cover deterministic simulator starts, not every possible
opponent interaction or perturbed pose. Treat this controller as the fast-data
expert and retain the earlier controllers as fallbacks and comparison points.
