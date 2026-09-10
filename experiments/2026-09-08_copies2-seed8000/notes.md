# Increased self-play traffic (copies_per_side=2) -- regression, not a fix (2026-09-08)

Directed to try increasing self-play traffic (`--copies-per-side`) as a
training-procedure lever for reducing hesitation/car-contact time,
distinct from the three reward-shaping attempts that already failed this
session (`WEIGHT_STEERING_REVERSAL` untested, `WEIGHT_ROBOT_PROXIMITY`
unrestricted and angle-restricted, both regressed). Trained fresh on
seed 8000 (same network initialization as the current best checkpoint's
origin) with `--copies-per-side 2`, otherwise identical config
(n_step=3, obstacle lidar, no robot-proximity, races=40,
round_seconds=120, buffer_capacity=800000, eval_round_seconds=120) --
compared against `2026-09-08_seed-sweep-v2-8000` (copies_per_side=1).

Note: doubling copies_per_side roughly doubles transitions collected per
race (more simultaneous cars generating data), so this run filled its
800,000-transition buffer and ran 287,110 gradient updates -- about 2x
the ~143,000 updates typical of a copies_per_side=1 run at races=40, and
took roughly 2x the wall-clock time (899.8s vs ~450s). This is a
confound worth noting: the comparison below isn't purely "more traffic
diversity," it's also "roughly double the effective training."

## Result: a clear, uniform regression on pace, minimal hesitation improvement

| | copies=1 (reference) | current best (fine-tuned) | copies=2 |
| --- | --- | --- | --- |
| avg low-progress | 2.83s | 3.30s | 3.23s (flat) |
| avg car-contact | 1.21s | 1.70s | 1.38s (modest improvement) |
| avg laps | 7.30 | 6.75 | **3.00** |
| avg best lap time | 14.96s | 15.54s | **32.95s** |
| avg max speed | 26.7 m/s | 30.3 m/s | **12.0 m/s** |
| avg damage | 0.0298 | 0.0004 | 0.0019 |
| max damage | 0.5942 | 0.0047 | 0.0378 |
| eliminated | 0/20 | 0/20 | 0/20 |

Per-race detail: every single one of 20 evaluation races landed at
exactly 3 laps and ~11.8-12.4 m/s max speed -- remarkably uniform, not a
skewed average hiding a few bad races. Car-contact time is highly
variable race to race (0.10s-4.83s) without a clear systematic
improvement pattern.

## Read

Denser self-play traffic did not teach better collision avoidance --
it produced a much more uniformly cautious, slow policy across the
board, with only a modest (~19%) reduction in average car-contact time
that doesn't come close to justifying the pace cost. Given the
transitions/gradient-update confound noted above, this result can't
cleanly separate "more opponent traffic helps avoidance" from "roughly
double the effective training budget pushes toward the same
more-conservative equilibrium seen elsewhere this session" (e.g. causal
test 11's races=40->80 regression, and the seed 1000/2000 resume
experiments) -- but either way, this specific configuration is not an
improvement.

## Decision and rationale

Not adopted. `2026-09-08_seed8000-resumed-short` remains the reference
checkpoint. Given this is now the fourth attempt this session (after
three reward-shaping ones) to reduce hesitation/car-contact time and
regress pace instead, the evidence increasingly suggests the current
checkpoint's modest hesitation level (2.8% of race time, no severe
outlier) may be close to a practical floor for this reward/observation/
training setup, rather than an easily-fixable gap.

## Next steps

1. If self-play traffic is revisited, control for the gradient-update
   confound by reducing `--races` proportionally when increasing
   `--copies-per-side` (e.g. races=20 at copies=2, to roughly match
   copies=1's transition count at races=40) for a cleaner comparison.
2. Otherwise, treat the current checkpoint's hesitation level as
   acceptable and consider this thread closed for now.
