# Genuine-init seed sampling: seed 10000 -- near-record pace, dramatically cleaner safety (2026-09-08)

Part of a second batch of 5 fresh genuine-init seeds (9000, 10000, 11000,
12000, 13000) sampled specifically looking for a checkpoint matching or
beating the pace of `2026-09-08_seed-sweep-v2-8000` (14.96s avg lap time)
without its documented near-miss (0.5942 damage in 1/20 races). Same
locked-in config as every other genuine-init sweep run (n_step=3,
obstacle lidar, no robot-proximity term, `WALL_WARNING_DISTANCE_M`
reverted to 6.0, races=40, round_seconds=120, buffer_capacity=800000).

## Result: didn't beat the raw lap-time record, but is dramatically safer at nearly the same pace

| | champion (seed 8000) | seed 10000 |
| --- | --- | --- |
| avg best lap time | 14.96s | 16.70s (+12%) |
| avg laps | 7.30 | 6.80 |
| avg max speed | 26.7 m/s | **35.4 m/s (higher!)** |
| avg damage | 0.0298 | 0.0008 |
| **max damage (worst race)** | **0.5942** | **0.0154 (38x smaller)** |
| avg off-track | 0.15s | 0.17s |
| avg wall-contact | 0.11s | 0.06s |
| eliminated | 0/20 | 0/20 |

Per-race detail: 18 of 20 races have exactly 0.0000 damage, 6-7 laps
every time, and a remarkably consistent ~35-36 m/s max speed across
literally every race (far tighter variance than the champion's). The
only two blips are trivial: `seed=8675309 vs crash_fast race=1` had a
2.10s off-track excursion with zero damage or wall contact (recovered
cleanly), and `seed=7 vs default_student_controller race=2` had a tiny
0.0154 damage graze (1.27s off-track, 1.23s wall-contact, 1 marshal).
Neither is remotely comparable to the champion's near-elimination-level
event.

## Read

This checkpoint reaches a *higher* raw top speed than the current
champion (35+ vs 26.7 m/s) but a slightly slower lap time -- consistent
with the pattern established much earlier this project (top speed and
lap pace are not the same thing; this checkpoint likely corners more
conservatively despite faster straights). It doesn't strictly beat the
14.96s record, but it directly answers the follow-up question raised
after adopting seed 8000: is there a checkpoint with comparable pace and
a meaningfully better safety margin? This one clearly qualifies -- a 12%
pace cost for a worst-case-damage reduction of roughly 38x.

## Decision and rationale

Not automatically adopted -- presenting as a genuine alternative for
Charlotte's decision, the same way seed 8000 itself was presented rather
than swapped in unilaterally. Whether the ~1.7s/lap pace difference is
worth trading for the much safer worst case is a judgment call.

## Next steps

1. Awaiting direction: adopt seed 10000 in place of seed 8000, keep
   seed 8000, or continue sampling for something that beats 14.96s
   outright without a near-miss.
2. If adopted, repackage `controllers.race_faster` from this checkpoint.
