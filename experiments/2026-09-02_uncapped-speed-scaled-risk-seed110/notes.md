# Uncap speed, scale wall-proximity risk by speed (2026-09-02)

Directed: "the car is currently too safe... find a way to increase
throttle/speed... implement drifting mechanisms around curves." A literal
hand-coded drift controller isn't compatible with the self-play/SAC
architecture (that would be rule-based control fighting the learned
policy, and out of this track's scope). Instead: removed the flat,
context-blind `MAX_REWARDED_SPEED_MPS` cap entirely (progress reward is
now uncapped), and made the wall-proximity penalty scale with current
speed (`WALL_PROXIMITY_SPEED_SCALE_MPS = 10.0`, so proximity penalty
doubles at 10 m/s, triples at 20 m/s, etc.) instead of being speed-blind.
Intent: price risk by how dangerous the *current situation* actually is
(fast + close to a wall), not by a flat speed number that treats "close
to a wall at walking pace" the same as "close to a wall at 35 m/s" -- and
leave room for the policy to discover aggressive, slide-through-the-apex
cornering if that's actually faster in this physics model, rather than
forbidding any speed above an arbitrary line. Same seed (110), races=40,
round_seconds=120, buffer_capacity=800000 as every prior comparison
today, trained from scratch.

## Result: genuinely different from all seven prior attempts -- a real trade-off, not a clean win or a regression

| | races=40 reference (flat cap=10.0) | uncapped + speed-scaled risk |
| --- | --- | --- |
| avg max speed | 15.53 m/s | **17.13 m/s (+10%)** |
| avg damage | 0.000 | 0.003 (still ~perfect) |
| eliminated | 0/20 | **0/20 (unchanged)** |
| avg off-track | 0.000s | 0.082s (small, still near-perfect) |
| avg wall contact | 0.000s | 0.076s (small, still near-perfect) |
| avg marshal/race | 0.15 | 0.25 (still very low) |
| avg laps | 4.10 | 3.90 (slightly fewer) |
| avg best lap time | 24.76s | **28.46s (slower)** |
| wins (both baselines) | 20/20 | 20/20 (unchanged) |

This is the **first of eight speed-focused experiments today (including
2026-09-01's) to achieve a genuine, meaningful increase in top speed
without regressing safety** -- every previous attempt either broke safety
outright (cap=20.0, trajectory-bonus buggy) or preserved safety only by
also reducing speed (cap=12.0, more training via two different paths,
steering smoothness, trajectory-bonus fixed). This one moved max speed up
while keeping damage/eliminations/off-track/wall-contact all still
excellent (not literally perfect anymore, but close).

The catch: that extra top speed didn't translate into a faster or more
complete race. Laps and lap time both got slightly worse. The likely
read: the policy is reaching a higher peak speed somewhere (plausibly on
straights, where the now-uncapped progress term rewards it and wall
proximity is a non-issue), but isn't yet converting that into a better
overall lap -- possibly spending more time modulating speed up and down,
or being more conservative through corners specifically now that the
proximity penalty bites harder there at any given speed. This can't be
distinguished from headless stats alone; would need a live watch (or
per-tick sensor logging via `sensor_sample_callback`) to see whether it's
actually attempting anything drift-like through corners, or just going
faster on the straights without changing cornering technique.

## Decision and rationale

Provisionally adopting this as the new reference point, since it's the
first speed-focused change all day to move the needle in the requested
direction (raw speed) without breaking safety -- but flagging explicitly
that it is not an unambiguous improvement over the prior best on overall
race performance (laps/lap time), only on top speed specifically. Whether
to prefer this or the old races=40 checkpoint depends on what's being
optimized: raw speed capability favors this one; lap time/consistency
still favors the old reference.

## Next steps (proposed)

1. **(recommended)** Train further under this new reward structure
   specifically. Every previous "more training" experiment was under the
   old flat-cap reward, where more training provably converged speed
   *down* toward the cap (causal tests 11 and 16) -- that mechanism no
   longer applies here since there is no cap to converge toward. More
   training under an uncapped, risk-scaled reward is a genuinely
   different question, not a repeat of an already-answered one.
2. Watch it live (`controllers.sac_candidate`, pointed at this
   checkpoint) to see qualitatively whether it's attempting anything
   drift-like through corners, or just carrying more speed on straights.
3. If lap time doesn't improve with more training either, consider
   loosening `WEIGHT_CENTER_OFFSET` (currently penalizes any lateral
   deviation from centerline uniformly) alongside this change -- a real
   racing line, let alone a drift, requires deviating from centerline
   through corners, which the current reward still penalizes regardless
   of whether that deviation is genuinely faster.
