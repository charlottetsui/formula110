# Training against a fixed expert opponent instead of self-play: a severe regression (2026-09-10)

Tests "approach 1" from the combined-approach discussion: instead of
self-play (challenger and incumbent both `TrainableController` instances
sharing one policy/buffer), the incumbent is a frozen
`controllers.leaderboard_expert.Controller` instance -- only the
challenger learns/pushes transitions. Motivation: the SAC track's own
causal tests 21-23 found reward-shaping attempts at competitor-avoidance
repeatedly failed or backfired; practicing against a genuinely different
driving style (rather than a mirror of itself) was hypothesized to teach
real avoidance instead. Implemented as a new `--opponent {self,expert}`
flag on `scripts/train_sac.py` (default `self`, preserving all existing
behavior/reproducibility). Same seed/races/round-length/n-step/hidden-size
as the reference (`2026-09-08_seed-sweep-v2-8000`) -- `--opponent` is the
only new variable.

## Result: catastrophic, not an improvement

| | reference (self-play) | expert-opponent (this run) |
| --- | --- | --- |
| avg damage | 0.0298 | **1.0000 (100%)** |
| eliminated | 0/20 | **30/30 (every race)** |
| avg laps | 7.30 | **0.63** |
| avg off-track | 0.15s | 1.89s |
| avg wall-contact | 0.11s | 0.69s |
| avg car-contact (opponent) | 1.21s | **0.27s (lower)** |
| avg max speed | 26.7 m/s | 37.3 m/s |
| avg best lap time (of races with a lap) | 14.96s | 16.85s |
| wins vs crash_fast | 20/20 | 10/10 |
| wins vs default_student_controller | 20/20 | **2/10** |
| wins vs leaderboard_expert (its own training opponent) | n/a | **0/10** |

Every single one of 30 evaluated races (5 seeds x 3 baselines x 2 races)
ended in full elimination (`average_damage=1.0`). It also lost to
`default_student_controller` (previously beaten 20/20) and to the exact
expert it trained against, badly (its own scored distance 168-610m per
seed vs. the expert's 5300-5450m). Notably, avg car-contact (opponent
collision time) is *lower* than the reference, not higher -- so this
isn't the opponent-collision failure mode from causal tests 21-23
recurring. The car crashes into walls/off-track at very high speed
instead.

## Diagnosis

Training itself only collected 111,472 transitions over 40 races x 120s
-- far short of the ~288,000 a full-duration single-learner run would
produce (40 * 120 * 60), meaning the challenger was frequently getting
eliminated *during training too*, cutting races short well before the
round timeout (the simulator stops calling an eliminated car's
controller). This is roughly 1/5 of the reference run's 575,840
transitions (which came from *two* learning cars sharing one buffer, so
some reduction is expected, but not this much).

`metrics.csv` shows real training instability, not smooth convergence:
critic loss swings widely and trends upward over training (3.5 -> 0.34 ->
10.7 -> 48.7 -> 18.0 at even checkpoints through the run, vs. a
well-behaved run's loss settling down), while the entropy temperature
(`alpha`) collapses hard (0.9997 -> 0.03-0.04) -- the policy became
confidently deterministic very early, before the critic had anything
reliable to be confident about.

**Likely root cause:** self-play's incumbent co-evolves with the
challenger throughout training -- both start equally unskilled and
improve together, so the "opponent difficulty" is always roughly matched
to the challenger's current skill. A fixed, already-competent expert
(cruising at 23-38 m/s from tick zero) does not co-evolve -- for most of
training, the challenger is a much weaker driver sharing a track with a
much faster, unpredictable-to-it opponent from the very first race. That
plausibly biases the *entire* training distribution toward "recovering
from being near/passed by a fast car" rather than "driving a clean solo
lap," and towards reckless, high-speed, low-exploration behavior trying
to keep pace (consistent with the alpha collapse and the much higher
crash-inducing top speed) -- never giving the policy enough of the calm,
opponent-free experience that produced the reference checkpoint's
clean driving. This is a training-distribution-shift problem, not a
reward-function bug or a code bug in the new `--opponent` plumbning
(unit-tested and smoke-tested separately before this run; the harness
correctly restricts `--opponent expert` to `--copies-per-side 1` and
correctly adds `leaderboard_expert` as an eval baseline).

## Decision and rationale

**Not adopted.** `--opponent` defaults to `self`; `race_faster.py` and
the reference checkpoint (`2026-09-08_seed8000-resumed-short`) are
unchanged. This directly contradicts the hypothesis that a fixed,
different-style opponent would straightforwardly improve
competitor-awareness -- at least trained from a fresh/random
initialization for the full run, it produced a uniformly worse and less
safe policy on every tracked metric except opponent-collision time
specifically (which did improve, but at the cost of total collapse
everywhere else).

## Next steps

1. The literal "always train against the expert from scratch" approach is
   not viable as tested -- do not make `--opponent expert` the default.
2. A **curriculum** variant is untested and more consistent with what
   actually differs here: fine-tune an already-good self-play checkpoint
   (e.g. `--resume-from` the current reference) against the expert for a
   *small* dose of races, rather than training from random init against
   it for the full run -- mirrors the project's own successful pattern
   (causal test 32's "+10 races beats both 0 and +40") of small,
   post-hoc nudges to an already-competent policy rather than changing
   the entire training trajectory from the start.
3. A **mixed-opponent** variant (alternate self-play and expert-opponent
   races within one run, rather than 100% either way) is also untested
   and would avoid committing the whole training run to the harder
   distribution before the policy has any competence at all.
4. If either of those is tried, watch `alpha` and `critic_loss` in
   `metrics.csv` during training (not just final eval) -- this run's
   instability was visible well before the final eval confirmed it, and
   could serve as an early stopping signal in future attempts.
