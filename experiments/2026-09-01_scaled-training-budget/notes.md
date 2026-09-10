# Scaled training budget: longer rounds + more races (2026-09-01)

Follow-up to the 2026-09-01 triage finding that the car had never completed
a lap (183m) in any evaluation race, and that 15s training rounds meant
self-play never experienced most of the track from a given spawn point.

## What changed

Single bundled variable ("training budget", `docs/rl_design.md` §6 item 0),
vs. `experiments/2026-09-01_center-weight-6x` (same reward weights, same
network size, same base seed `110`, same warmup/batch/update-frequency):

| | center-weight-6x (prior) | scaled-training-budget (this run) |
| --- | --- | --- |
| training races | 6 | 10 |
| training round length | 15s | 60s |
| replay buffer capacity | 50,000 | 150,000 (raised only so the larger run doesn't evict early data; not a hypothesis variable) |
| transitions collected | 10,788 | 72,000 |
| gradient updates | 2,448 | 17,751 |
| training wall-clock | 7.5s | 55.3s |
| eval round length | 20s | 120s (raised so evaluation can actually show a completed lap if one happens) |

## Result

Raw totals look like a big win: average raw distance per race went from
~24-54m (13-30% of the 183m lap) to ~140-175m (75-95% of a lap), and one
race (seed 2024) **completed a full lap** (223m, lap time 98.6s) — the
first completed lap in any evaluation run so far.

**But pace, normalized by round length, tells a more honest story:**

| | center-weight-6x (20s rounds) | scaled-training-budget (120s rounds) |
| --- | --- | --- |
| avg raw pace vs crash_fast | 1.84 m/s | 1.31 m/s |
| avg raw pace vs default_student_controller | 1.53 m/s | 1.21 m/s |
| avg off-track fraction (crash_fast races) | 17.1% | 11.8% |
| avg off-track fraction (default races) | 19.5% | 11.5% |
| avg damage rate | 0.095 %/s | 0.098 %/s (flat) |
| avg marshal rate (crash_fast races) | 7.2/min | 5.65/min |
| avg marshal rate (default races) | 7.8/min | 7.9/min (flat) |

Most of the raw-distance gain is explained by the round simply being 6x
longer, not by the car driving faster. Off-track fraction did improve
meaningfully (~18% -> ~12%); damage rate and marshal rate are roughly flat.

Against `default_student_controller` (a real baseline, ~5 m/s sustained,
~600m in 120s): still 0/10 race wins, and the pace gap (1.21 vs ~5 m/s) is
still large.

## What we observed

- First-ever completed lap in evaluation (seed 2024, both baselines,
  ~99-106s lap time) — real evidence the controller *can* complete a lap
  under the right conditions, not yet evidence it does so reliably.
- Off-track time improved as a fraction of the race; this is consistent
  with the "longer training episodes force the policy to encounter more of
  the track, including corners its previous 15s training rounds never
  reached" hypothesis from the triage.
- Average forward pace did not clearly improve, and if anything went down
  slightly. Plausible explanations: (a) real trade-off -- more careful
  driving costs some speed but reduces off-track excursions, consistent
  with the off-track-fraction improvement; (b) the newly-reached, previously
  -unseen track sections are genuinely harder and drag the average down;
  (c) pure run-to-run noise -- **this is a single training run with a single
  seed**, so we cannot yet distinguish signal from noise on the pace number.

## Decision and rationale

Adopting this checkpoint as the new reference point (it's a real
improvement on off-track behavior with no cost on damage/marshal rate, and
demonstrates a completed lap for the first time), but explicitly **not**
declaring it definitively faster, since the pace regression could be noise
from a single run. The next experiment should repeat this same
configuration with only the training random seed changed, to check whether
the pace difference reproduces.

## Next steps

- Repeat with a different training seed (only variable changed) to check
  whether the pace change is signal or noise.
- Watch `controllers.sac_candidate` (now pointed at this checkpoint) in a
  live race to see qualitatively whether it's driving more cautiously
  through specific corners, consistent with the off-track/pace trade-off
  hypothesis.
- Once pace is understood, continue scaling training budget further --
  lap completion being reachable at all is new evidence this is worth
  pursuing further, not a finished result.
