# Fine-tuning against the expert: a 6-variant dose/resume-mode sweep (2026-09-10)

Follow-up to this directory's main `notes.md` (training against the expert
from scratch was a severe regression). Per direction to keep fine-tuning
until a positive change is found, tested resuming the reference checkpoint
(`2026-09-08_seed8000-resumed-short`) and continuing training against the
expert for a small number of additional races, at several doses and two
resume modes.

## Resume modes

- **Full resume** (`--resume-from`, existing flag): loads policy, critics,
  and the entropy temperature `log_alpha` from the checkpoint. The
  checkpoint's `log_alpha` had already converged to a near-zero value
  (~0.035, i.e. almost no exploration noise) from its own (self-play)
  training.
- **Policy-only resume** (`--resume-policy-only`, new flag; see
  `SACAgent.load_policy_only` in `src/training/sac.py`): loads only the
  actor's weights, leaving critics and `log_alpha` at their fresh
  `__init__` values (`alpha` starts at 1.0). Added specifically because
  the full-resume attempts below failed catastrophically and fast,
  consistent with a policy that has almost no exploration budget left
  having no way to recover once it starts making mistakes against an
  unfamiliar, faster opponent.

## Results (all vs. the reference: avg_damage 0.0004, 0/20 eliminated,
6.75 avg laps, 15.54s avg lap time, 30.33 m/s avg max speed, 1.702s avg
car-contact, 20/20 wins vs. both `crash_fast` and
`default_student_controller`)

| variant | avg damage | eliminated /30 | avg laps | avg lap time | avg max speed | avg car-contact |
| --- | --- | --- | --- | --- | --- | --- |
| full-resume, dose=2 | 0.9667 | 29 | 1.30 | 18.36s | 37.94 m/s | 0.456s |
| full-resume, dose=10 | 0.9948 | 27 | 0.20 | 15.79s | 38.71 m/s | 0.126s |
| policy-only, dose=2 | 0.4988 | 14 | 0.40 | 64.48s | 25.96 m/s | 1.777s |
| policy-only, dose=10 | 0.1391 | 1 | 2.00 | 48.41s | 16.07 m/s | 3.906s |
| policy-only, dose=20 | 0.3540 | 1 | 3.07 | 32.07s | 22.29 m/s | 3.265s |
| policy-only, dose=40 | 0.1545 | 2 | 3.87 | 27.22s | 20.41 m/s | 4.326s |

Wins vs. the two standard baselines: full-resume variants lost badly to
both (consistent with the from-scratch collapse in the parent `notes.md`).
Policy-only dose=20 and dose=40 both swept 20/20 against `crash_fast` and
`default_student_controller` -- matching the reference's win record on
those two baselines. All variants lost every race (0/10) against
`leaderboard_expert` itself, which averages ~5300m of scored distance per
race regardless of who it's racing.

## Read

**Full resume is not viable at any dose tested.** The checkpoint's
near-zero exploration budget means it has no way to correct course once
the unfamiliar expert-opponent distribution starts producing mistakes --
both doses collapse to ~90-97% elimination within a handful of races,
matching (not improving on) the from-scratch failure this was meant to
fix.

**Policy-only resume is a qualitatively different, much more stable
regime** -- elimination stays low (1-14 out of 30) across every dose
tried, vs. 27-29 for full resume. Within this regime there's a genuine,
mostly-monotonic dose/response trend as races increase from 2 to 40: damage
and lap time both improve substantially, laps completed rises, and by
dose=20 it's already matching the reference's win record against the two
standard baselines. This is real progress relative to the from-scratch and
full-resume failures.

**But even at dose=40 (the same total race count as the original
from-scratch run), no variant closes the gap to the reference, and the
best-performing variant regresses on the exact metric this whole line of
experimentation was meant to improve:** avg car-contact time at dose=40
(4.326s) is *higher* than the reference's (1.702s), not lower -- the
opposite of the original goal (teaching real opponent-avoidance). Every
other tracked metric (damage, laps, lap time, max speed) is also still
meaningfully worse than the reference at every dose tested, with no sign
of crossing over even as the trend improves.

## Decision and rationale

**Not adopted; the reference checkpoint (`2026-09-08_seed8000-resumed-
short`) and `race_faster.py` remain unchanged.** Across 7 tested variants
(1 from-scratch run + 2 full-resume doses + 4 policy-only doses), nothing
has produced a checkpoint that matches, let alone beats, the reference --
and the metric motivating the entire combined-approach exploration
(opponent-collision time) is worse, not better, in the best candidate
found. Continuing to chase larger doses looks like it would keep
approaching the reference asymptotically at best, for a mechanism that
has not yet, at any tested point, actually helped the one thing it was
for.

**Recommendation: pause this specific direction** (training/fine-tuning
against a fixed expert opponent, in either resume mode) rather than
running further doses by default. `--opponent expert` and
`--resume-policy-only` are kept in code (both default off / require
explicit opt-in) as tested, documented, working mechanisms, consistent
with this track's convention of preserving negative results rather than
deleting the code that produced them.

## Next steps

1. If revisited: a **mixed-opponent curriculum** (alternating self-play
   and expert-opponent races within one run, rather than switching
   entirely to the expert after an initial self-play phase) is still
   untested and structurally different from every variant tried here --
   it might avoid ever fully leaving the self-play distribution rather
   than trying to recover speed after leaving it.
2. If revisited: doses beyond 40 races could be tried to see whether the
   improving trend eventually crosses the reference, but this is
   increasingly expensive for an uncertain payoff given the trend has
   not yet helped the original target metric (car-contact time) at any
   point tested.
3. Otherwise, treat the reference checkpoint as the best available and
   redirect combined-approach effort elsewhere (e.g. the shielded/
   safety-override idea discussed earlier, which doesn't carry any of
   this training-instability risk since it doesn't retrain the policy at
   all).
