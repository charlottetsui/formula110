# Causal test: terminal penalty on top of the idle penalty (2026-09-01)

Same configuration as `experiments/2026-09-01_idle-penalty-seed909` (seed
909 held fixed, races=10, round_seconds=60, eval_round_seconds=120), with
one further change: `WEIGHT_TERMINAL_PENALTY = 10.0` added to
`src/training/reward.py` -- a one-time penalty applied specifically on the
tick that crosses `NEAR_ELIMINATION_DAMAGE`, on top of the existing
delta-based `WEIGHT_DAMAGE` penalty. Intent: make dying itself
unambiguously costly, to counteract the idle-penalty-vs-elimination
asymmetry hypothesized in the previous experiment's notes.

## Result: zero measurable effect

`eval_results.json` and `metrics.csv` for this run are **byte-identical**
to `2026-09-01_idle-penalty-seed909`'s (diffed directly, 0 differences).
The reward change had no effect on the trained policy whatsoever.

## Why: the terminal penalty never fired during training

Ran a diagnostic: reproduced the exact same self-play training call (seed
909, 10 races, 60s rounds) with a `sensor_sample_callback` logging the
maximum `contact.damage` seen and how many ticks satisfied `is_terminal()`
(`damage >= 0.9`) for each side.

```
max damage seen during training: {'challenger': 0.196, 'incumbent': 0.325}
ticks with is_terminal=True during training: {'challenger': 0, 'incumbent': 0}
```

Damage never got anywhere near the terminal threshold in a 60s training
round -- `WEIGHT_TERMINAL_PENALTY` had literally zero opportunities to
apply. The 100% elimination rate seen in the *2026-09-01_idle-penalty-
seed909* evaluation happens in the 120s eval rounds, which are 2x longer
than any training round has ever been. **The policy has never once
experienced, in training, the conditions that lead to its own elimination
in evaluation** -- it has no gradient signal at all pointing away from
whatever it's doing in the back half of a long race, good or bad.

## What this means

This is a sharper version of the same training/eval scenario-coverage gap
first raised in the 2026-09-01 lap-completion triage, now demonstrated
directly rather than inferred: reward tuning aimed at damage/survival
cannot have any effect on behavior the policy never encounters during
training. A reward fix is necessary but not sufficient here -- training
round length also has to reach into the territory the fix is meant to
govern.

## Decision and rationale

Keep `WEIGHT_TERMINAL_PENALTY` in the reward (it's inert here but harmless,
and should matter once training rounds are long enough for terminal events
to occur). Do not conclude it "doesn't work" -- it was never tested, only
computed and never triggered. The idle-penalty checkpoint from the prior
experiment remains the best candidate from this seed, and it's still
unacceptable (100% elimination rate).

## Next steps (proposed, not yet run)

1. Increase training round length so training rounds actually reach
   elimination-relevant territory -- likely needs to approach or exceed
   the 120s eval length, not just the 60s used so far. This directly
   continues the training-budget line of experiments (`docs/rl_design.md`
   section 6, item 1) but now motivated by a much more specific,
   diagnosed reason than "cover more of the track."
2. Once training rounds are long enough for real terminal events to occur,
   re-evaluate whether `WEIGHT_TERMINAL_PENALTY` actually prevents the
   sprint-to-death behavior -- this experiment did not test that
   hypothesis, it only showed the hypothesis was untestable at the
   current training round length.
