# n-step=5, a regression vs. n=3 (2026-09-07)

Next point on the n-step axis after n=3's clean win (`2026-09-07_nstep3-
seed110/notes.md`). Same seed (110), races=40, round_seconds=120,
buffer_capacity=800000, reward -- only `--n-step 3 -> 5` changed.

## Result: regression on everything except safety

| | n-step=3 | n-step=5 |
| --- | --- | --- |
| avg damage | 0.0000 | 0.0000 |
| avg off-track | 0.000s | 0.000s |
| avg wall-contact | 0.000s | 0.000s |
| avg low-progress | 3.18s (2.7%) | **5.07s (4.2%, worse)** |
| avg max speed | 14.73 m/s | 15.74 m/s |
| avg laps | 4.00 | **1.50** |
| avg best lap time | 25.06s | **59.37s** |
| avg scored distance | 869.8m | **363.5m** |
| eliminated | 0/20 | 0/20 |

Training budget was essentially identical (571,812 vs. 570,114
transitions) so this isn't a training-time confound. Consistent with the
off-policy-staleness/variance tradeoff discussed before running this: a
longer real-reward window (n=5) increasingly reflects a policy several
gradient updates out of date by the time it's replayed, which grows worse
than n=3's tradeoff.

## Decision and rationale

Not adopted. Reverting to n=3 as the reference for further work
(`2026-09-07_nstep3-seed110`). Not continuing further up the n-step axis
(e.g. n=10) without first understanding why n=5 regressed this sharply --
per the earlier discussion, this isn't guaranteed to keep improving
monotonically the way training budget did.
