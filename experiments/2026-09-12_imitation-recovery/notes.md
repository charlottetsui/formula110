# Rejected corrective imitation continuation — September 12, 2026

Objective: address second-round seed 1's failures under repeated steering/braking
disturbances and denser, slower traffic.

Collected 46,949 additional samples on training layouts and extra default-track
starts, using seed 1 for learner blocks and the applied-action-aware teacher for
labels. Combined with 180,159 prior samples, then warm-started from seed 1 for
60 epochs with sampling seed 3. Whole validation layouts remained excluded.

Ordinary driving validation remained close to the expert, but forced-disturbance
stress eliminations increased from two to four. Default-track recovery improved
while radial/stretch cases worsened. The candidate was **not installed**, and
was not used for the new final test suite. Keep this evidence as a failed
experiment, not as the latest/best model merely because it was trained later.

`collection.json`, `dataset_sources.json`, `seed-3/config.json`, `validation.json`
and `stress.json` retain the evidence. Commands are described in MODEL_TRAINING.md.
The raw combined data is `artifacts/imitation-v2-recovery/combined.npz`.
