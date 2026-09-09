# First expert clone — 2026-09-08

See [MODEL_TRAINING.md](../../MODEL_TRAINING.md) for commands, implementation,
interpretation, corrections and next steps.

One 128×128 behavioral clone, trained for 80 epochs with initialization seed 0,
using expert trajectories from training seeds 1001–1010 and validation seeds
2001–2002. The deployed mean network has 59,906 parameters. No RL reward or
SAC optimization was used. Frozen inference uses NumPy and the exact neutral
brake-release wrapper.

The clone matched/slightly exceeded expert distance in the first five-seed
head-to-head comparison, but incurred damage on two seeds. In local subprocess
solo trials on seeds 110 and 2026, both models completed three laps without
damage/contact; the clone traveled 663.794 m and 682.878 m compared with the
teacher's 656.613 m and 680.625 m. This is promising initial evidence, not
cross-track, multi-training-seed or Linux isolated-worker validation.

- `config.json`, `metrics.csv`: exact split, settings and epoch losses.
- `policy.npz`: selected frozen mean network, also copied to
  `src/controllers/imitation_policy.npz`.
- `actor.pt`: corresponding PyTorch actor; log-standard-deviation head is
  untrained, and SAC noise/critics/observation integration still needs work.
- `collection_metadata.json`, `collection_race_results.json`: expert dataset
  provenance and demonstration outcomes; raw dataset is in ignored
  `artifacts/imitation-v1-data`.
- `eval_results.json`: authoritative matched head-to-head rerun with fresh
  expert factory state for every car/race.
- `eval_initial_state_reuse.json`: superseded first evaluation; bare expert
  instances could retain state across races. Retained to explain the correction.
- `solo_results.json`: autograder local subprocess trial outputs.
- `source_manifest.json`: explicitly post-run hashes, including later validation
  and provenance improvements; not a training-start source snapshot.

Next: investigate contact cases, broaden scenarios and train more independent
seeds before deciding that the clone is ready for SAC refinement.
