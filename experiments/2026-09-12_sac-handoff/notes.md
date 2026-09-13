# Prepared SAC initialization — September 12, 2026

Selected model: round-2 seed 1, retained after the additional recovery imitation
pass failed to improve overall performance. This is a competent baseline for
controlled SAC initialization/critic warmup, not a fully trained SAC agent or
a guarantee of arbitrary-track/recovery performance.

- `initial_sac.pt`: mean actor preserved, small constant exploration variance,
  fresh critics/targets, alpha initialized to 0.01.
- `actor.pt`: compatible actor with initialized variance head.
- `manifest.json`: configuration, hashes, action/data contracts and phase status.
- `final_test.json`: six fresh layouts (301–306), two 60-second races each.
  Selected clone and expert had no damage, wall contact or eliminations.
- `exploration_check.json`: deterministic and stochastic inference checks on
  three scenarios; 12 races without damage/eliminations. Recorded-observation
  mean parity error below 2.84e-7. No policy optimization was performed.

Load using `training.imitation_handoff.load_prepared`, which restores architecture
and settings as well as weights. `HandoffController` supplies the correct history
and brake-release wrapper. The original 17-input SAC training loop is incompatible.
The replay/terminal handling and critic-warmup stage must be implemented before
actor updates; see the explicit contract in [MODEL_TRAINING.md](../../MODEL_TRAINING.md).

Severe repeated disturbances still cause failures, and the actual race-day
track generator is unknown. Synthetic layout tests use the existing fixed
track width, sensors and vehicle physics.
