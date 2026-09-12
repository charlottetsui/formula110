# Layout-diverse imitation round 2

Audited and completed September 12, 2026 after the interrupted session. The
directory's original date is preserved. See [MODEL_TRAINING.md](../../MODEL_TRAINING.md)
for commands, selection details and the SAC handoff.

- 180,159 samples; 17 training layouts, four validation layouts, six held-out
  test layouts. Both geometry and spawn seeds are saved in `collection.json`.
- Three independent 80-epoch fits: `seed-0`, `seed-1`, `seed-2`.
- `selection.json` selected seed 1 using driving validation before final tests.
- Seed 1 had no damage/contact/eliminations on six test layouts. Other training
  seeds had failures; do not describe the training procedure as uniformly robust.
- `stress.json` exposed severe forced-disturbance failures, despite good normal
  traffic performance. A later corrective pass was attempted and rejected.
- Current deployed policy is seed 1; its SHA-256 is
  `d73047e19c5c0b85cf385af6f55c365accd8266a7041c3a69f787a3b8e741af7`.

The sample dataset is `artifacts/imitation-v2-data/dataset.npz`; raw trajectories
are in the same ignored directory. These are imitation labels, not a ready-made
SAC replay buffer. No reward optimization was performed.
