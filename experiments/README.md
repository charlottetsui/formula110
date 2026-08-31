# Experiment evidence convention

Every SAC training or evaluation run that's worth citing in
[`docs/lab_notebook.md`](../docs/lab_notebook.md) gets its own directory
here:

```
experiments/<YYYY-MM-DD>_<short-slug>/
  config.yaml        # hyperparameters, observation/reward weights, seeds, git commit
  metrics.csv         # per-episode or per-gradient-step training metrics
  eval_results.json   # HeadToHeadResult.to_dict() output from evaluation seeds
  checkpoints/         # saved policy weights, if any (gitignored if large)
  notes.md             # what this run was for, what happened, one paragraph
```

Conventions:

- `config.yaml` must record enough to reproduce the run: the evaluation
  seed set, `round_seconds`, `race_count`, `copies_per_side`, reward
  weights, network sizes, and the git commit the run was made from.
- `eval_results.json` reports the simulator's own public race stats
  (`HeadToHeadTeamRaceStats` / `HeadToHeadResult`), not just the training
  proxy reward — see `docs/rl_design.md` §2.3 on why both are tracked.
- Name the slug after what changed (e.g. `2026-09-03_wall-penalty-2x`),
  not the algorithm — this directory is SAC-only, so the algorithm name
  isn't distinguishing information.
- Large checkpoint binaries: keep only the best/most recent checkpoint per
  run if size becomes a problem; do not commit training-only datasets or
  virtualenvs here (same rule as the top-level packaging contract in
  `README.md`).
- Every run directory should be referenced from at least one
  `docs/lab_notebook.md` entry under that entry's "Evidence" section.
