# SAC Track — Laboratory Notebook

Chronological log of the SAC exploration/refinement track for the Formula
110 COMP 590H project. One entry per substantive work session, added during
or immediately after the session rather than reconstructed later. This
notebook covers the SAC track only — see the project team split for why the
second (PPO) approach is tracked independently.

Entry template (also duplicated in [`CLAUDE.md`](../CLAUDE.md) so it's
followed consistently):

```
## YYYY-MM-DD HH:MM

**Participants and contributions:**

**Question or objective:**

**What we investigated or changed:**

**Evidence:**
- Sources or documentation:
- AI-agent assistance:
- Commits or code:
- Experiment output:
- Leaderboard result, if applicable:

**What we observed:**

**Decision and rationale:**

**Next steps:**
```

---

## 2026-08-31

**Participants and contributions:** Charlotte Tsui — requested and
reviewed the SAC design/documentation scaffold. Claude Code (AI agent) —
read the existing simulator contract, drafted `docs/rl_design.md`,
`experiments/README.md`, this notebook, and `CLAUDE.md`.

**Question or objective:** Before writing any training code, establish (a)
a concrete SAC design grounded in what the simulator's public API actually
allows, and (b) the documentation infrastructure (lab notebook, experiment
evidence convention, per-session process doc) the course requires to be
maintained throughout the project.

**What we investigated or changed:**

- Read `README.md`, `GETTING_STARTED.md`, and `SENSORS.md` for the
  controller runtime contract, sensor fields, and CPU/memory constraints.
- Read `src/racing/race/head_to_head.py` in full to confirm the public
  headless training entry point (`run_headless_head_to_head`) and its
  parameters (`sensor_sample_callback`, `copies_per_side`,
  `copy_for_car`).
- Confirmed there is no public single-car step API and no access to
  official lap progress, the physics world, or another controller's
  private state — only `run_headless_head_to_head`, which runs full
  races and returns aggregate `HeadToHeadResult` stats.
- Confirmed `sensors.tick == 0` is a documented, public new-episode
  marker (`SENSORS.md`), usable as a `done` signal substitute.
- Checked repo state: no `docs/`, `experiments/`, or `src/training/`
  existed yet in this checkout — this session created the doc/evidence
  scaffolding described here; no training code was written.
- Wrote `docs/rl_design.md`: MDP formulation (observation vector, action
  space, proxy reward, episode-boundary detection), the self-play/
  controller-as-agent training architecture that works within the
  public-API constraint, dependency plan (`torch`+`numpy`, no
  `gymnasium`/`stable-baselines3`), the exploration-stage minimum
  experiment, and a refinement plan.
- Wrote `experiments/README.md` describing the per-run evidence directory
  convention (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/`, `notes.md`).
- Wrote `CLAUDE.md` at the repo root with the explicit steps to follow
  every session on this track, including updating this notebook.

**Evidence:**
- Sources or documentation: `README.md`, `GETTING_STARTED.md`,
  `SENSORS.md`, `src/racing/race/head_to_head.py`, `pyproject.toml`.
- AI-agent assistance: Claude Code read the simulator source directly
  (rather than assuming API shape) before proposing the self-play
  architecture; the "controller is its own RL agent" design and the
  `tick == 0` done-signal were both verified against actual source, not
  inferred.
- Commits or code: none yet (documentation only this session).
- Experiment output: none yet — no training code exists.
- Leaderboard result: n/a.

**What we observed:** The simulator's public surface genuinely has no
step-by-step single-car API — `run_headless_head_to_head` is the only
headless entry point, and it blocks for a full race. This rules out a
conventional external `for step in range(...)` training loop and confirms
the design has to put the RL agent (observation building, reward
computation, replay buffer push, and periodic gradient step) inside the
controller's own `__call__`, driven by repeated self-play races.

**Decision and rationale:** Adopt the design in `docs/rl_design.md`:
self-play (`challenger` and `incumbent` both pointing at the same
in-training policy/shared replay buffer via `copy_for_car`), a proxy
reward built from public sensor fields (since official lap progress is
private), and `sensors.tick == 0` as the episode-boundary signal. This is
the only architecture found that trains SAC using exclusively the public
API — it doesn't touch private/underscore-prefixed simulator internals,
which the project's non-goals explicitly rule out.

**Next steps:**
- Implement `src/training/` (replay buffer, actor/critic networks, SAC
  update step, `TrainableController`) per `docs/rl_design.md` §3.
- Run the minimum experiment from §5 (fixed 5-seed evaluation set) and
  record results under `experiments/`.
- Compare against `crash_fast` baseline on scored distance, elimination
  rate, and cross-seed consistency; add a lab notebook entry with that
  evidence before deciding whether to continue investing in SAC.

---

## 2026-08-31 (continued)

**Participants and contributions:** Charlotte Tsui — directed the
implementation and requested the branch/commit/next-steps workflow. Claude
Code (AI agent) — implemented `src/training/`, `scripts/train_sac.py`, unit
tests, fixed lint/type errors, and ran the first real training/evaluation
pass.

**Question or objective:** Implement the SAC design from `docs/rl_design.md`
end to end and get first evidence of whether it works: does the self-play
plumbing actually run, and does the resulting policy beat a trivial
baseline consistently across the fixed evaluation seeds?

**What we investigated or changed:**

- Committed the prior session's docs/CLAUDE.md scaffold on a new
  `charlotte-rl` branch.
- Added `torch` and `numpy` as project dependencies (`uv add torch numpy`);
  registered `training` as a top-level package in
  `[tool.uv.build-backend] module-name` (`pyproject.toml`) alongside
  `racing`/`controllers` so `import training` resolves the same way the
  rest of the project does.
- Implemented `src/training/`: `observation.py` (17-dim feature encoding
  from `RobotSensors`), `reward.py` (proxy reward + episode-boundary
  detection), `replay_buffer.py` (fixed-capacity circular buffer),
  `sac.py` (tanh-Gaussian policy, twin critics, twin targets, auto entropy
  temperature — hand-rolled on `torch`, no `gymnasium`/`stable-baselines3`,
  per `docs/rl_design.md` §4), and `controller.py`
  (`TrainableController` + shared `TrainingState`, implementing
  `RobotController` and `copy_for_car`).
- Added `scripts/train_sac.py`: runs one self-play `run_headless_head_to_head`
  call to train, then evaluates the frozen policy against
  `controllers.crash_fast` across the fixed 5-seed set, writing
  `config.yaml`/`metrics.csv`/`eval_results.json`/`checkpoints/` under
  `experiments/<dir>/` per `experiments/README.md`.
- Added unit tests (`tests/test_training_*.py`) for observation encoding,
  reward shaping, the replay buffer, the SAC agent (bounded/repeatable
  actions, finite losses, weights actually change on update,
  save/load round-trips), and the controller (no push on first call, one
  push per subsequent call, near-elimination damage marks a transition
  done, an eval controller never writes to the buffer, `copy_for_car`
  shares learning state but not per-car episode history, warmup actions
  stay random). All run without a real headless race, matching how
  `tests/test_head_to_head.py` avoids exercising the real Panda3D/physics
  stack in unit tests.
- Added a `[[tool.pyright.executionEnvironments]]` override scoped to
  `src/training` relaxing the three `reportUnknown*` strict-mode rules
  (`pyproject.toml`) — torch's public API resolves to `Unknown` through
  many call chains the same way panda3d/ursina already do elsewhere in
  this project even with `reportMissingTypeStubs = false`; real-bug rules
  (wrong attributes, bad returns, unreachable code) stay enforced.
- Ran `scripts/train_sac.py` with default hyperparameters (6 self-play
  races, 15s each, hidden size 128) and the fixed 5-seed evaluation set.

**Evidence:**
- Sources or documentation: re-read `src/racing/race/head_to_head.py` while
  implementing to confirm exact `HeadToHeadResult`/`HeadToHeadRaceResult`
  shapes (a first draft of `scripts/train_sac.py` incorrectly assumed the
  aggregate `HeadToHeadResult` exposed `.challenger`/`.incumbent`; pyright
  caught this — see "what we observed").
- AI-agent assistance: Claude Code wrote all of `src/training/`,
  `scripts/train_sac.py`, and the new tests; verified correctness by
  running `uv run ruff check`, `uv run pyright`, and `uv run pytest -q`
  (138 passed) before treating anything as done, and by hand-running small
  smoke scripts (SAC update step on random data, `TrainableController`
  against synthetic `RobotSensors`) before wiring into a real race.
- Commits or code: `src/training/__init__.py`, `observation.py`,
  `reward.py`, `replay_buffer.py`, `sac.py`, `controller.py`;
  `scripts/train_sac.py`; `tests/test_training_*.py` (5 files);
  `pyproject.toml` (deps + build-backend module list + pyright override).
- Experiment output: `experiments/2026-08-31_sac-minimum-experiment/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a — this is a local self-play checkpoint, not yet
  packaged as a `src/controllers/` submission.

**What we observed:**

- The original design doc assumed `contact.damage >= 1.0` was observable a
  tick early via the damage delta. It isn't: `run_headless_head_to_head`
  stops calling an eliminated car's controller, and the damage that
  crosses `1.0` is applied *after* the last tick the controller actually
  sees. The exact terminal reading is never delivered in-band. Corrected
  the design to use a `NEAR_ELIMINATION_DAMAGE = 0.9` threshold as an
  approximate terminal signal instead — see the updated
  `docs/rl_design.md` §2.4 for the full explanation. This is exactly the
  kind of assumption-vs-reality gap the design doc flagged as a risk in
  §1; worth stating outright since it changed committed code, not just a
  parameter.
- A first draft of the evaluation/logging code referenced
  `HeadToHeadResult.challenger`/`.incumbent`, which don't exist on the
  aggregate result type (only on each per-race `HeadToHeadRaceResult`).
  `pyright` (strict mode) caught this as a real type error before the
  script ever ran against real data — fixed by summing
  `team_sum_distance_m` across `result.races`.
- Self-play training was fast: 10,788 transitions and 2,448 gradient
  updates in 7.6s wall-clock on a small (128,128) MLP, on CPU. Training
  time is not a bottleneck at this scale.
- `crash_fast` (full throttle, no steering) scores exactly `0.0 m` on
  every seed in the fixed evaluation set — it crashes almost immediately,
  and scored distance excludes contact time. It's a real floor, not a
  strong baseline; beating it is a low bar, not evidence of a competitive
  controller.
- The SAC-trained controller won every evaluation race (2/2) on all 5
  fixed seeds after under 8 seconds of training, with scored distances
  from 41.0 m to 80.5 m. Consistent across all 5 seeds, not one lucky
  start.

**Decision and rationale:** The self-play/`TrainableController`
architecture from `docs/rl_design.md` works end to end on the real
simulator, not just in isolated unit tests — training runs, the policy
updates, and the resulting controller is evaluable via the same public
`run_headless_head_to_head` API used for the leaderboard. That's enough
evidence to keep investing in this approach rather than revisit the design.
The near-elimination-threshold correction is adopted as documented, since
the alternative (waiting for an unobservable exact signal) isn't available
from the public API at all.

**Next steps:**
- Re-run against a stronger baseline than `crash_fast` — likely
  `racing.student.api.default_student_controller`, which actually tries to
  stay on-track — since beating `crash_fast` alone doesn't demonstrate
  competitiveness.
- Track proxy reward vs. real scored distance over a longer training run
  to check for the divergence risk flagged in `docs/rl_design.md` §2.3.
- Widen the training seed distribution (currently a single seed, `110`,
  drives all self-play spawns) so the policy doesn't overfit to one set of
  starting positions; evaluate on seeds never used in training.
- Start on the refinement plan in `docs/rl_design.md` §6, roughly in the
  order listed there (reward-shaping/divergence check first).

---

## 2026-09-01

**Participants and contributions:** Charlotte Tsui — requested a triage of
why the trained controller wasn't driving well, then directed two follow-up
experiments and asked how to watch the controller drive. Claude Code (AI
agent) — ran the triage (read-only), then implemented and ran both
experiments and a local-dev viewer controller.

**Question or objective:** Triage why the 2026-08-31 checkpoint doesn't
drive well on the track (requested read-only first, no changes), then run
two follow-up experiments: (1) a stronger baseline than `crash_fast`, since
"beats crash_fast" was suspected to be a weak signal, and (2) a
single-variable reward reweight targeting the root cause the triage found.
Also: how to watch the trained controller drive, for the presentation.

**What we investigated or changed:**

- **Triage (read-only, no code changes):** pulled per-race stats from the
  2026-08-31 `eval_results.json` and read `src/racing/race/runtime.py`.
  Found the trained controller spent 7.5-27% of each 20s race off-track,
  needed marshal recovery 2-3 times per race, and had substantial wall
  contact and low-progress time. Traced two root causes in the code:
  `WEIGHT_CENTER_OFFSET` (0.05) is 20x smaller than `WEIGHT_PROGRESS`
  (1.0) in `src/training/reward.py`, and there's a real sensing gap — the
  drivable-surface threshold is `TRACK_WIDTH/2` (3.3 m) but the
  marshal-reset threshold is 3.3 + `TRACK_EDGE_BUFFER` (4.7 m), so there's
  a ~1.4 m runoff band where the car is off-track with no wall nearby to
  trigger `wall_lidar`/`contact.wall`.
- **Experiment 2 (baseline change, no retraining):** added
  `racing.student.api.default_student_controller` as a second evaluation
  baseline. Refactored the shared evaluation logic out of
  `scripts/train_sac.py` into `src/training/evaluation.py`
  (`evaluate_against_baselines`, `BASELINE_CONTROLLERS`) so both
  `train_sac.py` and a new `scripts/eval_sac.py` (evaluates an existing
  checkpoint with no training) use the same code. Re-evaluated the
  2026-08-31 checkpoint against both baselines with `eval_sac.py`.
- **Experiment 1 (reward reweight, retrained):** raised
  `WEIGHT_CENTER_OFFSET` 0.05 -> 0.3 in `src/training/reward.py` (the only
  changed variable), re-ran `scripts/train_sac.py` with identical
  hyperparameters/seed to the 2026-08-31 baseline.
- **Viewer:** added `src/controllers/sac_candidate.py` (loads a saved
  checkpoint, deterministic inference, `FORMULA110_SAC_CHECKPOINT` env var
  to pick a checkpoint) so the trained controller can be watched via the
  normal `uv run racing` / `h2h --watch` CLI. Explicitly documented as not
  submission-packaged yet (imports `training`, which isn't included by
  `scripts/export_student_controllers.py`).

**Evidence:**
- Sources or documentation: `src/racing/race/runtime.py`
  (`_track_projection_is_off_track`, `_track_projection_is_outside_drivable_surface`,
  `maybe_marshal_race_runtimes`, `RACE_OFF_TRACK_RESET_DISTANCE_M`,
  `TRACK_EDGE_BUFFER`, `TRACK_WIDTH`) for the triage; re-read
  `src/racing/student/api.py` to confirm `default_student_controller`'s
  signature for use as a baseline.
- AI-agent assistance: Claude Code did the triage read-only per explicit
  instruction before any code changed; verified every new/changed file
  with `uv run ruff check`, `uv run pyright` (strict, 0 errors), and
  `uv run pytest -q` (138 passed) before running experiments; smoke-tested
  `controllers.sac_candidate` by loading it through
  `load_student_submission` and calling it once before recommending it be
  watched live.
- Commits or code: `src/training/reward.py` (weight change + rationale
  comment), `src/training/evaluation.py` (new, shared eval logic),
  `scripts/train_sac.py` (refactored to use it), `scripts/eval_sac.py`
  (new), `src/controllers/sac_candidate.py` (new), `docs/rl_design.md`
  (§5, §6 updated).
- Experiment output:
  `experiments/2026-09-01_original-vs-default-baseline/` (re-eval of the
  2026-08-31 checkpoint against both baselines) and
  `experiments/2026-09-01_center-weight-6x/` (retrained with the new
  reward weight, evaluated against both baselines).
- Leaderboard result: n/a.

**What we observed:**

- **Experiment 2 result:** against `crash_fast`, the 2026-08-31 checkpoint
  reproduces its original numbers exactly (62.0/57.3/41.0/80.5/48.6 m,
  confirming `eval_sac.py`'s deterministic re-evaluation is correct).
  Against `default_student_controller`, the same checkpoint **loses every
  race on every seed** (0/10), scoring 16.3-67.2 m versus the baseline's
  92.5-106.9 m. "Beats crash_fast" was not evidence of a competitive
  controller — it was evidence of beating a controller that doesn't try.
- **Experiment 1 result (negative):** the reward reweight produced no
  measurable improvement. Averaged over 10 races against `crash_fast`:
  scored distance 28.9m -> 24.8m, off-track time 18.2%->17.1%, wall contact
  3.12s->2.97s, marshal count 2.30->2.40 — statistically indistinguishable
  given the sample size, and if anything slightly worse on distance. Same
  pattern against `default_student_controller`. This was a well-controlled
  single-variable comparison: both runs used the same base spawn seed
  (`110`) and the same default SAC network initialization seed (`0`), so
  everything except the reward's center-offset weight was identical by
  construction, not just by matching CLI flags.
- Best current interpretation of the negative result: at only ~2,448
  gradient updates (most spent in warmup), the policy likely hasn't had
  enough updates to exploit *any* reshaped incentive yet. Training budget
  looks like the current bottleneck, not reward shape — see
  `docs/rl_design.md` §6 item 0.

**Decision and rationale:** Keep the `WEIGHT_CENTER_OFFSET = 0.3` change
(a reasonable prior, not measurably worse) but do not draw further
conclusions about reward shaping until training budget is scaled up as its
own separately-measured variable — reprioritized `docs/rl_design.md` §6 to
put training budget first. Keep both baselines
(`crash_fast`+`default_student_controller`) as the standard evaluation
going forward via `training.evaluation.evaluate_against_baselines`, since
the single-baseline comparison was shown to be misleading.

**Next steps:**
- Scale up training budget (more/longer self-play races) as an isolated
  experiment, same reward and hyperparameters otherwise, to test whether
  it's really the bottleneck.
- Re-test reward shaping once training budget is no longer the confound.
- Watch `controllers.sac_candidate` in a live `h2h --watch` race for a
  qualitative read (does it look like it's cutting corners, stalling after
  wall contact, etc.) to sanity-check the quantitative off-track numbers.
- Widen the training seed distribution (still single-seed as of this
  entry) — unchanged from the 2026-08-31 next-steps item.
