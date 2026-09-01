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

---

## 2026-09-01 (continued)

**Participants and contributions:** Charlotte Tsui — asked for a triage of
why the car never finishes a race, then approved running the top-priority
"training budget" experiment. Claude Code (AI agent) — ran the triage
(read-only, using existing eval data plus the track's total length), then
implemented and ran the experiment and the pace analysis that followed.

**Question or objective:** Why does the trained controller never finish a
race (complete a lap), and what's the next highest-leverage experiment to
try? Then: run that experiment.

**What we investigated or changed:**

- **Triage (no code changes):** computed the track's total lap length
  (`TrackProgressModel.total_length_m` = 183.07m) and checked `lap_counts`
  across all 20 evaluation races run so far (both 2026-09-01 experiments):
  every single one was `0`. Raw distance per race was only 24-54m (13-30%
  of a lap) in 20s rounds. Concluded the round length itself was the
  dominant constraint, compounded by training rounds being only 15s — self-
  play had likely never given the policy experience of the track beyond a
  short stretch from each spawn point.
- **Experiment:** scaled training budget as one bundled variable (races
  6→10, training round length 15s→60s, buffer capacity raised 50k→150k to
  avoid excess eviction), holding reward weights, network size, and the
  base training seed (`110`) fixed at the `2026-09-01_center-weight-6x`
  values. Also raised eval round length 20s→120s so evaluation could
  actually reveal a completed lap if one happened. Ran via
  `scripts/train_sac.py --races 10 --round-seconds 60 --buffer-capacity
  150000 --eval-round-seconds 120`.
- Updated `src/controllers/sac_candidate.py`'s default checkpoint to this
  run's, and `docs/rl_design.md` §6 item 0 with the result and the
  immediate next step.

**Evidence:**
- Sources or documentation: computed `TrackProgressModel.total_length_m`
  directly rather than assuming a track length.
- AI-agent assistance: Claude Code ran the triage entirely from data
  already on disk (no new experiment needed to answer "why doesn't it
  finish"); after the training-budget run produced eye-catching raw-
  distance numbers, it independently normalized every metric by round
  length before drawing conclusions, which is what caught that average
  pace hadn't actually improved despite raw distance looking like a big
  win — flagged as a "verify, don't just report the flattering number"
  moment worth recording explicitly.
- Commits or code: `scripts/train_sac.py` (no code change, new CLI args
  used), `src/controllers/sac_candidate.py` (default checkpoint updated),
  `docs/rl_design.md` §6 (items 0 and 1 updated).
- Experiment output: `experiments/2026-09-01_scaled-training-budget/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

- Training: 72,000 transitions, 17,751 gradient updates (vs. 10,788 /
  2,448 before), in 55.3s wall-clock (vs. 7.5s) — training time scales
  roughly linearly with total simulated ticks and is still cheap.
- Evaluation raw totals looked like a large win: average raw distance rose
  from 13-30% of a lap to 75-95%, and **one race (seed 2024) completed a
  full lap** — 223m raw distance, 98.6s lap time, the first completed lap
  in any evaluation run so far, reproduced against both baselines for that
  seed.
- Normalizing every metric by round length told a more careful story:
  - Average raw pace (m/s) **did not improve** and slightly regressed:
    1.84 → 1.31 m/s vs `crash_fast`, 1.53 → 1.21 m/s vs
    `default_student_controller`.
  - Off-track fraction **did improve**: ~17-19.5% → ~11.5-11.8% of the
    race.
  - Damage rate (%/s) and marshal rate (interventions/min) were roughly
    flat (damage: 0.095→0.098 %/s; marshal: mixed, 7.2→5.65/min vs
    `crash_fast` but 7.8→7.9/min vs `default_student_controller`).
  - Against `default_student_controller` (~5 m/s sustained, ~600m in
    120s): still 0/10 race wins; the pace gap remains large.
- The raw-distance/lap-completion headline is mostly explained by the
  round simply being 6x longer, not by the car driving faster. The real,
  smaller win is the off-track-fraction improvement, which is at least
  directionally consistent with the triage's "longer training rounds
  expose more of the track" hypothesis. The pace regression is a genuine
  open question, not yet distinguishable from run-to-run noise, because
  **this is a single training run with a single training seed** — no
  repeat trial exists yet for either configuration.

**Decision and rationale:** Adopt this checkpoint as the new reference
point (real off-track improvement, no cost on damage/marshal rate, first
completed lap) but do not yet claim it's a faster controller — the pace
number needs a repeated-seed run before it's trustworthy either way. This
is exactly the kind of number the course notebook guidance warns against
reporting on a single run. Updated `docs/rl_design.md` §6 to make the
repeat-seed test the explicit next step before further scaling.

**Next steps:**
- Repeat this exact training configuration with only the training random
  seed changed, to determine whether the pace figure is signal or noise.
- Watch `controllers.sac_candidate` (now pointed at this checkpoint) in a
  live race to qualitatively check the speed/track-following trade-off
  hypothesis.
- Widen the training seed distribution used for self-play spawns (still
  single base seed `110` as of this entry).
- Once pace is understood, continue scaling training budget further, and
  re-test the reward-weight change now that training budget is less of a
  confound.

---

## 2026-09-01 (continued, 2)

**Participants and contributions:** Charlotte Tsui — asked how to always
run the controller with the latest training updates, and flagged (by
asking) that `controllers.sac_candidate`'s checkpoint path might need to be
re-pointed by hand after every run. Claude Code (AI agent) — found and
fixed the actual gap.

**Question or objective:** Is `uv run racing --seed 110 --student-module
controllers.sac_candidate` always correct for watching the latest trained
controller?

**What we investigated or changed:** `src/controllers/sac_candidate.py`'s
`DEFAULT_CHECKPOINT` was a hardcoded path to one specific experiment
directory, manually updated after each training run (as done in the prior
two entries) — its own docstring claimed it "defaults to the most recent
experiment's checkpoint," which wasn't actually true. Replaced the hardcoded
constant with `_latest_checkpoint()`, which globs
`experiments/*/checkpoints/policy_final.pt` and picks whichever file has
the newest modification time, computed fresh every time the controller
loads (not cached at import time).

**Evidence:**
- Sources or documentation: none beyond the file itself.
- AI-agent assistance: Claude Code verified the fix by calling
  `_latest_checkpoint()` directly (confirmed it resolves to
  `experiments/2026-09-01_scaled-training-budget/checkpoints/policy_final.pt`,
  the newest one on disk) and by loading the controller through
  `load_student_submission("controllers.sac_candidate")` and calling it
  once, before reporting it fixed. Ran `ruff`, `pyright` (strict, 0
  errors), and `pytest -q` (138 passed).
- Commits or code: `src/controllers/sac_candidate.py`.
- Experiment output: n/a (no training run this entry).
- Leaderboard result: n/a.

**What we observed:** The command the user ran was correct as a way to run
the controller, but "always" was not true before this fix — it depended on
a manual edit after every training run that was easy to forget. `uv run
racing --seed 110 --student-module controllers.sac_candidate` is now
actually correct "always" without further action, as long as new
checkpoints get saved under `experiments/<run>/checkpoints/policy_final.pt`
(which `scripts/train_sac.py` already does).

**Decision and rationale:** Auto-discovery by file mtime is simpler and
more reliable than continuing to hand-maintain a pointer, and it was
already what the docstring claimed to do.

**Next steps:** None new — this was a correctness fix, not a modeling
change. Resume with the repeated-seed training-budget run next.

---

## 2026-09-01 (continued, 3)

**Participants and contributions:** Charlotte Tsui — directed the
repeated-seed run to continue improving the model. Claude Code (AI agent)
— ran it, found the result was far more consequential than a noisy pace
number, and analyzed why.

**Question or objective:** Repeat the scaled-training-budget configuration
with only the training seed changed, to check whether the earlier pace
regression (1.84 -> 1.31 m/s) was signal or noise.

**What we investigated or changed:** Ran
`scripts/train_sac.py --races 10 --round-seconds 60 --buffer-capacity
150000 --eval-round-seconds 120 --seed 909` — identical to
`2026-09-01_scaled-training-budget` except `--seed 110` -> `--seed 909`.
Compared full per-race stats (not just scored distance) between the two
runs.

**Evidence:**
- Sources or documentation: none beyond the two experiments' own
  `eval_results.json`.
- AI-agent assistance: Claude Code noticed the headline number
  (avg scored distance 100.7m -> 2.2m) was extreme enough to warrant
  checking *why*, not just recording it as "more noise" — pulled damage,
  off-track, wall-contact, and low-progress per race for both runs before
  writing anything down, which is what surfaced the zero-damage/zero-off-
  track/zero-wall-contact pattern.
- Commits or code: `docs/rl_design.md` §6 (reprioritized, new item 0;
  renumbered items 2-7 -> 3-8).
- Experiment output:
  `experiments/2026-09-01_scaled-training-budget-seed909/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a.

**What we observed:** Not a noisy version of the seed-110 result — a
qualitatively different, worse failure mode. Across all 10 evaluation
races for seed 909: **damage was exactly 0.000, off-track time was exactly
0.0s, and wall contact was exactly 0.0s, every single time.** But
low-progress time averaged 26.8% of each race (vs. 17.4% for seed 110),
avg scored distance was 2.2m (vs. 100.7m), and it lost a race outright
(0/2 vs. `crash_fast` on seed 7, vs. 10/10 wins for the seed-110 policy).
This looks like the policy converged to a "do nothing" local optimum:
standing still or crawling never incurs `WEIGHT_DAMAGE = 5.0`,
`WEIGHT_CONTACT`, or (near spawn) `WEIGHT_CENTER_OFFSET`, and nothing in
the reward penalizes near-zero forward speed specifically
(`WEIGHT_REVERSE` only fires on negative speed) — so freezing can be
locally rational given how the reward is currently weighted. The seed-110
policy apparently avoided this trap and learned to drive with some risk
instead, but we now have no idea how often each outcome occurs (n=2).

**Decision and rationale:** This overrides the read from earlier today
("training budget is the bottleneck, not reward shape") — training budget
was clearly *insufficient on its own* to reliably produce a competent
policy; a second seed produced a policy that is safe but nearly useless.
Reprioritized `docs/rl_design.md` §6: reward risk-asymmetry (specifically,
the lack of any penalty for near-zero speed, and the size of
`WEIGHT_DAMAGE` relative to `WEIGHT_PROGRESS`) is now item 0, ahead of
further training-budget scaling. Not adopting the seed-909 checkpoint
(strictly worse for racing than seed-110's). Not changing the reward yet
without running the proposed causal test first.

**Next steps:**
- Proposed to Charlotte (not yet run): re-run with seed `909` held fixed
  and a targeted reward change (mild near-zero-speed penalty, or a lower
  `WEIGHT_DAMAGE`) to test whether it prevents the freeze -- a clean
  causal test since the seed is held constant.
- Alternative considered: run 2-3 more seeds at the current config first
  to characterize how often each regime occurs, before changing the
  reward. Either is reasonable; asked Charlotte which to prioritize.
- Longer-term: the training-seed-vs-eval-seed overlap noticed while doing
  this analysis (training has used seed `110`, which is also in the fixed
  5-seed evaluation set, in every run so far) should eventually be fixed
  so evaluation seeds are genuinely held out -- not urgent given the
  current findings are about qualitative behavior, not close numeric
  comparisons, but worth fixing before final leaderboard evaluation.

---

## 2026-09-01 (continued, 4)

**Participants and contributions:** Charlotte Tsui — directed continued
refinement of the reward, prioritizing speed and safety. Claude Code (AI
agent) — ran two more single-variable causal tests plus a diagnostic run,
all with training seed `909` held fixed for a clean causal chain, and
found the second test's null result pointed to a training/eval scenario-
coverage gap rather than the reward change being wrong.

**Question or objective:** Fix the "do nothing" freeze found in the
previous entry (seed 909) without sacrificing speed or safety.

**What we investigated or changed:**

- **Causal test 1:** added `WEIGHT_IDLE = 0.2` to `src/training/reward.py`
  (penalizes `abs(speed_mps) < IDLE_SPEED_MPS = 0.5`), leaving every
  safety-side term (`WEIGHT_DAMAGE`, `WEIGHT_WALL_PROXIMITY`,
  `WEIGHT_CONTACT`) untouched on purpose, per the "prioritize speed and
  safety" instruction — the idle penalty targets speed without weakening
  safety incentives. Re-ran with seed `909` held fixed (same
  races/round-length/buffer-capacity as the prior seed-909 run).
- **Causal test 2:** added `WEIGHT_TERMINAL_PENALTY = 10.0` -- a one-time
  penalty on the tick that crosses `NEAR_ELIMINATION_DAMAGE`, on top of
  the existing per-tick `WEIGHT_DAMAGE * damage_delta` term -- to
  counteract a hypothesized exploit where dying early escapes the new
  idle penalty's ongoing cost. Re-ran again with seed `909` held fixed.
- **Diagnostic run:** when causal test 2 produced byte-identical output to
  causal test 1, wrote a one-off script reproducing the exact training
  call with a `sensor_sample_callback` logging max `contact.damage` and
  `is_terminal()` counts per side, to find out why.
- Added `tests/test_training_reward.py` coverage for both new terms
  (idle-speed penalty and its threshold boundary, and the terminal-tick
  penalty).
- Updated `docs/rl_design.md` §6 item 0 with the full causal chain and a
  revised understanding (item 0 and item 1 "training budget" are no
  longer separable).

**Evidence:**
- Sources or documentation: none beyond the experiments' own output.
- AI-agent assistance: Claude Code did not accept the causal-test-2 null
  result as "the fix doesn't work" without checking why -- diffed
  `eval_results.json` and `metrics.csv` between the two runs directly
  (confirmed byte-identical), then wrote and ran the diagnostic script
  before writing any conclusion. This is the same "verify the surprising
  number before reporting it" discipline as the earlier pace-normalization
  finding, applied to a code change instead of a raw metric this time. Ran
  `ruff`, `pyright` (strict, 0 errors), and `pytest -q` (141 passed) before
  and after both reward changes.
- Commits or code: `src/training/reward.py` (`WEIGHT_IDLE`,
  `IDLE_SPEED_MPS`, `WEIGHT_TERMINAL_PENALTY`), `tests/test_training_reward.py`
  (3 new tests), `docs/rl_design.md` §6.
- Experiment output: `experiments/2026-09-01_idle-penalty-seed909/`,
  `experiments/2026-09-01_terminal-penalty-seed909/` (each with
  `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

- Causal test 1 (idle penalty) fixed the freeze cleanly: race wins vs.
  `crash_fast` 5/10 -> 10/10, avg scored distance 2.2m -> 31.8m,
  low-progress time 26.8% -> 4.8%. But it overcorrected: `damage == 1.0`
  (full elimination) in **all 10** evaluation races (vs. 0/10 before), max
  speed up to 11-19 m/s (vs. ~4-5 m/s before). Off-track/wall-
  contact/low-progress times looked *better* in isolation, but that's an
  artifact of eliminated cars stopping mid-race -- those trackers simply
  had less time left to accumulate before the car died.
- Causal test 2 (terminal penalty) produced **zero measurable effect** --
  outputs byte-identical to causal test 1's. The diagnostic run explained
  why: max damage seen during the entire 60s training run was 0.196
  (challenger) and 0.325 (incumbent) -- nowhere near the 0.9 terminal
  threshold. `is_terminal()` was `True` on **zero** ticks during training.
  The policy that reaches full elimination in every 120s evaluation race
  has never once, during training, experienced the conditions that lead
  to its own death.
- This reframes the whole "training budget vs. reward shape" question
  from the earlier entries: they were never actually separable variables.
  A reward term that's supposed to discourage an outcome literally cannot
  do so if training never reaches that outcome -- the fix needs both the
  reward change (already made) and a longer training round (not yet
  tried at a length matching evaluation).

**Decision and rationale:** Keep both `WEIGHT_IDLE` and
`WEIGHT_TERMINAL_PENALTY` in the reward -- the idle penalty is proven to
work as intended, and the terminal penalty is untested (not disproven) and
should matter once training rounds are long enough to reach it. Do not
adopt either seed-909 checkpoint from this entry (both are unacceptable:
one freezes, one dies every race). Updated `docs/rl_design.md` §6 to merge
items 0 and 1 conceptually -- they're the same underlying problem.

**Next steps:**
- Re-run with seed `909` held fixed, current reward (idle + terminal
  penalties), and training round length extended toward or past the 120s
  evaluation length, so the terminal penalty gets a chance to actually
  apply during training. This is the direct, motivated next experiment.
- Once that produces a policy that neither freezes nor dies every race,
  re-check the repeated-seed question again (seed 110) to see if the
  combined fix generalizes, not just works for seed 909.
- Still pending from earlier entries: training/eval seed overlap, held-out
  seed robustness, opponent traffic.

---

## 2026-09-01 (continued, 5)

**Participants and contributions:** Charlotte Tsui — none this entry
(continuation of the same "proceed, continue refining" instruction).
Claude Code (AI agent) — ran the proposed next causal test, found the
hypothesis was wrong, verified it wasn't a simulator bug, and paused the
experiment chain to report back rather than continuing to guess.

**Question or objective:** Test the previous entry's proposed fix: extend
training round length so the terminal penalty (which never fired at 60s
training) gets a chance to actually apply.

**What we investigated or changed:** Re-ran with seed `909` held fixed,
same reward as the previous two causal tests (idle + terminal penalties),
`--round-seconds 60 -> 120` (matching the eval length),
`--buffer-capacity 150000 -> 200000`. Directory:
`experiments/2026-09-01_longer-training-round-seed909/`.

**Evidence:**
- Sources or documentation: read `reset_robot_vehicle` in
  `src/racing/race/runtime.py` to check whether the extreme speed reading
  could be a marshal-reset artifact.
- AI-agent assistance: before writing up the 40.5 m/s figure as a real
  finding, Claude Code checked whether it could be a bug — read the reset
  code directly rather than assuming, and confirmed it correctly zeroes
  linear/angular velocity and clears forces on every reset, ruling out a
  spurious velocity-spike explanation. Ran `ruff`/`pyright`/`pytest`
  unchanged (no code changed this entry, only experiment configuration).
- Commits or code: `docs/rl_design.md` §6 item 0 (added causal test 3 and
  a revised, hedged hypothesis).
- Experiment output:
  `experiments/2026-09-01_longer-training-round-seed909/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a.

**What we observed:** The hypothesis was wrong. Elimination rate stayed
at 10/10 (unchanged from the 60s-training run), and average max speed got
*worse* — 40.5 m/s, versus 15.9 m/s at 60s training and ~6.9 m/s for the
seed-110 reference. Off-track/wall-contact/low-progress times all dropped
to under 1% of the race, consistent with crashing almost immediately
(probably flooring the throttle in a straight line). Self-play's own
in-training distance was large this time (853-913m per 120s race, vs.
~0m in every earlier run's printed self-play total) — the training
process itself looks like it's exploring real, fast racing behavior, but
the frozen deterministic evaluation policy is far more extreme and
dangerous than what training self-play showed. Best current guess: a
train/eval behavior mismatch, where the deterministic (no exploration
noise) action at evaluation time diverges from what was actually sampled
and experienced during noisy training — not confirmed.

**Decision and rationale:** Do not adopt this checkpoint (worse than the
prior one on the metric that matters most for safety). Do not continue
guessing at reward/config tweaks for seed 909 without more diagnosis --
four experiments deep into this causal-test chain (idle penalty, terminal
penalty, longer training round, this one) is a reasonable point to pause
and get direction rather than keep iterating blind. The **best checkpoint
from all of today's work remains `2026-09-01_scaled-training-budget`**
(training seed 110): 0/10 eliminations, ~6.9 m/s max speed, 10/10 wins vs.
`crash_fast`, first completed lap. `controllers.sac_candidate` currently
auto-selects today's newest (and most dangerous) checkpoint by file time --
flagged this explicitly so it isn't watched by accident.

**Next steps (options, not yet decided):**
1. Investigate the stochastic-training-vs-deterministic-eval mismatch
   directly -- e.g. compare the entropy/spread of sampled training actions
   against the deterministic eval action at similar states, or evaluate
   with `deterministic=False` (sampled, like training) instead of the mean
   action to see if the extreme behavior is eval-only.
2. Try seed 110 (the "good" seed) through the same idle+terminal reward
   change, to see whether the safety regression is specific to seed 909's
   particular training trajectory or a general property of the new reward
   terms.
3. Step back from single-seed causal tests and run a small sweep (3-5
   seeds) at the current best-known config to get a real sense of the
   outcome distribution, rather than continuing to chase one seed's
   specific pathology.
4. Consider a hard action/speed cap at the controller level (not just a
   reward penalty) as a safety backstop independent of what the reward
   teaches -- reward shaping alone has now twice produced an unsafe
   policy for this seed.
