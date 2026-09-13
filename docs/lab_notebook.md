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

---

## 2026-09-01 (continued, 6)

**Participants and contributions:** Charlotte Tsui — directed the
investigation of the stochastic-vs-deterministic hypothesis specifically.
Claude Code (AI agent) — implemented the diagnostic capability, ran it,
got a clean rejection, and caught its own earlier misreading of a number
in the process.

**Question or objective:** Test directly whether the previous entry's
dangerous checkpoint (40.5 m/s, 100% elimination) behaves that way only
under deterministic (mean-action) evaluation, or whether it's equally
dangerous when actions are sampled stochastically, the way training
itself chooses them.

**What we investigated or changed:**

- Added a `deterministic: bool | None = None` parameter to
  `TrainableController.__init__` (`src/training/controller.py`),
  decoupling action-selection determinism from the `training` flag
  (previously hard-coded as `deterministic = not training`).
  `training.evaluation.evaluate_against_baselines` got a matching
  `deterministic: bool = True` parameter so existing callers are
  unaffected by default.
- Added 3 tests (`tests/test_training_controller.py`): deterministic
  action selection is repeatable for identical sensors, an explicit
  `deterministic=False` override samples differently each call even with
  `training=False`, and `copy_for_car` preserves the override.
- Wrote `scripts/compare_stochastic_eval.py`: loads a checkpoint, runs
  `evaluate_against_baselines` twice (deterministic and stochastic) across
  the same seeds/baseline, and reports a side-by-side summary (avg damage,
  avg max speed, avg scored distance, elimination rate).
- Ran it against the causal-test-3 checkpoint
  (`2026-09-01_longer-training-round-seed909`, the 40.5 m/s / 100%-
  elimination one) — no retraining, just re-evaluating the existing
  weights two ways.

**Evidence:**
- Sources or documentation: none beyond the checkpoint and code already on
  disk.
- AI-agent assistance: Claude Code verified the new code with
  `ruff`/`pyright` (strict, 0 errors)/`pytest -q` (144 passed) before
  running the diagnostic. After getting the result, it re-checked its own
  prior reasoning (the "self-play's own distance looked reasonable" claim
  from the previous entry) against the new numbers instead of just
  reporting the rejection and moving on — found that claim was based on
  not dividing a printed total by the race count, and corrected it
  explicitly rather than leaving the earlier entry's reasoning
  uncorrected.
- Commits or code: `src/training/controller.py`, `src/training/evaluation.py`,
  `tests/test_training_controller.py` (3 new tests),
  `scripts/compare_stochastic_eval.py` (new), `docs/rl_design.md` §6.
- Experiment output:
  `experiments/2026-09-01_stochastic-vs-deterministic-diagnosis/`
  (`eval_results_deterministic.json`, `eval_results_stochastic.json`,
  `summary.json`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** The hypothesis is rejected, cleanly. Deterministic
vs. stochastic evaluation of the identical checkpoint: avg max speed 40.46
vs. 40.02 m/s, avg damage 1.000 vs. 1.000, elimination rate 100% vs. 100%,
avg scored distance 91.3 vs. 94.5m — essentially identical across the
board. Sampling actions with training-style exploration noise produces the
same extreme-speed, always-eliminated behavior as the deterministic mean
action. This also resolved a loose end from the previous entry: self-play
training's printed total ("913.4m over 10 races") was never evidence of
safe behavior — 913.4 / 10 = 91.3m/race, which matches the eval averages
almost exactly. The training process itself has been producing this same
crash-after-a-fast-burst pattern all along; it just wasn't checked
per-race at the time.

**Decision and rationale:** Close the eval-mode-artifact line of
investigation — the danger is substantively learned, not a deployment-time
quirk, so tuning inference-time noise/temperature would not fix it.
Updated `docs/rl_design.md` §6 with a new leading (untested) hypothesis:
`forward_progress_m` in the reward has no upper bound on speed, so nothing
structurally stops the policy from valuing ever-higher speed, and the
one-time `WEIGHT_TERMINAL_PENALTY = 10.0` may just be smaller than the
cumulative reward from a fast enough burst before crashing.

**Next steps:** Options unchanged from the previous entry, plus one new,
more specific candidate:
1. **(new, current leading hypothesis)** Test capping/saturating the speed
   term in `forward_progress_m` so reward stops scaling linearly with
   speed above some threshold, removing the structural incentive to push
   speed indefinitely.
2. Try the current reward (idle + terminal penalties) on seed 110 (the
   "good" seed) to see if this failure is seed-909-specific or general.
3. Run a small seed sweep (3-5 seeds) at the current reward to see the
   real outcome distribution.
4. Add a hard action/speed cap at the controller level as a safety
   backstop independent of reward shaping.

---

## 2026-09-01 (continued, 7)

**Participants and contributions:** Charlotte Tsui — directed testing the
speed-cap hypothesis and re-evaluating. Claude Code (AI agent) —
implemented it, got another null result, and this time worked out a
quantitative explanation rather than just a qualitative one.

**Question or objective:** Test the previous entry's new leading
hypothesis: does capping the reward benefit of speed (removing the
incentive to exceed some threshold) stop the seed-909 policy from driving
at 40 m/s and dying every race?

**What we investigated or changed:** Added `MAX_REWARDED_SPEED_MPS = 10.0`
to `src/training/reward.py`; the speed used in `forward_progress_m` is now
`copysign(min(abs(speed_mps), 10.0), speed_mps)` instead of the raw,
unbounded `speed_mps`. Chose 10.0 as above both the only zero-elimination
checkpoint's speed range (seed 110, ~6.9 m/s avg max) and the competent
heuristic baseline (~5 m/s), but well below the 15-40+ m/s crash regime.
Added 2 tests (`tests/test_training_reward.py`): progress reward still
scales with speed below the cap, and is exactly equal (saturates) for any
speed at or above it. Re-ran with seed `909` held fixed, same 120s
training round as the previous entry, for direct comparison.

**Evidence:**
- Sources or documentation: none beyond the experiments' own output.
- AI-agent assistance: Claude Code did not stop at "the cap didn't work" --
  worked out the actual arithmetic (per-tick reward at the capped speed,
  compared against `WEIGHT_TERMINAL_PENALTY` over a range of sustained
  durations) to explain *why*, before writing anything down. Ran
  `ruff`/`pyright` (strict, 0 errors)/`pytest -q` (146 passed) before
  running the experiment.
- Commits or code: `src/training/reward.py` (`MAX_REWARDED_SPEED_MPS`),
  `tests/test_training_reward.py` (2 new tests), `docs/rl_design.md` §6.
- Experiment output: `experiments/2026-09-01_speed-cap-seed909/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** Another null result on the metric that matters: avg
max speed 40.5 -> 38.8 m/s (essentially unchanged), elimination rate
unchanged at 10/10. The quantitative explanation: at the capped speed,
progress reward is `1.0 * 10.0 / 60 ≈ 0.167` per tick. The observed scored
distance (~85-92m) at the observed speed (~38 m/s) implies the car covers
that ground in roughly 2-3 seconds before crashing -- and 2-3 seconds of
capped-speed reward (20-30) already exceeds `WEIGHT_TERMINAL_PENALTY =
10.0`, 2-3x over. The cap removed the reward for exceeding 10 m/s, but did
nothing to change the more basic fact that a short burst at (or even
under) the cap speed is already profitable enough to make dying net-
positive for the whole episode. Two hypotheses rejected now (train/eval
mismatch, uncapped-speed reward) -- the common thread across both is that
`WEIGHT_TERMINAL_PENALTY = 10.0` is simply too small relative to
achievable per-episode reward, not the shape of any other term.

**Decision and rationale:** Keep `MAX_REWARDED_SPEED_MPS` (harmless, closes
off unbounded reward-hacking in principle even though it didn't fix this
seed's problem alone). Do not adopt this checkpoint. Reframe the leading
hypothesis around magnitude, not shape: `WEIGHT_TERMINAL_PENALTY` needs to
be large enough to dominate several seconds of achievable reward, not
comparable to a single tick or to `WEIGHT_DAMAGE`'s per-hit scale.

**Next steps (options, refined with the new quantitative reasoning):**
1. **(now the most quantitatively motivated)** Raise
   `WEIGHT_TERMINAL_PENALTY` substantially -- into the tens (50-100+),
   not single digits, so it reliably dominates a multi-second high-speed
   burst rather than breaking even against about one second of it.
2. Switch from "no additional credit above the cap" to an *active*
   penalty for exceeding it, on top of raising the terminal penalty.
3. Try the current reward on seed 110 (the "good" seed) to see if this
   failure is seed-909-specific or general -- still open.
4. Run a small seed sweep (3-5 seeds) at the current reward to see the
   real outcome distribution -- still open.
5. Add a hard action/speed cap at the controller level as a safety
   backstop independent of reward shaping -- still open, and increasingly
   worth considering given two reward-side fixes in a row haven't moved
   the elimination rate at all.

---

## 2026-09-01 (continued, 8)

**Participants and contributions:** Charlotte Tsui — directed raising
`WEIGHT_TERMINAL_PENALTY` and focusing on wall-avoidance specifically.
Claude Code (AI agent) — implemented both, ran the causal test, found a
pattern across three consecutive attempts, and recommended a change of
direction rather than a fourth attempt on the same seed.

**Question or objective:** Raise the terminal penalty (per the previous
entry's quantitative recommendation) and strengthen the controller's
ability to detect and react to nearing walls.

**What we investigated or changed:**

- `src/training/reward.py`: `WEIGHT_TERMINAL_PENALTY` 10.0 -> 100.0
  (needs ~10s of at-cap driving to break even, vs. ~2s before, per last
  entry's arithmetic). As a direct, bundled wall-avoidance strengthening
  pass: `WEIGHT_WALL_PROXIMITY` 0.5 -> 1.0 (doubled, now comparable in
  scale to `WEIGHT_PROGRESS`) and `WALL_WARNING_DISTANCE_M` 3.0 -> 6.0
  (3.0m gives under a tenth of a second of reaction time at the 38+ m/s
  speeds every crashing checkpoint has shown; 6.0m gives ~0.6s even at
  the reward-cap cruising speed).
- Removed a duplicate `WEIGHT_WALL_PROXIMITY` declaration left over from
  editing (the constant is now declared once, alongside
  `WALL_WARNING_DISTANCE_M` since they're used together).
- Re-ran with seed `909` held fixed, same 120s training round as the
  previous three entries, for direct comparability.

**Evidence:**
- Sources or documentation: none beyond the experiments' own output.
- AI-agent assistance: Claude Code caught and fixed its own mistake before
  running anything -- the wall-avoidance edit initially left two
  `WEIGHT_WALL_PROXIMITY` definitions in the file (Python would have
  silently used the second, but it was confusing and not what was
  intended structurally), found by re-reading the file after editing
  rather than assuming the edit was clean. Ran `ruff`/`pyright` (strict,
  0 errors)/`pytest -q` (146 passed, including the reward tests with the
  new values) before running the experiment.
- Commits or code: `src/training/reward.py`, `docs/rl_design.md` §6.
- Experiment output:
  `experiments/2026-09-01_terminal100-walldist6-seed909/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a.

**What we observed:** Essentially no change on the metric that matters.
Avg max speed 38.8 -> 39.0 m/s (unchanged). Elimination rate 10/10 -> 9/10
-- a marginal improvement, but the one race that avoided full elimination
isn't a wall-avoidance success story: it took 0.469 damage (a serious
partial hit) and then spent 107 of 120 seconds (89%) stuck/low-progress,
looking more like "survived a bad hit and then couldn't function" than
"detected and steered around a wall in time." Every other race (9/10)
still reached full elimination at essentially the same extreme speeds
(29-47 m/s) seen in every reward configuration tested today.

This is the third reward-tuning attempt in a row (speed cap, then this
combined change) where average max speed stayed pinned in the same 38-40
m/s range, despite each fix being confirmed *active* during training
(unlike the very first terminal-penalty test, which never fired at all).
That consistency across genuinely different reward shapes points at
something more structural than "the reward needs one more tweak": seed
909's policy may simply be stuck in a resistant local optimum -- a
"floor it straight" behavior that's easy for the network to represent and
may have been reinforced early, before any of today's reward changes
existed to redirect it.

**Decision and rationale:** Keep both changes (not harmful, directionally
correct even if insufficient alone). Do not adopt this checkpoint. Given
three consecutive reward-only attempts on the same seed with no movement
on top speed, recommending a change of direction rather than a fourth
attempt: test the current, now substantially revised reward on seed 110
(the only checkpoint so far with zero eliminations) to see whether it
helps, hurts, or doesn't matter there -- this is the most informative
single next experiment, since it's untested and would distinguish
"seed-909-specific stuck optimum" from "something more fundamentally
wrong with the reward that seed 110 happened not to expose yet."

**Next steps:**
1. **(recommended)** Train seed 110 from scratch with the full current
   reward and compare against its own prior result (0/10 eliminations,
   ~6.9 m/s, 10/10 wins) -- not yet run, proposed to Charlotte.
2. Run a small seed sweep (3-5 seeds) at the current reward for a real
   sense of the outcome distribution.
3. Add a hard action/speed cap at the controller level as a safety
   backstop -- increasingly the most reliable lever given reward shaping
   alone has now made four attempts (idle, terminal-penalty-only, speed
   cap, terminal+wall-avoidance) without controlling top speed on seed
   909.

---

## 2026-09-01 (continued, 9)

**Participants and contributions:** Charlotte Tsui — directed running the
recommended seed-110 experiment. Claude Code (AI agent) — ran it (moved
to background after exceeding the 120s foreground timeout, picked back up
via the completion notification), found a materially more encouraging
result than any seed-909 test today, and verified it carefully rather
than taking the first big number at face value.

**Question or objective:** Does the current reward (idle penalty,
`WEIGHT_TERMINAL_PENALTY = 100.0`, speed cap, wall-avoidance changes) help,
hurt, or not matter on seed 110 -- the only checkpoint so far with zero
eliminations -- distinguishing "seed 909 is stuck" from "the reward
doesn't really work"?

**What we investigated or changed:** Trained seed `110` from scratch
(`scripts/train_sac.py --races 10 --round-seconds 120 --buffer-capacity
200000 --eval-round-seconds 120 --seed 110`), same config as every
seed-909 test today, with today's fully revised reward. The run took
88.6s training + evaluation (longer than prior runs -- 117,542 transitions
vs. ~72-86k before, since this policy survives longer per race and
therefore generates more ticks), exceeding the 120s foreground command
timeout; it was moved to a background task and picked up via the
completion notification rather than polled for.

**Evidence:**
- Sources or documentation: none beyond this run's own output.
- AI-agent assistance: Claude Code did not stop at the headline
  scored-distance numbers (up to 3069.9m in one race) -- given today's
  repeated lesson about big totals hiding elimination, it immediately
  pulled per-race damage/lap/max-speed detail before characterizing the
  result, and reported both the genuine improvement (speed, lap
  completion, beating the strong baseline) and the still-open problem
  (40% elimination) rather than leading with only the positive framing.
- Commits or code: `docs/rl_design.md` §6 (causal test 7).
- Experiment output: `experiments/2026-09-01_full-reward-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** A materially different outcome from every seed-909
test today. Per-race: 4 of 10 races vs. `crash_fast` completed 8-9 laps
with best lap times of 15.8-22.3s (~9-11.6 m/s average pace, right around
`MAX_REWARDED_SPEED_MPS = 10.0` -- exactly what the reward is supposed to
encourage) and max speeds of 25-30 m/s; the other 6 ended in full
elimination, some after productive laps (3-4 laps then a crash), some
almost immediately (near-zero off-track/wall-contact time, a fast direct
crash). Aggregate max speed 30.2 m/s -- notably lower than every seed-909
checkpoint today (38-47 m/s), suggesting the wall-avoidance/terminal-
penalty changes did have a real effect here. Against
`default_student_controller` (~5 m/s sustained, previously undefeated):
**6/10 race wins -- the first time any SAC checkpoint has beaten this
baseline at all.**

This is a genuinely different result from three consecutive seed-909
tests that stayed pinned at ~90-100% elimination and 38-40 m/s regardless
of reward changes. The same reward, starting from a different seed,
produces real competitive speed. This supports last entry's hypothesis
that seed 909 was stuck in a resistant local optimum specific to its own
training trajectory, not that the reward doesn't work.

**Decision and rationale:** This is the most interesting checkpoint from
today, but not an unambiguous "best" -- it trades reliability for speed
against the original `2026-09-01_scaled-training-budget` (seed 110, old
reward: 0/10 eliminated, ~6.9 m/s). Average distance over many races
heavily favors the new one (835-842m vs. 100.7m, even counting the
crashes); guaranteed survival favors the old one. Given the course
rubric's emphasis on reliability "across random starting-point seeds," a
~40% elimination rate is not acceptable to submit as-is, but this is real
progress -- the problem has shifted from "no speed, no wall-avoidance" to
"inconsistent," which is a different and more tractable problem.

**Next steps:**
1. Investigate what distinguishes the crash races (immediate deaths like
   seed 7 race 1, seed 8675309 both races) from the success races --
   e.g. whether a specific spawn point or early-track feature is harder
   to handle.
2. More training (more races/updates) on this same reward+seed
   combination to see if reliability improves with more gradient steps.
3. Run the same full reward on 2-3 more seeds to check whether this
   speed-with-partial-reliability pattern generalizes or was specific to
   how well seed 110 responded.
4. Small seed sweep at the current reward, still open from earlier
   entries.
5. Hard action/speed cap at the controller level, still open, though less
   urgent now that the reward changes have shown they *can* work well
   starting from the right conditions.

---

## 2026-09-01 (continued, 10)

**Participants and contributions:** Charlotte Tsui — directed more
training on the same seed/reward combination, focused on lap completion.
Claude Code (AI agent) — ran it (backgrounded again, ~3 minutes), verified
the result thoroughly (per-race detail, held-out-seed check) before
reporting it as a genuine success rather than another misleading total.

**Question or objective:** Does more training on the same (seed 110,
current full reward) combination improve the ~40% elimination rate found
in the previous entry, now that the reward is confirmed to point somewhere
productive?

**What we investigated or changed:** Re-ran
`scripts/train_sac.py --races 20 --round-seconds 120 --buffer-capacity
400000 --eval-round-seconds 120 --seed 110` -- identical to
`2026-09-01_full-reward-seed110` except `--races 10 → 20` (buffer capacity
raised to match). No reward or hyperparameter changes. Took 184.4s
training (exceeded the 120s foreground timeout, moved to background and
picked up via the completion notification).

**Evidence:**
- Sources or documentation: none beyond this run's own output.
- AI-agent assistance: Claude Code did not report the headline numbers
  (consistent ~1550-1650m scored distance, 2/2 wins everywhere) as
  success without checking per-race damage, lap count, and max speed
  first -- the same discipline applied to the previous entry's big
  numbers. Additionally checked the training/eval seed overlap directly
  (training used seed 110, also an eval seed) by isolating the 4 genuinely
  held-out seeds and confirming they show the identical pattern, rather
  than letting that known caveat go unaddressed for a result this
  significant.
- Commits or code: `docs/rl_design.md` §6 (causal test 8).
- Experiment output: `experiments/2026-09-01_more-training-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** The reliability problem is resolved, not just
improved. **Zero eliminations across all 20 evaluation races** (5 seeds x
2 races x 2 baselines). Every single race completed exactly 4 laps.
Max speed settled to a controlled, consistent 17.8-21.2 m/s (down from
25-47 m/s at races=10). Best lap times 21.2-27.9s (~6.5-8.6 m/s average
lap pace -- genuine cornering competence, not just a fast straightaway).
**20/20 race wins against both baselines.** The 4 held-out seeds (42, 7,
2024, 8675309 -- never used for training spawns) show the identical
pattern, ruling out the training/eval seed overlap as an explanation.

Only the amount of training changed (2x races, ~29k -> ~60k gradient
updates, same already-productive reward) -- this alone took elimination
rate from 60% to 0% and converged what had been a highly variable policy
(0-9 laps depending on the race) into a highly consistent one (4 laps,
every time).

**Decision and rationale:** This is the new best checkpoint by every
metric that matters -- reliability, cross-seed consistency, and
competitiveness -- and is now the reference point for further refinement.
`controllers.sac_candidate` auto-selects it as the newest checkpoint.
Given how strong and clean this result is, treating today's causal-test
chain (which started from the seed-909 freeze/crash investigation) as
substantially resolved: the current reward, given enough training, reaches
a reliable, competitive controller from seed 110.

**Next steps:**
1. Investigate whether 4 laps is a round-length artifact (would a longer
   round show 5+ laps, or is progress capping out for another reason).
2. Try further scaling training to see if performance keeps improving or
   has plateaued.
3. Test on more seeds beyond the fixed 5 to build confidence before
   treating this as leaderboard-ready.
4. Consider packaging this checkpoint as a self-contained, submission-
   ready controller (no `training/` dependency) per the refinement plan.
5. Watch it live (`controllers.sac_candidate`) for a qualitative check --
   all evidence so far is headless stats.

---

## 2026-09-01 (continued, 11)

**Participants and contributions:** Charlotte Tsui -- asked for the
Gradescope submission to be created for the SAC controller; when told the
rubric actually names two required modules (`minimum_viable` +
`race_faster`) and only the SAC side exists, chose to submit SAC-only as
`race_faster` rather than also writing a `minimum_viable` heuristic.
Claude Code (AI agent) -- read `autograder/README.md` and
`scripts/export_student_controllers.py` to understand the actual
submission contract (not just `README.md`'s general packaging section),
wrote the self-contained controller, and verified it before treating the
export as done.

**Question or objective:** Package the current best SAC checkpoint
(`2026-09-01_more-training-seed110`, refinement plan item 8) as an
actual Gradescope-uploadable submission, not just a local-dev viewer.

**What we investigated or changed:** Read `autograder/README.md` (not
previously covered in this doc) and found the real submission contract:
`scripts/export_student_controllers.py` produces the upload zip, rooted at
`controllers/`, and the grading config names two module slots --
`controllers.minimum_viable` and `controllers.race_faster` -- each graded
independently (missing module = zero on that module's points, not a
failure of the whole submission). `sac_candidate.py` (the existing
checkpoint viewer) is explicitly not submission-ready: it imports
`training.observation`/`training.sac`, which live outside
`src/controllers/` and are not followed by the export script's
`controllers.*`-only dependency scanner.

Wrote `src/controllers/race_faster.py`: inlines the observation encoding
(`training/observation.py`) and just the inference half of the actor
network (`training/sac.py`'s `GaussianPolicy`, dropping the critics/
sampling/optimizer code that's training-only) so the module has zero
dependency outside `src/controllers/`. Extracted only the `"policy"` key
from `experiments/2026-09-01_more-training-seed110/checkpoints/
policy_final.pt` (396KB, includes critics/targets/log_alpha for resuming
training) into a trimmed `src/controllers/checkpoints/
race_faster_policy.pt` (81KB, policy weights only). Always acts
deterministically (inference, not training).

**Evidence:**
- Sources or documentation: `autograder/README.md` (submission layout,
  rubric, module-naming convention), `README.md`'s packaging-a-controller
  and CPU/memory-boundary sections, `scripts/export_student_controllers.py`
  (read, not edited, to confirm what the dependency scanner does and does
  not follow).
- AI-agent assistance: Claude Code read the actual export/autograder
  scripts rather than assuming the general packaging-contract README
  section was the whole story -- that's what surfaced the two-module
  rubric structure and the checkpoint-not-followed-by-static-import gap.
  Verified the packaged module three ways before calling it done: (1) a
  Python check that `race_faster`'s policy output matches
  `SACAgent.act(..., deterministic=True)` from the full checkpoint
  bit-for-bit (`atol=1e-6`) on 5 random observations; (2) `ruff check`,
  `ruff format --check`, and `pyright` (project's strict-mode config,
  matching the rubric's Pyright-strict check) all clean; (3) an actual
  `uv run racing h2h` race (seed 110, 30s, vs. `crash_fast`) confirming
  real driving behavior, not just synthetic-observation output. Asked the
  user via `AskUserQuestion` whether to also write a `minimum_viable`
  heuristic controller before proceeding, since that's materially
  different work than "package my SAC controller" and CLAUDE.md scopes
  this track to Charlotte's own SAC work -- user chose SAC-only.
- Commits or code: `src/controllers/race_faster.py`,
  `src/controllers/checkpoints/race_faster_policy.pt`,
  `docs/rl_design.md` section 6 item 8 (marked done).
- Experiment output: n/a (packaging, not a new training run); source
  checkpoint is `experiments/2026-09-01_more-training-seed110/`.
- Leaderboard result: not yet submitted to Gradescope.

**What we observed:** The trimmed, self-contained module reproduces the
full checkpoint's behavior exactly (bit-for-bit deterministic-action
match) and drives correctly in a real race: seed 110, 30s round, 0
eliminations, 18.3 m/s max speed, 1 lap, 210.6m raw distance vs.
`crash_fast`'s 16.4m, 0.075 damage -- consistent with the checkpoint's
documented behavior in the previous entry. `scripts/export_student_
controllers.py --all-controllers` (needed because the checkpoint is a
non-Python file the module-selection mode's import-following wouldn't
pick up) produced `artifacts/formula110-student-controllers.zip`
containing `controllers/race_faster.py`, its checkpoint, `__init__.py`,
`py.typed`, and (harmlessly, since Gradescope only grades configured
module names) the existing `crash_fast.py` starter and `sac_candidate.py`
dev viewer.

**Decision and rationale:** Submit only `controllers.race_faster` for
now. The rubric's `minimum_viable` module (a hand-tuned heuristic judged
on zero damage/wall-contact) is a different, non-RL deliverable that
CLAUDE.md's scope guardrails and the user's explicit choice both place
outside this session's work -- it's a gap to close later (by this track
or a separate assignment pass), not something to improvise into today's
SAC-packaging task. `docs/rl_design.md` updated in the same session per
CLAUDE.md step 5.

**Next steps:**
1. Decide who/when writes `controllers.minimum_viable` -- without it, the
   submission is capped at the 35 `race_faster`-only rubric points (out of
   100) and is not leaderboard-eligible until both modules are present per
   `autograder/README.md`.
2. Actually upload `artifacts/formula110-student-controllers.zip` to the
   Gradescope assignment (not done by this session -- packaging only).
3. Watch `controllers.race_faster` live for a qualitative check, same as
   the standing item on `sac_candidate`.

---

## 2026-09-01 (continued, 12)

**Participants and contributions:** Charlotte Tsui -- attempted the
upload from the previous entry, hit a Gradescope rejection, reported the
exact error text, and supplied the fix (manifest schema and which two
extra files to include) once she'd found it. Asked for the fix to be
codified in `CLAUDE.md` and for the root cause to be debugged. Claude
Code (AI agent) -- applied the fix to the zip, then investigated why the
previously-documented process (which had been based on reading
`autograder/README.md` and `scripts/export_student_controllers.py`
directly) hadn't surfaced this requirement.

**Question or objective:** The Gradescope upload of
`artifacts/formula110-student-controllers.zip` from the previous entry
failed with "expected formula110-submission.json at the root of the
submission." Why didn't the documented submission process catch this,
and how do we make sure future submissions don't hit it again?

**What we investigated or changed:** Re-grepped the entire repo
(case-insensitive) for `submission.json` and `manifest` -- zero hits
anywhere in `autograder/`, `scripts/export_student_controllers.py`,
`README.md`, or `autograder/README.md`. Read the full 489 lines of
`autograder/gradescope/grade.py` (previously only its first ~60 lines had
been read) specifically for any manifest-reading logic -- none exists;
it reads the modules to grade from an instructor-baked
`/opt/formula110-autograder/config.json` built by
`scripts/build_gradescope_autograder.py`, not from anything inside the
student's submission zip. Conclusion: the requirement is real (fixing it
worked), but it is not represented anywhere in this repo's checked-in
`autograder/` bundle or docs -- the **live** Gradescope autograder for
this assignment has diverged from the local trusted bundle, most likely
an instructor-side update to the deployed autograder that was never
back-ported into this repository's `autograder/` snapshot.

Fixed the immediate submission by appending three files to the zip root
(not inside `controllers/`, and not produced by
`scripts/export_student_controllers.py`, which has no support for this
and wasn't edited to add any): `formula110-submission.json`
(`{"schema_version": 1, "controller_module": "controllers.race_faster"}`),
plus unmodified copies of `pyproject.toml` and `uv.lock`. Documented the
requirement in `CLAUDE.md` under a new "Packaging a Gradescope
submission" section so this is a standing step rather than something
rediscovered by trial and error on the next submission.

**Evidence:**
- Sources or documentation: full read of `autograder/gradescope/grade.py`
  (489 lines); repo-wide grep for `submission.json`/`manifest` (no
  matches); the user's literal Gradescope error text and the working fix
  they supplied.
- AI-agent assistance: Claude Code did not guess at the manifest schema
  or claim to know why the live autograder differs from the checked-in
  bundle -- when first asked to create the submission, it said plainly
  that the required file didn't appear anywhere in the repo and asked the
  user to supply the actual requirement rather than fabricating a schema.
  Once the user supplied the schema and the fix worked, Claude Code did
  the root-cause read (full `grade.py`, repo-wide grep) to confirm the gap
  was real and repo-wide, not a file it had simply missed on the first
  pass.
- Commits or code: `CLAUDE.md` (new "Packaging a Gradescope submission"
  section); `artifacts/formula110-student-controllers.zip` (rebuilt with
  the three added root files -- not committed, gitignored build output).
- Experiment output: n/a.
- Leaderboard result: submission uploaded successfully to Gradescope
  after the fix (per user confirmation); grading outcome not yet known.

**What we observed:** The fix resolved the upload error. The gap is
entirely on the documentation/tooling side, not the controller itself --
`controllers/race_faster.py` and its checkpoint were already correct and
verified in the previous entry.

**Decision and rationale:** Codified the three-extra-files recipe in
`CLAUDE.md` rather than in `docs/rl_design.md` or
`experiments/README.md`, since it's a submission-mechanics step that
applies regardless of which experiment or checkpoint is being submitted,
not something tied to a specific run's evidence. Framed it as "the live
autograder is authoritative over the local `autograder/` bundle when they
conflict" rather than trying to fix or explain the divergence itself,
since nothing in this repo shows *why* the live side changed and
`autograder/` is course-owned infrastructure this track does not edit.

**Next steps:**
1. If the live autograder's contract changes again, update the
   `CLAUDE.md` recipe rather than assuming the old one still holds.
2. Once a `controllers.minimum_viable` module exists (still open from the
   previous entry), confirm whether the manifest schema supports naming
   two modules or only one `controller_module` -- the schema handed to us
   only had a single field, which may mean the live autograder grades one
   submitted module at a time rather than the two-module scheme described
   in the local `autograder/README.md`.

---

## 2026-09-01 (continued, 13)

**Participants and contributions:** Charlotte Tsui -- asked to push
training further on the `2026-09-01_more-training-seed110` (races=20)
result to see if it keeps improving. Claude Code (AI agent) -- ran a
further-doubled training run, found a clean monotonic improvement, and
on picking the conversation back up found (via the "changed on disk"
notice) that entries 11-12 had packaged and submitted the races=20
checkpoint as `controllers.race_faster` to Gradescope in the meantime --
flagged the mismatch rather than silently continuing past it.

**Question or objective:** Does performance keep improving with more
training beyond `2026-09-01_more-training-seed110` (races=20, 0/20
eliminations)?

**What we investigated or changed:** Same seed (110), reward, and 120s
round length as the last two entries;
`scripts/train_sac.py --races 40 --round-seconds 120 --buffer-capacity
800000 --eval-round-seconds 120 --seed 110` (`--races` doubled again from
20). Took 413.2s training (backgrounded past the 120s foreground timeout,
picked up via completion notification, same pattern as the races=20 run).

On resuming work in this doc, noticed via the harness's "file changed on
disk" notice that `docs/rl_design.md`, `docs/lab_notebook.md`, and
`CLAUDE.md` had all been modified since last read here, and two full
entries (11, 12) already existed describing work done outside this
conversation thread: packaging the races=20 checkpoint into
`src/controllers/race_faster.py` (self-contained, no `training/`
dependency) and successfully submitting it to Gradescope, including
fixing an undocumented live-autograder submission-manifest requirement
now codified in `CLAUDE.md`'s new "Packaging a Gradescope submission"
section. Read those entries and the current `race_faster.py` in full
before continuing, rather than overwriting or ignoring them.

**Evidence:**
- Sources or documentation: entries 11-12 of this notebook (read, not
  written, this session); `CLAUDE.md`'s new packaging section;
  `src/controllers/race_faster.py` and
  `src/controllers/checkpoints/race_faster_policy.pt` (read to confirm
  which checkpoint they package -- `2026-09-01_more-training-seed110`,
  races=20, not the races=40 result from this entry).
- AI-agent assistance: Claude Code did not proceed with new work as if
  the conversation's own history were the complete picture -- it noticed
  the disk-state mismatch, read the intervening entries fully, and is
  reporting the resulting packaging gap explicitly rather than silently
  leaving `race_faster.py` pointed at a now-superseded checkpoint. Ran
  `ruff`/`pyright`/`pytest -q` (146 passed) before treating this entry's
  own changes as done.
- Commits or code: `docs/rl_design.md` §6 (causal test 9).
- Experiment output: `experiments/2026-09-01_more-training2-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a for this run; the races=20 checkpoint was
  submitted per entry 12, grading outcome not yet known.

**What we observed:** Training kept improving, cleanly and monotonically:

| training budget | avg damage | avg off-track | avg wall contact | avg max speed | eliminated |
| --- | --- | --- | --- | --- | --- |
| races=10 | 0.746 | 1.88s | 0.92s | 30.2 m/s | 6/10 |
| races=20 (submitted as `race_faster`) | 0.062 | 0.84s | 0.12s | 18.4 m/s | 0/10 |
| races=40 (this entry) | **0.000** | **0.00s** | **0.00s** | 15.5 m/s | 0/20 |

At races=40: zero damage, zero off-track time, and zero wall contact in
literally every one of 20 evaluation races (all 5 seeds, both baselines,
including the 4 held-out ones). Max speed converged tighter (15.4-16.1
m/s). Still 20/20 race wins, now with a larger average margin. Full detail
in `experiments/2026-09-01_more-training2-seed110/notes.md`.

**This means the checkpoint currently submitted to Gradescope
(races=20) is no longer this track's best result** -- races=40 is
strictly better on every tracked metric (damage, off-track time, wall
contact all exactly zero vs. small-but-nonzero; comparable or better
speed/lap count).

**Decision and rationale:** Did not touch `race_faster.py` or resubmit
without asking -- repackaging and resubmitting to a live, already-graded
Gradescope assignment is exactly the kind of external/shared-system
action that warrants confirmation first, not something to do
unilaterally on the strength of "the numbers are better." Reporting the
gap to Charlotte and asking whether/how to proceed (repackage now, wait,
or something else) rather than deciding for her.

**Next steps:**
1. **Awaiting direction:** repackage `controllers.race_faster` from the
   races=40 checkpoint and re-submit, using the now-documented
   `CLAUDE.md` packaging recipe -- or hold off, e.g. if the races=20
   submission is already graded and stable and Charlotte would rather not
   disturb it.
2. Training-budget scaling could plausibly continue (the trend hasn't
   broken), but safety metrics are already at floor -- further gains
   would likely be speed/lap-count, not reliability. Optional, given
   growing wall-clock cost (7:45 for this run).
3. Still open from entry 11: who/when writes `controllers.minimum_viable`
   -- without it the submission is capped at partial rubric points per
   `autograder/README.md`.
4. Still open: broader seed testing, live qualitative watch, further
   training-budget scaling if desired.

---

## 2026-09-01 (continued, 14)

**Participants and contributions:** Charlotte Tsui -- asked to optimize
for speed specifically (higher throttle/speed while staying safe and
completing laps), given how safe the races=40 checkpoint already was.
Claude Code (AI agent) -- implemented the most direct lever (raise the
reward's speed cap), tested it, found a clear regression, and reverted
rather than presenting a worse checkpoint as progress.

**Question or objective:** The current best checkpoint
(`2026-09-01_more-training2-seed110`) is very safe (zero incidents) but
only reaches ~15.5 m/s average max speed and ~6.8-7.7 m/s average lap
pace. How can the controller run faster while staying safe and still
completing laps?

**What we investigated or changed:** Noted that `MAX_REWARDED_SPEED_MPS`
(10.0) was already being exceeded by the reference checkpoint's actual
max speed (15.4-16.1 m/s) while its average lap pace stayed well under
even 10.0 -- meaning the cap wasn't blocking top speed, just not
crediting the policy for holding higher speed longer. Raised it to 20.0
in `src/training/reward.py`, leaving every safety-side weight
(`WEIGHT_DAMAGE`, `WEIGHT_WALL_PROXIMITY`, `WALL_WARNING_DISTANCE_M`,
`WEIGHT_TERMINAL_PENALTY`, `WEIGHT_CONTACT`) untouched, and trained from
scratch with the same seed (110), races=40, and round_seconds=120 as the
reference, for direct comparison.

**Evidence:**
- Sources or documentation: none beyond this run's own output and the
  reference checkpoint's known numbers.
- AI-agent assistance: Claude Code did not report the higher top-speed
  number as success -- pulled full per-race detail (laps, lap times,
  damage, marshal counts) before characterizing the result, the same
  discipline applied to every big-number claim today, and it's what
  caught that lap times got *slower* despite higher top speed. Reverted
  the change immediately once the regression was clear rather than
  leaving a worse value in place for discussion. Ran
  `ruff`/`pyright`/`pytest -q` (146 passed) both before running the
  experiment and after reverting.
- Commits or code: `src/training/reward.py` (`MAX_REWARDED_SPEED_MPS`
  tried at 20.0, reverted to 10.0), `docs/rl_design.md` §6 (causal test
  10).
- Experiment output: `experiments/2026-09-01_speedcap20-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a; not adopted.

**What we observed:** A clear regression, not an improvement. Max speed
roughly doubled (34.4-37.7 m/s), but laps completed dropped from an
average of 4.1 to 0-2 per race, best lap times got *slower* despite the
higher top speed (21.2-27.9s -> 36.3-95.9s), damage returned (0.000 ->
0.14-0.66, though no full eliminations in this sample), low-progress time
spiked to 10-48% of the race, and marshal recoveries reached as high as
21 in a single race (from ~0). It lost 5 of 10 races against
`default_student_controller` -- the first losses against that baseline
since the races=40 breakthrough (previously 10/10).

Best explanation: the cap was never limiting top speed (already exceeded
it), so raising it didn't unlock more speed -- it doubled the maximum
achievable per-tick progress reward while every control/safety term
stayed the same absolute size, shifting the reward's relative balance
toward raw speed at the expense of cornering control. This reopened the
same speed-vs-control trade-off from the seed-909 causal chain earlier
today, just triggered from a different (previously good) starting point.

**Decision and rationale:** Reverted `MAX_REWARDED_SPEED_MPS` to `10.0`.
The evidence was unambiguous enough not to need a second confirming run --
regression on nearly every metric, not a borderline or mixed result.
`2026-09-01_more-training2-seed110` remains the current best checkpoint.
Reward-magnitude tuning is not the right lever for improving lap times
from here; the lever that has worked cleanly and repeatedly today is more
training on the existing, already-productive reward (causal tests 8-9).

**Next steps:**
1. **(recommended)** Pursue speed via more training on the unchanged
   (cap=10.0) reward, continuing the races=10→20→40 trend rather than
   reshaping the reward again -- lap times/consistency may keep improving
   the same way damage/off-track/wall-contact did.
2. If reward-side speed tuning is revisited later, use a much smaller
   step (e.g. 10.0 -> 12.0) and compare directly against the races=40
   reference rather than training from scratch.
3. Consider the standing refinement-plan item on reducing hesitation
   (penalizing oscillating steering) as a more targeted lap-time lever.
4. Note for anyone watching live: `controllers.sac_candidate` auto-
   selects the newest checkpoint by file time, which is now the regressed
   speedcap20 run -- use `FORMULA110_SAC_CHECKPOINT` to point at
   `2026-09-01_more-training2-seed110` explicitly.

---

## 2026-09-01 (continued, 15)

**Participants and contributions:** Charlotte Tsui -- directed pushing
training further on the current (reverted, cap=10.0) reward. Claude Code
(AI agent) -- ran it (backgrounded ~14 minutes), found speed and lap
count went *down* rather than up, and worked out why before reporting it.

**Question or objective:** Continue the races=10→20→40 training-budget
trend on the unchanged reward -- does it keep improving?

**What we investigated or changed:** Same seed (110), reward
(`MAX_REWARDED_SPEED_MPS = 10.0`, as reverted last entry), round length
(120s); `scripts/train_sac.py --races 80 --round-seconds 120
--buffer-capacity 1600000 --eval-round-seconds 120 --seed 110`. Took
849.0s training (1,106,262 transitions, 276,316 gradient updates),
exceeding the foreground timeout as expected at this scale; backgrounded
and picked up via the completion notification.

**Evidence:**
- Sources or documentation: none beyond this run's own output and the
  races=40 reference's known numbers.
- AI-agent assistance: Claude Code noticed the aggregate scored-distance
  number had *dropped* (~1780m avg at races=40 -> ~1130m avg here) before
  assuming "more training = better" and pulled full per-race detail,
  which is what surfaced that safety stayed perfect while speed and laps
  both declined -- the opposite of what "push training further" was
  aimed at. Worked out the mechanistic explanation (the reward doesn't
  credit speed past the cap, so more training converges toward the cap
  rather than past it) rather than just reporting numbers without
  analysis.
- Commits or code: `docs/rl_design.md` §6 (causal test 11).
- Experiment output: `experiments/2026-09-01_more-training3-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a; not adopted as the speed-goal checkpoint.

**What we observed:** Damage, off-track time, and wall contact stayed
exactly zero across all 20 evaluation races -- more training didn't break
anything. But max speed dropped (15.4-16.1 -> 13.1-14.7 m/s), laps
completed dropped (4-5 -> 2-3), and best lap times got slower (21.2-27.9s
-> 34.3-42.1s). `MAX_REWARDED_SPEED_MPS = 10.0` gives zero reward benefit
for exceeding 10 m/s, only unrewarded risk -- races=40's ~15-16 m/s looks
like a residual of less-converged training rather than a reward-seeking
choice, and further training pulled it back down toward the actual
reward-maximizing speed (at/near the cap).

**Decision and rationale:** Not adopting this checkpoint for the stated
speed goal -- `2026-09-01_more-training2-seed110` (races=40) remains
better for "fast but safe" (races=80 is arguably even safer/more
consistent, but that's not what's being optimized right now). This
closes off "just train more" as a path to higher speed: combined with the
previous entry's regression from doubling the cap outright, the evidence
now points at a **smaller** cap increase (e.g. 10.0 -> 12-13.0, close to
races=40's own organic ceiling) as the most promising untried lever,
rather than either extreme already tried.

**Next steps:**
1. **(recommended)** Try a smaller speed-cap increase (10.0 -> 12.0 or
   13.0) and retrain, comparing directly against the races=40 reference.
2. `scripts/train_sac.py` always initializes a fresh `SACAgent` -- if
   fine-tuning from an existing checkpoint would be valuable (continuing
   races=40's policy rather than retraining from scratch each time), that
   would need a `--resume-from` option added; not currently supported.
3. Standing item: reducing hesitation (oscillating steering) as a
   non-reward-magnitude lap-time lever.
4. `controllers.sac_candidate` now auto-selects this run's checkpoint (the
   newest by file time) -- still point `FORMULA110_SAC_CHECKPOINT` at
   `2026-09-01_more-training2-seed110` for the best speed+safety
   checkpoint.

---

## 2026-09-02

**Participants and contributions:** Charlotte Tsui -- directed a smaller,
more careful speed-cap increase (10.0 -> 12.0 or 13.0) as a direct
comparison against the races=40 reference. Claude Code (AI agent) --
implemented it, ran the comparison, and found a third consecutive null
result on this specific lever.

**Question or objective:** Does a small (rather than doubled) increase to
`MAX_REWARDED_SPEED_MPS` improve lap times over the races=40 reference
without reopening the speed-vs-control trade-off that a full doubling
caused?

**What we investigated or changed:** `src/training/reward.py`:
`MAX_REWARDED_SPEED_MPS` 10.0 -> 12.0 (a 20% step, chosen to stay below
the reference checkpoint's own observed peak speed, ~16 m/s, rather than
the 2x jump tried on 2026-09-01). Retrained from scratch with the exact
same seed (110), races=40, round_seconds=120, buffer_capacity=800000, and
eval_round_seconds=120 as the reference, for a clean, direct comparison.

**Evidence:**
- Sources or documentation: none beyond this run's output and the
  reference checkpoint's known numbers.
- AI-agent assistance: Claude Code computed exact averages (best lap
  time, max speed, raw distance) for both this run and the reference
  before characterizing the result, rather than eyeballing the printed
  totals -- the same discipline applied throughout yesterday's session.
  Ran `ruff`/`pyright`/`pytest -q` (146 passed) before and after the
  reward change.
- Commits or code: `src/training/reward.py` (`MAX_REWARDED_SPEED_MPS`
  tried at 12.0, reverted to 10.0), `docs/rl_design.md` §6 (causal test
  12).
- Experiment output: `experiments/2026-09-02_speedcap12-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a; not adopted.

**What we observed:** Essentially a tie with the reference, not an
improvement. Safety identical (0.000 damage, 0.00s off-track/wall-contact,
20/20 wins, unchanged). Lap completion slightly *more* consistent (4 laps
in literally every one of 20 races, vs. the reference's mix of 4s and 5s).
But average best-lap time was **slower** (24.76s -> 28.20s), and average
max speed was not meaningfully different (15.53 -> 15.88 m/s, ~2%). This
is the third point tried on the `MAX_REWARDED_SPEED_MPS` axis (10.0
reference, 12.0 this entry, 20.0 on 2026-09-01) and the third to not beat
the original 10.0 value -- the middling attempt landed closer to a tie
than the extreme attempt's clear regression, but neither improved on the
baseline.

**Decision and rationale:** Reverted `MAX_REWARDED_SPEED_MPS` to `10.0`.
Treating this constant as exhausted as a lever for the stated speed goal
-- three attempts across a 2x range (10, 12, 20) produced one reference
result and two non-improvements, which is enough evidence to stop
searching this specific axis rather than trying more intermediate values.
`2026-09-01_more-training2-seed110` (races=40, cap=10.0) remains the best
checkpoint for speed+safety.

**Next steps:**
1. Treat `MAX_REWARDED_SPEED_MPS` tuning as a dead end for now; if
   further speed gains are wanted, pursue a different lever -- the
   standing "reduce hesitation" (oscillating steering) refinement item,
   or adding a `--resume-from` option to `scripts/train_sac.py` so
   training can fine-tune from the races=40 checkpoint instead of
   restarting from scratch each time.
2. Alternatively, treat races=40 as a strong enough result on speed for
   now and shift focus to other open items: broader seed testing beyond
   the fixed 5, repackaging `controllers.race_faster` from the current
   best checkpoint (still packages races=20, not races=40, per the
   2026-09-01 packaging entries), or the `controllers.minimum_viable`
   module gap noted in those same entries.
3. If `MAX_REWARDED_SPEED_MPS` tuning is revisited anyway, note that each
   attempt so far is n=1 per value (a fresh from-scratch run) -- repeating
   a value with a different seed would help separate a genuine causal
   effect from ordinary run-to-run variance, which hasn't been
   characterized on this specific axis.

---

## 2026-09-02 (continued)

**Participants and contributions:** Charlotte Tsui -- directed pursuing
"path 1" (a different lever than reward magnitude) from the prior
options. Claude Code (AI agent) -- chose the "reduce hesitation" option
over the checkpoint-resume option (more directly testable without new
infrastructure), implemented it, and found the clearest regression of
any experiment run today.

**Question or objective:** Does penalizing oscillating/hesitant steering
improve lap time over the races=40 reference, as an alternative to
`MAX_REWARDED_SPEED_MPS` tuning (which had failed three times)?

**What we investigated or changed:** Added
`WEIGHT_STEERING_SMOOTHNESS = 0.1` to `src/training/reward.py`, penalizing
tick-to-tick change in `imu.yaw_rate_degrees_per_s` (scaled by
`YAW_RATE_CHANGE_SCALE_DEGREES_PER_S = 200.0`, capped at 1.0) as a proxy
for jerky steering -- `step_reward` only receives sensor transitions, not
the raw steer action (the action for a given tick is chosen *after*
reward is computed for the previous transition, inside
`TrainableController.__call__`), so a direct action-delta penalty would
have needed a larger refactor; the yaw-rate-change proxy avoided that.
Added 2 tests, then reworked to 1 after the revert (see below). Retrained
from scratch with the exact same seed (110), races=40, round_seconds=120,
and buffer_capacity=800000 as the reference.

**Evidence:**
- Sources or documentation: none beyond this run's output and the
  reference checkpoint's known numbers.
- AI-agent assistance: Claude Code computed exact averages before
  characterizing the result (same discipline as every prior entry), which
  is what surfaced how uniform the regression was (every one of 10 races
  vs. `crash_fast` completed exactly 2 laps). After reverting the weight
  to 0.0, recognized that the two tests written for the *active*
  mechanism would now assert something false (a real behavioral
  difference that no longer exists at weight=0) -- rather than leaving
  them to silently pass by coincidence or fail confusingly, replaced them
  with one test that explicitly documents and verifies the mechanism is
  currently inert. Ran `ruff`/`pyright`/`pytest -q` (147 passed) after
  the revert and test rewrite.
- Commits or code: `src/training/reward.py`
  (`WEIGHT_STEERING_SMOOTHNESS` tried at 0.1, reverted to 0.0),
  `tests/test_training_reward.py` (2 tests replaced with 1),
  `docs/rl_design.md` §6 (causal test 13).
- Experiment output:
  `experiments/2026-09-02_steering-smoothness-seed110/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a; not adopted.

**What we observed:** The clearest, most uniform regression of any
experiment today. Safety unaffected (still 0.000 damage, 0.00s off-track/
wall-contact, 20/20 wins). But laps completed halved (4.10 avg -> exactly
2.00 in every one of 10 evaluated races), best lap time nearly doubled
(24.76s -> 45.48s), and max speed dropped 31% (15.53 -> 10.75 m/s). Best
explanation: penalizing raw yaw-rate *change* can't distinguish wasteful
oscillation from a legitimate, necessary steering input for cornering --
both involve the yaw rate changing quickly -- so the penalty suppressed
real cornering itself, not just hesitation.

**Decision and rationale:** Reverted `WEIGHT_STEERING_SMOOTHNESS` to
`0.0`, keeping the mechanism in code (documented, disabled) rather than
deleting it. Not adopting this checkpoint.
`2026-09-01_more-training2-seed110` remains the best checkpoint. This is
the fourth consecutive reward-tuning attempt at the speed goal (cap=12.0,
cap=20.0, more training at cap=10.0, this entry) to fail against the
plain races=40 reference -- recommending a pause on further reward-tuning
attempts at pure speed optimization specifically, rather than trying a
fifth variant.

**Next steps:**
1. **(recommended)** Pause reward-magnitude/shape tuning for the speed
   goal. If a smarter steering-smoothness proxy is wanted later:
   distinguish *sustained* turning (real cornering) from *oscillating*
   turning (yaw rate repeatedly changing sign/direction in a short
   window) -- e.g. penalize sign changes rather than raw magnitude of
   change, which wouldn't penalize a held turn the way this did.
2. Consider adding a `--resume-from` option to `scripts/train_sac.py`
   (currently always initializes a fresh `SACAgent`) so training could
   fine-tune from the races=40 checkpoint instead of re-deriving a new
   policy from a random init under each modified reward -- a genuinely
   different mechanism than anything tried today.
3. Otherwise, treat races=40 as a strong, practical result and shift
   focus to other open items: broader seed testing, repackaging
   `controllers.race_faster` from the current best checkpoint (still
   packages races=20), or the `controllers.minimum_viable` module gap.

---

## 2026-09-02 (continued, 2)

**Participants and contributions:** Charlotte Tsui -- asked for the best
way to continue improving speed/reduce hesitation, and specifically
proposed tracking the optimal track path per iteration and learning best
deviations/speeds at different points, iterating toward the most optimal
path. Claude Code (AI agent) -- validated the idea's reasoning
(explained why uniform global reward constants can't express "fast here,
careful there," unlike a position/time-specific signal), designed and
implemented a "best-known-trajectory" reward bonus, tested it, and found
a severe regression with a diagnosable root cause rather than just a bad
outcome.

**Question or objective:** Design and test a location/time-specific
reward mechanism -- reward relative to the best-known pace ever achieved
at the same point in an episode -- as a fundamentally different lever
than the four failed global-constant tweaks from earlier today.

**What we investigated or changed:**

- New module `src/training/trajectory.py`: `BestTrajectoryTracker`,
  recording the best cumulative `odometry.distance_m` ever reached at
  each `sensors.tick` across a training run (both reset to 0 per fresh
  car/episode per SENSORS.md, making them comparable across races/copies
  without needing the private track-position API). `bonus_m()` returns
  how much more (or less) distance was gained this tick vs. the
  best-known run at the same point; `update()` ratchets the record up,
  never down. `WEIGHT_TRAJECTORY_BONUS = 1.0`, same units/scale as the
  existing progress reward.
- `src/training/controller.py`: added `TrainingState.trajectory:
  BestTrajectoryTracker | None = None` (opt-in) and wired the bonus into
  `TrainableController.__call__`'s reward computation, gated behind
  `self.training` like the rest of the training-only logic.
- `scripts/train_sac.py`: new `--trajectory-bonus` flag (default off, so
  existing behavior is unaffected unless explicitly requested), sizing
  the tracker to the configured round length.
- Added 7 unit tests for the tracker (`tests/test_training_trajectory.py`)
  and 2 for its controller integration
  (`tests/test_training_controller.py`) -- caught and fixed an arithmetic
  mistake in a self-written clamping test before it could hide a real bug.
- Ran with `--trajectory-bonus`, otherwise identical to the races=40
  reference (seed 110, races=40, round_seconds=120, buffer_capacity=800000).

**Evidence:**
- Sources or documentation: read `_run_headless_student_runtime_step` in
  `src/racing/race/head_to_head.py` (not previously read this closely) to
  investigate the regression, which is what surfaced the root cause.
- AI-agent assistance: Claude Code did not stop at "this failed" --
  noticed the failure had a *different shape* than every prior regression
  today (incoherent/unstable rather than one consistent bad strategy),
  formed a specific hypothesis about shared-state timing within self-play,
  and verified it by reading the actual tick-loop code before writing up
  a root cause rather than a guess. Ran
  `ruff`/`pyright`/`pytest -q` (156 passed) before running the experiment.
- Commits or code: `src/training/trajectory.py` (new),
  `src/training/controller.py` (`TrainingState.trajectory` field, bonus
  wiring), `scripts/train_sac.py` (`--trajectory-bonus` flag),
  `tests/test_training_trajectory.py` (new, 7 tests),
  `tests/test_training_controller.py` (2 new tests),
  `docs/rl_design.md` §6 (causal test 14).
- Experiment output: `experiments/2026-09-02_trajectory-bonus-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a; not adopted.

**What we observed:** The worst outcome of the day. Self-play's own
training distance collapsed to 3.0m/0.0m over 40 whole races (every prior
run: hundreds to tens of thousands of meters). 0 laps completed in all 20
evaluation races. Marshal recoveries up to 56 in a single race (previous
worst across every experiment today: 21). Up to 93% of a race spent
stuck. Unlike every prior regression today -- each a coherent single
strategy (uniformly faster-and-crashier, or uniformly
slower-and-cautious) -- this was genuinely unstable, incoherent training.

Root cause: self-play's two copies (challenger, incumbent) are controlled
sequentially within each physics tick and share one `TrainingState`,
including now one `BestTrajectoryTracker`. Whichever copy is processed
first in a tick writes that tick's "record" *before* the second copy's
bonus is computed from it -- so a copy can be compared against a "best"
its own rival just set in the same race, at the same instant, not a
genuinely separate historical best. Since one grid position starts ahead
of the other (README), this is a systematic, adversarial corruption
between the two copies, not the intended self-improving curriculum.

**Decision and rationale:** `--trajectory-bonus` defaults to off, so
existing default behavior is unaffected. Not adopting this checkpoint.
The underlying idea remains well-motivated -- the bug is specifically in
*when* the shared record is written relative to when concurrent copies
read it mid-race, not in the concept of a best-known-trajectory reward at
all. Worth fixing and retrying, not abandoning.

**Next steps:**
1. Fix the tracker's update timing before trying again -- most direct
   option: only write records from a fully-completed episode (after
   elimination or round end), using that episode's whole trajectory,
   rather than updating continuously while other copies are still racing
   and reading it mid-episode.
2. Separately worth checking once the primary bug is fixed: whether
   warmup's random actions (pure noise, the first `--warmup-steps`
   transitions) can plant spurious early-tick records that later,
   better-controlled episodes then have to fight against.
3. If a proper fix is more design work than warranted right now,
   deprioritize below the `--resume-from` checkpoint-continuation idea or
   simply treating races=40 as the practical best result for the
   remaining project time.

---

## 2026-09-02 (continued, 3)

**Participants and contributions:** Charlotte Tsui -- directed "proceed
with 1 [fix the trajectory bug and retest] then continue with 2
[--resume-from / other work]." Claude Code (AI agent) -- fixed the bug,
verified the fix worked, found the fixed mechanism still didn't beat the
reference, then built and started testing `--resume-from` as the
genuinely different next lever.

**Question or objective:** Fix the same-tick, same-race cross-copy bug in
`BestTrajectoryTracker` found last entry, confirm the fix actually
resolves the instability, and add checkpoint-resume support to
`scripts/train_sac.py` so training can continue from an existing
checkpoint instead of always starting from a random init.

**What we investigated or changed:**

- **Bug fix** (`src/training/trajectory.py`, `src/training/controller.py`):
  decoupled reads from writes. `BestTrajectoryTracker.snapshot()` returns
  a frozen copy of the live record; a new free function
  `bonus_from_snapshot()` computes the bonus against a passed-in array
  instead of the live tracker. `TrainableController` now takes a snapshot
  once, on its own episode's first call (safe because self-play
  constructs every copy for a race before any of them run a single tick,
  confirmed by re-reading `_run_headless_student_race` in
  `head_to_head.py`), and reads bonuses from that frozen snapshot for the
  whole episode, while still writing live updates to the shared tracker
  for future episodes. Added 2 regression tests
  (`tests/test_training_trajectory.py`) specifically encoding the
  causal-test-14 bug scenario, plus a snapshot-isolation test.
- **Re-ran the identical `--trajectory-bonus` config** (seed 110,
  races=40, round_seconds=120, buffer_capacity=800000) to check whether
  the fix actually resolved the instability, and whether the corrected
  mechanism beats the plain reference.
- **Added `--resume-from`** to `scripts/train_sac.py`: loads an existing
  checkpoint's policy/critics/log_alpha via `SACAgent.load()` (already
  supported at the agent level, just not exposed on the CLI) instead of
  constructing a fresh `SACAgent`. Documented in the flag's help text that
  the replay buffer and critic-optimizer momentum are *not* resumed
  (start fresh), and that `--warmup-steps 0` is recommended when resuming
  (otherwise the resumed policy's actions are discarded for the first
  `--warmup-steps` transitions in favor of random noise). Smoke-tested
  with a 1-race, 5-second run against the races=40 checkpoint before
  trusting it for a real experiment.
- Started a real resume test: `--resume-from` the races=40 checkpoint,
  `--warmup-steps 0`, otherwise the same races=40/round_seconds=120/seed
  110 config, plain (non-trajectory) reward -- to isolate whether
  continuing from a strong prior beats another from-scratch run,
  independent of the trajectory-bonus question. Still running as this
  entry is being written.

**Evidence:**
- Sources or documentation: re-read `_run_headless_student_race` in
  `src/racing/race/head_to_head.py` to confirm all self-play copies for a
  race are constructed before any of them are called, which is what makes
  first-call snapshotting safe.
- AI-agent assistance: Claude Code wrote the two new tests to directly
  encode the exact bug scenario from the previous entry (same-tick,
  same-race rival corruption) as a regression test, not just generic
  coverage -- so a future change that reintroduces this specific bug
  would be caught. Smoke-tested `--resume-from` with a fast, cheap run
  before committing to a full 40-race experiment. Ran
  `ruff`/`pyright`/`pytest -q` (158 passed) before running anything.
- Commits or code: `src/training/trajectory.py` (snapshot mechanism),
  `src/training/controller.py` (snapshot wiring),
  `tests/test_training_trajectory.py` (2 new tests),
  `scripts/train_sac.py` (`--resume-from` flag), `docs/rl_design.md` §6
  (causal test 15).
- Experiment output:
  `experiments/2026-09-02_trajectory-bonus-fixed-seed110/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`); `experiments/2026-09-02_resumed-more-training-seed110/`
  in progress.
- Leaderboard result: n/a.

**What we observed:** The fix worked. Marshal recoveries dropped from
22.25/race (the bug) back to 0.50/race (near the reference's 0.15),
off-track/wall-contact time back near zero, self-play's own training
distance back to a normal order of magnitude (23,603m/25,141m over 40
races, vs. the bug's 3.0m/0.0m). This is no longer unstable, broken
training.

But the corrected mechanism still didn't beat the plain races=40
reference: 2.70 avg laps (vs. 4.10), 32.73s avg best lap (vs. 24.76s),
2/20 eliminated (vs. 0/20). Max speed was marginally higher (17.33 vs.
15.53 m/s) but didn't translate into better lap times or reliability.
This is the sixth consecutive reward-tuning/mechanism attempt today to
fail against races=40, though the first that's a genuine design bug fix
rather than a hyperparameter guess -- and with n=1 per condition (every
experiment today is a single from-scratch run), it's not yet possible to
separate "the idea doesn't help here" from "this particular run had worse
luck than the reference run did."

**Decision and rationale:** Keep the trajectory-bonus fix (correct,
tested, default-off, no impact on existing behavior). Not adopting the
trajectory-bonus checkpoint. Given six straight from-scratch attempts
haven't beaten races=40, shifted the next experiment to a genuinely
different mechanism -- `--resume-from` -- rather than a seventh reward
variant trained from scratch.

**Next steps:**
1. Finish and evaluate the in-progress `--resume-from` test.
2. If continuing from the strong prior helps, consider a follow-up
   combining `--resume-from` with `--trajectory-bonus` -- refining an
   already-competent policy with the bonus, rather than learning both
   "how to drive" and "beat your own record" from scratch at once.
3. Otherwise, treat races=40 as the practical best result for this
   project's remaining time and shift focus to other open items: broader
   seed testing, repackaging `controllers.race_faster`, or the
   `controllers.minimum_viable` module gap.

---

## 2026-09-02 (continued, 4)

**Participants and contributions:** Charlotte Tsui -- none this entry
(continuation of "proceed with 1 then continue with 2"). Claude Code (AI
agent) -- finished the resume-from test, recognized it confirmed rather
than contradicted an earlier finding, and synthesized the full
seven-experiment arc into one clear recommendation.

**Question or objective:** Finish testing whether `--resume-from`
(continuing training from the already-good races=40 checkpoint) beats
training a new reward variant from scratch.

**What we investigated or changed:** Evaluated the completed
`--resume-from` run (races=40 checkpoint, +40 more races, plain reward,
`--warmup-steps 0`, otherwise identical config) against the races=40
reference.

**Evidence:**
- Sources or documentation: none beyond this run's output and the two
  checkpoints being compared.
- AI-agent assistance: Claude Code recognized the result's significance
  immediately rather than treating it as a standalone data point --
  compared the resumed run's landing speed (13.91 m/s) against both the
  reference (15.53 m/s) *and* the previous day's races=80 result
  (13.1-14.7 m/s, same reward, different training path), which is what
  revealed the two independent confirmations of the same mechanism. Ran
  `ruff`/`pyright`/`pytest -q` before this entry's doc-only changes.
- Commits or code: `docs/rl_design.md` §6 (causal test 16, plus an
  overall seven-experiment synthesis).
- Experiment output:
  `experiments/2026-09-02_resumed-more-training-seed110/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a; not adopted.

**What we observed:** Safety exactly preserved (identical 0.000 damage,
0.00s off-track/wall-contact, 0.15 marshal/race, before and after
resuming). But laps halved (4.10 -> 2.00) and best lap time nearly
doubled (24.76s -> 46.36s), landing at 13.91 m/s -- between races=40's
15.53 m/s and races=80's 13.1-14.7 m/s (2026-09-01's causal test 11, same
reward, a comparable total gradient-update count reached via a
from-scratch run instead of resuming). Two independent training paths
(from-scratch-then-more, and resume-then-more) now agree: under
`MAX_REWARDED_SPEED_MPS = 10.0`, more optimization converges the policy
toward ~13-15 m/s regardless of how it gets there -- races=40's ~15.53
m/s looks like a residual of not-yet-fully-converged training, not a
stable point the reward actually favors.

**Decision and rationale:** Not adopting this checkpoint.
`2026-09-01_more-training2-seed110` (races=40) remains best. Zooming out
across the full "proceed with 1 then continue with 2" arc plus everything
before it: **seven consecutive experiments today** (cap=12.0, cap=20.0,
more training from scratch [races=80], steering smoothness,
trajectory-bonus buggy, trajectory-bonus fixed, this resumed run) have
failed to beat races=40 on speed -- either by regressing safety or by
regressing speed while keeping safety intact. This is no longer "haven't
found it yet" -- the mechanism is understood: `MAX_REWARDED_SPEED_MPS`
sets a real ceiling that any amount of further optimization converges
toward, and directly raising it (tried twice, small and large) either did
nothing or broke the speed/control balance. **Recommending races=40 as
the practical stopping point for this reward structure and shifting focus
to consolidation** rather than an eighth variant.

**Next steps:**
1. **(recommended)** Treat races=40 as the practical best result for this
   SAC track's remaining time. Shift to: broader seed testing beyond the
   fixed 5 to build robustness confidence, repackaging
   `controllers.race_faster` from this checkpoint (it currently packages
   the older races=20 one, per the 2026-09-01 packaging entries), and the
   `controllers.minimum_viable` module gap.
2. If more speed is wanted later, the next genuinely new idea needs to
   change the reward's structural ceiling itself, not just its value --
   e.g. a progress term that scales with *safe* speed (conditioned on
   wall-proximity margin) rather than a flat cap. This is a bigger design
   change than anything tried today and would warrant its own dedicated
   investigation, not another quick variant.
3. Still open: the `trajectory-bonus` + `resume-from` combination (refine
   an already-competent policy with the bonus rather than learning both
   "how to drive" and "beat your own record" from scratch at once) was
   never tried -- noted for completeness, but deprioritized given the
   seven-experiment pattern above.

---

## 2026-09-02 (continued, 5)

**Participants and contributions:** Charlotte Tsui -- said the car is
currently too safe, asked to keep iterating until throttle/speed
increases, and asked specifically for drift-style cornering around
curves. Claude Code (AI agent) -- explained why a literal hand-coded
drift controller is out of scope for this track, designed and ran a
structural (not incremental) reward change instead, and found the first
genuine speed increase of the day that didn't cost safety.

**Question or objective:** Increase throttle/speed specifically (the
races=40 checkpoint was judged too safe/conservative), and explore
whether drift-style cornering can be encouraged, after seven consecutive
experiments (2026-09-01/02) failed to beat races=40 via value changes on
the `MAX_REWARDED_SPEED_MPS` axis or new reward terms.

**What we investigated or changed:** Reframed "implement drifting" for
what's actually buildable here: not a hand-coded rule-based drift
controller (which would fight the self-play/SAC architecture and this
track's scope, and its effectiveness couldn't be verified without first
testing whether the physics model rewards it at all), but reward
conditions that let a drift-like technique emerge and be rewarded if it's
actually faster, without forbidding it outright. Concretely, in
`src/training/reward.py`:

- Removed `MAX_REWARDED_SPEED_MPS` entirely -- `forward_progress_m` is
  now the raw, uncapped `odometry.speed_mps` term. Every previous
  attempt on this axis (values 10.0, 12.0, 20.0) changed a number within
  a structure that had a hard ceiling; this removes the ceiling itself.
- Added `WALL_PROXIMITY_SPEED_SCALE_MPS = 10.0`: the existing wall-
  proximity penalty now scales with current speed (2x its base value at
  10 m/s, 3x at 20 m/s, ...) instead of being speed-blind, so "close to a
  wall at 1 m/s" and "close to a wall at 35 m/s" are no longer priced
  the same -- risk now tracks how dangerous the *current situation*
  actually is.

Trained from scratch, same seed (110), races=40, round_seconds=120,
buffer_capacity=800000 as every comparison this week.

**Evidence:**
- Sources or documentation: none beyond this run's output and the
  races=40 reference's known numbers.
- AI-agent assistance: Claude Code was explicit up front about what it
  would and would not build (no rule-based drift override) before
  writing any code, rather than either refusing the request or silently
  reinterpreting it. After the run, computed precise per-metric averages
  rather than trusting the printed totals -- this run's totals looked
  like an unambiguous win at first glance (higher summed distance), and
  the precise averages revealed a more honest, mixed picture (higher
  speed, but slightly worse lap time/count). Ran `ruff`/`pyright`/
  `pytest -q` (159 passed, 2 old cap-tests replaced with 3 new ones for
  the uncapped/speed-scaled behavior) before running the experiment.
- Commits or code: `src/training/reward.py` (removed
  `MAX_REWARDED_SPEED_MPS`, added `WALL_PROXIMITY_SPEED_SCALE_MPS`),
  `tests/test_training_reward.py` (3 new tests replacing 2 obsolete
  ones), `docs/rl_design.md` §6 (causal test 17).
- Experiment output:
  `experiments/2026-09-02_uncapped-speed-scaled-risk-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** The first genuine speed increase of the entire
2026-09-01/02 arc that did *not* cost safety. Avg max speed 15.53 -> 17.13
m/s (+10%), with damage/eliminations/off-track/wall-contact all still
excellent (0/20 eliminated, unchanged; other metrics near-zero, not
literally 0.000 anymore but close). Every one of the seven prior speed
attempts either broke safety outright or only preserved it by also
reducing speed -- this is the first to move both in the intended
direction (well, one of the two: speed up, safety held).

The catch: avg laps went 4.10 -> 3.90 and avg best lap time 24.76s ->
28.46s (slower) -- the extra top speed didn't translate into a better
overall race. Not yet diagnosed why (can't tell from headless stats alone
whether it's driving differently through corners specifically, or just
reaching a higher peak speed on straights without that helping lap time).

**Decision and rationale:** Provisionally adopting this checkpoint as the
new reference point, but explicitly flagged as an unresolved *trade-off*
(speed up, lap time/count down), not an unambiguous improvement -- unlike
every other adoption decision this week, which were clear wins on every
metric that mattered. Reasoning for why this experiment differs from the
seven that failed: those all changed a *value* on an axis with a real
ceiling; this removes the ceiling's existence. "More training converges
speed down toward the cap" (causal tests 11, 16) has no obvious reason to
still apply once there's no cap to converge toward -- worth testing
directly rather than assuming the old finding still holds.

**Next steps:**
1. **(recommended)** Train further under this new, uncapped reward
   structure specifically -- a genuinely different question from every
   prior "more training" experiment, all of which were run under the old
   capped reward.
2. Watch it live (`controllers.sac_candidate`, now pointed at this
   checkpoint) to see qualitatively whether it's attempting anything
   drift-like through corners, or just carrying more speed on straights
   without changing cornering technique.
3. If lap time still doesn't improve, consider loosening
   `WEIGHT_CENTER_OFFSET` alongside this change -- it currently penalizes
   any lateral deviation from centerline uniformly, which would suppress
   a real racing line (let alone a drift) even if it's genuinely faster.

---

## 2026-09-03

**Participants and contributions:** Charlotte Tsui -- asked what the
best approach would be for a full restart-and-retrain, optimizing for
speed and safety and open to a different mechanism such as drifting;
after hearing the recommendation, said to set it up as the next
experiment. Claude Code (AI agent) -- reviewed the full causal-test
history, recommended against a true from-scratch restart, designed and
ran the experiment, caught and fixed its own evaluation mistake, and
documented the (negative) result.

**Question or objective:** If restarting SAC training from scratch,
what's the highest-leverage single next step to improve speed while
keeping safety, and possibly enable drift-style cornering?

**What we investigated or changed:** First, reviewed
`docs/rl_design.md` §6's full history (17 prior causal tests) rather than
proposing something new unprompted. Recommended *not* a literal restart:
keep seed `110` (the only seed that avoided the `909`-style "do nothing"
dead end) and the current best reward (`2026-09-02_uncapped-speed-
scaled-risk-seed110` — uncapped `forward_progress_m` +
`WALL_PROXIMITY_SPEED_SCALE_MPS`, the only prior attempt that gained
speed without losing safety), and test the one lever flagged in that
run's own notes but not yet tried: loosening `WEIGHT_CENTER_OFFSET` so
the policy has room for a wider racing line through corners. User agreed
and asked to proceed.

Implemented: `WEIGHT_CENTER_OFFSET` 0.3 -> 0.15 in
`src/training/reward.py` (a deliberate half-step, not a full return to
the old 0.05 that caused the 2026-09-01 off-track regression this weight
originally fixed), everything else identical to the reference (seed 110,
races=40, round_seconds=120, buffer_capacity=800000), trained from
scratch via `scripts/train_sac.py`.

**Evidence:**
- Sources or documentation: re-read `docs/rl_design.md` §6 in full and
  `experiments/2026-09-02_uncapped-speed-scaled-risk-seed110/notes.md`
  before proposing the change, to avoid re-deriving or contradicting
  already-recorded findings (per this file's step 1).
- AI-agent assistance: Claude Code ran
  `ruff check`/`ruff format --check`/`pyright`/`pytest -q` (159 passed)
  before training. Launched training in the background and picked it up
  automatically on completion. **Caught its own mistake**: the first
  training invocation omitted `--eval-round-seconds 120`, so the initial
  evaluation silently ran at `eval_sac`'s 20s default instead of the
  reference's 120s -- noticed by diffing the saved `config.yaml` against
  the reference's before reporting any numbers (rather than reporting
  the 20s-round numbers, which looked like extreme regressions purely
  from the shorter round, e.g. 41-122m vs. the reference's hundreds of
  meters). Fixed by re-evaluating the same saved checkpoint via
  `scripts/eval_sac.py --eval-round-seconds 120` (no retraining needed --
  training itself had used the correct 120s round throughout). Computed
  precise per-metric averages directly from `eval_results.json` (not just
  the printed per-seed summary lines) to build the comparison table.
- Commits or code: `src/training/reward.py`
  (`WEIGHT_CENTER_OFFSET` tried at 0.15, reverted to 0.3, comment
  records both), `docs/rl_design.md` §6 (causal test 18).
- Experiment output:
  `experiments/2026-09-03_center-offset-half-seed110/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`, plus the mistaken `config_WRONG_20s.yaml` /
  `eval_results_WRONG_20s.json` kept for the record, unused in any
  comparison).
- Leaderboard result: n/a; not adopted.

**What we observed:** A clear regression, not the hoped-for trade-off.
Avg max speed **17.13 -> 8.80 m/s (-49%)**, avg laps **3.90 -> 1.00
(-74%)**, avg best lap time **28.46s -> 78.80s (+177%)**, wins vs.
`default_student_controller` **10/10 -> 3/10** (still 10/10 vs.
`crash_fast`). Safety stayed essentially flat: damage 0.003 -> 0.006
(still near-zero), off-track/wall-contact both still near-zero, 0/20
eliminated both before and after. So loosening the centerline penalty
did not trade safety for speed -- it just made the policy notably less
competent while safety, already at floor, stayed at floor.

**Decision and rationale:** Reverted `WEIGHT_CENTER_OFFSET` to `0.3`.
Not adopting this checkpoint. `2026-09-02_uncapped-speed-scaled-risk-
seed110` remains the reference/current best. Best-guess explanation:
under the already-uncapped speed reward, a uniform per-tick centerline
penalty was apparently doing real work keeping cornering deliberate, not
merely suppressing a better line underneath it -- weakening it broadly
(touching straights and safe cruising as much as corners) removed a
signal the policy depended on rather than unlocking one. Same failure
pattern as the steering-smoothness and idle-penalty attempts earlier
this week: a plausible-sounding loosened constraint that backfires.
Treating global `WEIGHT_CENTER_OFFSET` reduction as a dead end at this
step size, same as `MAX_REWARDED_SPEED_MPS` tuning was treated after its
three failed attempts.

**Next steps:**
1. **(recommended)** Treat `2026-09-02_uncapped-speed-scaled-risk-
   seed110` as the practical best result for this track's remaining time
   and shift to consolidation: broader seed testing beyond the fixed 5,
   repackaging `controllers.race_faster` from this checkpoint (it still
   packages the older races=20 checkpoint per the 2026-09-01 packaging
   entries), and the `controllers.minimum_viable` module gap.
2. If wider/drift-style cornering is still wanted, the next idea should
   target corners specifically -- e.g. a term conditioned on
   `camera.lookahead_offsets_m` curvature, or only relaxing the
   centerline penalty when wall-proximity margin is large -- rather than
   another global weight change that also touches straights and safe
   cruising, which is what made this attempt backfire.
3. Not yet tried: the `trajectory-bonus` (fixed, 2026-09-02) +
   `resume-from` combination, refining an already-competent policy with
   the best-known-trajectory bonus rather than learning both "how to
   drive" and "beat your own record" from scratch at once. Still
   deprioritized given the current focus, noted for completeness.

---

## 2026-09-07

**Participants and contributions:** Charlotte Tsui -- asked what a better
RL implementation could look like to produce a faster controller (rather
than another reward-weight tweak), then directed implementing and testing
n-step returns at n=3 against the current reference checkpoint. Claude
Code (AI agent) -- explained the algorithmic idea and its bias/variance
tradeoff at different n, implemented it, verified it against the existing
test suite plus new tests, smoke-tested it on a real (tiny) headless
race, then ran and analyzed the full causal test.

**Question or objective:** Every prior speed-focused experiment (causal
tests 9-18, 2026-09-01/09-02) changed the reward function or trained
longer -- none changed the SAC algorithm itself. Does switching the
critic's target from a 1-step TD backup to an n-step return (n=3) produce
a faster and/or safer controller than the current best checkpoint,
`2026-09-02_uncapped-speed-scaled-risk-seed110`?

**What we investigated or changed:**

- Explained the mechanism and its tradeoffs before writing any code:
  1-step TD only lets a delayed penalty (e.g. `WEIGHT_TERMINAL_PENALTY`
  for a crash several ticks after a risky action) reach the responsible
  earlier states via many sequential Bellman backups -- exactly the
  problem several 2026-09-01 causal tests (5, 6) worked around by raising
  the penalty's *magnitude* rather than shortening that path. n-step
  returns sum several real ticks of reward before bootstrapping, so a
  crash's consequence reaches nearby earlier states directly, in one
  update. Also walked through why n=3-5 is the standard starting range
  (off-policy replay means older transitions' summed rewards become
  increasingly stale relative to the current policy as n grows) rather
  than jumping straight to a larger n.
- Implemented n-step returns:
  - `src/training/replay_buffer.py`: `ReplayBuffer.push`/`ReplayBatch`
    gained a per-transition `discount` field (`gamma**actual_n`) instead
    of relying on a single shared scalar gamma, since a window truncated
    by early episode termination bootstraps over fewer than `n_step`
    ticks.
  - `src/training/sac.py`: `_update_critics`'s target now multiplies the
    bootstrap term by the batch's per-sample `discounts` tensor instead
    of `self.gamma`.
  - `src/training/controller.py`: `TrainableController` now holds a
    small per-car sliding window (`deque`) of raw 1-tick transitions;
    once it reaches `n_step` ticks it emits one n-step transition to the
    shared buffer and slides forward by one, or (if the episode
    terminates mid-window) immediately flushes every pending window --
    each remaining window emits its own transition with `done=True`,
    which is why termination produces multiple pushes, not one.
    `TrainingState` gained an `n_step: int` field (default 1).
  - `scripts/train_sac.py`: new `--n-step` CLI flag (default 1, so no
    existing run's behavior changes unless passed explicitly).
- Added 11 new tests: `ReplayBuffer` discount storage and round-trip,
  a critic-update test proving the target actually uses the per-sample
  discount (two agents with identical init, updated on transitions
  differing only in `discount`, must produce different critic losses),
  and four `TrainableController` tests (n_step=1 reproduces the exact
  prior single-push-per-tick behavior, n_step=3 holds transitions until
  the window fills, the emitted n-step return is the correct discounted
  sum with `discount == gamma**3`, and termination flushes every pending
  window rather than just one).
- Verified before running any real experiment: `ruff check` and
  `pyright` (project's strict-mode config) both clean; `pytest -q`
  (165 passed, up from 154 baseline); a tiny real headless race
  (`--races 1 --round-seconds 5 --n-step 3 --eval-round-seconds 5`)
  confirmed the plumbing runs against the actual simulator, not just
  synthetic unit-test sensors, before committing to the full run.
- Ran `scripts/train_sac.py --races 40 --round-seconds 120 --n-step 3
  --buffer-capacity 800000 --eval-round-seconds 120 --seed 110`,
  identical to the `2026-09-02_uncapped-speed-scaled-risk-seed110`
  reference except `--n-step 3` (default 1) as the single changed
  variable. Backgrounded (448.7s, exceeding the foreground timeout),
  picked up via the completion notification.

**Evidence:**
- Sources or documentation: none beyond this run's own output and the
  reference checkpoint's known numbers.
- AI-agent assistance: Claude Code did not report the headline
  scored-distance numbers (768.1m -> 869.8m) as the whole story --
  pulled full per-race detail (damage, off-track, wall-contact, max
  speed, laps, best lap time) across all 20 evaluation races for both
  this run and the reference before characterizing the result, the same
  discipline applied to every experiment on this track. Also caught and
  fixed a dating error before writing any evidence to disk: the
  experiment directory and every new doc reference were initially
  written as `2026-09-04` (a stale date carried over from context)
  instead of the actual current date, `2026-09-07` -- caught by
  cross-checking the environment's current-date context, renamed the
  experiment directory and corrected every reference (`docs/rl_design.md`,
  the experiment's own `notes.md`) before this entry was written, rather
  than leaving a wrong date in permanent lab-notebook evidence.
- Commits or code: `src/training/replay_buffer.py`, `src/training/sac.py`,
  `src/training/controller.py`, `scripts/train_sac.py`,
  `tests/test_training_replay_buffer.py`, `tests/test_training_sac.py`,
  `tests/test_training_controller.py`, `docs/rl_design.md` (section 4
  and section 6, causal test 19).
- Experiment output: `experiments/2026-09-07_nstep3-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** n=3 beat the 1-step reference on every tracked
metric except top speed, and even there the net effect (lap time) was
still better:

| | reference (1-step) | n-step=3 |
| --- | --- | --- |
| avg damage | 0.0028 | **0.0000** |
| avg off-track | 0.082s | **0.000s** |
| avg wall contact | 0.076s | **0.000s** |
| avg max speed | 17.13 m/s | 14.73 m/s (-14%) |
| avg laps | 3.90 | **4.00** |
| avg best lap time | 28.46s | **25.06s (-12%)** |
| avg scored distance | 768.1m | **869.8m** |
| eliminated | 0/20 | 0/20 |
| wins (both baselines) | 20/20 | 20/20 |

Every safety metric reached exact zero across all 20 evaluation races (5
seeds x 2 baselines, including the 4 held-out seeds), matching the best
safety result seen anywhere on this track. Unlike every one of the eight
prior speed-focused experiments (causal tests 9-18), this was not a
trade-off in either direction: safety improved, lap time improved, and
top speed alone went down. The most likely explanation, consistent with
the n-step motivation, is less time lost to hesitation or an indirect
line through corners rather than a pure straight-line speed change --
not confirmed mechanistically, since headless stats alone can't
distinguish that from other explanations; a live watch or per-tick
sensor logging would be needed to check cornering technique directly.

**Decision and rationale:** Adopted `2026-09-07_nstep3-seed110` as the
new best/reference checkpoint -- it dominates the prior reference on
every metric except top speed, where the net race-level effect is still
an improvement. This is also, methodologically, the first change on this
track to alter the RL algorithm itself rather than the reward or training
budget, and it's the first speed-adjacent experiment all week to not cost
something else in exchange. `docs/rl_design.md` section 4 updated in the
same session (architecture change) per CLAUDE.md step 5, alongside
section 6's causal-test log.

**Next steps:**
1. Watch the checkpoint live (`controllers.sac_candidate`, once pointed
   at this checkpoint) to check qualitatively whether cornering technique
   changed, per the open mechanism question above.
2. Try `n=5` or `n=10` next now that `n=3` has shown a real, positive
   effect -- not guaranteed to keep improving monotonically, since larger
   n trades lower bias for higher variance and more off-policy staleness
   in the replay-buffer rewards (discussed with Charlotte before
   implementing).
3. **Repackage `controllers.race_faster`** from this checkpoint -- still
   open since the 2026-09-01 packaging entries; the packaged module ships
   the much older races=20 checkpoint, now several strictly-better
   checkpoints behind.
4. Still open, unchanged from every prior entry: a genuine multi-training-
   seed sweep at the current best config, to check whether n-step's
   benefit (like every result on this track so far) is specific to seed
   110 or general.
5. Still open: the `controllers.minimum_viable` module gap from the
   2026-09-01 packaging entries.

---

## 2026-09-07 (continued)

**Participants and contributions:** Charlotte Tsui -- directed trying
n=5 next and explicitly prioritized reducing hesitation over other goals;
when the first attempt at a hesitation fix turned out to be a guess, asked
to investigate the actual cause before writing more reward code. Claude
Code (AI agent) -- ran n=5 (a regression), built a per-tick diagnostic
that overturned its own initial hypothesis, disabled the now-unsupported
fix it had already started, implemented the diagnosis-driven fix instead,
and reported a mixed (not unambiguous) result including a self-found
outlier.

**Question or objective:** Does n=5 improve on n=3's result? Separately,
and prioritized above further n-step tuning: what is actually causing the
low-progress/"hesitation" time seen in every evaluation run so far, and
can it be fixed directly rather than guessed at?

**What we investigated or changed:**

- **n=5 (causal test 20):** same seed/races/round-length/reward as
  causal test 19, only `--n-step 3 -> 5`. Regression, not further
  improvement (see "what we observed").
- **First hesitation attempt (built, then superseded before testing):**
  assumed steering oscillation was the cause (matching the standing
  "reduce hesitation" refinement item) and implemented
  `WEIGHT_STEERING_REVERSAL` in `src/training/reward.py` -- unlike the
  already-failed `WEIGHT_STEERING_SMOOTHNESS` (penalizes raw yaw-rate
  *magnitude* change, causal test 13, regressed badly because it
  couldn't distinguish real cornering from wobble), this penalizes yaw-
  rate *sign reversals* specifically, so a held turn (consistent sign,
  changing magnitude) doesn't trigger it. Added 4 tests. Before running
  it, Charlotte asked to investigate the actual cause first rather than
  test another guess.
- **Diagnosis (read-only):** wrote a one-off script (not committed to the
  repo) using `sensor_sample_callback` to log `contact.robot`, speed,
  position, and yaw rate per tick across a real evaluation race with the
  `2026-09-07_nstep3-seed110` checkpoint. Found `wall_contact` was exactly
  0.00s across every evaluated race in that checkpoint's own
  `eval_results.json`, while `car_contact` was frequently 2-4+ seconds per
  race. Ran the diagnostic against `crash_fast` (which never moves --
  confirmed by reading `src/controllers/crash_fast.py`, throttle=0.0
  always) and found repeated `contact.robot` windows at track positions
  spaced ~180-190m apart, matching the track's known ~183m lap length --
  the car collides with the stationary opponent at roughly the same point
  once per lap, every lap. Read `src/training/observation.py` and
  confirmed neither `camera.competitors` nor `sensors.lidar` (the only
  public fields that detect other robots) was in the 17-dim observation
  vector -- both were explicitly deferred by the original design pending
  solo-driving competence, which has been solid since 2026-09-01 and was
  never revisited.
- **Disabled the unsupported fix:** set `WEIGHT_STEERING_REVERSAL = 0.0`
  (mechanism kept, tests updated to the "currently disabled" convention
  already used for `WEIGHT_STEERING_SMOOTHNESS`) so it wouldn't confound
  the real fix.
- **Implemented the diagnosis-driven fix:** added `sensors.lidar` (7
  beams, same angles/encoding as the existing `wall_lidar`) to
  `src/training/observation.py`'s observation vector.
  `OBSERVATION_DIM` 17 -> 24. Added 2 tests (infinite obstacle-lidar
  beams map to 1.0; a nearby car is detected via the new beams even with
  no wall nearby, and does not affect the existing wall-lidar beams).
  Verified `ruff`/`pyright` clean and 168 tests passing, then a real
  smoke race (`--races 1 --round-seconds 5`) before committing to a full
  run, since this changes the network's input dimension, not just a
  constant.
- **Causal test 21:** trained with the new observation, otherwise
  identical to `2026-09-07_nstep3-seed110` (seed 110, races=40,
  round_seconds=120, buffer_capacity=800000, n_step=3, unchanged reward).
- **Outlier investigation:** the aggregate result looked mixed (low-
  progress time up, not down) -- pulled every individual race's stats
  rather than trusting the average, found one race
  (`seed=2024 vs default_student_controller race=2`) with low-progress=
  22.17s vs. 1.65-4.40s everywhere else, plus the run's only nonzero
  wall-contact and real damage. Re-ran the same per-tick diagnostic
  against that specific seed/baseline/checkpoint to characterize it
  rather than discard it as noise.

**Evidence:**
- Sources or documentation: `src/racing/race/runtime.py`
  (`_race_runtime_is_stuck`, confirming "low progress" is a speed/contact-
  based signal, not a raw-odometer-distance one -- a first diagnostic
  pass using odometer distance found zero low-progress windows, which is
  what prompted reading this file); `src/controllers/crash_fast.py`
  (confirmed it never moves); `src/racing/student/api.py`
  (`CameraCompetitorReading`, `LidarSensors`, `RobotSensors` docstrings,
  confirming which fields see other cars).
- AI-agent assistance: Claude Code's first hesitation hypothesis
  (steering oscillation) was built into working, tested code before being
  investigated -- when asked to check the actual cause, it did not defend
  or rationalize the existing implementation; it disabled it and pursued
  the diagnostic with no attachment to the prior guess, which is what
  surfaced a completely different, better-supported root cause. Also
  did not report causal test 21's improved-on-average numbers without
  checking why the average was worse on one metric (low-progress) --
  pulling every individual race is what found the outlier, and the
  outlier was itself diagnosed with the same rigor as the main result
  rather than being dropped from the average silently. Ran
  `ruff`/`pyright` (strict, 0 errors) and `pytest -q` (168 passed) before
  each real training run.
- Commits or code: `src/training/reward.py` (`WEIGHT_STEERING_REVERSAL`,
  `YAW_RATE_REVERSAL_THRESHOLD_DEGREES_PER_S`, disabled at weight 0.0),
  `src/training/observation.py` (`OBSTACLE_LIDAR_BEAM_ANGLES_DEGREES`,
  `OBSTACLE_LIDAR_CAP_M`, `OBSERVATION_DIM` 17 -> 24),
  `tests/test_training_reward.py`, `tests/test_training_observation.py`,
  `docs/rl_design.md` (section 2.1 and section 6, causal tests 20-21).
- Experiment output: `experiments/2026-09-07_nstep5-seed110/`,
  `experiments/2026-09-07_obstacle-lidar-nstep3-seed110/` (each with
  `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

n=5 regressed sharply relative to n=3 despite near-identical training
budget: avg low-progress time rose 3.18s -> 5.07s (worse than even the
1-step reference), avg laps 4.00 -> 1.50, avg best lap time 25.06s ->
59.37s. Not adopted; reverted to n=3.

The obstacle-lidar addition produced a real, consistent improvement on
19 of 20 evaluation races (avg car-contact 1.595s -> 0.952s, marshal/race
0.25 -> 0.10, best lap time 25.06s -> 24.79s, max speed 14.73 -> 23.24
m/s) but the full-20-race average for low-progress time got *worse*
(3.18s -> 3.61s) because of one severe outlier (22.17s, vs. 1.65-4.40s in
every other race) -- excluding just that one race, low-progress improves
to 2.63s, better than the reference. Diagnosing the outlier directly
found the car spent ~54 of 120 seconds below 1.2 m/s, with a contact
window showing negative (reversing) speed while off-center, consistent
with getting physically wedged against the *moving* opponent near the
track edge -- a different, harder-to-recover-from failure mode than the
brief bump-and-clear contacts seen everywhere else, including in this
same run's other 19 races.

**Decision and rationale:** Adopted
`2026-09-07_obstacle-lidar-nstep3-seed110` as the new best/reference
checkpoint -- the aggregate improvement is real and consistent, and the
outlier, while severe, is rare (1/20) and non-fatal (no elimination).
Reporting the outlier explicitly rather than only the flattering 19-race
subset, per this notebook's standing practice of verifying surprising
numbers (favorable or not) before writing them down. Not adopting n=5.
Keeping `WEIGHT_STEERING_REVERSAL` disabled (mechanism preserved, not
proven wrong -- just not the dominant cause, and untested in isolation).
`docs/rl_design.md` section 2.1 (observation) and section 6 (causal-test
log) both updated in the same session per CLAUDE.md step 5, since this
changed the architecture (`OBSERVATION_DIM`), not just a reward constant.

**Next steps:**
1. A seed sweep at this checkpoint to check whether the stuck-against-
   opponent failure mode is specific to one spawn/track geometry or
   general -- still no checkpoint on this track has had a real multi-seed
   sweep.
2. A competitor-proximity reward term (mirroring
   `WEIGHT_WALL_PROXIMITY`'s speed-scaled design) now that the
   observation can support it, to teach proactive avoidance rather than
   only reactive sensing.
3. Re-test `WEIGHT_STEERING_REVERSAL` on its own now that the dominant
   hesitation cause has a different fix, to see if it matters for any
   residual wobble in isolation.
4. Still open, unchanged from every prior entry: a genuine multi-
   training-seed sweep more broadly, and repackaging
   `controllers.race_faster` (still ships the races=20 checkpoint from
   2026-09-01, now several generations behind).
5. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-07 (continued, 2) / 2026-09-08

**Participants and contributions:** Charlotte Tsui -- directed proceeding
with the natural next step (a competitor-proximity reward term). Claude
Code (AI agent) -- implemented it, ran the causal test, found a clear
regression by checking per-race detail rather than the average, and
diagnosed why before writing anything down.

**Question or objective:** Does adding a proactive competitor-proximity
penalty (mirroring `WEIGHT_WALL_PROXIMITY`'s design) reduce car-contact
and prevent the stuck-against-opponent outlier found in the previous
entry's obstacle-lidar experiment, now that the observation includes
`sensors.lidar`?

**What we investigated or changed:**

- Added `WEIGHT_ROBOT_PROXIMITY` to `src/training/reward.py`: the direct
  structural analog of `WEIGHT_WALL_PROXIMITY`, penalizing proximity to
  the nearest reading from `camera.competitors` (ramping from 0 at
  `ROBOT_WARNING_DISTANCE_M = 8.0` to a max at 0m, scaled by the same
  speed-risk multiplier already computed for wall proximity). No angle
  restriction on the competitor reading, unlike the wall-proximity beams
  (front-only) -- flagged explicitly in the code comment as a possible
  weak point before running anything.
- Added `_robot_proximity_penalty` helper and generalized the existing
  `_proximity_ratio` helper (previously wall-specific) to take a
  `warning_distance_m` parameter so both wall and robot proximity share
  it. Added 3 tests (nearby competitor penalized, distant competitor
  ignored, nearest-of-multiple used).
- Verified `ruff`/`pyright` (strict, 0 errors) and `pytest -q`
  (171 passed) before running anything, then a real, tiny smoke race
  before committing to a full run.
- Ran `scripts/train_sac.py --races 40 --round-seconds 120 --n-step 3
  --buffer-capacity 800000 --eval-round-seconds 120 --seed 110`, otherwise
  identical to `2026-09-07_obstacle-lidar-nstep3-seed110` -- the new
  reward term is the only changed variable.

**Evidence:**
- Sources or documentation: none beyond this run's own output and the
  reference checkpoint's known numbers.
- AI-agent assistance: Claude Code did not report the average car-contact
  improvement (0.952s -> 0.825s) as a win -- pulled every individual
  race's stats, which is what surfaced that `seed=110 vs crash_fast`
  (both races, reproducibly) had gotten *worse* than any prior
  configuration on the exact metric this term targeted, and that laps/
  lap-time regressed broadly across nearly every race, not just one
  outlier. Formed and stated a specific mechanistic hypothesis (no angle
  restriction causing generalized rather than targeted caution) rather
  than reporting the regression as unexplained. Ran
  `ruff`/`pyright`/`pytest -q` (169 passed, after updating 3 tests to the
  file's "kept but disabled" convention) after disabling the term.
- Commits or code: `src/training/reward.py` (`WEIGHT_ROBOT_PROXIMITY`,
  `ROBOT_WARNING_DISTANCE_M`, `_robot_proximity_penalty`, generalized
  `_proximity_ratio`; tried at weight=1.0, disabled to 0.0),
  `tests/test_training_reward.py`, `docs/rl_design.md` section 6
  (causal test 22).
- Experiment output:
  `experiments/2026-09-07_robot-proximity-obstacle-lidar-nstep3-seed110/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** A regression, not an improvement, on the metric
that mattered most. Avg car-contact improved only slightly (0.952s ->
0.825s), but avg laps dropped 4.00 -> 2.90, avg best lap time rose 24.79s
-> 35.31s (+42%), avg marshal/race rose 5.5x (0.10 -> 0.55), and max
speed converged to a suspiciously uniform ~17.3-17.7 m/s across nearly
every race (previously 14.7-23+ m/s, situation-dependent) -- a broad,
uniform slowdown rather than a targeted fix. Worse, the specific case
this term was meant to prevent got *worse*: `seed=110 vs crash_fast`
(both races) rose to 6.3-6.8s of car-contact with 2 marshal recoveries
and only 2 laps completed -- exceeding the prior worst case (2.68s) this
term targeted.

**Decision and rationale:** Not adopted. Disabled `WEIGHT_ROBOT_PROXIMITY`
(weight 0.0, mechanism kept in code) -- same convention as this file's
other two reverted terms (`WEIGHT_STEERING_SMOOTHNESS`,
`WEIGHT_STEERING_REVERSAL`). Likely mechanism: with no angle restriction,
the penalty fires for any nearby competitor regardless of whether it's
actually in the way, teaching generalized caution rather than targeted
collision avoidance -- for a stationary mid-track blocker specifically,
that caution plausibly makes committing to a clean pass harder, not
easier. Same failure family as other plausible-sounding caution terms on
this track that backfired into overcaution rather than a targeted fix.
`2026-09-07_obstacle-lidar-nstep3-seed110` remains the reference
checkpoint.

**Next steps:**
1. An angle-restricted variant (mirroring
   `WALL_WARNING_BEAM_ANGLES_DEGREES`'s front-only beams) is untested and
   may avoid this failure mode.
2. A closing-speed-scaled variant (`closing_speed_mps` from
   `CameraCompetitorReading` instead of own absolute speed) is also
   untested -- would only penalize proximity while actually gaining on
   the competitor.
3. Two consecutive hesitation-specific reward fixes (steering-reversal,
   then this) have not been the answer -- consider treating the
   obstacle-lidar observation change alone (real improvement on 19/20
   races) as sufficient for now, and revisit competitor-avoidance reward
   shaping later with more diagnostic detail on the seed=110/crash_fast
   case specifically.
4. Still open, unchanged from every prior entry: a genuine multi-
   training-seed sweep, and repackaging `controllers.race_faster`.
5. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08

**Participants and contributions:** Charlotte Tsui -- asked for a list and
explanation of available next approaches, then directed proceeding with
the angle-restricted competitor-proximity variant followed by many
iterations of a multi-training-seed sweep. Claude Code (AI agent) --
implemented and ran the angle-restricted variant, found it catastrophically
worse than the already-reverted unrestricted version, reverted it, then
launched the seed sweep against the locked-in working config.

**Question or objective:** Does restricting the (already-reverted)
competitor-proximity penalty to a forward angle cone fix the broad
overcaution regression from the previous entry, without reintroducing the
stuck-against-opponent problem it was meant to solve? Then: characterize
how much this track's results depend on the training seed, since every
result so far comes from a single seed (110).

**What we investigated or changed:**

- Re-enabled `WEIGHT_ROBOT_PROXIMITY` (1.0) with a new
  `ROBOT_WARNING_ANGLE_DEGREES = 45.0` restriction: only a competitor
  within 45 degrees of straight ahead counts toward the penalty, mirroring
  `WALL_WARNING_BEAM_ANGLES_DEGREES`'s front-only beams. Updated
  `_robot_proximity_penalty` to filter by angle before finding the
  nearest qualifying competitor. Replaced the "disabled" test with active
  tests covering the angle filter (ahead penalized, beside/behind not,
  nearest-qualifying selection). Verified `ruff`/`pyright`/`pytest`
  (172 passed) and a smoke race before running the full experiment.
- Ran identical to the previous (reverted) robot-proximity experiment --
  seed 110, races=40, round_seconds=120, buffer_capacity=800000, n_step=3
  -- the angle restriction is the only new variable.
- Result was a severe regression (see below) -- reverted
  `WEIGHT_ROBOT_PROXIMITY` to 0.0 again, kept the angle-restriction code
  (verified correct via a direct unit test using a manual weight
  override, isolating "is the filtering logic right" from "did the
  policy's response to it work").
- Launched the multi-training-seed sweep against the now-locked-in config
  (n_step=3, obstacle lidar in the observation, no robot-proximity term):
  4 new training runs in parallel background tasks, seeds 909 (the
  historically unstable seed from 2026-09-01's causal-test chain, worth
  rechecking under the full current reward/observation stack), 1000,
  2000, 3000 (fresh, never used on this track), each otherwise identical
  to the existing seed-110 reference (`2026-09-07_obstacle-lidar-
  nstep3-seed110`) so all five are directly comparable.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code did not treat the angle-restricted
  run's improved car-contact average or unchanged elimination rate (0/20)
  as evidence of success -- pulled every individual race's lap count and
  low-progress time, which is what surfaced that 17 of 20 races completed
  zero laps, uniformly across every seed, not a scattered or ambiguous
  result. Ran `ruff`/`pyright`/`pytest -q` (170 passed after reverting)
  before moving on to the seed sweep.
- Commits or code: `src/training/reward.py`
  (`ROBOT_WARNING_ANGLE_DEGREES` added; `WEIGHT_ROBOT_PROXIMITY` re-tried
  at 1.0, reverted to 0.0), `tests/test_training_reward.py`,
  `docs/rl_design.md` section 6 (causal test 23).
- Experiment output:
  `experiments/2026-09-08_robot-proximity-angle-restricted-seed110/`;
  seed-sweep runs in progress at the time of writing:
  `experiments/2026-09-08_seed-sweep-{909,1000,2000,3000}/` (each with
  `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt` once complete).
- Leaderboard result: n/a.

**What we observed:** The angle restriction made things much worse, not
better. Avg laps dropped 2.90 (unrestricted) -> 0.20; avg low-progress
time rose to 46.2s/race (38.5% of the round; one race hit 119.58s,
essentially the whole round); avg marshal/race rose to 6.35; avg max
speed dropped to 8.30 m/s. This was uniform across every seed and both
baselines -- 17 of 20 evaluation races completed exactly zero laps.
Car-contact time was actually the lowest of any configuration tested
(0.662s) -- the car genuinely avoided the opponent, but only by nearly
never moving at all.

**Decision and rationale:** Not adopted. The mechanism (avoiding a wall
requires active steering; avoiding an *ahead* competitor is trivially
satisfiable by simply never closing distance, which crawling at 6-10 m/s
achieves while staying above `WEIGHT_IDLE`'s threshold) explains why this
failed differently from, and worse than, the unrestricted version.
Reverted `WEIGHT_ROBOT_PROXIMITY` to 0.0 again; not tuning this specific
mechanism further without a different underlying idea, given two
consecutive attempts have each failed in a distinct way.
`2026-09-07_obstacle-lidar-nstep3-seed110` is the locked-in config for
the seed sweep that follows. `docs/rl_design.md` section 6 updated in the
same session per CLAUDE.md step 5.

**Next steps:**
1. Await the four in-progress seed-sweep runs, then compare all five
   (110, 909, 1000, 2000, 3000) across the fixed evaluation set to
   characterize how much this track's results depend on training seed --
   the one prior seed comparison (110 vs. 909, 2026-09-01) found a
   qualitatively different, much worse policy from a different seed, and
   that finding has never been rechecked under the current (much
   improved) reward/observation stack.
2. If a closing-speed-scaled competitor-proximity variant is revisited
   later, note it may have a similar "never gain on them" escape hatch to
   the one found here -- not a guaranteed fix.
3. Still open: repackaging `controllers.race_faster`, and the
   `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued)

**Participants and contributions:** Charlotte Tsui -- directed proceeding
with the multi-training-seed sweep after the angle-restricted attempt was
reverted. Claude Code (AI agent) -- launched 4 training runs in parallel
against the locked-in config, aggregated all 5 seeds' results, and found
the most reassuring result on this track to date.

**Question or objective:** How much does this track's results depend on
the training seed? Every checkpoint decision since 2026-08-31 was made
from training seed 110 alone. The one prior seed comparison (110 vs. 909,
2026-09-01) found a catastrophic qualitative difference under a much
cruder reward/observation setup -- does that instability still exist
under the current stack (idle penalty, terminal penalty, n-step returns,
obstacle-lidar observation)?

**What we investigated or changed:** Trained 4 new seeds in parallel
background tasks -- 909 (the seed that froze completely on 2026-09-01),
plus 1000, 2000, 3000 (fresh, never used on this track, and not
overlapping the fixed evaluation seed set) -- each with the locked-in
config from the previous entry (n_step=3, obstacle lidar in the
observation, no robot-proximity term), races=40, round_seconds=120,
buffer_capacity=800000, identical to every seed-110 comparison this week.
Combined with the existing seed-110 checkpoint
(`2026-09-07_obstacle-lidar-nstep3-seed110`) for a 5-seed comparison.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code aggregated all 5 seeds' full
  `eval_results.json` output (not just elimination rate) into one
  comparison table before drawing any conclusion, which is what surfaced
  both the reassuring safety finding and the less-reassuring pace
  variance in the same pass, rather than stopping at "no eliminations,
  done."
- Commits or code: `docs/rl_design.md` section 6 (causal test 24) and
  the "Robustness across seeds" refinement item (marked partially
  resolved).
- Experiment output: `experiments/2026-09-08_seed-sweep-{909,1000,2000,3000}/`
  (each with `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

| seed | avg damage | off-track | wall-contact | avg laps | avg lap time | avg max speed | eliminated | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110 (original reference) | 0.0027 | 0.03s | 0.01s | 4.00 | 24.79s | 23.2 m/s | 0/20 | 20/20 |
| 909 (2026-09-01's unstable seed) | 0.0000 | 0.00s | 0.00s | 5.00 | 22.50s | 17.4 m/s | 0/20 | 20/20 |
| 1000 (fresh) | 0.0013 | 0.02s | 0.00s | 6.50 | 17.19s | 31.0 m/s | 0/20 | 20/20 |
| 2000 (fresh) | 0.0000 | 0.00s | 0.00s | 2.95 | 34.00s | 16.3 m/s | 0/20 | 20/20 |
| 3000 (fresh) | 0.0127 | 0.22s | 0.15s | 6.20 | 17.26s | 33.7 m/s | 0/20 | 20/20 |

Every one of the 5 seeds is safe and competent: 0/20 eliminated and
20/20 wins against both baselines across the board. Seed 909 -- the seed
that produced a "do nothing" freeze on 2026-09-01 -- is now one of the
*better* performers (5.00 avg laps, 22.50s avg lap time), not a broken
outlier. The 2026-09-01 instability does not recur under the current
stack.

There is real, worth-reporting variance in pace, though: lap time ranges
17.19s-34.00s (~2x) and laps completed ranges 2.95-6.50 across the 5
seeds. Seed 1000 is the standout (fastest lap time, most laps, strong
safety); seed 2000 is the weakest (safe, but meaningfully slower and less
complete). Seed 110 -- the only seed used for every prior checkpoint
decision on this track -- turns out to be middle-of-the-pack, not
representative of the best available outcome.

**Decision and rationale:** Treating seed-level *safety* robustness as
resolved for the current reward/observation config -- 5/5 safe seeds is
real evidence, not proof for all possible seeds, but a large improvement
over the single-seed evidence this track has relied on throughout.
Treating seed-level *pace* as a separate, still-open finding, since it
varies meaningfully enough that defaulting to "whichever seed was used
first" leaves real performance on the table. **Recommending
`2026-09-08_seed-sweep-1000` as the new best checkpoint** for
speed-sensitive purposes given its combination of fastest pace and
strong safety. `docs/rl_design.md` section 6 and the "Robustness across
seeds" refinement item both updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Repackage `controllers.race_faster` from `seed-sweep-1000` -- now
   clearly motivated by evidence, not just "a newer checkpoint exists."
2. A larger sweep (10+ seeds) would tighten the pace-variance estimate,
   but 5/5 safe is already a meaningful confidence improvement; not
   urgent.
3. Investigate why pace varies so much across seeds (2000 vs. 1000/3000)
   -- a live watch comparing them could show whether the difference is
   visible qualitatively (hesitant vs. committed cornering) or not.
4. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 2)

**Participants and contributions:** Charlotte Tsui -- directed
repackaging `controllers.race_faster` from the seed-1000 checkpoint.
Claude Code (AI agent) -- found this required more than a checkpoint
swap (the observation encoding itself had changed shape since the module
was last packaged), updated it correctly, and re-verified the full chain
before treating it as done.

**Question or objective:** Repackage `controllers.race_faster` from
`2026-09-08_seed-sweep-1000` (causal test 24's recommended checkpoint),
replacing the stale races=20 checkpoint from 2026-09-01 that it had
shipped since entry 11.

**What we investigated or changed:**

- Read the existing `src/controllers/race_faster.py` before touching it
  and noticed its inlined `_encode_observation` still matched the old
  17-dim vector -- the seed-1000 checkpoint was trained with the 24-dim
  vector (obstacle LiDAR added 2026-09-07, this notebook's 2026-09-07
  entry). A bare checkpoint swap would have failed to load (shape
  mismatch on the first `nn.Linear`) or, worse in a differently-shaped
  coincidence, loaded successfully while silently misinterpreting the
  observation. Updated the inlined encoding to add the obstacle-LiDAR
  beams (`sensors.lidar`, mirroring the existing `wall_lidar` block),
  matching `training.observation`'s current 24-dim layout exactly.
- Extracted the `"policy"` key from
  `experiments/2026-09-08_seed-sweep-1000/checkpoints/policy_final.pt`
  (413.8KB, includes critics/targets/log_alpha) into
  `src/controllers/checkpoints/race_faster_policy.pt` (84.7KB, policy
  weights only) -- same trim-for-packaging convention as every prior
  `race_faster` export.
- Updated the module docstring: which checkpoint is packaged, its
  headless-eval numbers, the full supersession chain
  (races=20 -> seed-sweep-1000), and an explicit note that the
  observation encoding must stay in lockstep with whichever checkpoint is
  loaded, with no migration path across the dimension change.

**Evidence:**
- Sources or documentation: `training/observation.py` (current 24-dim
  encoding, read to confirm the exact beam angles/ordering needed to
  match it).
- AI-agent assistance: Claude Code did not treat "swap the checkpoint
  path" as the whole task -- reading the existing module first (rather
  than assuming the last packaging pass was still current) is what
  surfaced the dimension mismatch before it became a load-time or,
  worse, a silent-misinterpretation bug. Verified three ways before
  calling it done, same bar as every prior `race_faster` packaging pass:
  (1) a bit-for-bit match between the packaged module's output and
  `SACAgent.act(..., deterministic=True)` from the full checkpoint on 20
  random 24-dim observations (`max abs diff = 0.0`); (2) `ruff
  check`/`ruff format --check`/`pyright` (project's strict-mode config)
  all clean; (3) a real `racing h2h` race (seed 110, 30s, vs.
  `crash_fast`) confirming actual driving behavior, not just synthetic-
  observation output -- 0 damage, 1 lap in 16.88s, 0 off-track/wall-
  contact, 291.4m margin, consistent with the checkpoint's documented
  headless-eval numbers.
- Commits or code: `src/controllers/race_faster.py`,
  `src/controllers/checkpoints/race_faster_policy.pt`,
  `docs/rl_design.md` section 6 item 8 (repackaging update).
- Experiment output: n/a (packaging, not a new training run); source
  checkpoint is `experiments/2026-09-08_seed-sweep-1000/`.
- Leaderboard result: `artifacts/formula110-student-controllers.zip`
  rebuilt via `scripts/export_student_controllers.py --all-controllers`;
  not yet re-uploaded to Gradescope.

**What we observed:** The dimension mismatch would not have been obvious
from a quick diff of just the checkpoint path -- `race_faster.py`'s own
`_OBSERVATION_DIM` constant and inlined encoding needed to change too.
Once corrected, the packaged module reproduces the full training
checkpoint exactly and drives correctly in a real race.

**Decision and rationale:** Treating this repackaging as complete and
correct, verified to the same three-part standard (numeric match, static
checks, real race) used for every prior `race_faster` export. Not
re-uploading to Gradescope in this session -- that's a separate,
external action (per CLAUDE.md's "Packaging a Gradescope submission"
section, actually uploading needs three additional root-level files
appended to this zip) that should be confirmed explicitly before
submitting, not bundled into a repackaging task.

**Next steps:**
1. If ready to submit: build the three-file manifest addition per
   CLAUDE.md and upload `artifacts/formula110-student-controllers.zip`
   to Gradescope.
2. Still open: the `controllers.minimum_viable` module gap -- without it
   the submission is capped at partial rubric points regardless of how
   good `race_faster` is.
3. Still open from the previous entry: investigate why pace varies so
   much across training seeds; consider watching `seed-sweep-1000` live
   for a qualitative check now that it's the packaged reference.

---

## 2026-09-08 (continued, 3)

**Participants and contributions:** Charlotte Tsui -- directed
understanding why pace varies across training seeds, then continuing to
iterate on pace based on the finding. Claude Code (AI agent) -- ran a
per-tick diagnostic, then a chain of resume-from experiments that
overturned the initial framing (this isn't a per-seed problem at all) and
pointed at a specific, previously-untested reward constant.

**Question or objective:** Why does seed 1000 (17.19s avg lap) drive
roughly 2x faster than seed 2000 (34.00s avg lap) under an otherwise
identical reward/observation/architecture? Then: use that understanding
to keep improving pace.

**What we investigated or changed:**

- **Diagnosis:** wrote a one-off per-tick diagnostic (not committed)
  comparing seed 1000 and seed 2000 head-to-head: speed and yaw rate
  logged per tick, aligned by track position (`distance_m` modulo the
  ~183m lap length) so the two could be compared corner-by-corner and
  straight-by-straight. Finding: seed 2000 wasn't hesitating at specific
  corners -- it was uniformly slower across nearly every point on the
  track (speed-by-position profile a scaled-down version of seed 1000's
  own), never exceeded 20 m/s at all (vs. seed 1000's 7.5% of ticks above
  20 m/s), and spent 55.3% of ticks below 5 m/s (vs. 29.4% for seed
  1000). This pointed at a track-wide risk-tolerance difference, not a
  localized bug -- consistent with SAC's entropy-regularized objective
  settling into different local optima for the same speed-vs-wall-risk
  trade-off depending on training seed.
- **Causal test 1 (does more training fix it?):** used `--resume-from`
  to continue training seed 2000's checkpoint for 40 more races
  (`--warmup-steps 0`, otherwise identical config). Real improvement:
  avg lap time 34.00s -> 26.71s, laps 2.95 -> 3.95, max speed 16.33 ->
  19.14 m/s, safety stayed effectively perfect. Not a stuck local
  optimum -- more optimization helped.
- **Causal test 2 (does it keep improving, and is seed 1000 also below
  its own ceiling?):** ran two more resume experiments in parallel: seed
  2000 resumed again (+40 more races, ~120 races-equivalent total) and
  seed 1000 (the *fast* one) resumed for the first time (+40 races).
  Seed 2000 kept improving, but with diminishing returns (26.71s ->
  26.10s). **Seed 1000 got worse, not better** (17.19s -> 25.08s, max
  speed 31.02 -> 18.65 m/s) despite starting from the best result on the
  whole track. Both ended up in the same ~25-26s/~19-23 m/s range
  regardless of starting point (one climbing up to it, one falling down
  to it) -- strong evidence of a shared training-dynamics attractor that
  more optimization pulls every seed toward, not a per-seed lottery that
  more training resolves in one favored direction.
- **Reward change, motivated by the attractor finding:** raised
  `WALL_PROXIMITY_SPEED_SCALE_MPS` 10.0 -> 15.0 in `src/training/reward.py`
  -- the constant controlling how fast the wall-proximity penalty scales
  with speed, chosen when written "to match the old
  `MAX_REWARDED_SPEED_MPS` cruising target," making it the most directly
  implicated lever for where the attractor sits. A moderate step (50%),
  not a large one, per the "small careful step" lesson from the
  `MAX_REWARDED_SPEED_MPS` axis (10->12 tied, 10->20 regressed badly,
  2026-09-01/09-02). Ran `ruff`/`pyright`/`pytest -q` (170 passed)
  before training. Training from scratch on seed 1000 (not resumed, to
  isolate the reward change from the "more training" confound just
  found) -- run in progress at the time of writing.

**Evidence:**
- Sources or documentation: none beyond this session's own diagnostic and
  experiment output.
- AI-agent assistance: Claude Code did not stop at "more training fixed
  seed 2000" as the answer -- running the symmetric test on seed 1000 (an
  experiment not strictly necessary to answer the original question, but
  the right control to check whether *everyone* was still improving or
  whether something more specific was happening) is what overturned the
  initial hypothesis and found the more interesting, more useful result
  underneath it. Chose the reward-side follow-up based on which specific
  constant was actually implicated by the finding (the constant's own
  stated rationale, "match the old cruising target"), rather than
  reaching for the nearest previously-tried lever.
- Commits or code: `src/training/reward.py`
  (`WALL_PROXIMITY_SPEED_SCALE_MPS` 10.0 -> 15.0).
- Experiment output: `experiments/2026-09-08_seed2000-resumed/`,
  `experiments/2026-09-08_seed2000-resumed2/`,
  `experiments/2026-09-08_seed1000-resumed/`,
  `experiments/2026-09-08_wallproxscale15-seed1000/` (in progress).
- Leaderboard result: n/a.

**What we observed:**

| checkpoint | avg lap time | max speed | laps |
| --- | --- | --- | --- |
| seed 2000, races=40 | 34.00s | 16.33 m/s | 2.95 |
| seed 2000, +40 more | 26.71s | 19.14 m/s | 3.95 |
| seed 2000, +80 more | 26.10s | 22.92 m/s | 4.40 |
| seed 1000, races=40 | 17.19s | 31.02 m/s | 6.50 |
| seed 1000, +40 more | 25.08s | 18.65 m/s | 4.00 |

Both seeds converge toward the same ~25-26s/~19-23 m/s regime with more
training, regardless of which side of it they started on. Seed 1000's
17.19s result (the checkpoint currently packaged in `race_faster.py`)
was a fast point in its trajectory, not a stable achievement -- more
training pulls it back down toward the same attractor seed 2000 climbs
up to.

**Decision and rationale:** Reframing the whole question: pace variance
across training seeds is not really a "seed lottery" to be solved by
sampling more seeds and picking a winner (last entry's framing) -- it's
that different seeds are caught at different points along a shared
trajectory toward a common, moderate-pace equilibrium, and training
duration determines how close to that equilibrium a given checkpoint is.
**This means "more training" is not a lever for improving pace here --
it actively erodes the best results found so far.** The productive lever
is understanding and shifting *where the equilibrium itself sits*, which
points at reward magnitude (specifically `WALL_PROXIMITY_SPEED_SCALE_MPS`,
given its own stated design rationale) rather than seed selection or
training budget. Not drawing a conclusion on the reward change until its
result is in -- documented here so the reasoning chain is recorded even
though the causal test itself is still running.

**Next steps:**
1. Evaluate the `WALL_PROXIMITY_SPEED_SCALE_MPS = 15.0` result once
   training finishes: does the equilibrium shift to a faster regime
   without reintroducing damage/off-track/wall-contact regressions, per
   the "small step first" plan?
2. If it helps, worth rechecking whether the currently-packaged
   `race_faster` checkpoint (seed 1000, races=40, scale=10.0) should be
   replaced -- it's a snapshot of a transient fast point under the *old*
   scale value, not a stable optimum under either reward.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 4)

**Participants and contributions:** Charlotte Tsui -- none this entry
(continuation of "keep iterating to improve pace"). Claude Code (AI
agent) -- got a null result on the reward-magnitude test, then found and
fixed a foundational bug that reframes every training-seed conclusion
reached earlier today.

**Question or objective:** Does raising `WALL_PROXIMITY_SPEED_SCALE_MPS`
(the constant implicated by the shared-attractor finding from the
previous entry) shift the training equilibrium toward a faster pace?

**What we investigated or changed:**

- Ran the planned test: `WALL_PROXIMITY_SPEED_SCALE_MPS` 10.0 -> 15.0,
  trained fresh on seed 1000 (races=40, otherwise identical to the
  17.19s/31.02 m/s reference). Result: worse, not better -- 30.31s avg
  lap time, 20.39 m/s avg max speed, landing in the same ~25-30s range
  already characterized as the training attractor rather than a faster
  regime. Reverted to 10.0.
- Before accepting that as the final word, checked how `--seed` is
  actually wired through `scripts/train_sac.py`'s `train()` function --
  and found `SACAgent(...)` is constructed without `seed=args.seed`,
  so it always uses the class default (`seed=0`). Verified directly:
  constructing two agents with `seed=1000` and `seed=2000` produced
  bit-for-bit identical initial policy weights before the fix.
- Fixed it: added `seed=args.seed` to the `SACAgent(...)` call in
  `train()`. Verified the fix mechanically (two different seeds now
  produce different, individually-reproducible initial weights) and
  updated the `--seed` flag's help text to describe everything it now
  controls. Ran `ruff`/`pyright`/`pytest -q` (170 passed) after the fix.

**Evidence:**
- Sources or documentation: `src/training/sac.py`
  (`SACAgent.__init__`'s `seed: int = 0` default, read to confirm the
  fallback value actually in effect).
- AI-agent assistance: Claude Code did not treat the
  `WALL_PROXIMITY_SPEED_SCALE_MPS` null result as the end of the
  investigation -- checking the seed-wiring code directly (rather than
  assuming a CLI flag does what its name says) is what surfaced a bug
  that had been silently in effect for every training run on this track,
  including the entire 5-seed sweep from two entries ago. Flagged the
  implication precisely rather than either overstating it (the sweep's
  raw numbers are still real observations) or understating it (the
  "seed sensitivity"/"shared attractor" framing was narrower than
  claimed).
- Commits or code: `src/training/reward.py`
  (`WALL_PROXIMITY_SPEED_SCALE_MPS` tried at 15.0, reverted to 10.0),
  `scripts/train_sac.py` (`seed=args.seed` added to the `SACAgent(...)`
  call; `--seed` help text updated), `docs/rl_design.md` section 6
  (causal test 25).
- Experiment output:
  `experiments/2026-09-08_wallproxscale15-seed1000/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`).
- Leaderboard result: n/a.

**What we observed:** Every "training seed" experiment run on this track
to date -- the 2026-09-01 seed-110-vs-909 comparison, today's 5-seed
sweep, every resume-from test, and this reward test -- started from
identical network initialization. `--seed` only ever varied self-play
spawn positions and the replay-buffer/warmup sampling order. The
shared-attractor pattern found in the previous entry (both the fastest
and slowest seeds converging toward ~25-26s lap time with more training)
may be partly or entirely explained by every run beginning at the same
point in weight-space rather than a property of the reward landscape
itself.

**Decision and rationale:** Adopted the fix. Not adopting the
`WALL_PROXIMITY_SPEED_SCALE_MPS` change (null/negative result, and now
additionally confounded by the bug). Not re-running the 5-seed sweep or
the reward test automatically in this session -- both would cost the
same order of compute already spent (5+ training runs), and re-spending
it should be a deliberate choice, not an automatic reflex, given how much
has already run today. `2026-09-08_seed-sweep-1000` remains the
checkpoint packaged in `race_faster.py` and the best directly-evaluated
result so far regardless of this bug -- the fix changes what *future*
training explores, not what that specific checkpoint already achieved.
`docs/rl_design.md` section 6 updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Re-run the multi-training-seed sweep with the fix in place -- this is
   now the scientifically meaningful version of the question causal test
   24 asked, separating genuine network-initialization sensitivity from
   trajectory sensitivity from a fixed start.
2. Re-test `WALL_PROXIMITY_SPEED_SCALE_MPS` (and reconsider any other
   seed-dependent conclusion reached before this fix) once genuine
   initialization diversity is available.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 5)

**Participants and contributions:** Charlotte Tsui -- directed re-running
the seed sweep now that genuine network-initialization diversity is
available. Claude Code (AI agent) -- ran all 5 seeds in parallel and
found the seed-to-outcome mapping does not carry over across the fix at
all.

**Question or objective:** Does the multi-training-seed sweep's finding
(all 5 seeds safe, pace varies ~2x) hold up once `--seed` actually varies
network initialization, not just spawn/sampling order?

**What we investigated or changed:** Re-ran the identical 5-seed sweep
(110, 909, 1000, 2000, 3000; same n_step=3/obstacle-lidar/no-robot-
proximity config, races=40, round_seconds=120, buffer_capacity=800000)
with the `seed=args.seed` fix from the previous entry in place, all 5 in
parallel background tasks.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code checked each run's result as it
  landed rather than waiting silently for all 5, giving an early read
  (seed 110 already looked notably different from its v1 counterpart)
  without treating a single data point as conclusive -- held the full
  comparison until all 5 were in before drawing any conclusion.
- Commits or code: `docs/rl_design.md` section 6 (causal test 26).
- Experiment output: `experiments/2026-09-08_seed-sweep-v2-{110,909,1000,2000,3000}/`
  (each with `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

| seed | v1 (shared init=0, buggy) lap time | v2 (genuine init) lap time | v2 max speed | v2 laps | v2 damage |
| --- | --- | --- | --- | --- | --- |
| 110 | 24.79s | 17.96s (now fastest) | 30.6 m/s | 6.05 | 0.0045 |
| 909 | 22.50s | 26.60s | 11.0 m/s | 4.00 | 0.0000 |
| 1000 | 17.19s (was fastest) | 48.86s (now slowest) | 8.9 m/s | 2.00 | 0.0000 |
| 2000 | 34.00s | 41.37s | 9.9 m/s | 2.10 | 0.0000 |
| 3000 | 17.26s | 22.73s | 19.8 m/s | 4.95 | 0.0000 |

The seed-to-outcome mapping does not carry over at all: seed 1000 (the
winner of the buggy sweep, and the checkpoint currently packaged in
`race_faster.py`) is now the *worst* of the five under its own genuine
initialization. Seed 110, previously middling, is now the best. The pace
range widened, not narrowed: 17.96s-48.86s (~2.7x) vs. the buggy sweep's
already-wide 17.19s-34.00s (~2x) -- genuine initialization diversity
carries more variance than the trajectory-only variance measured before
the fix. Safety held up across all five regardless: 0/20 eliminated,
near-zero damage/off-track/wall-contact everywhere, 20/20 wins -- the "no
catastrophic freeze" finding from the original sweep is reconfirmed, now
on solid methodological footing.

**Decision and rationale:** `2026-09-08_seed-sweep-1000` remains the
best directly-evaluated checkpoint (17.19s, 6.50 laps, 0.0013 damage) and
stays packaged in `race_faster.py` -- the new sweep's best result (v2
seed 110: 17.96s, 6.05 laps, 0.0045 damage) is close but doesn't beat it
on either speed or safety. That checkpoint's own validity was never in
question; only whether asking for "seed 1000" again would reproduce it,
which it no longer does under the fixed code. Given the now-confirmed
wide variance, best-of-N seed sampling is itself a legitimate, cheap
strategy for further pace improvement, separate from reward tuning.
`docs/rl_design.md` section 6 updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Sample more genuine-init seeds if further pace improvement is wanted
   -- 5 draws may not have found the tail of the distribution yet.
2. Re-test `WALL_PROXIMITY_SPEED_SCALE_MPS` under genuine multi-seed
   diversity before drawing any conclusion about that constant --
   the earlier null result was a single seed under the old bug.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 6)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to sample genuine-init seeds cheaply in search of something better than
17.19s. Claude Code (AI agent) -- sampled 5 more, found one that broke
the record, and diagnosed its one flagged outlier before recommending
anything.

**Question or objective:** Does sampling more genuine-init seeds surface
a checkpoint that beats the 17.19s/6.50-lap record
(`2026-09-08_seed-sweep-1000`)?

**What we investigated or changed:** Trained 5 more fresh seeds (4000,
5000, 6000, 7000, 8000) in parallel, identical locked-in config to the
previous entry's sweep. Four landed in the already-seen 26-30s range.
Seed 8000 broke the record: 14.96s avg best lap time (vs. 17.19s), 7.30
avg laps (vs. 6.50), 1444.6m avg distance (vs. 1267.2m) -- but with
higher aggregate damage/off-track/wall-contact (0.0298/0.15s/0.11s vs.
0.0013/0.02s/0.00s). Pulled every individual race before accepting or
dismissing that difference: 18 of 20 races are exceptionally clean
(0.0000 damage, 7-8 laps every time), and the aggregate is driven by one
outlier (`seed=8675309 vs default_student_controller race=1`, 0.5942
damage -- a serious near-crash, short of the 0.9 elimination threshold).
Diagnosed that race directly with the same per-tick logging approach
used for every prior outlier this session: over ~0.5s the car
accelerated hard (8.8 -> 14.1 m/s) while drifting off-center (-0.20m ->
-1.96m) as wall clearance shrank (3.90m -> 1.30m) -- committing to a fast
line through what looks like a tightening corner, then taking one hard
wall impact in a single tick, not a repeated pattern. `contact.robot`
was zero throughout, ruling out the opponent-collision failure mode from
earlier in this session.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output and diagnostic.
- AI-agent assistance: Claude Code did not report the aggregate
  damage/off-track numbers as a disqualifying regression, nor accept the
  faster lap time at face value without checking why the aggregate
  looked worse -- pulling every individual race is what found both the
  outlier and how clean the other 19 races were. Explicitly did not
  auto-adopt this checkpoint the way every earlier "adopt despite an
  outlier" decision this session was made -- judged this trade-off
  (a near-crash, not a stuck-and-slow race) as sharper and genuinely
  worth a direction check rather than a unilateral call, since it affects
  the packaged submission.
- Commits or code: `docs/rl_design.md` section 6 (causal test 27).
- Experiment output:
  `experiments/2026-09-08_seed-sweep-v2-{4000,5000,6000,7000,8000}/`
  (each with `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** A genuine, understood speed-vs-cornering-margin
trade-off, not a bug: seed 8000 is faster and more consistent than the
current record on the 90% of races where nothing goes wrong, but
occasionally (1/20) commits to a corner entry it can't quite hold,
taking a real but non-fatal hit. Never resulted in elimination or a lost
race in this sample.

**Decision and rationale:** Not yet adopted. Every earlier "keep it
despite a flagged outlier" decision this session involved a clearly
net-positive trade (e.g. the obstacle-lidar checkpoint's stuck-and-slow
outlier against otherwise-uniform improvement); this one trades a
substantial pace gain against a real near-crash risk, which is a closer
call that should be Charlotte's to make rather than assumed.
`2026-09-08_seed-sweep-1000` remains packaged in `race_faster.py`
pending that decision. `docs/rl_design.md` section 6 updated in the same
session per CLAUDE.md step 5.

**Next steps:**
1. Awaiting direction: adopt and repackage from `seed-sweep-v2-8000`, or
   keep sampling for a cleaner-margin checkpoint, or keep the current
   record as the safer choice.
2. If this near-miss pattern recurs across future fast checkpoints, a
   corner-aware wall-proximity term (scaling with
   `camera.lookahead_offsets_m` curvature rather than only the current
   tick's distance) is a candidate fix, not yet tested.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 7)

**Participants and contributions:** Charlotte Tsui -- directed adopting
the seed-8000 checkpoint (accepting the documented near-miss trade-off),
then continuing with safety improvement from there. Claude Code (AI
agent) -- repackaged and verified `controllers.race_faster` from the new
checkpoint.

**Question or objective:** Repackage `controllers.race_faster` from
`2026-09-08_seed-sweep-v2-8000` (the pace-record checkpoint from causal
test 27), replacing `2026-09-08_seed-sweep-1000`.

**What we investigated or changed:**

- Extracted the `"policy"` key from
  `experiments/2026-09-08_seed-sweep-v2-8000/checkpoints/policy_final.pt`
  into `src/controllers/checkpoints/race_faster_policy.pt` (413.8KB ->
  84.7KB, same trim convention as every prior export). Unlike the
  previous repackaging, the observation encoding did not need to change
  this time -- both the old and new checkpoints were trained under the
  same 24-dim obstacle-LiDAR observation, so only the checkpoint file
  and docstring changed.
- Updated the module docstring: which checkpoint is packaged, its
  headless-eval numbers, and an explicit "known risk, accepted on
  purpose" note documenting the 1/20-race near-miss (0.5942 damage) found
  and diagnosed in the previous entry, plus the full supersession chain.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code verified the same three ways as every
  prior `race_faster` packaging pass: (1) bit-for-bit match against
  `SACAgent.act(..., deterministic=True)` on 20 random observations
  (`max abs diff = 0.0`); (2) `ruff check`/`ruff format --check`/`pyright`
  all clean; (3) a real `racing h2h` race (seed 110, 30s, vs.
  `crash_fast`) -- 0 damage, 1 lap in 15.75s (faster than the previous
  package's 16.88s), 0 off-track/wall-contact, consistent with the
  checkpoint's documented headless-eval behavior.
- Commits or code: `src/controllers/race_faster.py`,
  `src/controllers/checkpoints/race_faster_policy.pt`,
  `docs/rl_design.md` section 6 item 8 (repackaging update).
- Experiment output: n/a (packaging, not a new training run); source
  checkpoint is `experiments/2026-09-08_seed-sweep-v2-8000/`.
- Leaderboard result: `artifacts/formula110-student-controllers.zip`
  rebuilt via `scripts/export_student_controllers.py --all-controllers`;
  not yet re-uploaded to Gradescope.

**What we observed:** The packaged module reproduces the full checkpoint
exactly and drives correctly and faster in a real race than the
previous package.

**Decision and rationale:** Adopted per direction -- the known near-miss
risk is documented in the module's own docstring (not just the lab
notebook) so anyone reading the packaged controller's source sees the
trade-off directly, not only in evidence files that don't ship with it.
Not re-uploading to Gradescope this session -- a separate, external
action per CLAUDE.md's submission-manifest process.

**Next steps:**
1. Per direction, pursue safety improvement from this faster baseline --
   the corner-aware wall-proximity idea (scaling with
   `camera.lookahead_offsets_m` curvature, giving reaction time before a
   corner rather than only reacting to the current tick's distance) is
   the most specific candidate raised so far, targeting the exact
   failure mode diagnosed in the previous entry.
2. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 8)

**Participants and contributions:** Charlotte Tsui -- directed improving
safety from the newly-adopted checkpoint. Claude Code (AI agent) -- found
a principled, evidence-grounded lever (a stale calibration) rather than
guessing at a new mechanism, tested it, and found the same overcorrection
pattern as two earlier attempts this session.

**Question or objective:** Fix the near-miss documented in
`2026-09-08_seed-sweep-v2-8000` (now packaged in `race_faster.py`)
without losing the pace gain that motivated adopting it.

**What we investigated or changed:** Rather than proposing a new,
untested mechanism, checked whether an existing one was simply
miscalibrated for this checkpoint's speed: `WALL_WARNING_DISTANCE_M =
6.0` was explicitly tuned on 2026-09-01 for "~0.6s at 10 m/s" reaction
time, back when checkpoints cruised near that speed. This checkpoint
reaches ~27 m/s, where the same 6.0m gives only ~0.22s -- consistent
with the diagnosed near-miss, where wall clearance shrank from 3.90m to
1.30m in ~0.4s, faster than the mechanism had lead time to act on.
Doubled the distance to 12.0 (matching the relative size of the original
3.0->6.0 step) and trained fresh on seed 8000 -- the same seed that
produced the near-miss -- otherwise identical config.

**Evidence:**
- Sources or documentation: `src/training/reward.py`'s own comment
  history (read to find the "~0.6s at 10 m/s" calibration rationale
  rather than assuming the constant was already well-tuned for this
  checkpoint's speed).
- AI-agent assistance: Claude Code chose a specific, already-implicated
  lever (checking whether an existing mechanism was miscalibrated) over
  introducing a new one, and reported the honest result -- a large safety
  improvement bought at a cost that undermines the reason the checkpoint
  was adopted -- rather than either overselling the safety win or hiding
  the pace cost. Ran `ruff`/`pyright`/`pytest -q` (170 passed) before and
  after the change.
- Commits or code: `src/training/reward.py`
  (`WALL_WARNING_DISTANCE_M` tried at 12.0, reverted to 6.0),
  `docs/rl_design.md` section 6 (causal test 28).
- Experiment output: `experiments/2026-09-08_wallwarn12-seed8000/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:**

| | seed 8000, warn=6.0 (packaged) | seed 8000, warn=12.0 |
| --- | --- | --- |
| avg damage | 0.0298 | 0.0000 |
| avg off-track | 0.152s | 0.003s |
| avg wall-contact | 0.107s | 0.000s |
| avg laps | 7.30 | 3.35 |
| avg best lap time | 14.96s | 31.46s |
| avg max speed | 26.65 m/s | 18.74 m/s |

The near-miss is essentially gone, but the resulting policy is a
genuinely different, much more conservative one -- not a marginally
safer version of the fast policy. This is the third safety-motivated
reward change this session (after both `WEIGHT_ROBOT_PROXIMITY`
attempts) to produce this same shape of result: an all-or-nothing trade,
not a small-pace-for-large-safety middle ground.

**Decision and rationale:** Reverted `WALL_WARNING_DISTANCE_M` to 6.0.
Not adopting -- it solves the stated problem at a cost that erases most
of the reason `seed-sweep-v2-8000` was adopted.
`2026-09-08_seed-sweep-v2-8000` remains packaged in `race_faster.py`,
near-miss and all. Given three consecutive attempts on this general
axis (reward-side risk penalties) have each hit the same all-or-nothing
pattern, deprioritizing further search along it in favor of two
qualitatively different levers. `docs/rl_design.md` section 6 updated in
the same session per CLAUDE.md step 5.

**Next steps:**
1. Keep sampling genuine-init seeds (the strategy that already found
   `seed-sweep-v2-8000`) looking specifically for one matching or beating
   its pace without a near-miss of its own.
2. A controller-level hard safety backstop (override throttle when a
   forward wall reading is both very close and speed is high, regardless
   of the learned policy's output) remains untested and wouldn't have the
   "policy learns to avoid the situation entirely" side effect reward
   shaping keeps producing.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 9)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to sample genuine-init seeds. Claude Code (AI agent) -- sampled 5 more,
found none beat the 14.96s record outright, but found one with a
dramatically better safety margin at close to the same pace.

**Question or objective:** Does sampling more genuine-init seeds surface
a checkpoint matching or beating `2026-09-08_seed-sweep-v2-8000`'s pace
(14.96s avg lap time) without its documented near-miss (0.5942 damage in
1/20 races)?

**What we investigated or changed:** Trained 5 more fresh seeds (9000,
10000, 11000, 12000, 13000) in parallel, identical locked-in config to
the previous sweeps (with `WALL_WARNING_DISTANCE_M` reverted to 6.0 per
the previous entry). Checked each one's max damage (not just average) as
it landed, given the whole point of this search is avoiding a hidden
near-miss the average could obscure.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code checked worst-case damage per seed as
  each result landed, not just the average -- this is what let it
  immediately recognize seed 10000 as a materially different find from
  the other four (mostly-clean-but-slower) results, rather than only
  reporting the closest lap time.
- Commits or code: `docs/rl_design.md` section 6 (causal test 29).
- Experiment output: `experiments/2026-09-08_seed-sweep-v2-{9000,10000,11000,12000,13000}/`
  (each with `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** None of the 5 beat 14.96s outright (9000: 24.79s,
0.0000 damage; 11000: 24.86s, 0.0447 max damage; 12000: 29.71s, 0.0000
damage; 13000: 20.99s, 0.0000 damage). Seed 10000 came closest to the
pace target with a dramatically better safety margin:

| | champion (seed 8000) | seed 10000 |
| --- | --- | --- |
| avg best lap time | 14.96s | 16.70s (+12%) |
| avg laps | 7.30 | 6.80 |
| avg max speed | 26.7 m/s | 35.4 m/s (higher) |
| avg damage | 0.0298 | 0.0008 |
| max damage (worst race) | 0.5942 | 0.0154 (38x smaller) |

18 of seed 10000's 20 races have exactly 0.0000 damage, 6-7 laps every
time, remarkably consistent ~35-36 m/s max speed across every race. The
only two blips are trivial (a clean off-track excursion, one tiny
0.0154-damage graze) -- nothing remotely comparable to the champion's
near-elimination-level event. Notably, seed 10000's raw top speed
exceeds the champion's, consistent with the long-standing "top speed and
lap pace aren't the same thing" finding -- it likely corners more
conservatively despite faster straights.

**Decision and rationale:** Not automatically adopted -- presenting as a
genuine alternative the same way seed 8000 itself was presented, since
the trade (12% slower lap time for a ~38x smaller worst-case damage
event) is a judgment call, not an unambiguous win in either direction.
`docs/rl_design.md` section 6 updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Awaiting direction: adopt seed 10000 in place of seed 8000, keep seed
   8000 as packaged, or continue sampling for something that beats
   14.96s outright without a near-miss.
2. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 10)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to sample until something better is found. Claude Code (AI agent) --
sampled 5 more, found nothing better, and reported the dropping search
yield honestly rather than continuing to sample by default.

**Question or objective:** Does a third batch of genuine-init seeds beat
seed 8000 (14.96s, one near-miss) or seed 10000 (16.70s, clean)?

**What we investigated or changed:** Trained 5 more fresh seeds (14000,
15000, 16000, 17000, 18000) in parallel, identical locked-in config.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code tallied the yield across all 20
  genuine-init seeds sampled today (not just this batch of 5) before
  deciding how to characterize the result, which is what surfaced the
  "~10% hit rate for near-record pace, and dropping" framing rather than
  just reporting "this batch found nothing."
- Commits or code: `docs/rl_design.md` section 6 (causal test 30).
- Experiment output: `experiments/2026-09-08_seed-sweep-v2-{14000,15000,16000,17000,18000}/`
  (each with `config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** All 5 were perfectly clean (0.0000 damage) but
slower than both existing candidates: 20.25s (seed 18000, best of this
batch) to 45.36s. Across all 15 fresh seeds sampled today plus the
original 5-seed sweep (20 total genuine-init seeds), only two have
landed in the 15-17s range -- seed 8000 and seed 10000. The other 18
cluster in a 20-45s range.

**Decision and rationale:** Not continuing to sample automatically.
Three consecutive batches of 5 have found one improvement (8000), one
good alternative (10000), and zero in the most recent batch -- the
marginal yield of blind sampling is visibly dropping, and continuing by
default rather than by decision doesn't match the evidence.
`docs/rl_design.md` section 6 updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Awaiting direction: keep sampling (possibly in a larger batch given
   the ~10% hit rate observed so far), settle on 8000 or 10000, or try a
   qualitatively different lever (the untested controller-level safety
   backstop, or fine-tuning/resuming from seed 10000 specifically to try
   closing its pace gap to seed 8000 rather than searching blindly for a
   new seed).
2. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 11)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to sample until a strict improvement is found. Claude Code (AI agent) --
doubled the batch size to test search efficiency, confirmed the plateau
with a larger sample, and recommended against continuing to sample
blindly without a decision to do so.

**Question or objective:** Does a larger batch (10 seeds instead of 5)
find a checkpoint that strictly beats both existing candidates (seed
8000: 14.96s, one near-miss; seed 10000: 16.70s, clean)?

**What we investigated or changed:** Trained 10 fresh seeds (19000-28000)
in parallel, identical locked-in config to every prior sweep -- doubling
the batch size specifically to test whether a bigger single draw would
be more sample-efficient than repeated batches of 5.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code computed the cumulative hit rate
  across all 30 seeds sampled today (not just this batch) before
  characterizing the result, which is what confirmed the ~6.7-10% rate is
  stable rather than an artifact of a small early sample -- a finding
  that directly informs whether continuing to sample is worth the
  compute, not just whether this particular batch succeeded.
- Commits or code: `docs/rl_design.md` section 6 (causal test 31).
- Experiment output: `experiments/2026-09-08_seed-sweep-v2-{19000..28000}/`
  (10 directories, each with `config.yaml`, `metrics.csv`,
  `eval_results.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** No strict improvement. Closest: seed 25000 (18.50s,
5.75 laps, 0.0179 max damage), seed 24000 (19.49s, perfectly clean), seed
21000 (20.04s, near-clean). The other 7 landed at 22.57-36.15s. Across
all 30 genuine-init seeds sampled today, only 2 (8000, 10000) have
landed in the 15-17s range -- a ~6.7% hit rate, consistent with the
~10% estimate from the smaller sample two entries ago. Doubling the
batch size did not reveal a hidden intermediate tier.

**Decision and rationale:** Not launching further batches automatically.
With 30 samples taken and the hit rate stable rather than improving,
continuing pure random seed sampling is unlikely to reliably turn up a
strict improvement without substantially more compute than has been
spent so far today. This is a natural point to make a deliberate choice
(accept one of the two existing candidates, or switch to a qualitatively
different lever) rather than continue by default. `docs/rl_design.md`
section 6 updated in the same session per CLAUDE.md step 5.

**Next steps:**
1. Awaiting direction: settle on 8000 or 10000, keep sampling anyway at
   the now-known odds, or pursue fine-tuning from seed 10000 / the
   controller-level safety backstop instead.
2. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 12)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to fine-tune and optimize seed 8000 specifically, and asked for the
command to watch it live. Claude Code (AI agent) -- gave the view
command (already packaged, no env var needed), then ran two parallel
resume experiments that found a checkpoint dominating both prior
candidates, and repackaged from it.

**Question or objective:** Can fine-tuning seed 8000 (rather than
sampling new seeds) reduce its near-miss without losing its pace
advantage?

**What we investigated or changed:**

- Gave the direct view commands for the currently-packaged checkpoint
  (`uv run racing --student-module controllers.race_faster --seed 110`,
  and an h2h/`--watch` variant, plus the exact seed/baseline combination
  that reproduces the diagnosed near-miss).
- Ran two `--resume-from` experiments in parallel from
  `2026-09-08_seed-sweep-v2-8000/checkpoints/policy_final.pt`
  (`--warmup-steps 0`, same seed 8000/round-length/reward/observation):
  +10 races and +40 races. The shorter resume was chosen specifically
  because earlier resume experiments this session (seeds 1000 and 2000)
  showed +40-race resumes tend to converge toward a shared, more
  conservative equilibrium -- a smaller nudge might catch a partial
  improvement before that convergence took hold.
- Verified the +10-resumed result was genuine (not an averaging artifact)
  by pulling every individual race before treating it as a finding.
- Repackaged `controllers.race_faster` from the +10-resumed checkpoint:
  extracted the `"policy"` key, verified bit-for-bit match against
  `SACAgent.act(..., deterministic=True)` (max diff 0.0), `ruff
  check`/`ruff format --check`/`pyright` clean, and a real `racing h2h`
  race (seed 110, 30s, vs. `crash_fast`) confirming correct driving.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code pulled every individual race for the
  +10-resumed result before reporting it as a win, the same discipline
  applied to every surprising number this session -- confirmed 18/20
  exactly-zero-damage races and that the two non-zero cases were trivial
  grazes, not a hidden second near-miss. Directly compared the
  +10-resumed checkpoint against seed 10000 (not just the original seed
  8000) to check whether it was a genuine dominant improvement or just
  better than one of the two prior candidates -- it beat both.
- Commits or code: `src/controllers/race_faster.py`,
  `src/controllers/checkpoints/race_faster_policy.pt`,
  `docs/rl_design.md` section 6 (causal test 32, item 8 repackaging
  update).
- Experiment output: `experiments/2026-09-08_seed8000-resumed-short/`,
  `experiments/2026-09-08_seed8000-resumed/` (each with `config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: `artifacts/formula110-student-controllers.zip`
  rebuilt via `scripts/export_student_controllers.py --all-controllers`;
  not yet re-uploaded to Gradescope.

**What we observed:**

| | original (races=40) | +10 resumed | +40 resumed |
| --- | --- | --- | --- |
| avg best lap time | 14.96s | 15.54s | 16.36s |
| avg laps | 7.30 | 6.75 | 7.05 |
| avg damage | 0.0298 | 0.0004 | 0.0138 |
| max damage (worst race) | 0.5942 | 0.0047 | 0.2764 |
| avg max speed | 26.7 m/s | 30.3 m/s | 34.4 m/s |

+10 races finds a clear sweet spot -- the near-miss is essentially
eliminated (126x smaller worst case) at a small pace cost (+3.9% lap
time). +40 races overshoots it: the near-miss partially reappears and
pace gets slower too, a non-monotonic curve rather than "more training
= more safety." The +10-resumed checkpoint also strictly beats seed
10000 (the separately-found safety alternative from two entries ago) on
lap time, avg damage, and max damage.

**Decision and rationale:** Adopted `2026-09-08_seed8000-resumed-short`
as the new best/reference checkpoint and repackaged `race_faster.py`
from it -- unlike the seed-8000-vs-seed-10000 choice, this one dominates
every prior candidate rather than requiring a judgment call. Not
adopting the +40 resume (worse than +10 on both axes, kept as evidence
of the non-monotonicity). `docs/rl_design.md` section 6 updated in the
same session per CLAUDE.md step 5.

**Next steps:**
1. A finer search around the sweet spot (+5, +15, +20 races) could find
   an even better point, though the non-monotonicity means this isn't
   guaranteed to improve smoothly.
2. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 13)

**Participants and contributions:** Charlotte Tsui -- asked how to
reduce hesitation further; when told the current level was already
modest and three reward-shaping attempts had failed, directed trying
increased self-play traffic instead. Claude Code (AI agent) -- checked
the current hesitation level against the record before proposing
anything, gave an honest options list with the track record attached,
then ran and reported a negative result on the chosen option.

**Question or objective:** Can increasing self-play traffic
(`--copies-per-side`) reduce car-contact/hesitation time on the current
best checkpoint without the reward-shaping side effects seen three times
already this session?

**What we investigated or changed:** Before proposing anything, pulled
the current checkpoint's own low-progress/car-contact numbers (2.83s
avg low-progress at copies=1's origin, 3.30s on the fine-tuned current
checkpoint, both with a modest worst case) to confirm hesitation was
already fairly low, not a severe problem -- this is what motivated
presenting a menu of options with an explicit track record (3 failed
reward-shaping attempts) rather than jumping straight to a fourth
variant. Ran `--copies-per-side 2` fresh on seed 8000, otherwise
identical config, and compared against the copies=1 reference.

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code verified the copies=2 result was a
  genuine, uniform regression (every one of 20 races at exactly 3 laps
  and ~12 m/s) rather than an averaging artifact before reporting it,
  and explicitly flagged the transitions/gradient-update confound
  (doubling copies_per_side roughly doubles data collected per race) so
  the result isn't over-attributed to "traffic diversity" specifically
  when "effectively more training" is an equally plausible explanation
  given this session's other findings about training-duration
  sensitivity.
- Commits or code: `docs/rl_design.md` section 6 (causal test 33).
- Experiment output: `experiments/2026-09-08_copies2-seed8000/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** A clear, uniform regression, not a fix. Avg
car-contact time improved only modestly (1.70s -> 1.38s, ~19%) while avg
best lap time more than doubled (15.54s -> 32.95s) and avg laps dropped
by more than half (6.75 -> 3.00) -- consistently across every evaluated
race, not a skewed average.

**Decision and rationale:** Not adopted.
`2026-09-08_seed8000-resumed-short` remains the reference checkpoint.
This is the fourth consecutive hesitation-reduction attempt this session
(after `WEIGHT_STEERING_REVERSAL`, both `WEIGHT_ROBOT_PROXIMITY`
attempts, now this) to regress pace instead of cleanly improving
hesitation -- treating the current checkpoint's hesitation level (2.8%
of race time, no severe outlier) as close to a practical floor for this
setup rather than an easily-closable gap. `docs/rl_design.md` section 6
updated in the same session per CLAUDE.md step 5.

**Next steps:**
1. If self-play traffic is revisited, control for the gradient-update
   confound (reduce `--races` proportionally when increasing
   `--copies-per-side`) for a cleaner comparison.
2. Otherwise, treat this thread as closed for now given the current
   hesitation level.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-08 (continued, 14)

**Participants and contributions:** Charlotte Tsui -- directed continuing
to iterate until a clear improvement is found. Claude Code (AI agent) --
ran the confound-controlled follow-up, got a decisive null result, and
recommended closing the thread with the accumulated evidence rather than
continuing to guess at further variants.

**Question or objective:** With the previous entry's `copies_per_side=2`
result confounded by ~2x the effective training budget, does a properly
controlled version (denser traffic as a small fine-tune increment, not a
full retrain) show a genuine hesitation improvement?

**What we investigated or changed:** Fine-tuned the current best
checkpoint (`2026-09-08_seed8000-resumed-short`) via `--resume-from`
with `--copies-per-side 2 --races 10` -- a small dose, matching the
successful "+10 races" pattern that fixed the earlier near-miss --
rather than retraining from scratch, specifically to isolate "denser
traffic" from "more total training."

**Evidence:**
- Sources or documentation: none beyond this session's own experiment
  output.
- AI-agent assistance: Claude Code designed this experiment specifically
  to remove the exact confound it had flagged in the previous entry,
  rather than treating the earlier ambiguous result as good enough to
  act on either way. Compared the target metric (car-contact time)
  directly against the reference before looking at anything else, which
  is what made the "zero effect" finding immediate and unambiguous
  rather than needing further interpretation.
- Commits or code: `docs/rl_design.md` section 6 (causal test 34).
- Experiment output: `experiments/2026-09-08_copies2-finetune-seed8000/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a.

**What we observed:** Car-contact time was completely unchanged (1.70s
-> 1.72s) while avg best lap time still regressed substantially (15.54s
-> 24.37s) and avg laps dropped (6.75 -> 4.50). Unlike the confounded
from-scratch version (which at least showed a modest ~19% car-contact
improvement), this controlled version shows zero measurable effect on
the metric it was specifically testing, while still costing pace.

**Decision and rationale:** Not adopted. With the confound removed, the
result is unambiguous: self-play traffic density is not a productive
lever for hesitation reduction on this checkpoint, at any dose or
procedure tested. Recommending this thread be closed -- across five
independently-designed experiments this session (three reward-shaping
terms, two self-play-traffic configurations), none has cleanly improved
hesitation without a pace cost, and this last one specifically found no
improvement at all even controlling for every confound identified along
the way. Treating the current checkpoint's hesitation level (2.8% of
race time, no severe outlier) as a practical floor for this setup.
`docs/rl_design.md` section 6 updated in the same session per CLAUDE.md
step 5.

**Next steps:**
1. Two genuinely untried levers remain if this is revisited: a
   closing-speed-scaled competitor-proximity term (lower confidence --
   may share the escape-hatch failure mode of the two distance-based
   attempts), or a controller-level deterministic safety backstop
   (structurally different -- an inference-time override, immune to the
   failure mode every reward-based attempt has hit).
2. Otherwise, accept the current checkpoint's hesitation level as final
   for this track.
3. Still open: the `controllers.minimum_viable` module gap.

---

## 2026-09-09

**Participants and contributions:** Charlotte Tsui -- accepted the
current checkpoint's hesitation level as final and directed creating the
Gradescope submission of the best controller so far. Claude Code (AI
agent) -- rebuilt the submission zip per the documented process and
verified its contents before calling it ready.

**Question or objective:** Package the current best checkpoint
(`2026-09-08_seed8000-resumed-short`, already loaded into
`controllers/race_faster.py`) as a Gradescope-ready submission zip.

**What we investigated or changed:** Confirmed `race_faster.py` still
pointed at the current best checkpoint (it did -- no repackaging needed,
this is the same module verified and packaged two sessions ago). Ran the
documented submission recipe from `CLAUDE.md`'s "Packaging a Gradescope
submission" section: `scripts/export_student_controllers.py
--all-controllers` (rebuilt fresh, not reused from a stale prior run),
then added the three required root-level files
(`formula110-submission.json` with `controller_module:
"controllers.race_faster"`, unmodified `pyproject.toml`, unmodified
`uv.lock`) to `artifacts/formula110-student-controllers.zip`.

**Evidence:**
- Sources or documentation: `CLAUDE.md`'s "Packaging a Gradescope
  submission" section (the documented recipe from the 2026-09-01 upload
  failure).
- AI-agent assistance: Claude Code did not reuse the existing
  `artifacts/` zip from an earlier session without checking it was
  current -- rebuilt it fresh via the export script, then verified with
  `unzip -l` that the packaged checkpoint
  (`controllers/checkpoints/race_faster_policy.pt`, 84,711 bytes) matches
  the current best checkpoint on disk (same size, same modification time
  as when it was packaged from `seed8000-resumed-short`) before calling
  the submission ready, per the standing checklist.
- Commits or code: none (packaging output only; `artifacts/` is
  gitignored build output, not source).
- Experiment output: n/a -- packaging, not a new training run. Source
  checkpoint: `experiments/2026-09-08_seed8000-resumed-short/`.
- Leaderboard result: `artifacts/formula110-student-controllers.zip`
  built and verified locally; not yet uploaded to Gradescope this
  session (uploading itself is a separate, external action for Charlotte
  to take).

**What we observed:** The zip contains all three required root files
(`formula110-submission.json`, `pyproject.toml`, `uv.lock`) alongside the
`controllers/` tree (`__init__.py`, `crash_fast.py`, `py.typed`,
`race_faster.py`, its trimmed checkpoint, and the `sac_candidate.py` dev
viewer, included harmlessly since Gradescope only grades the configured
module name).

**Decision and rationale:** Treating this submission as ready for upload.
No code changes were needed -- the best-controller decision and its
packaging were already done two sessions ago; this session's job was
producing a fresh, verified zip rather than trusting a possibly-stale
build artifact left over from then.

**Next steps:**
1. Charlotte to upload `artifacts/formula110-student-controllers.zip` to
   Gradescope (external action, not done by this session).
2. Still open: the `controllers.minimum_viable` module gap -- without it
   the submission is capped at partial rubric points regardless of how
   good `race_faster` is.

## 2026-09-10 15:40

**Participants and contributions:** Charlotte (direction), Claude Code
(implementation, training run, diagnosis, documentation).

**Question or objective:** Following up on `combined-approach`'s merge of
Lucy's imitation-learning work (`lucy-il`, merged via the `Build first
clone model` commit) into this branch: explore ways to combine the SAC
and imitation-learning tracks that are safer and could produce a faster
car. After ruling out fine-tuning Lucy's behavior-cloned network directly
(incompatible observation encodings -- 24-dim single-tick SAC vs. 336-dim
8-frame-history clone -- and the risk of an untrained critic unlearning
good BC behavior before it catches up), landed on: train the SAC
controller against Lucy's rule-based `leaderboard_expert.py` as a fixed
opponent instead of self-play, on the hypothesis that a genuinely
different sparring partner would teach real opponent-avoidance where
~5 reward-shaping attempts (causal tests 21-23, 33-34) had failed or
regressed. Directed to always train against the expert going forward and
look for improvement.

**What we investigated or changed:**
- Added a `--opponent {self,expert}` flag to `scripts/train_sac.py`
  (default `self`, preserving all existing behavior/reproducibility of
  every prior documented experiment). With `--opponent expert`, the
  incumbent is a frozen `controllers.leaderboard_expert.Controller`
  instance instead of a second `TrainableController`; only the challenger
  learns/pushes transitions to the shared buffer. Guarded
  `--copies-per-side` to 1 in this mode, since
  `leaderboard_expert.Controller` has no `copy_for_car` and its mutable
  per-car state (recovery timer, previous steer) would otherwise be
  shared and corrupted across multiple incumbent copies.
  `main()` additionally adds `leaderboard_expert` itself as an extra
  evaluation baseline when `--opponent expert` is used, so results show
  whether the trained policy can beat the opponent it practiced against,
  not just the two standard baselines.
- Ran the full test suite (186 passed), `ruff check`/`ruff format --check`
  (clean), and `pyright` in strict mode (0 errors) on the changed file
  before treating the plumbing as trustworthy.
- Smoke-tested the new code path with a tiny run (2 races, 15s rounds) to
  confirm it executes end-to-end and correctly adds the new eval baseline,
  before spending the ~3-minute wall-clock cost of a full comparable run.
- Ran the full comparison: identical hyperparameters to the current
  reference checkpoint's original training run
  (`2026-09-08_seed-sweep-v2-8000`: seed 8000, races=40, round_seconds=120,
  n_step=3, hidden_size=128, buffer_capacity=800000), changing only
  `--opponent` from `self` to `expert`.

**Evidence:**
- Sources or documentation: re-read `src/controllers/leaderboard_expert.py`
  in full before proposing this direction, to confirm it's a genuinely
  different (rule-based, deterministic) driving style rather than another
  learned policy, and specifically that it already encodes exactly the
  hazard rules (competitor-proximity speed cap, speed-scaled wall-braking
  horizon, stuck recovery) that this track's reward-shaping attempts at
  the same problems had failed on.
- AI-agent assistance: Claude Code wrote the `--opponent` flag and its
  guard, ran `ruff`/`pyright`/`pytest`, ran the smoke test and the full
  comparison training run, computed aggregate eval metrics directly from
  `eval_results.json` (avg damage/off-track/wall-contact/car-contact/laps/
  lap-time/speed/elimination across all races) rather than trusting only
  the per-race console output, and diagnosed the regression from
  `metrics.csv`'s critic-loss/alpha trends rather than reporting the eval
  numbers alone.
- Commits or code: `scripts/train_sac.py` (`--opponent` flag, guard,
  extra eval baseline).
- Experiment output:
  `experiments/2026-09-10_expert-opponent-seed8000/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a -- not adopted, `race_faster.py` unchanged.

**What we observed:** Training against the fixed expert opponent for the
full run, from random initialization, produced a **severe regression**,
not an improvement -- 100% elimination across all 30 evaluated races
(vs. 0/20 for the reference), avg laps 0.63 (vs. 7.30), and losses to
`default_student_controller` (2/10 wins, down from 20/20) and to the
expert itself (0/10). Notably, avg opponent-collision time was actually
*lower* than the reference (0.27s vs. 1.21s) -- so this is not a
recurrence of the causal-test-21-23 opponent-collision failure mode; the
car crashes into walls at very high speed (avg max speed 37.3 vs. 26.7
m/s) instead. `metrics.csv` shows real training instability (critic loss
swinging 3.5 -> 48.7 over the run rather than settling; the entropy
temperature `alpha` collapsing from ~1.0 to ~0.03 very early, i.e. the
policy became confidently deterministic before the critic had anything
reliable to be confident about), and total transitions collected
(111,472) were far short of the theoretical maximum for 40 uninterrupted
races -- meaning the challenger was also being eliminated frequently
*during* training, not just in evaluation. Likely root cause: unlike
self-play (where the incumbent co-evolves with the challenger, so
opponent difficulty always roughly matches current skill), a fixed,
already-competent expert opponent from tick zero exposes an unskilled
early-training policy to a training distribution dominated by
"recovering from/chasing a much faster car" rather than clean solo
driving, plausibly biasing the whole run toward reckless, low-exploration
behavior rather than the safe driving self-play produces.

**Decision and rationale:** Not adopted. `--opponent` defaults to `self`;
the reference checkpoint (`2026-09-08_seed8000-resumed-short`) and
`race_faster.py` are unchanged. This rejects the literal "always train
against the expert from scratch" version of the combined-approach
direction as tested -- the hypothesis that a different sparring partner
would straightforwardly teach better opponent-awareness was wrong, or at
least wrong for a full from-scratch run; reporting this plainly rather
than treating it as validating the underlying idea. The `--opponent`
flag and its test coverage are kept in code (default off), following this
track's convention of preserving tested-but-rejected mechanisms rather
than deleting them.

**Next steps:**
1. A curriculum variant is untested and more consistent with what
   actually differs here: `--resume-from` an already-good self-play
   checkpoint and fine-tune against the expert for a small dose of races
   (mirroring causal test 32's successful small-dose fine-tuning pattern),
   rather than training against it from random init for the full run.
2. A mixed-opponent variant (alternating self-play and expert-opponent
   races within one run) is also untested and would avoid committing the
   entire training trajectory to the harder distribution before the
   policy has any baseline competence.
3. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-10 (continued) 17:30

**Participants and contributions:** Charlotte (direction: "continue to
fine tune until we find where we can make positive change"), Claude Code
(implementation, six training runs, diagnosis, documentation).

**Question or objective:** Following up on this session's earlier
regression (training against `leaderboard_expert` from scratch), test
whether *fine-tuning* the existing reference checkpoint against the
expert -- a small, targeted nudge rather than retraining from random
init -- can find a genuine improvement, iterating across doses and resume
strategies until either a positive result or a clear stopping point is
found.

**What we investigated or changed:**
- Tested resuming `2026-09-08_seed8000-resumed-short` and continuing
  training against `leaderboard_expert` at two doses each of two resume
  strategies (4 runs): full resume (`--resume-from`, existing flag: loads
  policy + critics + entropy temperature) at 2 and 10 races.
- Diagnosed both full-resume doses as a near-total collapse (27-29/30
  eliminated) within a handful of races. Hypothesized the checkpoint's
  already-near-zero entropy temperature (`log_alpha`, converged to ~0.035
  from its own self-play training) left it with almost no exploration
  budget to recover once the unfamiliar opponent distribution started
  producing mistakes.
- To test that hypothesis, added `SACAgent.load_policy_only`
  (`src/training/sac.py`) and a `--resume-policy-only` flag
  (`scripts/train_sac.py`): loads only the actor's weights, leaving
  critics and `log_alpha` at fresh `__init__` values (full exploration
  budget restored). Ran ruff/pyright/pytest (186 passed) before using it.
- Ran this policy-only mode at four doses (2, 10, 20, 40 races) against
  the expert, each evaluated against the same fixed 5-seed set and all
  three baselines (`crash_fast`, `default_student_controller`,
  `leaderboard_expert`).

**Evidence:**
- AI-agent assistance: Claude Code wrote `load_policy_only` and
  `--resume-policy-only`, ran `ruff check`/`ruff format --check`/`pyright`
  (clean) and the full test suite (186 passed) before using the new code
  path, ran all 6 training+eval runs, and computed aggregate metrics
  (avg damage/off-track/wall-contact/car-contact/laps/lap-time/speed,
  elimination counts, per-baseline win counts) directly from each run's
  `eval_results.json` rather than relying on console output alone, plus
  inspected `metrics.csv`'s critic-loss/alpha trends to diagnose *why*
  each variant behaved as it did, not just what the final numbers were.
- Commits or code: `src/training/sac.py` (`SACAgent.load_policy_only`),
  `scripts/train_sac.py` (`--resume-policy-only` flag + guard).
- Experiment output: six new run directories under `experiments/`
  (`2026-09-10_expert-finetune-dose2-seed8000`,
  `...-dose10-seed8000`, `...-policyonly-dose2-seed8000`,
  `...-policyonly-dose10-seed8000`, `...-policyonly-dose20-seed8000`,
  `...-policyonly-dose40-seed8000`, each with its own `config.yaml`/
  `metrics.csv`/`eval_results.json`/`checkpoints/`/`notes.md`), plus a
  consolidated comparison and decision in
  `experiments/2026-09-10_expert-opponent-seed8000/followup_finetune_sweep.md`.
- Leaderboard result: n/a -- not adopted, `race_faster.py` unchanged.

**What we observed:** Full resume collapsed at both doses tested (27-29/30
eliminated), confirming the near-zero-exploration hypothesis was at least
plausible enough to act on. Policy-only resume was a qualitatively
different, much more stable regime at every dose (1-14/30 eliminated, vs.
27-29 for full resume), with a real, mostly-monotonic improving trend as
dose increased from 2 to 40 races: avg damage, laps completed, and lap
time all improved substantially, and by dose=20 the policy already swept
20/20 against both standard baselines, matching the reference's win
record there. **But even at dose=40 (matching the original from-scratch
run's total training length), no variant closed the gap to the
reference on any metric, and the best-performing variant's avg
car-contact time (4.326s) was *higher* than the reference's (1.702s) --
worse, not better, on the exact metric this entire combined-approach
direction was meant to improve.** Full comparison table in
`followup_finetune_sweep.md`.

**Decision and rationale:** Not adopted at any tested dose or resume
mode; `2026-09-08_seed8000-resumed-short` remains the reference checkpoint
and `race_faster.py` is unchanged. Recommending a pause on this specific
direction (fine-tuning against a fixed expert opponent) rather than
running further doses by default -- the trend is real but has not yet, at
any point tested, actually improved opponent-avoidance, which was the
original motivation, and closing the remaining gap to the reference looks
like it would require substantially more compute for an uncertain payoff.
`--opponent expert` and `--resume-policy-only` are kept in code (both
default off) as tested, working, documented mechanisms, per this track's
convention of preserving negative results rather than deleting the code
that produced them.

**Next steps:**
1. If revisited: a mixed-opponent curriculum (alternating self-play and
   expert-opponent races within one run, rather than switching to the
   expert entirely after an initial phase) is structurally different from
   every variant tried here and remains untested.
2. Otherwise, treat the reference checkpoint as the best available for
   this direction and redirect combined-approach effort elsewhere -- e.g.
   the earlier-discussed shielded/safety-override idea (borrowing
   `leaderboard_expert`'s hazard rules as an inference-time override
   rather than a training signal), which carries none of this
   training-instability risk since it never retrains the policy.
3. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-10 (continued) 19:20

**Participants and contributions:** Charlotte (direction: explore other
combined-approach ideas after the fine-tuning search found nothing),
Claude Code (design, implementation, experiment, documentation).

**Question or objective:** Since training against the expert (as opponent,
in either resume mode) never once helped the target problem
(opponent-collision time) despite 7 tested variants, try a mechanism that
combines the two tracks' work without ever leaving self-play's
proven-stable training distribution: reward SAC for matching the expert's
action, but only during the specific hazard states already tracked by the
reward function.

**What we investigated or changed:**
- Added `WEIGHT_EXPERT_MATCH` and helper functions `_in_hazard`/
  `_expert_match_penalty` to `src/training/reward.py`: `step_reward` now
  optionally accepts `previous_action`/`expert_action` (both default
  `None`, so every existing call site and test is unaffected) and adds a
  bonus for the two being close, restricted to ticks where the existing
  wall/robot-proximity checks already judge the state hazardous.
- Added a private, non-controlling shadow instance of
  `controllers.leaderboard_expert.Controller` to `TrainableController`
  (`src/training/controller.py`, new `expert_match` constructor flag,
  propagated through `copy_for_car`), fed every tick's real sensors purely
  to compute what the expert would have done -- it never drives the car,
  and self-play's actual opponent is completely unchanged.
- Added a `--expert-match-bonus` flag to `scripts/train_sac.py`, wired to
  both challenger and incumbent (both remain ordinary `TrainableController`
  self-play instances -- `--opponent` was left at its default `self`).
- Wrote 7 new unit tests (`tests/test_training_reward.py`,
  `tests/test_training_controller.py`) covering: the bonus is a no-op
  without both actions present, it only applies in a hazard state, it's
  zero when actions already match, it scales with action distance, the
  weight is currently enabled, and the controller runs/pushes normally
  with the flag on (including through `copy_for_car`). Ran the full suite
  (193 passed), `ruff check`/`ruff format --check` (one line-length fix
  needed), and `pyright` strict mode (clean) before running any training.
- Ran a full training experiment matching the *original* from-scratch
  config of the current reference checkpoint's lineage
  (`2026-09-08_seed-sweep-v2-8000`: seed=8000, races=40,
  round_seconds=120, n_step=3, hidden_size=128, buffer_capacity=800000)
  with only `--expert-match-bonus` added.

**Evidence:**
- Sources or documentation: re-derived the hazard predicate directly from
  the existing `_wall_proximity_penalty`/`_robot_proximity_penalty`
  functions already in `training/reward.py`, rather than introducing new
  distance thresholds, so "hazard" means exactly what those
  (independently reward-shaped) mechanisms already treat as dangerous.
- AI-agent assistance: Claude Code designed and implemented the mechanism,
  wrote and ran the new tests, ran `ruff`/`pyright`/`pytest` before
  training, ran the training+eval experiment, computed aggregate metrics
  directly from `eval_results.json` (not console output alone), inspected
  `metrics.csv` for training-stability confirmation, and cross-checked
  per-race detail to confirm the result wasn't one outlier.
- Commits or code: `src/training/reward.py` (`WEIGHT_EXPERT_MATCH`,
  `_in_hazard`, `_expert_match_penalty`, extended `step_reward` signature),
  `src/training/controller.py` (`expert_match` flag, shadow expert),
  `scripts/train_sac.py` (`--expert-match-bonus` flag), 7 new tests.
- Experiment output:
  `experiments/2026-09-10_expert-match-bonus-seed8000/` (`config.yaml`,
  `metrics.csv`, `eval_results.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a -- not yet adopted, `race_faster.py` unchanged.

**What we observed:** Versus the matched from-scratch reference
(`2026-09-08_seed-sweep-v2-8000`): avg car-contact time (the target
metric) improved 1.212s -> 1.136s (-6%), avg damage improved 0.0298 ->
0.0129 (-57%), avg max speed rose 26.65 -> 35.11 m/s (+32%), and avg laps
(7.30) / avg best lap time (14.96s -> 14.98s) / eliminations (0/20) /
wins (20/20 vs. both baselines) all held essentially unchanged. The
trade-off: avg off-track (0.152s -> 0.372s) and avg wall-contact (0.107s
-> 0.236s) both rose, though both remain small in absolute terms and
`metrics.csv` shows no training instability (critic loss and entropy
settled normally throughout, unlike every expert-*opponent* experiment
earlier today) and per-race detail shows the increase spread across
several races/seeds rather than one outlier. This also compares
favorably to the currently-packaged reference
(`2026-09-08_seed8000-resumed-short`) on car-contact, lap time, and max
speed, at the cost of a still-negligible damage increase (0.0129 vs.
0.0004, both far from elimination).

**Decision and rationale:** This is the first combined-approach
experiment today where the actual target metric (opponent-collision time)
improved, together with real gains in damage and speed at essentially no
lap-time cost -- a genuine, if modest, positive result, and the first one
found all day. Not unilaterally repackaging `race_faster.py` -- presenting
this as a candidate for direction, consistent with how earlier
judgment-call improvements on this track (e.g. causal test 27) were
handled. This is a single run/seed; per this track's own established
caution about n=1 results, it demonstrates the mechanism *can* help, not
that it reliably will.

**Next steps:**
1. Await direction on whether to adopt this checkpoint or repackage
   `controllers.race_faster` from it.
2. A seed sweep would establish robustness versus one favorable draw --
   not yet done for this mechanism.
3. The off-track/wall-contact increase is small but unexplained -- a
   per-tick diagnostic on one of the nonzero races would help if this
   direction is pursued further.
4. `WEIGHT_EXPERT_MATCH = 0.5` was chosen by analogy to
   `WEIGHT_WALL_PROXIMITY`'s scale, not tuned by a dedicated sweep.
5. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-11 (combined-approach) 10:15

**Participants and contributions:** Charlotte (direction: build a
literal, visible combination of the SAC and imitation-learning tracks,
after confirming the training-time reward nudge was too subtle to count
as "using both approaches" in the final controller), Claude Code (design,
implementation, evaluation, documentation).

**Question or objective:** Build a controller where both tracks' finished
work is literally present and visible in the final artifact, not just a
training-time influence baked into weights -- SAC drives normally, and
`controllers.leaderboard_expert` (the separate imitation-learning track's
rule-based expert) takes over control during specific hazard moments.

**What we investigated or changed:**
- Added `src/controllers/hybrid_controller.py`: composes
  `controllers.race_faster` (this track's packaged SAC checkpoint,
  `2026-09-08_seed8000-resumed-short`) and `controllers.leaderboard_expert`
  (Lucy's hand-written expert) by calling both every tick and returning
  the expert's command during a wall- or robot-proximity hazard (duplicated
  threshold constants matching `training.reward`'s existing hazard
  definition), SAC's command otherwise. Both sub-controllers are always
  called, not just the active one, since `leaderboard_expert.Controller`
  is stateful (a stuck-recovery timer, previous-tick steer/throttle) and
  must see every tick to stay consistent with the real race. Implements
  its own `copy_for_car` (fresh sub-controller instances per car) so the
  expert's recovery state is never shared across cars in a race.
- Confirmed via `scripts/export_student_controllers.py`'s own
  AST-based dependency walker (`local_controller_dependencies`) that a
  controller importing two sibling `controllers.*` modules packages
  correctly (already how every real submission is built, per this
  track's established `--all-controllers` convention).
- Wrote 8 new unit tests (`tests/test_hybrid_controller.py`): hazard
  hand-off for a close wall and a close ahead competitor, no hand-off for
  a distant or off-angle competitor, command stays in documented ranges,
  and a copy's expert state is independent of the original's. Ran the
  full suite (200 passed), `ruff check`/`ruff format --check`, and
  `pyright` strict mode (all clean) before evaluating in a real race.
- Ran a real single race (`racing h2h`, headless, seed 110) to confirm it
  works end to end, then a proper evaluation: the hybrid across the fixed
  5-seed set against all three available opponents
  (`crash_fast`, `default_student_controller`, `leaderboard_expert`,
  2 races each, 120s rounds -- the same protocol
  `training.evaluation.evaluate_against_baselines` uses). Also ran plain
  `controllers.race_faster` (no shield) against `leaderboard_expert`
  under the identical protocol, since that specific matchup (pure SAC vs.
  the expert as a live, aggressive opponent) had never been tested before
  -- every prior reference number only came from racing against the two
  much slower/less aggressive standard baselines.

**Evidence:**
- AI-agent assistance: Claude Code designed and implemented the module,
  wrote and ran the new tests, verified real-race behavior via `racing
  h2h --json`, ran both evaluation sweeps via a short one-off script using
  `run_headless_head_to_head` directly, and computed aggregate metrics
  from the raw per-race results to compare fairly (same-opponent
  comparisons, not just against numbers from a different, easier
  matchup).
- Commits or code: `src/controllers/hybrid_controller.py`,
  `tests/test_hybrid_controller.py` (8 tests).
- Experiment output: `experiments/2026-09-11_hybrid-controller-eval/`
  (`notes.md`, `hybrid_eval.json`, `pure_sac_vs_expert.json`).
- Leaderboard result: n/a -- not adopted.

**What we observed:** The hybrid **reduces average wall/car-contact and
off-track time in both matchups tested, but increases the elimination
rate in both** -- vs. the two standard baselines, 0/20 -> 1/20 eliminated
(avg damage 0.0004 -> 0.0500, avg car-contact 1.702s -> 2.652s, though
avg off-track/wall-contact both *dropped*); vs. `leaderboard_expert` as a
live opponent (first time either the shielded or plain SAC controller had
been tested against it), 0/10 -> 6/10 eliminated (avg damage 0.1471 ->
0.6055) while avg car-contact/off-track/wall-contact all dropped
somewhat. Fewer, shorter close calls on average, but the close calls that
do happen are more often catastrophic -- not a clean improvement.

**Decision and rationale:** Not adopted as a final controller. The
mechanism is implemented correctly (routes as designed, fully tested,
races successfully end to end), but the net safety effect is a genuine
trade-off rather than an improvement, and the specific shape of that
trade-off (rarer but more severe failures) is a worse kind of risk than
the status quo. Likely mechanism (not yet directly confirmed with a
per-tick trace): a hard switch hands full control to a completely
different, independently-tuned control law with no continuity in the
commanded action across the boundary -- if SAC and the expert would have
commanded meaningfully different actions at the exact hand-off tick,
that's a sudden control discontinuity at precisely the moment the car is
already close to a wall or competitor. Consistent with a pattern already
seen elsewhere on this track (docs/rl_design.md section 6): safety
mechanisms here tend to produce all-or-nothing outcomes rather than
smooth trade-offs. `controllers.hybrid_controller` is kept in the
codebase as a real, tested artifact, not deleted, but not recommended for
submission or racing as-is.

**Next steps:**
1. The most likely fix is smoothing the hand-off -- blend toward the
   expert's action proportionally to hazard severity, or ease the
   transition over a few ticks, instead of a hard one-tick switch. This
   was flagged as an alternative when the shield idea was first proposed
   and not yet built.
2. A per-tick trace of one of the 6 `leaderboard_expert`-matchup
   eliminations would directly confirm or rule out the discontinuous-
   handoff hypothesis.
3. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-11 (combined-approach, continued) 10:58

**Participants and contributions:** Charlotte (direction: fix the
shield's discontinuity problem by having SAC train on top of Lucy's
controller as a base, i.e. "take Lucy's controller then use SAC to make
it faster/more efficient"), Claude Code (design, implementation,
experiment, evaluation, documentation).

**Question or objective:** The hard-switch shield (previous entry)
reduced average contact time but introduced a much higher catastrophic-
crash rate, diagnosed as likely coming from the discontinuity of
switching instantly between two unrelated control laws. Test the fix:
make the expert's influence continuous instead of switched, by having
SAC learn a bounded correction on top of the expert's action every tick
("residual reinforcement learning") rather than taking over control
entirely.

**What we investigated or changed:**
- Added a `residual_base` mode to `TrainableController`
  (`src/training/controller.py`): `controllers.leaderboard_expert
  .Controller`'s command becomes the base action every tick; the SAC
  policy's own (tanh-bounded) output is scaled by a new
  `RESIDUAL_ACTION_SCALE = 0.3` and added to it, clamped to `[-1, 1]`.
  Critically, the *raw* policy output (not the blended command) is what
  gets pushed to the replay buffer and what the SAC Bellman backup is
  defined over, since that's the actual action space the policy controls
  -- the reward still reflects the real physical consequence of the
  blended command the car received. Mutually exclusive with
  `expert_match` (raises `ValueError` if both set, since the
  expert-match bonus assumes `previous_action` is an absolute command,
  which it isn't once the policy outputs a correction instead).
- Added `--residual-expert-base` to `scripts/train_sac.py`, wired to both
  challenger and incumbent (both remain ordinary self-play
  `TrainableController` instances -- `--opponent` stayed at `self`).
- Fixed a real correctness gap this uncovered:
  `training.evaluation.evaluate_against_baselines` built its own internal
  `TrainableController` without knowing about `residual_base` -- silently
  evaluating a residual-trained network's *correction* output as if it
  were an absolute command. Added a `residual_base` parameter there and
  threaded it through `scripts/train_sac.py`'s own evaluation call and a
  matching `--residual-expert-base` flag on `scripts/eval_sac.py`, so
  re-evaluating a residual checkpoint later can't silently produce wrong
  numbers.
- Wrote 4 new unit tests (`tests/test_training_controller.py`): the two
  modes are mutually exclusive, the blended command exactly matches
  `expert_command + RESIDUAL_ACTION_SCALE * raw_policy_action` (clamped),
  the *raw* action (not the blended one) is what's pushed to the replay
  buffer, and `copy_for_car` propagates the mode correctly. Ran the full
  suite (204 passed), `ruff`/`ruff format --check`/`pyright` strict mode
  (all clean) before training.
- Ran a full training experiment matching the original from-scratch
  reference config exactly (`2026-09-08_seed-sweep-v2-8000`: seed=8000,
  races=40, round_seconds=120, n_step=3, hidden_size=128,
  buffer_capacity=800000) with only `--residual-expert-base` added.
- Evaluated the resulting checkpoint two ways: the standard protocol (vs.
  `crash_fast`/`default_student_controller`, 20 races) and, critically,
  the same live-opponent stress test that exposed the shield's failure
  (vs. `leaderboard_expert` directly, 10 races) -- via a short one-off
  script loading the checkpoint into a `residual_base=True` evaluation
  controller, since this mode isn't packaged as a `controllers.*` module
  yet.

**Evidence:**
- AI-agent assistance: Claude Code designed and implemented the
  mechanism, wrote and ran the new tests, fixed the evaluation gap before
  it could produce misleading numbers, ran the training experiment, ran
  both evaluation sweeps, computed aggregate metrics directly from the
  raw per-race results, and cross-checked `metrics.csv` for training
  stability.
- Commits or code: `src/training/controller.py` (`residual_base` mode,
  `RESIDUAL_ACTION_SCALE`), `src/training/evaluation.py` (`residual_base`
  parameter), `scripts/train_sac.py` and `scripts/eval_sac.py`
  (`--residual-expert-base` flag), 4 new tests.
- Experiment output: `experiments/2026-09-11_residual-expert-base-
  seed8000/` (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not yet adopted.

**What we observed:** Versus the matched from-scratch reference, standard
protocol: avg laps 7.30 -> **10.10**, avg best lap time 14.96s ->
**11.25s (-25%)**, still 0/20 eliminated, 20/20 wins -- damage/off-track/
wall-contact all rose somewhat (0.0298 -> 0.0662 damage) but stayed small
and zero eliminations held. Versus `leaderboard_expert` directly (the
real test, since this is what exposed the shield's failure): eliminated
0/10 (pure SAC) -> 6/10 (hard-switch shield) -> **1/10 (residual RL)** --
recovering almost all of plain SAC's safety while being the fastest and
most complete of all three variants (12.08s avg best lap time, 9.50 avg
laps, both better than pure SAC *and* the shield). `metrics.csv` showed
normal, stable training throughout, unlike every train-against-a-fixed-
opponent experiment (causal test 35) -- confirming that changing the
action *composition* (continuous base + correction) rather than the
training *opponent* avoids that instability entirely, as intended.

**Decision and rationale:** This is the strongest combined-approach
result found this session -- a continuous correction on top of the
expert, rather than a discontinuous switch, appears to be exactly the
fix the shield's diagnosis called for. Not a perfect result (one
elimination against the expert that plain SAC didn't have, and a real,
not-yet-diagnosed outlier), and not unilaterally adopted or repackaged
into `race_faster.py` -- presented as the strongest candidate for
direction, consistent with this session's established practice.

**Next steps:**
1. Await direction on whether to adopt this (e.g. package a self-
   contained `controllers.*` module analogous to `race_faster.py`).
2. Diagnose the one elimination against the expert with a per-tick trace.
3. A seed sweep would establish robustness versus one favorable draw.
4. `RESIDUAL_ACTION_SCALE = 0.3` was chosen without a dedicated sweep.
5. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-11 (combined-approach, continued) 11:32

**Participants and contributions:** Charlotte (direction: diagnose the
one crash from the previous entry and keep iterating until the combined
controller surpasses both individual controllers), Claude Code (per-tick
diagnosis, fix, retrain, evaluation, documentation).

**Question or objective:** The residual-RL controller (previous entry)
had one elimination out of 10 evaluation races against
`leaderboard_expert`. Diagnose the root cause with a per-tick trace (the
project's established method for outliers) and fix it, then check how
close the result comes to actually surpassing plain SAC and Lucy's raw
expert, not just the failed hybrid shield.

**What we investigated or changed:**
- Reproduced the exact crash (seed 2024, race 1) with a
  `sensor_sample_callback` logging the challenger's tick/speed/wall
  distance/contact/damage every tick, matching the project's established
  outlier-diagnosis method. Found the car hit the *identical* wall
  location 8 times over ~8 seconds (ticks 3576-4062+), each time backing
  off via `leaderboard_expert.Controller`'s stuck-recovery maneuver
  (reverse + a fixed hard steer for up to 28 ticks) and then driving
  straight back into the same spot -- not a single high-speed impact.
  Confirmed pure SAC (no residual) does not hit this at the same seed,
  isolating the cause to the residual composition itself.
- Root cause: the SAC correction was still being added on top of the
  expert's command *during its own deliberate recovery maneuver*, likely
  diluting a precise, fixed escape trajectory just enough that it never
  fully cleared the obstacle before the next attempt.
- Fixed in `src/training/controller.py`: detect when the base expert's
  command is a recovery command (checking `leaderboard_expert
  .Controller`'s `_recovery_ticks_remaining` counter both before and
  after calling it, since a recovery maneuver can both start and be
  mid-flight within a single call) and pass it through completely
  unmodified in that case -- zero residual applied during recovery.
- Verified the fix requires retraining, not just an inference patch:
  applying it to v1's already-trained checkpoint reproduced the identical
  crash (the policy had learned its behavior assuming its correction
  always applied). Retrained fresh with the fix in place, identical
  config to v1 otherwise (seed=8000, races=40, round_seconds=120,
  n_step=3, hidden_size=128, buffer_capacity=800000).
- Added a unit test (`tests/test_training_controller.py`) asserting the
  controller returns the expert's exact recovery command
  (`throttle == -1.0`, `abs(steer) == 0.9`) unmodified when triggered.
  Ran the full suite (205 passed) and `ruff`/`pyright` (clean) before
  retraining.
- Re-ran the standard evaluation and the live-opponent stress test
  against `leaderboard_expert` on the new checkpoint, specifically
  re-checking seed 2024.
- Also established Lucy's raw expert's own solo baseline (no
  combination, racing alone against `crash_fast` under the identical
  protocol) for a fair read on how close the combined controller comes to
  actually surpassing her work, not just plain SAC.

**Evidence:**
- AI-agent assistance: Claude Code wrote and ran the diagnostic script,
  identified the root cause from the trace, implemented and unit-tested
  the fix, verified it required retraining (not just an inference patch)
  before committing to the retrain, ran the retrain and both evaluation
  sweeps, and computed all aggregate metrics directly from the raw
  per-race results.
- Commits or code: `src/training/controller.py` (recovery-passthrough
  fix), `tests/test_training_controller.py` (1 new test).
- Experiment output: `experiments/2026-09-11_residual-expert-base-v2-
  seed8000/` (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not yet adopted.

**What we observed:** The fix fully resolved the crash -- seed 2024 now
completes cleanly both races (damage 0.0 and 0.009, 10 laps each, no
elimination), and the live-opponent stress test overall went from 1/10 to
**0/10 eliminated, matching pure SAC's perfect record**. This wasn't a
narrow fix either: every safety metric on the *standard* baseline
evaluation improved too (avg damage 0.0662 -> 0.0254, avg off-track
0.537s -> 0.325s, avg wall-contact 0.256s -> 0.140s, avg car-contact
1.669s -> 1.468s) while keeping the same large speed/lap gains (avg best
lap time ~11.3-11.4s vs. the reference's 14.96s, ~35% faster). Against
the expert specifically, this version is now the fastest and most
complete of every variant tested this session (11.78s avg best lap time,
9.80 avg laps -- better than pure SAC, the shield, *and* v1). For honest
context: Lucy's raw expert alone (no combination) still laps faster in
isolation (8.94s avg, 0.047 avg damage, 0/10 eliminated, same protocol) --
this combined controller has not closed that specific gap, though it now
clearly and unambiguously surpasses plain SAC alone and the failed hybrid
shield on every metric tracked.

**Decision and rationale:** Not unilaterally adopted or repackaged into
`race_faster.py` -- presented as the strongest, most complete combined-
approach candidate found this session, and a materially stronger case for
adoption than either prior candidate (the shield, or v1) since it now
dominates rather than trades off against plain SAC.

**Next steps:**
1. Await direction on adoption / packaging as a self-contained
   `controllers.*` module.
2. Closing the remaining pace gap to Lucy's raw expert (8.94s) would
   likely need more training, a larger `RESIDUAL_ACTION_SCALE`, or may
   partly reflect her expert's greater risk tolerance that a controller
   optimizing for both speed and safety may not fully match by design.
3. A seed sweep would establish robustness versus one favorable draw --
   not yet done for either residual version.
4. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-11 (combined-approach, continued) 11:57

**Participants and contributions:** Charlotte (direction: push further on
the remaining speed gap to Lucy's raw expert and try to close or surpass
it), Claude Code (implementation, experiment, honest negative result,
documentation).

**Question or objective:** v2's residual controller still laps
noticeably slower (11.37-11.78s) than Lucy's raw expert alone (8.94s,
same protocol). Test the most direct lever specific to this mechanism:
give the policy's correction more room to shift the expert's baseline
action, on the hypothesis that the current cap (`RESIDUAL_ACTION_SCALE =
0.3`) might be limiting how much the policy can improve on the expert's
line/pace.

**What we investigated or changed:**
- Made `RESIDUAL_ACTION_SCALE` configurable instead of a hardcoded
  constant: added `residual_scale` to `TrainableController.__init__`
  (default unchanged), threaded through `copy_for_car`,
  `training.evaluation.evaluate_against_baselines`, and a new
  `--residual-action-scale` flag on both `scripts/train_sac.py` and
  `scripts/eval_sac.py`. Added a unit test confirming a larger scale
  produces a larger deviation from the expert's command. Ran the full
  suite (206 passed) and `ruff`/`pyright` (clean) before training.
- Trained fresh at `--residual-action-scale 0.45` (up from 0.3), config
  otherwise identical to v2 (seed=8000, races=40, round_seconds=120,
  n_step=3, hidden_size=128, buffer_capacity=800000).
- Evaluated the same two ways as v2: the standard protocol (vs.
  `crash_fast`/`default_student_controller`) and the live-opponent stress
  test (vs. `leaderboard_expert` directly).

**Evidence:**
- AI-agent assistance: Claude Code implemented and unit-tested the
  configurable scale, ran the training experiment, ran both evaluation
  sweeps, computed aggregate metrics from the raw per-race results, and
  checked `metrics.csv` for training stability before concluding this
  was a genuine behavioral regression rather than an unstable run.
- Commits or code: `src/training/controller.py` (`residual_scale`
  parameter), `src/training/evaluation.py` (`residual_scale` parameter),
  `scripts/train_sac.py` and `scripts/eval_sac.py`
  (`--residual-action-scale` flag), 1 new test.
- Experiment output: `experiments/2026-09-11_residual-scale045-
  seed8000/` (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not adopted.

**What we observed:** A clean, consistent regression across both
evaluation protocols, not an unstable-training artifact (`metrics.csv`
showed normal, bounded critic loss and smoothly settling entropy). Versus
the standard baselines: avg off-track time rose 17.6x (0.325s -> 5.737s),
avg wall-contact rose 23.6x (0.140s -> 3.305s), avg damage rose 5x, avg
laps dropped, and one race was lost outright (19/20, down from 20/20) --
**and lap time itself got slightly slower, not faster** (11.37s ->
11.83s), which was the entire point of the change. The live-opponent
stress test showed the identical pattern (off-track 7.5x worse,
wall-contact 8.8x worse, lap time slower). The extra correction budget
was not spent on a faster line -- it produced more erratic, wall/off-
track-prone driving instead.

**Decision and rationale:** Not adopted; reverted.
`2026-09-11_residual-expert-base-v2-seed8000` (`residual_scale=0.3`, the
default) remains the best combined-approach checkpoint found this
session. `--residual-action-scale` is kept as a configurable, tested
parameter (default unchanged) per this track's convention. This specific
lever is treated as exhausted after one clear negative result -- not
continuing to search this axis without a different underlying idea. The
remaining pace gap to Lucy's raw expert (8.94s) may be partly structural:
her expert explicitly accepts more risk to maximize speed, while the RL
reward function balances speed against safety by design, which may set a
genuine ceiling below a pure-speed controller for anything optimizing
both objectives at once.

**Next steps:**
1. This axis (residual scale) is exhausted for now -- a different lever
   (more training at the existing scale, or residual-mode-specific
   reward tuning) would be needed to make further progress on pace,
   distinct from simply giving the correction more room.
2. Otherwise, treat v2 as the practical best result for this track's
   combined-approach work: it already unambiguously surpasses plain SAC
   alone and the hybrid shield on every safety and pace metric tracked,
   even though it hasn't closed the gap to Lucy's raw, safety-unconstrained
   pace.
3. A seed sweep would establish robustness versus one favorable draw for
   v2 specifically.
4. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 09:26

**Participants and contributions:** Charlotte (direction: keep improving
the combined controller via residual RL toward concrete improvements),
Claude Code (experiment, documentation).

**Question or objective:** Test the opposite direction from the failed
`scale=0.45` experiment -- a *narrower* correction (`residual_scale=0.15`,
down from v2's `0.3`), on the hypothesis that the wider correction's
erratic driving (from the prior entry) meant the correction's value was
mostly about smoothing rather than speed-finding, so tightening it
further might reduce erratic driving without much speed cost.

**What we investigated or changed:**
- Trained fresh at `--residual-action-scale 0.15`, config otherwise
  identical to v2 (seed=8000, races=40, round_seconds=120, n_step=3,
  hidden_size=128, buffer_capacity=800000).
- Evaluated the same two ways as every prior residual variant: the
  standard protocol and the live-opponent stress test vs.
  `leaderboard_expert`.

**Evidence:**
- Experiment output: `experiments/2026-09-12_residual-scale015-seed8000/`
  (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not adopted.

**What we observed:** Not simply safer. On the standard baselines,
average damage improved (0.0254 -> 0.0100), but every other metric got
worse (avg off-track 0.325s -> 0.593s, avg wall-contact 0.140s -> 0.280s,
avg laps 10.15 -> 9.45, avg best lap time 11.37s -> 12.08s). In the harder
live-opponent test, a real elimination reappeared (0/10 -> 1/10, exactly
what v2 had eliminated) alongside worse car-contact, laps, and pace.
`metrics.csv` showed normal, stable training -- a real behavioral
difference, not an unstable run. The hypothesis that a narrower
correction would simply reduce erratic driving was wrong: it has less
power to react when a real course change is genuinely needed (e.g.
avoiding a closing competitor), trading a small average-damage
improvement for worse outcomes almost everywhere else, including a
reintroduced crash.

**Decision and rationale:** Not adopted. Combined with the prior
`scale=0.45` result, this closes out the residual-scale axis -- three
points now tested (0.15, 0.3, 0.45), and `0.3` (v2) wins outright or ties
on nearly every metric in both directions. This is a genuine local
optimum, not an arbitrary first guess. Recommending further concrete
improvement come from a different lever, not more points on this axis.

**Next steps:**
1. A seed sweep (genuine network-initialization diversity, holding the
   locked-in v2 config fixed) is the project's own best-precedented way
   of finding real gains -- see the plain SAC track's causal tests 24-32,
   which found real, substantial improvements this way. Not yet tried
   for residual mode.
2. Otherwise, treat v2 (`2026-09-11_residual-expert-base-v2-seed8000`)
   as the practical best combined-approach result.
3. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 (continued) 12:59

**Participants and contributions:** Charlotte (direction: keep pushing
residual RL toward concrete improvements), Claude Code (seed sweep,
documentation).

**Question or objective:** With the residual-scale axis closed out
(prior entry), try this project's own best-precedented lever for finding
real gains: sample fresh network initializations at the locked-in best
config, mirroring the plain-SAC track's successful seed-sweep method
(causal tests 24-32).

**What we investigated or changed:** Trained two fresh seeds (9000,
10000) at the locked-in config (residual_base=True,
residual_action_scale=0.3, recovery-passthrough fix), config otherwise
identical to v2 (races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000). Evaluated both the same two ways as v2: the
standard protocol and the live-opponent stress test vs.
`leaderboard_expert`.

**Evidence:**
- Experiment output: `experiments/2026-09-12_residual-seedsweep-9000/`
  and `experiments/2026-09-12_residual-seedsweep-10000/` (each with
  `config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a -- neither adopted.

**What we observed:**

| | v2 (seed=8000) | seed=9000 | seed=10000 |
| --- | --- | --- | --- |
| eliminated (standard) | 0/20 | **1/20 (worse)** | 0/20 |
| avg damage (standard) | 0.0254 | 0.0682 | 0.0319 |
| avg laps (standard) | 10.15 | 9.45 | 9.25 |
| avg best lap time (standard) | 11.37s | 11.64s | 12.31s |
| eliminated (vs. expert) | 0/10 | -- | 0/10 |
| avg best lap time (vs. expert) | 11.78s | -- | 12.55s |

Seed 9000 was clearly worse (a reintroduced elimination plus worse
damage/laps/pace). Seed 10000 was mixed -- it matched v2's elimination
record and even improved car-contact time, but lost on off-track/
wall-contact/laps/pace in both evaluation protocols. Neither beat v2
overall. `metrics.csv` showed stable training for both -- these are
genuine differences in what each initialization converged to, not
broken runs.

**Decision and rationale:** Neither adopted.
`2026-09-11_residual-expert-base-v2-seed8000` remains the best checkpoint
after two sampled seeds. This is not surprising on its own -- the plain
SAC track's own seed sweeps found a similar low per-seed hit rate for a
strict improvement (roughly 10-20%, meaning multiple draws are often
needed) -- but two failed draws is a real, if inconclusive, data point.

**Next steps:**
1. Awaiting direction: continue sampling more seeds (a larger batch, per
   the plain SAC track's own convention of batches of 5), or conclude
   the seed sweep here given the compute already spent this session and
   settle on v2 as the final combined-approach result.
2. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 (continued) 13:16

**Participants and contributions:** Charlotte (direction: keep sampling,
explicitly targeting a checkpoint that beats Lucy's raw expert's pace,
not just plain SAC), Claude Code (two more seed draws, documentation).
Also coordinated with another Claude session (`formula110-8d`) that
reached out mid-task about to start its own combined-approach work in
this same repo -- confirmed no training/eval was left running and no
directory collisions, see that session's message and this session's
reply for the handoff details.

**Question or objective:** Sample two more fresh seeds (11000, 12000) at
the locked-in v2 config, specifically checking whether any individual
race or average approaches Lucy's raw solo pace (8.94s).

**What we investigated or changed:** Trained seeds 11000 and 12000
(residual_base=True, residual_action_scale=0.3, recovery-passthrough
fix; races=40, round_seconds=120, n_step=3, hidden_size=128,
buffer_capacity=800000 -- identical to v2 otherwise). Evaluated both the
standard way and against `leaderboard_expert` directly, and this time
also tracked the *minimum* individual lap time per checkpoint, not just
the average, since a single fast lap is directly relevant to the
"can it ever match Lucy's pace" question.

**Evidence:**
- Experiment output: `experiments/2026-09-12_residual-seedsweep-11000/`
  and `experiments/2026-09-12_residual-seedsweep-12000/` (each with
  `config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`,
  `notes.md`).
- Leaderboard result: n/a -- neither unilaterally adopted.

**What we observed -- full 5-seed table (standard-baseline protocol):**

| seed | avg damage | avg off-track | avg wall-contact | avg lap time | fastest lap | elim |
| --- | --- | --- | --- | --- | --- | --- |
| 8000 (v2, reference) | 0.0254 | 0.325s | 0.140s | 11.37s | 10.33s | 0/20 |
| 9000 | 0.0682 | 0.274s | 0.092s | 11.64s | -- | 1/20 |
| 10000 | 0.0319 | 0.935s | 0.567s | 12.31s | -- | 0/20 |
| 11000 | 0.0216 | 0.166s | 0.068s | 11.58s | 10.45s | 0/20 |
| 12000 | **0.0020** | **0.076s** | **0.007s** | 11.57s | **10.23s** | 0/20 |

Seed 11000 was a close, mixed result versus v2 (better off-track/
wall-contact, slightly worse laps/pace/car-contact) -- neither clearly
better nor worse. **Seed 12000 was a standout on safety**: 12.7x lower
damage, 4.3x lower off-track time, 20x lower wall-contact time than v2,
while essentially tying v2 on pace and actually setting a new fastest-
individual-lap record (10.23s). Against `leaderboard_expert` directly,
seed 12000 also had 11.6x lower damage than v2 and **became the first
checkpoint of any kind on this track -- plain SAC, hybrid shield, or any
residual variant -- to win a race outright against the expert** (seed
8675309, race 2), though its *average* lap time there was slower than
v2's (13.25s vs. 11.78s).

**Consolidated read across all 5 seeds:** no seed has come close to
Lucy's raw solo pace (8.94s) -- every sampled seed's average best lap
time stays in the 11.4-13.3s range on both protocols (25-45% slower).
The single fastest individual lap across all 5 seeds and both protocols
is seed 12000's 10.23s -- the closest anyone has gotten, but still ~14%
off Lucy's average, let alone her own best case. This consistency across
5 genuinely different network initializations is evidence (not proof)
that the pace gap is not primarily an initialization-luck problem -- it
more likely reflects the reward function's structural balance of speed
against safety, which every sampled seed converges toward in some form.

**Decision and rationale:** Neither seed unilaterally adopted. Seed 12000
is flagged as a genuinely strong *alternative* candidate to v2 -- arguably
better overall if safety is weighted heavily, and notable as the first
checkpoint to ever beat the expert outright -- but it does not advance
the specific pace-vs-Lucy's-expert goal this sweep targeted. v2 remains
the reference pending a decision on which axis (pace vs. safety) to
prioritize.

**Next steps:**
1. Awaiting direction: adopt seed 12000 (safety-focused), keep sampling
   more seeds one at a time (compute cost, uncertain payoff per the
   consolidated read above), or try a different lever entirely (e.g.
   residual-mode-specific reward tuning) if closing the pace gap
   specifically remains the priority.
2. Still open: the `controllers.minimum_viable` module gap.

**Addendum, same session:** added `src/controllers/residual_candidate.py`,
a local-dev viewer (mirrors `sac_candidate.py`'s convention) for watching
any residual-RL checkpoint race via `racing h2h --watch` -- no such
module existed yet, so there was no way to actually watch the combined
controller in the graphical viewer despite several checkpoints being
fully evaluated. Defaults to the current reference (v2, seed=8000);
`FORMULA110_RESIDUAL_CHECKPOINT` env var points it at any other
checkpoint (e.g. seed=12000). Verified with a real headless race
(`racing h2h`, seed 110) before treating it as working; full suite (206
tests) still passes.

## 2026-09-12 (continued) 14:10

**Participants and contributions:** Charlotte (direction: keep iterating
with RL + IL until the combined controller beats the expert on lap time
and speed, not just plain SAC), Claude Code (new mechanism, experiment,
honest negative result, documentation).

**Question or objective:** Try a genuinely different lever than the
residual-scale sweep and the seed sweep (both closed out, neither closed
the pace gap): reweight the reward function's speed-vs-caution balance
specifically for residual-mode training, on the hypothesis that the
caution terms -- tuned entirely for plain self-play, where the network
supplies 100% of its own collision avoidance -- may be more conservative
than residual mode needs, since the base action already comes from a
competent expert.

**What we investigated or changed:**
- Added `progress_weight` as an optional override of
  `training.reward.WEIGHT_PROGRESS`, threaded through `step_reward`,
  `TrainableController` (new constructor parameter, propagated via
  `copy_for_car`), and a new `--progress-weight` flag on
  `scripts/train_sac.py`. Training-only (evaluation never computes
  reward, so nothing needed changing in `training.evaluation` or
  `scripts/eval_sac.py`). Default unchanged, so every existing caller is
  unaffected unless the flag is passed explicitly.
- Added a unit test confirming the override changes only the progress
  term's contribution, holding everything else fixed. Ran the full suite
  (207 passed) and `ruff`/`pyright` (clean, after fixing one pre-existing
  test whose `**dict`-unpacking pattern no longer type-checked cleanly
  once a same-named parameter of a different type was added) before
  training.
- Trained fresh at `--progress-weight 1.5` (up from the default 1.0),
  config otherwise identical to v2. Evaluated both the standard protocol
  and the live-opponent stress test vs. `leaderboard_expert`.

**Evidence:**
- Experiment output: `experiments/2026-09-12_residual-progressweight15-
  seed8000/` (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not adopted.

**What we observed:** The opposite of the intended effect, on every axis
including the one this was meant to improve. Standard baselines: avg
damage worse (0.0254 -> 0.0569), avg off-track worse (0.325s -> 0.532s),
avg wall-contact worse (0.140s -> 0.252s), avg laps worse (10.15 -> 9.70),
and **avg best lap time got slower, not faster** (11.37s -> 12.03s), with
the fastest individual lap also getting slower (10.33s -> 11.48s). The
identical pattern held against `leaderboard_expert` directly (avg lap
time 11.78s -> 12.24s, fastest lap 11.48s, still 0/10 eliminated).
`metrics.csv` showed no training instability (critic loss bounded, alpha
settling smoothly) -- a genuine behavioral regression, not a broken run.

**Decision and rationale:** Not adopted. This is the **third** consecutive
combined-approach lever aimed at closing the pace gap to fail cleanly --
after the residual-scale sweep (both directions) and the 5-seed
initialization sweep, this is a third, structurally different mechanism
(what the correction is trained to optimize for, rather than how much it
can act or which policy it started from) pointing the same direction.
Recommending against further from-scratch runs purely aimed at beating
Lucy's raw pace via scale/seed/reward-weight tuning -- the evidence
increasingly reads as a real structural trade-off (the pace a safety-
unconstrained controller can hit vs. what one balancing speed against
damage/off-track/wall-contact can sustain) rather than a nearby local
optimum that the next tweak will escape.

**Next steps:**
1. If pace remains the priority, a genuinely different mechanism would
   be needed -- e.g. reward shaping targeting cornering technique
   specifically rather than a global speed/caution trade, or accepting a
   higher damage/elimination rate as the deliberate cost of matching
   Lucy's pace.
2. Otherwise, treat this line of investigation as concluded: v2
   (pace-balanced) and seed=12000 (safety-focused) are the two combined-
   approach checkpoints worth keeping.
3. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 (continued) 14:36

**Participants and contributions:** Charlotte (direction: try the more
invasive/riskier approach to see if any direction improves on the pace
gap), Claude Code (new mechanism, experiment, honest negative result,
documentation).

**Question or objective:** Try a cornering-technique-specific reward
shape, distinct from the three already-failed levers (residual scale,
seed init, flat progress-weight). Concrete gap identified: `training.
reward`'s `WEIGHT_CENTER_OFFSET` penalizes drifting off the centerline
uniformly everywhere on the track, while `controllers.leaderboard_expert`
itself explicitly treats corners and straights differently (its own
`bend_score`, from lookahead offsets, only triggers caution for real
bends). Hypothesis: gate the center-offset penalty by the same curvature
signal, reduced on straights, full strength in corners, so straight-line
speed isn't penalized for centerline drift that doesn't matter there --
without touching the load-bearing cornering safety signal, unlike a 2026-
09-03 attempt on the plain-SAC track that loosened this same weight
*uniformly* (including in corners) and regressed badly.

**What we investigated or changed:**
- Added `_bend_score`/`_center_offset_curvature_scale` helpers to
  `src/training/reward.py` (mirroring `leaderboard_expert.py`'s own
  bend-score formula exactly) and a `curvature_aware_center_offset: bool
  = False` parameter on `step_reward`, `TrainableController`, and a new
  `--curvature-aware-center-offset` flag on `scripts/train_sac.py`.
  Default unchanged, so every existing caller is unaffected unless
  explicitly enabled.
- Added 3 unit tests confirming: disabled by default, penalty reduced on
  a straight (flat lookahead offsets), penalty unchanged in a sharp
  corner (bend score above the reference threshold). Ran the full suite
  (210 passed) and `ruff`/`pyright` (clean) before training.
- Trained fresh with `--curvature-aware-center-offset`, config otherwise
  identical to v2. Evaluated both the standard protocol and the live-
  opponent stress test vs. `leaderboard_expert`.

**Evidence:**
- Experiment output: `experiments/2026-09-12_residual-curvature-aware-
  seed8000/` (`config.yaml`, `metrics.csv`, `eval_results.json`,
  `vs_leaderboard_expert.json`, `checkpoints/policy_final.pt`, `notes.md`).
- Leaderboard result: n/a -- not adopted.

**What we observed:** The fourth consecutive clean failure, and again not
even a trade-off -- it lost safety (an elimination reappeared in *both*
test protocols, 1/20 standard and 1/10 vs. the expert, both of which v2
had eliminated) without gaining speed on average (avg best lap time
11.37s -> 11.56s standard, 11.78s -> 11.98s vs. the expert; the fastest
individual lap also got slower, 10.33s -> 11.20s standard). One
genuinely encouraging data point -- a 10.28s individual lap against the
expert, close to the existing 10.23s record -- wasn't enough to outweigh
the reintroduced crashes. `metrics.csv` showed a critic-loss spike early
in training (30.0 at the first checkpoint, well above every other
residual run's typical range) that settled by the end -- not diverging,
but not as clean throughout as prior runs either.

**Decision and rationale:** Not adopted; reverted. This result is
informative beyond a simple rejection: the hypothesis was specifically
that the 2026-09-03 failure's *uniform* nature (touching corners too) was
why it regressed, and that a curvature-gated version would avoid that --
it didn't avoid it, just made it less severe. That suggests the center-
offset penalty's *shape* isn't the actual bottleneck on pace; something
else about how the residual correction handles full-track-width driving
is the limiting factor. **Four structurally different levers have now
failed to close the pace gap to Lucy's raw expert (8.94s)** -- how much
the correction can act, which network initialization it starts from,
what the reward optimizes for globally, and now a targeted cornering-
specific reward shape. This is a strong, consistent pattern, not one
unlucky axis. Recommending this line of investigation be concluded.

**Next steps:**
1. Recommending against further from-scratch experiments purely aimed at
   beating Lucy's raw pace via training-side tuning.
2. v2 and seed=12000 remain the two combined-approach checkpoints worth
   keeping.
3. If pace is still a priority, it likely needs a fundamentally
   different approach outside this reward-tuning family, or accepting
   that matching Lucy's raw pace requires giving up the safety balance
   this whole effort was built around.
4. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 (combined-approach) 15:05

**Participants and contributions:** Charlotte (direction: "make it
official" -- package the combined controller as a real, submission-ready
artifact rather than leaving it as an experiment checkpoint), Claude Code
(packaging, verification, documentation).

**Question or objective:** Package the residual combined-approach
checkpoint (`v2`, `2026-09-11_residual-expert-base-v2-seed8000`) the same
way `race_faster.py` packages the SAC-only checkpoint: self-contained, no
dependency outside `src/controllers/`, checkpoint trimmed to policy
weights only.

**What we investigated or changed:**
- Added `src/controllers/combined_candidate.py`: inlines the same actor
  network and 24-dim observation encoding `race_faster.py` uses, adds
  `controllers.leaderboard_expert` as the base action every tick (a
  sibling module within `src/controllers/`, not an external dependency),
  and applies the SAC network's output as a bounded correction on top
  (`_RESIDUAL_ACTION_SCALE = 0.3`, matching the checkpoint's trained
  value). Faithfully reproduces the recovery-passthrough fix from
  `training.controller` (checking `leaderboard_expert.Controller`'s
  `_recovery_ticks_remaining` before/after calling it, suppressing the
  correction entirely during the expert's stuck-recovery maneuver) --
  this packaged version must replicate that exact composition, since the
  checkpoint's weights were trained under it.
- Trimmed the checkpoint to policy weights only
  (`src/controllers/checkpoints/combined_candidate_policy.pt`, 414KB ->
  85KB, dropping critics/optimizer state, same convention as
  `race_faster_policy.pt`).
- Verified bit-for-bit against the training-time reference
  (`TrainableController(residual_base=True, residual_scale=0.3,
  deterministic=True)` loading the untrimmed checkpoint) on 20 random
  synthetic observations (`max abs diff = 0.0`) *and* specifically on
  the recovery-passthrough branch (a stuck scenario), not just general
  driving -- confirming the exact mechanism that fixed the earlier crash
  is faithfully reproduced in the packaged version, not just the general
  actor network.
- Confirmed `scripts/export_student_controllers.py`'s dependency walker
  correctly detects and would bundle `controllers.leaderboard_expert`
  when packaging this module (same mechanism already validated for
  `hybrid_controller.py`).
- Verified in two real (non-synthetic) headless races (`racing h2h`,
  seed 110): beats `crash_fast` (485.9m margin) and loses to
  `leaderboard_expert` directly, both exactly consistent with every
  other verification of this checkpoint this session.
- Added 4 new unit tests (`tests/test_combined_candidate.py`): valid
  command ranges, commits to forward progress on a clear track, exact
  recovery-command passthrough, and `copy_for_car` gives the expert
  independent per-car state. Ran the full suite (214 passed) and
  `ruff`/`pyright` (clean).

**Evidence:**
- AI-agent assistance: Claude Code wrote the packaged module, trimmed
  the checkpoint, wrote and ran the bit-for-bit and recovery-path
  verification scripts, ran real headless races to confirm end-to-end
  behavior, wrote and ran the new tests, and confirmed the export
  tooling's dependency detection before treating packaging as complete.
- Commits or code: `src/controllers/combined_candidate.py`,
  `src/controllers/checkpoints/combined_candidate_policy.pt`,
  `tests/test_combined_candidate.py`.
- Leaderboard result: n/a -- packaged and verified, but **not** wired
  into the Gradescope submission manifest and **not** uploaded. This is
  a separate, explicit decision (which `controller_module` a real
  submission zip names) that CLAUDE.md's packaging-a-submission section
  reserves for direct instruction -- `controllers.race_faster` (the
  pure-SAC checkpoint) remains whatever was last actually submitted.

**Decision and rationale:** `controllers.combined_candidate` now exists
as a real, tested, submission-shaped artifact -- the practical endpoint
of this session's combined-approach work, verified to reproduce the
exact validated training-time behavior (including the specific fix that
resolved the one known crash) rather than just approximating it. Not
promoted to replace `race_faster.py` as the actual graded submission
without explicit direction, since that's a separate, higher-stakes
decision than packaging.

**Next steps:**
1. Awaiting direction on whether to promote this to the actual
   Gradescope "improved" submission slot (replacing the pure-SAC
   `race_faster.py`), keep both available, or leave this as a
   demonstrated-but-not-submitted artifact.
2. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 16:10

**Participants and contributions:** Charlotte Tsui (direction: whether to
keep training `combined_candidate` to close the remaining pace gap to
Lucy's expert, or conclude that line of work), Claude Code (surveyed the
existing evidence, explained the combined-approach controllers, presented
options, recorded the decision).

**Question or objective:** Now that `controllers.combined_candidate` (the
packaged v2 residual checkpoint) exists, should training continue in
pursuit of closing its remaining pace gap to Lucy's raw expert (8.94s
solo vs. v2's ~11.4s)?

**What we investigated or changed:** No new training or code changes.
Reviewed `docs/rl_design.md` §6's causal test 37 chain and the separate
`experiments/2026-09-10_expert-opponent-seed8000/followup_finetune_sweep.md`
to establish what had already been tried: widening/narrowing
`RESIDUAL_ACTION_SCALE`, a 5-seed sweep, reward reweighting toward speed,
curvature-aware reward shaping, and (separately) a 6-variant fine-tune-
against-the-expert dose sweep in two resume modes — all documented
failures at closing the pace gap specifically. Presented this history
plus the one flagged-but-untried lever (a mixed-opponent training
curriculum) and asked for direction rather than starting a fifth/seventh
variation unprompted.

**Evidence:**
- Sources or documentation: `docs/rl_design.md` §6 (causal test 37 and
  its five follow-ups), `experiments/2026-09-10_expert-opponent-
  seed8000/followup_finetune_sweep.md`, `docs/lab_notebook.md`'s
  2026-09-11 packaging entry.
- AI-agent assistance: Claude Code read the design doc and experiment
  notes (no code run), summarized the four-plus-six already-tried
  variants, and asked which direction to take via a direct question
  rather than assuming.
- Commits or code: none this session.
- Experiment output: none new; referenced existing results only.

**What we observed:** The pace-gap question had already been tested from
enough independent angles (scale, seed, reward weight, reward shape,
fine-tuning dose, resume mode) that continuing to search the same family
of levers was assessed as low-yield, per the existing notes' own
recommendation to conclude this line.

**Decision and rationale:** Stop pursuing the pace gap. `2026-09-11_
residual-expert-base-v2-seed8000`, as packaged in
`src/controllers/combined_candidate.py`, is adopted as the final result
of the combined-approach work — not because it matches Lucy's raw pace,
but because it already dominates plain SAC and the hybrid shield on
every tracked metric, and four-plus-six independent attempts to close
the remaining gap have failed cleanly enough to read as a structural
trade-off (the RL reward balances speed against safety by design; the
hand-written expert doesn't) rather than an unfound local optimum.
Recorded the closing note in `docs/rl_design.md` §6 (end of the causal
test 37 chain).

**Next steps:**
1. No further from-scratch or fine-tuning runs planned against this
   specific goal.
2. Still open, if priorities change later: the mixed-opponent curriculum
   (untried) and the seed=12000 safety-focused alternative remain
   documented options.
3. Still open, separately: whether to promote `combined_candidate` to
   the actual Gradescope submission slot in place of `race_faster.py`
   (flagged in the 2026-09-11 entry, not decided here).
4. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 17:05

**Participants and contributions:** Charlotte Tsui (direction: reopen the
pace-gap question closed earlier this session, specifically to raise the
effective speed ceiling and accept more risk on purpose), Claude Code
(implemented both levers as opt-in reward overrides, ran the training/eval
runs, verified results, documented).

**Question or objective:** Two specific, concrete levers for closing
`combined_candidate`'s remaining pace gap to Lucy's raw expert (8.94s
solo vs. v2's ~11.4-11.8s): raise the speed reward ceiling, and loosen a
safety weight to deliberately accept more risk.

**What we investigated or changed:**
- Confirmed `MAX_REWARDED_SPEED_MPS` (the literal "speed ceiling" from
  early in this track's history) no longer exists -- removed structurally
  in causal test 17 (2026-09-02) and replaced by
  `WALL_PROXIMITY_SPEED_SCALE_MPS`, a speed-scaled wall-proximity risk
  price. Chose this as the closest surviving analog to "raise the speed
  ceiling."
- Added two opt-in overrides (both default to the unchanged module
  constant, so plain self-play is unaffected): `wall_proximity_speed_scale_mps`
  and `damage_weight`, threaded through `training.reward.step_reward`,
  `training.controller.TrainableController`, and new
  `scripts/train_sac.py` flags (`--wall-proximity-speed-scale`,
  `--damage-weight`). Added corresponding unit tests in
  `tests/test_training_reward.py`. Ran the full suite (216 passed) and
  `ruff`/`pyright` (clean) before training anything.
- Added a `--baseline leaderboard-expert` option to `scripts/eval_sac.py`
  (previously only evaluated against the two standard baselines) so an
  already-trained checkpoint can be re-evaluated against
  `controllers.leaderboard_expert` as a live opponent without a bespoke
  script -- the same stress test every causal test 37 follow-up has used.
- Trained two checkpoints, each matched to
  `2026-09-11_residual-expert-base-v2-seed8000`'s exact reference config
  (seed=8000, races=40, round_seconds=120, residual mode) with exactly
  one variable changed: `wall_proximity_speed_scale_mps=20.0` (2x
  default) and `damage_weight=2.5` (half default). Evaluated both against
  the standard baselines (train_sac.py's built-in eval) and against
  `leaderboard_expert` directly (the new eval_sac.py option).

**Evidence:**
- Sources or documentation: `docs/rl_design.md` section 6 (causal test 17
  for why the old speed cap is gone; causal test 37 and its five prior
  follow-ups for what had already been tried).
- AI-agent assistance: Claude Code wrote and tested the two reward
  overrides and the eval_sac.py flag, ran both training runs
  (~497s/~575,840 transitions/~143,750 gradient updates each, matching
  v2's scale) and both post-training evaluations end to end, computed
  aggregate stats from the raw `eval_results.json` output via a small
  scratch script (not committed), and wrote this entry plus each
  experiment's `notes.md` and the `docs/rl_design.md` update from the
  actual numbers observed, not assumed.
- Commits or code: `src/training/reward.py`, `src/training/controller.py`,
  `scripts/train_sac.py`, `scripts/eval_sac.py`,
  `tests/test_training_reward.py`.
- Experiment output:
  `experiments/2026-09-12_residual-wallscale20-seed8000/`,
  `experiments/2026-09-12_residual-damageweight25-seed8000/` (each with
  `config.yaml`, `train.log`, `eval_results.json`, a `vs_expert/`
  subdirectory for the leaderboard_expert comparison, `checkpoints/`, and
  `notes.md`).

**What we observed:** Both variants trained cleanly and stably (normal
critic-loss curves, 20/20 wins against the standard baselines, no
elimination there) but neither closed the pace gap, and one regressed
safety. wall-proximity-speed-scale=20: avg best lap time got slower in
both matchups (11.37s -> 11.88s standard, 11.78s -> 13.37s vs. expert),
avg damage worse, 0/10 eliminated vs. expert unchanged. damage-weight=2.5:
also slower in both matchups (11.37s -> 11.66s standard, 11.78s -> 12.32s
vs. expert) and, unlike every combined-approach checkpoint tested since
the recovery-passthrough fix, reintroduced real eliminations against the
expert (2/10, vs. v2's 0/10).

**Decision and rationale:** Neither adopted. This is the fifth and sixth
consecutive structurally different lever (after residual scale x2, a
5-seed sweep, progress-weight reweight, and curvature-aware
center-offset) to fail at closing this specific gap -- six for six across
every category of lever this reward structure offers. Both new overrides
are kept as tested, documented, opt-in parameters (defaults unchanged)
rather than reverted code, per this track's convention of preserving
negative results. `2026-09-11_residual-expert-base-v2-seed8000` /
`src/controllers/combined_candidate.py` remains the final combined-approach
result. Recorded in `docs/rl_design.md` section 6 as new causal test 37
follow-ups, immediately after the entry documenting this line's earlier
closure.

**Next steps:**
1. No further training-side attempts at this specific goal are planned;
   six independent attempts is being treated as conclusive for this
   reward-tuning family.
2. Still open, if priorities change later: the mixed-opponent curriculum
   (still untried and structurally different from every reward-tuning
   variant attempted) and the seed=12000 safety-focused alternative.
3. Still open, separately: whether to promote `combined_candidate` to
   the actual Gradescope submission slot in place of `race_faster.py`.
4. Still open: the `controllers.minimum_viable` module gap.

## 2026-09-12 17:35

**Participants and contributions:** Charlotte Tsui (direction: try the
mixed-opponent curriculum, the one lever from the prior entry noted as
still untried), Claude Code (implemented it, found and fixed a bug in the
prior session's overrides along the way, ran the experiment, verified
results, documented).

**Question or objective:** Try the mixed-opponent training curriculum —
alternating self-play and expert-opponent races within one run — as a
structurally different mechanism (not another reward-weight tweak) for
closing `combined_candidate`'s remaining pace gap to Lucy's raw expert.

**What we investigated or changed:**
- While threading through the change, found that `scripts/train_sac.py`'s
  `--opponent self` incumbent construction was missing the
  `wall_proximity_speed_scale_mps`/`damage_weight` overrides added last
  session (only the challenger got them) — roughly half of the two prior
  experiments' (`2026-09-12_residual-wallscale20-seed8000`,
  `2026-09-12_residual-damageweight25-seed8000`) transitions were computed
  under the wrong (default) weights. Fixed the bug and added caveat notes
  to both experiments' `notes.md` and to `docs/rl_design.md` rather than
  silently correcting the record.
- Added `--opponent mixed` and `--mixed-opponent-expert-every` (default 2)
  to `scripts/train_sac.py`: instead of one `run_headless_head_to_head`
  call spanning all races, loops per-race, alternating the incumbent
  between a fresh self-play copy and `controllers.leaderboard_expert`
  directly, varying the per-race random seed so races aren't identical
  repeats. Also extended the automatic post-training baseline eval to
  include `leaderboard_expert` for `--opponent mixed`, mirroring how
  `--opponent expert` already does this.
- Smoke-tested on a tiny config (4 races, 8s rounds) before committing to
  a full run — verified the expert/self race split and the eval baseline
  inclusion worked as intended.
- Ran the full experiment matched to
  `2026-09-11_residual-expert-base-v2-seed8000`'s config otherwise
  (seed=8000, races=40, round_seconds=120, residual mode,
  mixed_opponent_expert_every=2), evaluated against both the standard
  baselines and `leaderboard_expert` directly, and checked `metrics.csv`
  for training stability.
- Ran the full test suite (216 passed) and `ruff`/`pyright` (clean) before
  and after the bug fix and the new feature.

**Evidence:**
- Sources or documentation: `experiments/2026-09-10_expert-opponent-
  seed8000/followup_finetune_sweep.md`'s next-step note (where this idea
  originated), `docs/rl_design.md` section 6's causal test 37 chain for
  the six prior failed levers.
- AI-agent assistance: Claude Code wrote and smoke-tested the new
  `--opponent mixed` mechanism, found the pre-existing override-threading
  bug via direct code inspection (not from a symptom), ran the full
  training/eval run, inspected `metrics.csv` for critic-loss stability,
  and wrote this entry, the `docs/rl_design.md` update, and
  `experiments/2026-09-12_residual-mixedopponent-seed8000/notes.md` from
  the actual observed numbers.
- Commits or code: `scripts/train_sac.py`.
- Experiment output:
  `experiments/2026-09-12_residual-mixedopponent-seed8000/` (`config.yaml`,
  `train.log`, `eval_results.json`, `checkpoints/`, `notes.md`).

**What we observed:** Trained faster in wall-clock terms (344.8s vs.
v2's 497.1s) and collected fewer transitions (415,906 vs. 575,840) — as
expected, since expert-opponent races only have one learning copy. But
this was the worst combined-approach result observed on this track: avg
best lap time got slower in both matchups (11.37s -> 12.60s standard,
11.78s -> 12.69s vs. expert), and — for the first time in any residual
checkpoint since the recovery-passthrough fix — eliminations reappeared
on the standard baselines (1/20) as well as against the expert (2/10).
Off-track/wall-contact time exploded even against the easy baselines,
indicating a broad competence regression, not a targeted expert-matchup
effect. `metrics.csv` showed real training instability (critic loss
peaked at 290.8, well above this track's normal range for a converged
residual run).

**Decision and rationale:** Not adopted. This is the seventh
structurally different lever — and the first full mechanism change rather
than a reward-weight tweak — to fail at closing the pace gap, producing a
worse result than any of the six reward-tuning attempts before it.
`--opponent mixed` is kept as a tested, documented, opt-in flag (default
`self`, existing behavior unaffected). `2026-09-11_residual-expert-base-
v2-seed8000` / `src/controllers/combined_candidate.py` remains the final
combined-approach result. Recorded in `docs/rl_design.md` section 6 as a
new causal test 37 follow-up, with the line re-closed after it.

**Next steps:**
1. No further attempts at closing this specific pace gap are planned —
   seven independent, structurally different attempts (six reward-tuning
   levers plus this mechanism change) is being treated as conclusive.
2. Still open, if priorities change later: a lower expert-exposure ratio
   for the mixed curriculum (untried, lower-conviction after this result)
   and the seed=12000 safety-focused alternative.
3. Still open, separately: whether to promote `combined_candidate` to
   the actual Gradescope submission slot in place of `race_faster.py`.
4. Still open: the `controllers.minimum_viable` module gap.
