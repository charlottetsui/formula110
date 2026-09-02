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
