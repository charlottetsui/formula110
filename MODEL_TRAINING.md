# Model Training

This is the working guide and experiment log for the combined imitation → SAC
approach. Lucy's expert supplies demonstrations; Charlotte's SAC implementation
supplies the actor architecture and, later, reinforcement learning updates.
The original `docs/rl_design.md` and `docs/lab_notebook.md` describe Charlotte's
independent SAC experiments, including historical assumptions. This file tracks
the combined work from September 8, 2026 onward.

## Current status

**We have a small, runnable imitation model that matches the expert's distance
in the first local trials. It still needs broader reliability testing before
SAC fine-tuning or a leaderboard submission.**

- Controller: [`controllers.imitation`](src/controllers/imitation.py).
- Frozen artifact: [`imitation_policy.npz`](src/controllers/imitation_policy.npz),
  223,316 bytes (about 218 KiB).
- Training/evaluation CLI: [`scripts/train_imitation.py`](scripts/train_imitation.py).
- Evidence: [`experiments/2026-09-08_imitation-v1`](experiments/2026-09-08_imitation-v1).
- No expert controller runs inside the deployed model. Only the learned network,
  sensor history, and exact brake-release wrapper select its commands.

Watch it drive:

```bash
uv sync --managed-python
uv run racing --seed 110 --student-module controllers.imitation
```

## Steps to train and evaluate

### 1. Freeze the teacher and collect demonstrations

```bash
uv run python scripts/train_imitation.py collect \
  --output artifacts/imitation-v1-data \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 2001 2002 \
  --seconds 30
```

The collector runs expert versus expert using the public headless race API,
with a fresh teacher and history for each car. There is one 30-second race per
seed, two cars, 60 Hz physics, and no marshal resets. The seed changes the
starting placement on the existing track; it does not generate a new layout.

The output directory contains:

- `trajectories.jsonl.gz`: full public sensor snapshots before each action,
  previous applied action, expert/applied action, seed, tick, and episode ID.
  No-hit infinities use the simulator recording serializer's JSON convention.
- `dataset.npz`: finite model inputs, action labels, episode IDs and seed mapping.
- `race_results.json`: public race results for the demonstrations.
- `metadata.json`: teacher hash, settings and sample counts. New runs also
  record dependency versions, source hashes, git revision and collection time.

Every neutral brake-release tick is retained. Expert labels and applied actions
are identical in this expert-only dataset. Requested commands before the
expert's internal wrapper are not recorded; future intervention collection
will need a clearly defined teacher-state and requested/applied-action contract.

Datasets stay in ignored `artifacts/`; retain or transfer them separately when
sharing a trained experiment. Existing output directories are never overwritten.
Use a new name when repeating a run.

### 2. Fit the behavioral clone

```bash
uv run python scripts/train_imitation.py fit \
  --dataset artifacts/imitation-v1-data/dataset.npz \
  --output experiments/2026-09-08_imitation-v1 \
  --validation-seeds 2001 2002 \
  --epochs 80 --seed 0
```

Whole seed groups are held out: both cars and all ticks for seeds 2001 and 2002
are validation data. No random split of neighboring ticks is used. The final
five-seed evaluation suite is absent from collection and model selection.

The model uses Charlotte's `GaussianPolicy` trunk and mean head, trained by
supervised learning. The loss is throttle MSE + 4 × steering MSE. Samples are
balanced across braking, partial/neutral throttle, and full acceleration so
straightaway acceleration does not dominate training. Adam uses learning rate
0.0003, batch size 512, one CPU thread and a recorded training seed.

The checkpoint with minimum validation loss is saved as:

- `policy.npz`: NumPy inference weights, with observation version metadata.
- `actor.pt`: PyTorch actor weights and architecture metadata for future work.
- `metrics.csv`: training and validation losses for every epoch.
- `config.json`: dataset hash, train/validation seeds, settings and best loss.
  New runs also record source/dependency provenance and elapsed fitting time.

Checkpoint selection currently uses prediction loss, not autonomous driving.
Later iterations should compare checkpoints on a separate driving-validation
suite before evaluating the final held-out test suite.

### 3. Measure autonomous driving against the teacher

```bash
uv run python scripts/train_imitation.py evaluate \
  --model experiments/2026-09-08_imitation-v1/policy.npz \
  --output experiments/2026-09-08_imitation-v1/eval_results.json \
  --seeds 110 42 7 2024 8675309 --seconds 30
```

For each seed, run two races of expert versus expert and two of clone versus
expert. Compare challenger statistics across those matched runs. The expert is
loaded through the simulator's factory-aware public loader so state resets
between cars and races. Opponent behavior can change in response to the clone;
these are racing measurements, not identical solo trajectories. Marshals are
disabled to expose failures to recover.

Record distance, damage, contact, laps, survival and low-progress time. Passing
unit tests or achieving low action error does not establish driving quality.

### 4. Export a reviewed candidate

The first candidate has already been copied to
`src/controllers/imitation_policy.npz`. To install a subsequent evaluated model:

```bash
cp experiments/NEW_RUN/policy.npz src/controllers/imitation_policy.npz
uv run python scripts/export_student_controllers.py --all-controllers
```

The full-controller export is necessary to include the binary artifact. This
creates a local archive; it does not submit anything to a leaderboard.

## Model inputs, memory and actuator behavior

Each frame has 42 values:

- Speed divided by 50 m/s without saturating at the teacher's racing speeds;
  heading error, center offset, near/middle/far lookahead and yaw rate.
- Camera visibility, timestep, wall/robot contact flags and damage.
- Seven wall-only and seven full LiDAR distances, logarithmically scaled with
  an 80 m cap, plus a hit indicator for each beam.
- The previously applied throttle and steer.

Eight frames at lags `(0, 1, 2, 4, 8, 16, 24, 32)` produce **336 inputs**.
History is causal, bounded to 33 frames, reset for every new car/race, and padded
with the first observed frame at startup. At 60 Hz it spans about 0.53 seconds.
This is a practical feedforward alternative to the recurrent model originally
proposed in `GEOMETRY_EXPERT.md`; it does not guarantee that all teacher recovery
state can be inferred, particularly outside the collected scenarios.

Two 128-unit ReLU layers produce two tanh-bounded controls: **59,906 deployed
parameters**. The network runs deterministically in NumPy. It imports neither
PyTorch nor the teacher, and never trains during inference. The same feature
encoder is used during collection and deployment.

After the model requests positive throttle following a negative applied command,
the wrapper emits exactly zero throttle for one tick while preserving steering.
The next frame records that actual neutral command. The model predicts applied
expert commands during training; it is not guaranteed to predict neutral labels
exactly, which is why the explicit wrapper remains necessary.

## First experiment — 2026-09-08

**Objective:** Build and run the first complete expert-to-clone pipeline.

**Work:** Codex inspected both branches' combined code and notes, implemented
collection, causal observations, supervised training, model export, deterministic
inference and matched evaluation. Dependencies already declared on the combined
branch were installed; dependency declarations were not changed.

**Dataset and training:** 43,224 samples from 24 car episodes. Of these, 36,020
were training samples and 7,204 were validation samples. One model initialization
(seed 0), 80 epochs. Best weighted validation MSE: **0.045694**. No reward or SAC
updates were used, so an RL proxy reward is not applicable to this experiment.

**Matched head-to-head results:** challenger distance summed across two
30-second races per seed, against an expert opponent:

| Starting seed | Expert baseline | Clone | Clone wall contact | Maximum clone damage |
| --- | ---: | ---: | ---: | ---: |
| 110 | 1,290.21 m | 1,301.27 m | 0.133 s | 8.68% |
| 42 | 1,311.01 m | 1,329.87 m | 0.567 s | 13.28% |
| 7 | 1,368.81 m | 1,383.49 m | 0 s | 0% |
| 2024 | 1,348.60 m | 1,367.89 m | 0 s | 0% |
| 8675309 | 1,329.19 m | 1,351.01 m | 0 s | 0% |

Wall contact is summed over the two races; damage is the maximum final damage.
The expert baseline had zero damage and wall contact on this suite. The clone's
distance is slightly higher, but it has not matched the teacher's contact-free
reliability. Damage can also come from cars; the table does not attribute it
solely to walls. See the full `eval_results.json` for both sides' statistics.

**Solo trials through the autograder's local subprocess worker:** 30 seconds,
60 Hz, with no marshal recovery. These reproduce the teacher's documented
baseline and compare the clone in the same runner:

| Seed | Expert distance | Clone distance | Completed laps, both | Damage/contact, both |
| --- | ---: | ---: | ---: | --- |
| 110 | 656.613 m | 663.794 m | 3 | 0% / 0 s |
| 2026 | 680.625 m | 682.878 m | 3 | 0% / 0 s |

On seed 110, the best completed lap was 8.000 s for the expert and 7.900 s for
the clone. See `solo_results.json`. These local subprocess runs do not reproduce
the Linux privilege sandbox or its memory enforcement and are not leaderboard
submissions.

To reproduce a local solo trial:

```bash
FORMULA110_LOCAL_CONTROL=1 \
FORMULA110_CONTROL_WORKER=autograder/gradescope/control_worker.py \
uv run python autograder/gradescope/race_worker.py \
  --submission src --module-file src/controllers/imitation.py \
  --seed 110 --seconds 30
```

**Runtime check:** A fresh macOS process performing 10,000 synthetic sensor
calls reported 64,241,664 bytes peak RSS (61.3 MiB), about 22 microseconds per
call, and no PyTorch import. This is a local diagnostic, not a Linux memory-limit
certification or a timing benchmark on leaderboard hardware.

**Verification:** 154 tests passed, including six new tests covering observation
range, bounded/reset history, exact neutral release, PyTorch/NumPy export parity,
collector action alignment and whole-seed validation separation. Pyright passed
using the project interpreter. Changed Python files pass Ruff. Repository-wide
Ruff still reports existing issues in unrelated simulator/rendering files.

**Corrections during the experiment:** The initial evaluation used bare expert
objects, which the runner could reuse across races. This was corrected to the
public factory-aware loader and the suite was rerun without retraining. Only
`eval_results.json` is used above; `eval_initial_state_reuse.json` preserves the
superseded evidence. The first local solo-worker command failed because its
default worker path points into the Linux deployment; setting the documented
worker-path environment variable resolved it. No simulator internals were
modified. Initial Pyright invocation selected the wrong interpreter; the final
check explicitly uses `.venv/bin/python`.

**Provenance limitation:** The first run predates automatic timing/source
provenance fields added to the CLI later in this session. Its dataset hash,
teacher hash, losses, seed split, settings and model artifacts are retained.
`source_manifest.json` explicitly identifies its hashes as a post-run snapshot;
it must not be interpreted as the exact training-start source state.

## Next milestones

1. Diagnose the two contact/damage cases with watched races; collect useful
   corrections with teacher state aligned to the actions actually applied.
   DAgger/intervention collection is not implemented yet.
2. Repeat training with at least three independent model seeds. Evaluate longer
   races, different opponents and deliberately difficult recovery scenarios.
   Treat the seeds inspected here as a regression suite; reserve new seeds for
   final testing when tuning further. Entirely different tracks remain untested.
3. Gate SAC refinement on comparable distance and acceptable damage/contact
   across that broader suite, rather than this one successful initialization.
4. Adapt SAC to this 336-input representation and the same actuator wrapper.
   Seed replay with demonstrations and control exploration during fine-tuning.
   The saved actor includes an **untrained log-standard-deviation head**; its
   noise must be initialized deliberately. Existing 17-input SAC checkpoints are
   incompatible. Critic initialization, terminal handling and reward tuning
   remain separate required work; loading `actor.pt` alone is not sufficient.

Append new experiment entries here with the question, exact commands/settings,
evidence directory, observed results, failed attempts and next decision. Keep
large raw datasets outside the controller package and never replace the current
candidate solely because training loss or reward improved.
