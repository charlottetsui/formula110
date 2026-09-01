# CLAUDE.md — SAC track (Formula 110, COMP 590H)

This file governs work on the **SAC exploration/refinement track** of this
project. It does not cover the second (PPO) approach — that is a separate,
independently developed track and is out of scope here. Do not add
partner/PPO-coordination content to any file this document points at.

Read this file, then `docs/rl_design.md` and the most recent entries in
`docs/lab_notebook.md`, at the start of every session before writing code
or docs. They are the durable memory of this track; do not re-derive
decisions already recorded there.

## Where things live

| Path | Purpose |
| --- | --- |
| `docs/rl_design.md` | The technical design: MDP formulation, self-play training architecture, dependency plan, minimum experiment, refinement plan. Update it when the architecture, observation space, or reward actually changes — don't let it drift from the code. |
| `docs/lab_notebook.md` | Chronological log, one entry per substantive session. Required by the course; see template below. |
| `experiments/` | Per-run evidence (`config.yaml`, `metrics.csv`, `eval_results.json`, checkpoints, notes). Convention in `experiments/README.md`. |
| `src/training/` | SAC implementation (replay buffer, actor/critic networks, training loop) — not yet created as of 2026-08-31. |
| `src/controllers/` | Where the trained, frozen-weights controller gets packaged for racing/leaderboard submission, per `README.md`'s packaging contract. |

## What can and cannot be edited

This repo is a course-provided simulator plus a student-owned track.
Verified 2026-09-01: `git diff 2e2aee4 HEAD -- src/racing/ autograder/`
is empty, and none of Charlotte's SAC-track commits (`2b0b5c7`, `2c94b42`,
`92a0bc7`) touch `src/racing/`, `autograder/`, or any pre-existing
`tests/`/`scripts/` file — only new files were added alongside them.

**Do not edit — course-provided reference/infrastructure:**

| Path | Why |
| --- | --- |
| `src/racing/` | The simulator engine (physics, graphics, race rules, track, sensors, student API contract). The self-play design exists specifically to avoid needing to touch this — see step 9 below. |
| `autograder/` | Gradescope grading infrastructure, built from a trusted read-only bundle of the simulator. |
| `README.md`, `GETTING_STARTED.md`, `SENSORS.md`, `LICENSE` | Course-authored reference docs describing the public contract. |
| `tests/` — every file except `tests/test_training_*.py` | Simulator/autograder contract tests owned by the course. |
| `scripts/` — every file except `scripts/train_sac.py`, `scripts/eval_sac.py` (or later SAC-track scripts) | Course-provided tooling (asset capture, gamepad diagnostics, Gradescope packaging). |
| `src/controllers/crash_fast.py` | The course-provided starter controller, kept as a reference example. |

**Freely editable — this track's own work:**

| Path | Notes |
| --- | --- |
| `docs/rl_design.md`, `docs/lab_notebook.md` | This track's design doc and log. |
| `experiments/` | Per-run evidence. |
| `src/training/` | SAC implementation. |
| `src/controllers/` (any file other than `crash_fast.py`, e.g. `sac_candidate.py`) | Trained controllers packaged for racing. |
| `scripts/train_sac.py`, `scripts/eval_sac.py` (and later SAC-track scripts) | This track's training/eval entry points. |
| `tests/test_training_*.py` | Tests for `src/training/`, added on this track. |
| `CLAUDE.md` | This file. |
| `pyproject.toml`, `uv.lock` | Only via `uv add` / `uv sync --managed-python` per step 8 below — never hand-edited directly. |

If a task seems to require changing something in the "do not edit" list,
stop and flag it rather than editing — that usually means the task is
out of scope for this track (per step 9) rather than something to route
around silently.

## Packaging a Gradescope submission

Discovered 2026-09-01 via a real failed upload ("expected
formula110-submission.json at the root of the submission"): the **live**
Gradescope autograder for this assignment expects a submission contract
that this repo's checked-in `autograder/` bundle and
`scripts/export_student_controllers.py` do not know about or document —
confirmed by grep, there is no reference to a submission manifest
anywhere in `autograder/`, `README.md`, or `autograder/README.md` as
currently checked in. Treat the live Gradescope side as authoritative
over the local docs here whenever they conflict, per step 1's general
rule about trusting observed reality over stale plans.

Every time a submission zip is built for upload, do this in addition to
`scripts/export_student_controllers.py` (which only produces the
`controllers/` tree and has no flag for the following — it's
course-provided infra, do not edit it to add one; append these files to
its output zip after running it instead):

1. Run the export as usual, e.g.
   `uv run python scripts/export_student_controllers.py --all-controllers`
   (`--all-controllers` is required whenever the controller module bundles
   non-Python files, such as a checkpoint).
2. Add three files to the **root** of that zip (not under `controllers/`):
   - `formula110-submission.json`:
     ```json
     {
       "schema_version": 1,
       "controller_module": "controllers.<the module being submitted>"
     }
     ```
     Set `controller_module` to whichever module this submission is
     grading as the controller (e.g. `controllers.race_faster`).
   - An unmodified copy of `pyproject.toml` from the repo root.
   - An unmodified copy of `uv.lock` from the repo root.
3. Verify the zip's contents (`unzip -l`) before calling the submission
   ready — confirm all three root files are present alongside
   `controllers/`.

## Explicit steps to follow every session

1. **Orient before acting.** Read `docs/rl_design.md` and the last 1–2
   `docs/lab_notebook.md` entries. If the repo state (files, deps) doesn't
   match what those docs describe, trust the repo and flag the mismatch —
   don't silently build on a stale plan.
2. **Check git status** before anything that could discard uncommitted
   work (`git checkout`/`restore`/`reset`/`clean`, etc.), per standard
   safety practice.
3. **Make the change** (design doc edit, training code, experiment run,
   whatever the session's task is).
4. **Document the change in `docs/lab_notebook.md` before ending the
   session** — not batched later. Append a new entry using this exact
   template:

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

   - Convert relative dates/times to absolute ones.
   - Under "AI-agent assistance," record what Claude Code actually did and
     how it was verified (read the real source, ran a command, ran the
     tests) — not just "used AI." Record significant AI suggestions that
     were rejected or changed, and why.
   - Failed experiments get an entry too — document the failure and the
     analysis, don't discard it.
5. **If the architecture/observations/reward changed**, update
   `docs/rl_design.md` in the same session, and say so explicitly in the
   lab notebook entry's "Decision and rationale."
6. **If an experiment ran**, save its evidence under
   `experiments/<YYYY-MM-DD>_<slug>/` per `experiments/README.md`, and
   reference that directory from the lab notebook entry's "Evidence"
   section. Report both the training proxy reward *and* the simulator's
   own public race stats (`HeadToHeadResult`) — see `docs/rl_design.md`
   §2.3 for why both matter.
7. **Use the fixed evaluation seed set** (`110, 42, 7, 2024, 8675309`,
   defined in `docs/rl_design.md` §5) for any result meant to be compared
   across runs. A single seed is not evidence of robustness.
8. **Dependency changes:** `uv add <package>` then
   `uv sync --managed-python`; commit `pyproject.toml` and `uv.lock`
   together. Prefer CPU-capable packages (see `README.md`'s CPU/memory
   boundary — 512 MiB, no CUDA/MPS/ROCm at inference time).
9. **Stay on the public simulator surface.** Do not import or depend on
   private/underscore-prefixed simulator internals, the physics world, or
   official lap-progress data — the whole self-play design in
   `docs/rl_design.md` §3 exists specifically to avoid needing those.
10. **Only commit or push when explicitly asked.** Documentation and code
    changes within a session don't imply a commit request.

## Scope guardrails

- This track's docs (`docs/rl_design.md`, `docs/lab_notebook.md`) describe
  Charlotte's own SAC implementation and experiments only. Do not add
  sections managing or documenting the partner's PPO work, shared-infra
  write-ups, or "sync with teammate" steps.
- The two-approach comparison the course requires is satisfied at the team
  level via a joint write-up outside this track — not by duplicating both
  approaches here.
