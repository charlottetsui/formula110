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
