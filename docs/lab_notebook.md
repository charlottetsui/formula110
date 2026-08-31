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
