#!/usr/bin/env python3
"""Run a self-play SAC training run and evaluate it against baseline controllers.

Implements the minimum experiment from docs/rl_design.md section 5: train
via self-play (`run_headless_head_to_head` with both sides pointing at the
same in-training policy, per section 3), then evaluate the frozen policy
against every baseline in `training.evaluation.BASELINE_CONTROLLERS`
across a fixed multi-seed evaluation set. All evidence is written under
`experiments/<slug>/` per `experiments/README.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from controllers.leaderboard_expert import create_controller as create_expert_controller
from racing import RobotController
from racing.race.head_to_head import run_headless_head_to_head
from training.controller import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_STEP,
    DEFAULT_RESIDUAL_BASE_SOURCE,
    DEFAULT_UPDATE_EVERY_N_STEPS,
    DEFAULT_WARMUP_STEPS,
    RESIDUAL_ACTION_SCALE,
    RESIDUAL_BASE_SOURCES,
    TrainableController,
    TrainingState,
)
from training.evaluation import BASELINE_CONTROLLERS, evaluate_against_baselines, role_distance_m
from training.observation import OBSERVATION_DIM
from training.replay_buffer import ReplayBuffer
from training.reward import WALL_PROXIMITY_SPEED_SCALE_MPS, WEIGHT_DAMAGE, WEIGHT_PROGRESS
from training.sac import SACAgent
from training.trajectory import BestTrajectoryTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (110, 42, 7, 2024, 8675309)
ACTION_DIM = 2


@dataclass(frozen=True)
class TrainSacArguments:
    races: int
    round_seconds: float
    copies_per_side: int
    opponent: str
    mixed_opponent_expert_every: int
    seed: int
    buffer_capacity: int
    warmup_steps: int
    update_every_n_steps: int
    batch_size: int
    n_step: int
    hidden_size: int
    trajectory_bonus: bool
    expert_match_bonus: bool
    residual_expert_base: bool
    residual_action_scale: float
    residual_base_source: str
    residual_hazard_gated: bool
    progress_weight: float
    curvature_aware_center_offset: bool
    wall_proximity_speed_scale_mps: float
    damage_weight: float
    resume_from: Path | None
    resume_policy_only: bool
    eval_seeds: tuple[int, ...]
    eval_races: int
    eval_round_seconds: float
    experiment_dir: Path


def parse_args() -> TrainSacArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=int, default=6, help="self-play races to train over")
    parser.add_argument("--round-seconds", type=float, default=15.0, help="seconds per self-play race")
    parser.add_argument("--copies-per-side", type=int, default=1, help="cars per side during self-play")
    parser.add_argument(
        "--opponent",
        choices=("self", "expert", "mixed"),
        default="self",
        help=(
            "'self' (default): incumbent is another in-training TrainableController sharing this "
            "run's policy/buffer, per docs/rl_design.md section 3's self-play design. 'expert': "
            "incumbent is a frozen controllers.leaderboard_expert.Controller instance instead -- only "
            "the challenger learns/pushes transitions, so it practices against a genuinely different "
            "driving style rather than a mirror of itself. 'mixed': alternates race-by-race between "
            "the 'self' and 'expert' incumbents (training.reward.py's own docs/rl_design.md section 6 "
            "next-step note: a mixed-opponent curriculum, untested until now -- 'expert' alone shifts "
            "the whole training distribution at once and caused severe instability at every dose tried "
            "in the fine-tuning follow-up to causal test 35; alternating never fully leaves the "
            "self-play distribution). See --mixed-opponent-expert-every for the ratio. All three "
            "require --copies-per-side 1 whenever the incumbent can be the expert, since "
            "leaderboard_expert.Controller has no copy_for_car and its per-car state (recovery timer, "
            "previous steer) would otherwise be shared and corrupted across multiple incumbent copies."
        ),
    )
    parser.add_argument(
        "--mixed-opponent-expert-every",
        type=int,
        default=2,
        help=(
            "with --opponent mixed, race N (1-indexed) uses the expert as incumbent when N is a "
            "multiple of this value, self otherwise -- default 2 means every other race (50%% "
            "expert exposure). Ignored for --opponent self/expert."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=110,
        help=(
            "random seed for self-play race spawns, the SACAgent's network initialization, and the "
            "training RNG (warmup actions, replay-buffer sampling order) -- controls every source of "
            "run-to-run stochasticity except torch's own internal nondeterminism"
        ),
    )
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--update-every-n-steps", type=int, default=DEFAULT_UPDATE_EVERY_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--n-step",
        type=int,
        default=DEFAULT_N_STEP,
        help=(
            "ticks of real reward summed per transition before bootstrapping (default 1, "
            "i.e. standard 1-step TD); see docs/rl_design.md section 4 for the credit-"
            "assignment rationale"
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--trajectory-bonus",
        action="store_true",
        help="reward relative to the best-known distance-at-tick seen so far this run (training.trajectory)",
    )
    parser.add_argument(
        "--expert-match-bonus",
        action="store_true",
        help=(
            "reward actions close to what controllers.leaderboard_expert.Controller would have chosen, "
            "restricted to wall/robot-proximity hazard states (training.reward.WEIGHT_EXPERT_MATCH) -- "
            "runs a private, non-controlling shadow expert instance per training car; self-play's "
            "opponent is unaffected"
        ),
    )
    parser.add_argument(
        "--residual-expert-base",
        action="store_true",
        help=(
            "residual reinforcement learning: controllers.leaderboard_expert.Controller's command "
            "becomes the base action every tick, and the policy only learns a bounded correction on "
            "top of it (training.controller.RESIDUAL_ACTION_SCALE) -- pushed to the replay buffer as "
            "the policy's actual action space. Mutually exclusive with --expert-match-bonus. Remember "
            "to pass the same flag to scripts/eval_sac.py or training.evaluation.evaluate_against_"
            "baselines when re-evaluating a checkpoint trained this way -- otherwise its correction-"
            "only output gets misread as an absolute command."
        ),
    )
    parser.add_argument(
        "--residual-action-scale",
        type=float,
        default=RESIDUAL_ACTION_SCALE,
        help=(
            "with --residual-expert-base, how much the policy's correction can shift the expert's "
            "command in either action dimension before clamping to [-1, 1] -- larger values give the "
            "policy more room to deviate from the expert's own driving line/pace, smaller values keep "
            "it closer to a pure safety nudge. Must match the checkpoint's value when re-evaluating."
        ),
    )
    parser.add_argument(
        "--residual-base-source",
        choices=RESIDUAL_BASE_SOURCES,
        default=DEFAULT_RESIDUAL_BASE_SOURCE,
        help=(
            "with --residual-expert-base, which controller supplies the base command: 'expert' "
            "(default, controllers.leaderboard_expert -- exact, hand-written) or 'clone' "
            "(controllers.imitation -- the behavioral clone trained on the expert's trajectories, "
            "an approximation with its own clone error). Must match the checkpoint's value when "
            "re-evaluating."
        ),
    )
    parser.add_argument(
        "--residual-hazard-gated",
        action="store_true",
        help=(
            "with --residual-expert-base, apply the correction only on ticks training.reward"
            ".in_hazard judges a wall/robot-proximity hazard -- every other tick, the base "
            "command passes through completely unmodified. Built for --residual-base-source "
            "clone, where a uniform correction was found to cost pace everywhere for a fix "
            "only needed in rare proximity moments. Must match the checkpoint's value when "
            "re-evaluating."
        ),
    )
    parser.add_argument(
        "--progress-weight",
        type=float,
        default=WEIGHT_PROGRESS,
        help=(
            "overrides training.reward.WEIGHT_PROGRESS for this run's reward calculation -- the "
            "default was tuned entirely for plain self-play (a from-scratch policy supplying 100%% "
            "of its own collision avoidance); with --residual-expert-base, the base action already "
            "comes from a competent expert, so weighting speed more heavily relative to the fixed "
            "caution terms may be worth more here than it was for plain SAC. Only affects training, "
            "not evaluation (the reward is never computed for a frozen/eval controller)."
        ),
    )
    parser.add_argument(
        "--curvature-aware-center-offset",
        action="store_true",
        help=(
            "scale training.reward's WEIGHT_CENTER_OFFSET penalty by upcoming bend sharpness "
            "(training.reward.CENTER_OFFSET_CURVATURE_REFERENCE_M/MIN_CENTER_OFFSET_SCALE) instead of "
            "applying it uniformly -- full strength approaching a real corner, reduced on straights, "
            "mirroring how controllers.leaderboard_expert itself treats corners vs. straights "
            "differently. Only affects training, not evaluation."
        ),
    )
    parser.add_argument(
        "--wall-proximity-speed-scale",
        type=float,
        default=WALL_PROXIMITY_SPEED_SCALE_MPS,
        dest="wall_proximity_speed_scale_mps",
        help=(
            "overrides training.reward.WALL_PROXIMITY_SPEED_SCALE_MPS for this run's reward "
            "calculation -- raising it makes the wall-proximity penalty grow more slowly with "
            "speed (tolerates more speed before pricing it as risky), the closest analog left "
            "to a speed ceiling since MAX_REWARDED_SPEED_MPS was removed structurally. Only "
            "affects training, not evaluation."
        ),
    )
    parser.add_argument(
        "--damage-weight",
        type=float,
        default=WEIGHT_DAMAGE,
        help=(
            "overrides training.reward.WEIGHT_DAMAGE for this run's reward calculation -- lower "
            "values make taking damage cost less, a direct 'accept more risk on purpose' lever. "
            "Only affects training, not evaluation."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "fine-tune from an existing checkpoint's saved policy/critics/log_alpha "
            "(a SACAgent.save() file with include_training_state=True) instead of a fresh "
            "random init -- --hidden-size must match the checkpoint's architecture. The "
            "replay buffer and critic optimizer momentum are NOT resumed (start empty/fresh), "
            "and --warmup-steps still defaults to random actions before using the policy -- "
            "pass --warmup-steps 0 to use the resumed policy's actions from the first tick"
        ),
    )
    parser.add_argument(
        "--resume-policy-only",
        action="store_true",
        help=(
            "with --resume-from, load only the actor's weights (SACAgent.load_policy_only) instead "
            "of the full training state -- critics and the entropy temperature start fresh rather "
            "than inheriting the checkpoint's (typically near-zero) exploration budget and "
            "possibly-stale Q-estimates. Intended for resuming into a training distribution the "
            "checkpoint wasn't trained under, e.g. --opponent expert."
        ),
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-races", type=int, default=2, help="head-to-head races per evaluation seed")
    parser.add_argument("--eval-round-seconds", type=float, default=20.0)
    parser.add_argument("--experiment-dir", type=Path, required=True, help="output directory under experiments/")
    arguments = parser.parse_args()
    return TrainSacArguments(
        races=arguments.races,
        round_seconds=arguments.round_seconds,
        copies_per_side=arguments.copies_per_side,
        opponent=arguments.opponent,
        mixed_opponent_expert_every=arguments.mixed_opponent_expert_every,
        seed=arguments.seed,
        buffer_capacity=arguments.buffer_capacity,
        warmup_steps=arguments.warmup_steps,
        update_every_n_steps=arguments.update_every_n_steps,
        batch_size=arguments.batch_size,
        n_step=arguments.n_step,
        hidden_size=arguments.hidden_size,
        trajectory_bonus=arguments.trajectory_bonus,
        expert_match_bonus=arguments.expert_match_bonus,
        residual_expert_base=arguments.residual_expert_base,
        residual_action_scale=arguments.residual_action_scale,
        residual_base_source=arguments.residual_base_source,
        residual_hazard_gated=arguments.residual_hazard_gated,
        progress_weight=arguments.progress_weight,
        curvature_aware_center_offset=arguments.curvature_aware_center_offset,
        wall_proximity_speed_scale_mps=arguments.wall_proximity_speed_scale_mps,
        damage_weight=arguments.damage_weight,
        resume_from=arguments.resume_from,
        resume_policy_only=arguments.resume_policy_only,
        eval_seeds=tuple(arguments.eval_seeds),
        eval_races=arguments.eval_races,
        eval_round_seconds=arguments.eval_round_seconds,
        experiment_dir=arguments.experiment_dir,
    )


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def train(args: TrainSacArguments) -> tuple[SACAgent, TrainingState, float]:
    agent = SACAgent(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(args.hidden_size, args.hidden_size),
        seed=args.seed,
    )
    if args.resume_from is not None:
        if args.resume_policy_only:
            agent.load_policy_only(args.resume_from)
            print(f"[train] resumed policy only (fresh critics/log_alpha) from {args.resume_from}")
        else:
            agent.load(args.resume_from)
            print(f"[train] resumed policy/critics/log_alpha from {args.resume_from}")
    buffer = ReplayBuffer(capacity=args.buffer_capacity, observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)
    trajectory = BestTrajectoryTracker(max_ticks=int(args.round_seconds * 60) + 1) if args.trajectory_bonus else None
    state = TrainingState(
        agent=agent,
        buffer=buffer,
        rng=np.random.default_rng(args.seed),
        warmup_steps=args.warmup_steps,
        update_every_n_steps=args.update_every_n_steps,
        batch_size=args.batch_size,
        n_step=args.n_step,
        trajectory=trajectory,
    )
    if args.opponent in ("expert", "mixed") and args.copies_per_side != 1:
        raise ValueError("--opponent expert/mixed requires --copies-per-side 1 (see --opponent's help)")
    if args.resume_policy_only and args.resume_from is None:
        raise ValueError("--resume-policy-only requires --resume-from")
    if args.expert_match_bonus and args.residual_expert_base:
        raise ValueError("--expert-match-bonus and --residual-expert-base are mutually exclusive")

    def make_self_play_controller() -> TrainableController:
        return TrainableController(
            state=state,
            training=True,
            expert_match=args.expert_match_bonus,
            residual_base=args.residual_expert_base,
            residual_scale=args.residual_action_scale,
            residual_base_source=args.residual_base_source,
            residual_hazard_gated=args.residual_hazard_gated,
            progress_weight=args.progress_weight,
            curvature_aware_center_offset=args.curvature_aware_center_offset,
            wall_proximity_speed_scale_mps=args.wall_proximity_speed_scale_mps,
            damage_weight=args.damage_weight,
        )

    challenger = make_self_play_controller()

    started_at = time.monotonic()
    if args.opponent == "mixed":
        total_challenger_m = 0.0
        total_incumbent_m = 0.0
        expert_races = 0
        for race_index in range(1, args.races + 1):
            use_expert = race_index % args.mixed_opponent_expert_every == 0
            block_incumbent: RobotController = create_expert_controller() if use_expert else make_self_play_controller()
            block_result = run_headless_head_to_head(
                challenger_controller=challenger,
                incumbent_controller=block_incumbent,
                challenger_name="sac-in-training-a",
                incumbent_name="expert-opponent" if use_expert else "sac-in-training-b",
                race_count=1,
                round_seconds=args.round_seconds,
                # Varies race conditions across the loop -- calling run_headless_head_to_head
                # once per race would otherwise always land on its internal race_index=1 with
                # the same random_seed, repeating identical starting conditions every race.
                random_seed=args.seed + race_index,
                copies_per_side=1,
            )
            total_challenger_m += role_distance_m(block_result, "challenger")
            total_incumbent_m += role_distance_m(block_result, "incumbent")
            expert_races += use_expert
        training_seconds = time.monotonic() - started_at
        print(f"[train] mixed-opponent self-play finished in {training_seconds:.1f}s")
        print(f"[train] transitions collected: {len(buffer)}, gradient updates: {len(state.update_metrics)}")
        print(
            f"[train] mixed-opponent scored distance a={total_challenger_m:.1f}m b={total_incumbent_m:.1f}m "
            f"over {args.races} race(s) ({expert_races} vs. expert, {args.races - expert_races} self-play)"
        )
        return agent, state, training_seconds

    incumbent: RobotController = (
        create_expert_controller() if args.opponent == "expert" else make_self_play_controller()
    )
    result = run_headless_head_to_head(
        challenger_controller=challenger,
        incumbent_controller=incumbent,
        challenger_name="sac-in-training-a",
        incumbent_name="sac-in-training-b",
        race_count=args.races,
        round_seconds=args.round_seconds,
        random_seed=args.seed,
        copies_per_side=args.copies_per_side,
    )
    training_seconds = time.monotonic() - started_at

    print(f"[train] self-play finished in {training_seconds:.1f}s")
    print(f"[train] transitions collected: {len(buffer)}, gradient updates: {len(state.update_metrics)}")
    print(
        f"[train] self-play scored distance a={role_distance_m(result, 'challenger'):.1f}m "
        f"b={role_distance_m(result, 'incumbent'):.1f}m over {args.races} race(s)"
    )
    return agent, state, training_seconds


def write_config(
    path: Path, args: TrainSacArguments, *, training_seconds: float, transitions: int, updates: int
) -> None:
    payload = {
        **asdict(args),
        "experiment_dir": str(args.experiment_dir),
        "git_commit": git_commit_hash(),
        "training_seconds": round(training_seconds, 2),
        "transitions_collected": transitions,
        "gradient_updates": updates,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
    }
    with path.open("w") as config_file:
        for key, value in payload.items():
            config_file.write(f"{key}: {value}\n")


def write_metrics_csv(path: Path, update_metrics: list[dict[str, float]]) -> None:
    if not update_metrics:
        path.write_text("global_step,buffer_size,critic_loss,actor_loss,alpha_loss,alpha\n")
        return
    fieldnames = list(update_metrics[0].keys())
    with path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(update_metrics)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    agent, state, training_seconds = train(args)
    baselines = None
    if args.opponent in ("expert", "mixed"):
        baselines = {**BASELINE_CONTROLLERS, "leaderboard_expert": create_expert_controller()}
    eval_results = evaluate_against_baselines(
        agent,
        eval_seeds=args.eval_seeds,
        eval_races=args.eval_races,
        eval_round_seconds=args.eval_round_seconds,
        baselines=baselines,
        residual_base=args.residual_expert_base,
        residual_scale=args.residual_action_scale,
        residual_base_source=args.residual_base_source,
        residual_hazard_gated=args.residual_hazard_gated,
    )

    agent.save(checkpoints_dir / "policy_final.pt", include_training_state=True)
    write_config(
        experiment_dir / "config.yaml",
        args,
        training_seconds=training_seconds,
        transitions=len(state.buffer),
        updates=len(state.update_metrics),
    )
    write_metrics_csv(experiment_dir / "metrics.csv", state.update_metrics)
    (experiment_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))

    print(f"\n[done] evidence written to {experiment_dir}")


if __name__ == "__main__":
    main()
