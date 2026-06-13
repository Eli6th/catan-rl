"""Distill opponent-aware counterfactual midgame action values into a policy."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import catan_py

try:
    from .hybrid_policy_improvement import (
        evaluate_hybrid,
        export_candidate,
        hybrid_policy,
        load_policy_model,
        save_candidate,
    )
except ImportError:
    from hybrid_policy_improvement import (
        evaluate_hybrid,
        export_candidate,
        hybrid_policy,
        load_policy_model,
        save_candidate,
    )


# Frozen OBS_VERSION=1 offsets from rust/catan-env/src/obs.rs.
PLAYERS = 1196
PLAYER_PUBLIC_VP = 3
SELF_PRIVATE = 1264
DEV_VICTORY_POINT = 1
VICTORY_TARGET = 7


def observed_vp(obs: np.ndarray) -> float:
    public = float(obs[PLAYERS + PLAYER_PUBLIC_VP]) * VICTORY_TARGET
    hidden = float(obs[SELF_PRIVATE + 5 + DEV_VICTORY_POINT]) * 5.0
    return public + hidden


def observed_cities(obs: np.ndarray) -> int:
    return round(4.0 - float(obs[PLAYERS + 5]) * 4.0)


def score_target(
    scores: list[tuple[int, float]],
    *,
    temperature: float,
) -> tuple[np.ndarray, float]:
    if not scores:
        raise ValueError("counterfactual score set cannot be empty")
    actions = np.asarray([action for action, _ in scores], dtype=np.int64)
    values = np.asarray([value for _, value in scores], dtype=np.float64)
    logits = (values - values.max()) / temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    target = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float32)
    target[actions] = probabilities.astype(np.float32)
    return target, float(values.max() - values.min())


def should_retain_episode(
    terminal_rewards: np.ndarray,
    candidate_seat: int,
    *,
    losses_only: bool,
) -> bool:
    return not losses_only or float(terminal_rewards[candidate_seat]) <= 0.0


def relative_road_target(
    absolute_target: np.ndarray,
    candidate_seat: int,
) -> np.ndarray:
    absolute_target = np.asarray(absolute_target, dtype=np.float32)
    if absolute_target.shape != (5,) or not np.isclose(absolute_target.sum(), 1.0):
        raise ValueError("terminal road target must be a one-hot vector of length five")
    target = np.zeros(5, dtype=np.float32)
    holder = int(np.argmax(absolute_target))
    if not np.isclose(absolute_target[holder], 1.0):
        raise ValueError("terminal road target must be one-hot")
    target[4 if holder == 4 else (holder - candidate_seat) % 4] = 1.0
    return target


def collect_lineup(
    net_path: Path,
    teacher,
    *,
    deployment_base_net: Path | None,
    deployment_late_min_vp: int,
    deployment_opening_wheat_weight: float,
    deployment_immediate_vp_min: int,
    independent_rollout_seeds: bool,
    opponent: str,
    candidate_seat: int,
    games: int,
    seed: int,
    alpha_net: str,
    alpha_root_k: int,
    alpha_samples: int,
    alpha_depth: int,
    min_vp: float,
    max_vp: float,
    min_cities: int,
    max_cities: int,
    max_states_per_game: int,
    sample_probability: float,
    target_temperature: float,
    min_score_spread: float,
    late_weight: float,
    disagreement_weight: float,
    losses_only: bool,
    road_auxiliary: bool,
):
    seats = [opponent] * 4
    seats[candidate_seat] = "policy"
    env = catan_py.VecEnv(
        games,
        victory_target=VICTORY_TARGET,
        visibility="realistic",
        zero_sum=True,
        seed=seed,
        seats=seats,
        alpha_net=alpha_net if opponent == "alpha" else None,
        alpha_root_k=alpha_root_k,
        alpha_samples=alpha_samples,
        alpha_depth=alpha_depth,
    )
    policy = hybrid_policy(
        deployment_base_net or net_path,
        seed,
        opponent,
        late_net_path=net_path if deployment_base_net is not None else None,
        late_net_min_vp=deployment_late_min_vp,
        opening_wheat_weight=deployment_opening_wheat_weight,
        immediate_vp_min=deployment_immediate_vp_min,
        independent_rollout_seeds=independent_rollout_seeds,
    )
    obs, masks, _ = env.observe()
    obs = np.asarray(obs)
    masks = np.asarray(masks)
    finished = np.zeros(games, dtype=bool)
    scored = np.zeros(games, dtype=np.int32)
    eligible_seen = np.zeros(games, dtype=np.int32)
    samples = []
    episode_samples = (
        [[] for _ in range(games)] if losses_only or road_auxiliary else None
    )
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    decision = 0

    while not bool(finished.all()):
        actions = np.asarray(policy.actions(env), dtype=np.uint32)
        selected = []
        for index in np.flatnonzero(~finished):
            vp = observed_vp(obs[index])
            cities = observed_cities(obs[index])
            if (
                vp < min_vp
                or vp > max_vp + 1e-4
                or cities < min_cities
                or cities > max_cities
                or scored[index] >= max_states_per_game
                or int(masks[index].sum()) <= 1
            ):
                continue
            eligible_seen[index] += 1
            if eligible_seen[index] == 1 or rng.random() < sample_probability:
                selected.append(int(index))

        if selected:
            seeds = [
                seed
                ^ (decision + 1) * 0x9E3779B1
                ^ (index + 1) * 0x85EBCA77
                for index in selected
            ]
            required = [int(actions[index]) for index in selected]
            metric_rows = teacher.score_conversion_indices(
                env,
                selected,
                seeds,
                required,
            )
            for index, metrics in zip(selected, metric_rows):
                scores = [(int(row[0]), float(row[1])) for row in metrics]
                target, spread = score_target(
                    scores,
                    temperature=target_temperature,
                )
                if spread < min_score_spread:
                    continue
                deployed = int(actions[index])
                teacher_best = int(scores[0][0])
                best_metrics = metrics[0]
                vp = observed_vp(obs[index])
                progress = np.clip(
                    (vp - min_vp) / max(VICTORY_TARGET - min_vp, 1.0),
                    0.0,
                    1.0,
                )
                weight = 1.0 + late_weight * float(progress)
                if teacher_best != deployed:
                    weight *= 1.0 + disagreement_weight
                sample = {
                    "obs": obs[index].copy(),
                    "mask": masks[index].copy(),
                    "target": target,
                    "weight": weight,
                    "vp": vp,
                    "spread": spread,
                    "deployed": deployed,
                    "teacher_best": teacher_best,
                    "teacher_value": float(best_metrics[2]),
                    "teacher_vp_gain": float(best_metrics[3]),
                    "teacher_building_gain": float(best_metrics[4]),
                    "teacher_road_control": float(best_metrics[5]),
                }
                if episode_samples is None:
                    samples.append(sample)
                else:
                    episode_samples[index].append(sample)
                scored[index] += 1

        obs, masks, _, _, dones, terminal_rewards = env.step(actions)
        obs = np.asarray(obs)
        masks = np.asarray(masks)
        dones = np.asarray(dones)
        terminal_rewards = np.asarray(terminal_rewards)
        if episode_samples is not None:
            terminal_road_targets = np.asarray(env.terminal_road_targets())
            for index in np.flatnonzero(dones & ~finished):
                road_target = relative_road_target(
                    terminal_road_targets[index],
                    candidate_seat,
                )
                retain_policy = should_retain_episode(
                    terminal_rewards[index],
                    candidate_seat,
                    losses_only=losses_only,
                )
                for sample in episode_samples[index]:
                    sample["road_target"] = road_target
                    sample["policy_weight"] = float(retain_policy)
                if retain_policy or road_auxiliary:
                    samples.extend(episode_samples[index])
                episode_samples[index].clear()
        finished |= dones
        decision += 1
    return samples


def collect_counterfactuals(
    net_path: Path,
    *,
    deployment_base_net: Path | None,
    deployment_late_min_vp: int,
    deployment_opening_wheat_weight: float,
    deployment_immediate_vp_min: int,
    independent_rollout_seeds: bool,
    opponent: str,
    games: int,
    seed: int,
    teacher_root_k: int,
    teacher_samples: int,
    teacher_continuation: int,
    teacher_potential_weight: float,
    teacher_vp_gain_weight: float,
    teacher_building_gain_weight: float,
    teacher_road_control_weight: float,
    alpha_net: str,
    alpha_root_k: int,
    alpha_samples: int,
    alpha_depth: int,
    min_vp: float,
    max_vp: float,
    min_cities: int,
    max_cities: int,
    candidate_seats: tuple[int, ...],
    max_states_per_game: int,
    sample_probability: float,
    target_temperature: float,
    min_score_spread: float,
    late_weight: float,
    disagreement_weight: float,
    losses_only: bool,
    road_auxiliary: bool,
):
    if games % len(candidate_seats):
        raise ValueError("games must be divisible by the number of candidate seats")
    teacher = catan_py.OpponentAwarePlanner(
        str(net_path),
        root_k=teacher_root_k,
        samples=teacher_samples,
        continuation_decisions=teacher_continuation,
        value_weight=0.0,
        potential_weight=teacher_potential_weight,
        vp_gain_weight=teacher_vp_gain_weight,
        building_gain_weight=teacher_building_gain_weight,
        road_control_weight=teacher_road_control_weight,
        common_random_numbers=True,
        opponent_root_k=1,
        opponent_samples=1,
        opponent_depth=0,
    )
    samples = []
    games_per_seat = games // len(candidate_seats)
    for seat in candidate_seats:
        samples.extend(
            collect_lineup(
                net_path,
                teacher,
                deployment_base_net=deployment_base_net,
                deployment_late_min_vp=deployment_late_min_vp,
                deployment_opening_wheat_weight=deployment_opening_wheat_weight,
                deployment_immediate_vp_min=deployment_immediate_vp_min,
                independent_rollout_seeds=independent_rollout_seeds,
                opponent=opponent,
                candidate_seat=seat,
                games=games_per_seat,
                seed=seed + seat * 100_003,
                alpha_net=alpha_net,
                alpha_root_k=alpha_root_k,
                alpha_samples=alpha_samples,
                alpha_depth=alpha_depth,
                min_vp=min_vp,
                max_vp=max_vp,
                min_cities=min_cities,
                max_cities=max_cities,
                max_states_per_game=max_states_per_game,
                sample_probability=sample_probability,
                target_temperature=target_temperature,
                min_score_spread=min_score_spread,
                late_weight=late_weight,
                disagreement_weight=disagreement_weight,
                losses_only=losses_only,
                road_auxiliary=road_auxiliary,
            )
        )
    return samples


def fit_counterfactuals(
    model,
    reference,
    samples,
    *,
    learning_rate: float,
    epochs: int,
    minibatch_size: int,
    kl_scale: float,
    road_aux_scale: float,
):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [*model.policy.parameters(), *model.policy_type.parameters()]
    road_head = None
    if road_aux_scale > 0.0:
        road_head = nn.Linear(model.hidden, 5)
        parameters.extend(model.trunk.parameters())
        parameters.extend(road_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    totals = {
        "loss": 0.0,
        "target_loss": 0.0,
        "kl_loss": 0.0,
        "road_loss": 0.0,
    }
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(len(samples))
        for indexes in order.split(minibatch_size):
            batch = [samples[int(index)] for index in indexes]
            obs = torch.as_tensor(np.stack([sample["obs"] for sample in batch]))
            mask = torch.as_tensor(np.stack([sample["mask"] for sample in batch]))
            targets = torch.as_tensor(
                np.stack([sample["target"] for sample in batch])
            )
            weights = torch.as_tensor(
                [sample["weight"] for sample in batch],
                dtype=torch.float32,
            )
            policy_weights = weights * torch.as_tensor(
                [sample.get("policy_weight", 1.0) for sample in batch],
                dtype=torch.float32,
            )
            hidden = model.trunk(obs)
            logits = (
                model.policy(hidden)
                + model.policy_type(hidden)[:, model.action_types]
            ).masked_fill(~mask, -1e9)
            log_policy = torch.log_softmax(logits, dim=1)
            policy_weight_sum = policy_weights.sum()
            if float(policy_weight_sum) > 0.0:
                target_loss = (
                    -(targets * log_policy).sum(dim=1) * policy_weights
                ).sum() / policy_weight_sum
            else:
                target_loss = torch.zeros((), dtype=torch.float32)
            with torch.no_grad():
                reference_logits = reference(obs)["logits"].masked_fill(~mask, -1e9)
                reference_policy = torch.softmax(reference_logits, dim=1)
            kl_loss = (
                reference_policy
                * (
                    torch.log(reference_policy.clamp_min(1e-12))
                    - log_policy
                )
            ).masked_fill(~mask, 0.0).sum(dim=1).mean()
            road_loss = torch.zeros((), dtype=torch.float32)
            if road_head is not None:
                road_targets = torch.as_tensor(
                    np.stack([sample["road_target"] for sample in batch])
                )
                road_log_policy = torch.log_softmax(road_head(hidden), dim=1)
                road_loss = (
                    -(road_targets * road_log_policy).sum(dim=1) * weights
                ).sum() / weights.sum()
            loss = target_loss + kl_scale * kl_loss + road_aux_scale * road_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["target_loss"] += float(target_loss.item())
            totals["kl_loss"] += float(kl_loss.item())
            totals["road_loss"] += float(road_loss.item())
            updates += 1

    return {
        "samples": len(samples),
        "policy_samples": sum(
            sample.get("policy_weight", 1.0) > 0.0 for sample in samples
        ),
        "disagreements": sum(
            sample.get("policy_weight", 1.0) > 0.0
            and sample["deployed"] != sample["teacher_best"]
            for sample in samples
        ),
        "mean_vp": float(np.mean([sample["vp"] for sample in samples])),
        "mean_spread": float(np.mean([sample["spread"] for sample in samples])),
        "mean_teacher_value": float(
            np.mean([sample.get("teacher_value", 0.0) for sample in samples])
        ),
        "mean_teacher_vp_gain": float(
            np.mean([sample.get("teacher_vp_gain", 0.0) for sample in samples])
        ),
        "mean_teacher_building_gain": float(
            np.mean([sample.get("teacher_building_gain", 0.0) for sample in samples])
        ),
        "mean_teacher_road_control": float(
            np.mean([sample.get("teacher_road_control", 0.0) for sample in samples])
        ),
        **{key: value / updates for key, value in totals.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--deployment-base-net",
        type=Path,
        help="Use this base policy before routing to the checkpoint under training.",
    )
    parser.add_argument("--deployment-late-min-vp", type=int, default=5)
    parser.add_argument("--deployment-opening-wheat-weight", type=float, default=0.0)
    parser.add_argument("--deployment-immediate-vp-min", type=int, default=0)
    parser.add_argument("--independent-rollout-seeds", action="store_true")
    parser.add_argument(
        "--opponent",
        choices=("heuristic", "heuristic_v2", "alpha"),
        required=True,
    )
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=51_000_000)
    parser.add_argument("--teacher-root-k", type=int, default=6)
    parser.add_argument("--teacher-samples", type=int, default=4)
    parser.add_argument("--teacher-continuation", type=int, default=16)
    parser.add_argument("--teacher-potential-weight", type=float, default=1.0)
    parser.add_argument("--teacher-vp-gain-weight", type=float, default=0.0)
    parser.add_argument("--teacher-building-gain-weight", type=float, default=0.0)
    parser.add_argument("--teacher-road-control-weight", type=float, default=0.0)
    parser.add_argument("--min-vp", type=float, default=3.0)
    parser.add_argument("--max-vp", type=float, default=float(VICTORY_TARGET))
    parser.add_argument("--min-cities", type=int, default=0)
    parser.add_argument("--max-cities", type=int, default=4)
    parser.add_argument("--candidate-seats", default="0,1,2,3")
    parser.add_argument("--max-states-per-game", type=int, default=4)
    parser.add_argument("--sample-probability", type=float, default=0.25)
    parser.add_argument("--target-temperature", type=float, default=0.10)
    parser.add_argument("--min-score-spread", type=float, default=0.01)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--disagreement-weight", type=float, default=1.0)
    parser.add_argument(
        "--losses-only",
        action="store_true",
        help="Train only on sampled states from games the deployed policy loses.",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--kl-scale", type=float, default=2.0)
    parser.add_argument(
        "--road-aux-scale",
        type=float,
        default=0.0,
        help="Temporary eventual-Longest-Road auxiliary loss applied to the trunk.",
    )
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--alpha-net", default="models/catan-512.ctnn")
    parser.add_argument("--alpha-root-k", type=int, default=8)
    parser.add_argument("--alpha-samples", type=int, default=96)
    parser.add_argument("--alpha-depth", type=int, default=300)
    args = parser.parse_args()

    if args.target_temperature <= 0:
        raise ValueError("target temperature must be positive")
    if not 0.0 <= args.sample_probability <= 1.0:
        raise ValueError("sample probability must be between zero and one")
    if args.max_vp < args.min_vp:
        raise ValueError("max VP must be at least min VP")
    if not 0 <= args.min_cities <= args.max_cities <= 4:
        raise ValueError("city bounds must satisfy 0 <= min <= max <= 4")
    candidate_seats = tuple(
        dict.fromkeys(int(value.strip()) for value in args.candidate_seats.split(","))
    )
    if not candidate_seats or any(seat not in range(4) for seat in candidate_seats):
        raise ValueError("candidate seats must be a comma-separated subset of 0,1,2,3")
    if args.games % len(candidate_seats):
        raise ValueError("games must be divisible by the number of candidate seats")
    if args.road_aux_scale < 0.0:
        raise ValueError("road auxiliary scale must be non-negative")
    teacher_weights = (
        args.teacher_potential_weight,
        args.teacher_vp_gain_weight,
        args.teacher_building_gain_weight,
        args.teacher_road_control_weight,
    )
    if any(weight < 0.0 for weight in teacher_weights):
        raise ValueError("teacher score weights must be non-negative")
    if sum(teacher_weights) <= 0.0:
        raise ValueError("at least one teacher score weight must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(10)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = load_policy_model(args.checkpoint)
    reference = copy.deepcopy(model).eval()
    source_net = args.run_dir / "source.ctnn"
    export_candidate(args.checkpoint, source_net)

    samples = collect_counterfactuals(
        source_net,
        deployment_base_net=args.deployment_base_net,
        deployment_late_min_vp=args.deployment_late_min_vp,
        deployment_opening_wheat_weight=args.deployment_opening_wheat_weight,
        deployment_immediate_vp_min=args.deployment_immediate_vp_min,
        independent_rollout_seeds=args.independent_rollout_seeds,
        opponent=args.opponent,
        games=args.games,
        seed=args.seed,
        teacher_root_k=args.teacher_root_k,
        teacher_samples=args.teacher_samples,
        teacher_continuation=args.teacher_continuation,
        teacher_potential_weight=args.teacher_potential_weight,
        teacher_vp_gain_weight=args.teacher_vp_gain_weight,
        teacher_building_gain_weight=args.teacher_building_gain_weight,
        teacher_road_control_weight=args.teacher_road_control_weight,
        alpha_net=args.alpha_net,
        alpha_root_k=args.alpha_root_k,
        alpha_samples=args.alpha_samples,
        alpha_depth=args.alpha_depth,
        min_vp=args.min_vp,
        max_vp=args.max_vp,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        candidate_seats=candidate_seats,
        max_states_per_game=args.max_states_per_game,
        sample_probability=args.sample_probability,
        target_temperature=args.target_temperature,
        min_score_spread=args.min_score_spread,
        late_weight=args.late_weight,
        disagreement_weight=args.disagreement_weight,
        losses_only=args.losses_only,
        road_auxiliary=args.road_aux_scale > 0.0,
    )
    if not samples:
        raise RuntimeError("counterfactual collection produced no training samples")
    fit = fit_counterfactuals(
        model,
        reference,
        samples,
        learning_rate=args.lr,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        kl_scale=args.kl_scale,
        road_aux_scale=args.road_aux_scale,
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = args.run_dir / "counterfactual.pt"
    net = args.run_dir / "counterfactual.ctnn"
    save_candidate(checkpoint, source_payload, model, 1, config)
    export_candidate(checkpoint, net)
    evaluation = {
        opponent: evaluate_hybrid(
            net,
            opponent,
            args.eval_games,
            args.seed + 1_000_003,
            independent_rollout_seeds=args.independent_rollout_seeds,
        )
        for opponent in ("heuristic", "heuristic_v2")
    }
    result = {
        "checkpoint": str(checkpoint),
        "net": str(net),
        "fit": fit,
        "evaluation": evaluation,
        "seconds": time.time() - started,
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
