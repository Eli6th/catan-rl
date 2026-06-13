"""Evaluate a CTNN-backed planner against three fixed engine opponents."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import catan_py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.board import BoardTopology


ACTION_TYPE_NAMES = (
    "settlement",
    "city",
    "road",
    "robber",
    "steal",
    "discard",
    "monopoly",
    "year_of_plenty",
    "bank_trade",
    "propose_trade",
    "respond_trade",
    "confirm_trade",
    "roll",
    "buy_dev",
    "knight",
    "road_building",
    "end_turn",
)
RESOURCE_NAMES = ("wheat", "sheep", "wood", "brick", "stone")
PLAYER_OFFSET = 1196
PLAYER_STRIDE = 17
CONTEXT_OFFSET = 1314


def action_type_name(action: int) -> str:
    index = int(
        np.searchsorted(
            np.asarray(catan_py.ACTION_TYPE_BOUNDARIES),
            action,
            side="right",
        )
    )
    return ACTION_TYPE_NAMES[index]


def board_from_obs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resources = np.empty(19, dtype=np.int8)
    probabilities = np.empty(19, dtype=np.float32)
    for tile in range(19):
        base = tile * 8
        resources[tile] = int(np.argmax(obs[base : base + 6]))
        probabilities[tile] = float(obs[base + 6] / 7.2)
    return resources, probabilities


def opening_income(
    vertices: list[int],
    tile_resources: np.ndarray,
    tile_probabilities: np.ndarray,
    topology: BoardTopology,
) -> list[float]:
    income = np.zeros(5, dtype=np.float32)
    for vertex in vertices:
        for tile in topology.vertex_tiles[vertex]:
            if tile < 0:
                continue
            resource = int(tile_resources[tile])
            if resource < 5:
                income[resource] += tile_probabilities[tile]
    return [round(float(value), 6) for value in income]


def strategy_scores(income: list[float]) -> dict[str, float]:
    wheat, sheep, wood, brick, stone = income
    scores = {
        "expansion": min(wood / 2.0, brick / 2.0, wheat, sheep),
        "city": min(wheat / 2.0, stone / 3.0),
        "development": min(wheat, sheep, stone),
    }
    scores["best"] = max(scores.values())
    return {name: round(value, 6) for name, value in scores.items()}


def player_public_metrics(
    obs: np.ndarray,
    observer_seat: int,
    player_seat: int,
    actual_vp: float,
) -> dict[str, Any]:
    relative_seat = (player_seat - observer_seat) % 4
    base = PLAYER_OFFSET + relative_seat * PLAYER_STRIDE
    settlements = round(5.0 - float(obs[base + 4]) * 5.0)
    cities = round(4.0 - float(obs[base + 5]) * 4.0)
    roads = round(15.0 - float(obs[base + 6]) * 15.0)
    public_vp = round(float(obs[base + 3]) * 7.0)
    longest = bool(obs[base + 8] > 0.5)
    largest = bool(obs[base + 9] > 0.5)
    building_vp = settlements + 2 * cities
    bonus_vp = 2 * int(longest) + 2 * int(largest)
    return {
        "vp": round(actual_vp, 3),
        "public_vp": public_vp,
        "hidden_vp": round(actual_vp - public_vp, 3),
        "building_vp": building_vp,
        "award_vp": bonus_vp,
        "cards": round(float(obs[base]) * 19.0),
        "dev_cards_held": round(float(obs[base + 1]) * 25.0),
        "knights_played": round(float(obs[base + 2]) * 14.0),
        "settlements": settlements,
        "cities": cities,
        "roads": roads,
        "road_length": round(float(obs[base + 7]) * 15.0),
        "longest_road": longest,
        "largest_army": largest,
        "port_3_to_1": bool(obs[base + 10] > 0.5),
        "ports_2_to_1": [
            RESOURCE_NAMES[index]
            for index in range(5)
            if obs[base + 11 + index] > 0.5
        ],
    }


def mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "games": len(group),
            "win_rate": mean([float(game["won"]) for game in group]),
            "candidate_vp": mean([game["candidate"]["vp"] for game in group]),
            "vp_margin": mean([game["vp_margin"] for game in group]),
            "turns": mean([game["turns"] for game in group]),
            "opening_income": {
                resource: mean(
                    [game["opening"]["income"][index] for game in group]
                )
                for index, resource in enumerate(RESOURCE_NAMES)
            },
            "opening_total_income": mean(
                [sum(game["opening"]["income"]) for game in group]
            ),
            "opening_best_strategy": mean(
                [game["opening"]["strategy"]["best"] for game in group]
            ),
            "candidate_cities": mean(
                [game["candidate"]["cities"] for game in group]
            ),
            "candidate_settlements": mean(
                [game["candidate"]["settlements"] for game in group]
            ),
            "candidate_road_length": mean(
                [game["candidate"]["road_length"] for game in group]
            ),
            "candidate_knights": mean(
                [game["candidate"]["knights_played"] for game in group]
            ),
            "longest_road_rate": mean(
                [float(game["candidate"]["longest_road"]) for game in group]
            ),
            "largest_army_rate": mean(
                [float(game["candidate"]["largest_army"]) for game in group]
            ),
            "hidden_vp": mean([game["candidate"]["hidden_vp"] for game in group]),
            "action_mix": {
                action_type: mean(
                    [
                        float(game["candidate_actions"].get(action_type, 0))
                        for game in group
                    ]
                )
                for action_type in ACTION_TYPE_NAMES
            },
        }

    losses = [game for game in games if not game["won"]]
    wins = [game for game in games if game["won"]]
    return {
        "wins": group_summary(wins),
        "losses": group_summary(losses),
        "by_seat": {
            str(seat): group_summary(
                [game for game in games if game["candidate_seat"] == seat]
            )
            for seat in range(4)
        },
        "loss_vp_distribution": dict(
            sorted(Counter(str(round(game["candidate"]["vp"])) for game in losses).items())
        ),
        "loss_winner_awards": {
            "longest_road": mean(
                [float(game["winner_metrics"]["longest_road"]) for game in losses]
            ),
            "largest_army": mean(
                [float(game["winner_metrics"]["largest_army"]) for game in losses]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_net")
    parser.add_argument("--specialist-policy-net")
    parser.add_argument("--specialist-policy-min-vp", type=int, default=0)
    parser.add_argument("--specialist-policy-seat-mask", type=int, default=15)
    parser.add_argument("--late-policy-net")
    parser.add_argument("--late-policy-min-vp", type=int, default=5)
    parser.add_argument("--late-policy-seat-mask", type=int, default=15)
    parser.add_argument("--final-policy-net")
    parser.add_argument("--final-policy-min-vp", type=int, default=6)
    parser.add_argument("--final-policy-min-cities", type=int, default=0)
    parser.add_argument("--final-policy-max-cities", type=int, default=4)
    parser.add_argument("--final-policy-seat-mask", type=int, default=15)
    parser.add_argument("--opponent-net", default="models/catan-512.ctnn")
    parser.add_argument(
        "--opponent",
        choices=("alpha", "heuristic", "heuristic_v2", "random"),
        default="alpha",
    )
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=15_000_000)
    parser.add_argument(
        "--planner-seed",
        type=int,
        help="Search-policy seed; defaults to --seed so board and planner streams match.",
    )
    parser.add_argument("--root-k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--depth", type=int, default=300)
    parser.add_argument(
        "--planner",
        choices=("state", "opponent-aware", "hybrid-v1", "hybrid-v2"),
        default="opponent-aware",
    )
    parser.add_argument("--continuation-decisions", type=int, default=1)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--potential-weight", type=float, default=0.25)
    parser.add_argument("--opponent-root-k", type=int, default=1)
    parser.add_argument("--opponent-samples", type=int, default=1)
    parser.add_argument("--opponent-depth", type=int, default=0)
    parser.add_argument("--settlement-neural-mix", type=float, default=0.0)
    parser.add_argument("--strategy-settlement-weight", type=float, default=0.0)
    parser.add_argument("--opening-production-weight", type=float, default=0.0)
    parser.add_argument("--opening-wheat-weight", type=float, default=0.0)
    parser.add_argument("--opening-wheat-seat-mask", type=int, default=15)
    parser.add_argument("--opening-city-weight", type=float, default=0.0)
    parser.add_argument("--opening-city-seat-mask", type=int, default=15)
    parser.add_argument("--opening-settlement-lookahead", action="store_true")
    parser.add_argument("--opening-rollout-candidates", type=int, default=12)
    parser.add_argument("--opening-rollout-samples", type=int, default=0)
    parser.add_argument("--opening-rollout-finalists", type=int, default=2)
    parser.add_argument("--opening-rollout-final-samples", type=int, default=0)
    parser.add_argument("--opening-rollout-prior-weight", type=float, default=0.0)
    parser.add_argument("--rollout-vp-margin-weight", type=float, default=0.0)
    parser.add_argument("--common-rollout-random-numbers", action="store_true")
    parser.add_argument("--search-common-random-numbers", action="store_true")
    parser.add_argument("--independent-rollout-seeds", action="store_true")
    parser.add_argument("--second-settlement-rollout-samples", type=int, default=0)
    parser.add_argument("--heuristic-refinement", action="store_true")
    parser.add_argument("--endgame-conversion", action="store_true")
    parser.add_argument("--prefer-city-conversion", action="store_true")
    parser.add_argument("--prefer-city-conversion-seat-mask", type=int, default=15)
    parser.add_argument("--immediate-vp-min", type=int, default=5)
    parser.add_argument("--conversion-min-vp", type=int, default=5)
    parser.add_argument("--proposal-conversion-min-vp", type=int, default=5)
    parser.add_argument("--conversion-saving-min-vp", type=int, default=8)
    parser.add_argument("--conversion-saving-max-deficit", type=int, default=0)
    parser.add_argument("--endgame-road-push", action="store_true")
    parser.add_argument("--endgame-road-push-seat-mask", type=int, default=15)
    parser.add_argument("--opening-road-planning", action="store_true")
    parser.add_argument("--road-refinement", action="store_true")
    parser.add_argument("--road-length-weight", type=float, default=5.0)
    parser.add_argument("--road-settlement-weight", type=float, default=20.0)
    parser.add_argument("--knight-pressure", action="store_true")
    parser.add_argument("--knight-pressure-min-vp", type=int, default=0)
    parser.add_argument("--knight-pressure-seat-mask", type=int, default=15)
    parser.add_argument("--leader-robber-weight", type=float, default=0.0)
    parser.add_argument("--blocking-settlement-weight", type=float, default=0.0)
    parser.add_argument("--trade-refinement", action="store_true")
    parser.add_argument("--resource-tactics", action="store_true")
    parser.add_argument("--end-turn-trade-sweep", action="store_true")
    parser.add_argument("--end-turn-trade-sweep-max-vp", type=int, default=7)
    parser.add_argument("--evolved-state-refinement", action="store_true")
    parser.add_argument("--state-refinement-mix", type=float, default=0.0)
    parser.add_argument("--hybrid-search-root-k", type=int, default=1)
    parser.add_argument("--hybrid-search-samples", type=int, default=1)
    parser.add_argument("--hybrid-search-continuation", type=int, default=0)
    parser.add_argument("--hybrid-search-value-weight", type=float, default=1.0)
    parser.add_argument("--hybrid-search-potential-weight", type=float, default=0.0)
    parser.add_argument("--hybrid-search-vp-gain-weight", type=float, default=0.0)
    parser.add_argument("--hybrid-search-building-gain-weight", type=float, default=0.0)
    parser.add_argument("--hybrid-search-road-control-weight", type=float, default=0.0)
    parser.add_argument("--hybrid-search-min-vp", type=int, default=0)
    parser.add_argument("--hybrid-search-max-vp", type=int, default=7)
    parser.add_argument("--hybrid-search-max-hidden-vp", type=int, default=5)
    parser.add_argument("--hybrid-search-to-terminal", action="store_true")
    parser.add_argument("--hybrid-search-max-decisions", type=int, default=64)
    parser.add_argument("--hybrid-search-opponent-root-k", type=int, default=1)
    parser.add_argument("--hybrid-search-opponent-samples", type=int, default=1)
    parser.add_argument("--hybrid-search-opponent-depth", type=int, default=0)
    parser.add_argument(
        "--search-opening-decisions",
        type=int,
        default=-1,
        help="Use search only for the first N candidate decisions; -1 searches all.",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        help="Write per-game diagnostics and win/loss summaries as JSON.",
    )
    args = parser.parse_args()
    planner_seed = args.seed if args.planner_seed is None else args.planner_seed

    if args.planner == "state":
        planner = catan_py.AlphaTeacher(
            args.candidate_net,
            root_k=args.root_k,
            samples=args.samples,
            depth=args.depth,
            visibility="realistic",
        )
    elif args.planner == "opponent-aware":
        planner = catan_py.OpponentAwarePlanner(
            args.candidate_net,
            root_k=args.root_k,
            samples=args.samples,
            continuation_decisions=args.continuation_decisions,
            value_weight=args.value_weight,
            potential_weight=args.potential_weight,
            opponent_root_k=args.opponent_root_k,
            opponent_samples=args.opponent_samples,
            opponent_depth=args.opponent_depth,
        )
        greedy = catan_py.OpponentAwarePlanner(
            args.candidate_net,
            root_k=1,
            samples=1,
            continuation_decisions=0,
        )
    else:
        planner = catan_py.OpeningHybridPolicy(
            args.candidate_net,
            specialist_net_path=args.specialist_policy_net,
            specialist_net_min_vp=args.specialist_policy_min_vp,
            specialist_net_seat_mask=args.specialist_policy_seat_mask,
            late_net_path=args.late_policy_net,
            late_net_min_vp=args.late_policy_min_vp,
            late_net_seat_mask=args.late_policy_seat_mask,
            final_net_path=args.final_policy_net,
            final_net_min_vp=args.final_policy_min_vp,
            final_net_min_cities=args.final_policy_min_cities,
            final_net_max_cities=args.final_policy_max_cities,
            final_net_seat_mask=args.final_policy_seat_mask,
            heuristic=args.planner.removeprefix("hybrid-"),
            seed=planner_seed,
            settlement_neural_mix=args.settlement_neural_mix,
            strategy_settlement_weight=args.strategy_settlement_weight,
            opening_production_weight=args.opening_production_weight,
            opening_wheat_weight=args.opening_wheat_weight,
            opening_wheat_seat_mask=args.opening_wheat_seat_mask,
            opening_city_weight=args.opening_city_weight,
            opening_city_seat_mask=args.opening_city_seat_mask,
            opening_settlement_lookahead=args.opening_settlement_lookahead,
            opening_rollout_candidates=args.opening_rollout_candidates,
            opening_rollout_samples=args.opening_rollout_samples,
            opening_rollout_finalists=args.opening_rollout_finalists,
            opening_rollout_final_samples=args.opening_rollout_final_samples,
            opening_rollout_prior_weight=args.opening_rollout_prior_weight,
            rollout_vp_margin_weight=args.rollout_vp_margin_weight,
            common_rollout_random_numbers=args.common_rollout_random_numbers,
            search_common_random_numbers=args.search_common_random_numbers,
            second_settlement_rollout_samples=args.second_settlement_rollout_samples,
            heuristic_refinement=args.heuristic_refinement,
            endgame_conversion=args.endgame_conversion,
            prefer_city_conversion=args.prefer_city_conversion,
            prefer_city_conversion_seat_mask=args.prefer_city_conversion_seat_mask,
            immediate_vp_min=args.immediate_vp_min,
            conversion_min_vp=args.conversion_min_vp,
            proposal_conversion_min_vp=args.proposal_conversion_min_vp,
            conversion_saving_min_vp=args.conversion_saving_min_vp,
            conversion_saving_max_deficit=args.conversion_saving_max_deficit,
            endgame_road_push=args.endgame_road_push,
            endgame_road_push_seat_mask=args.endgame_road_push_seat_mask,
            opening_road_planning=args.opening_road_planning,
            road_refinement=args.road_refinement,
            road_length_weight=args.road_length_weight,
            road_settlement_weight=args.road_settlement_weight,
            knight_pressure=args.knight_pressure,
            knight_pressure_min_vp=args.knight_pressure_min_vp,
            knight_pressure_seat_mask=args.knight_pressure_seat_mask,
            leader_robber_weight=args.leader_robber_weight,
            blocking_settlement_weight=args.blocking_settlement_weight,
            trade_refinement=args.trade_refinement,
            resource_tactics=args.resource_tactics,
            end_turn_trade_sweep=args.end_turn_trade_sweep,
            end_turn_trade_sweep_max_vp=args.end_turn_trade_sweep_max_vp,
            evolved_state_refinement=args.evolved_state_refinement,
            state_refinement_mix=args.state_refinement_mix,
            search_root_k=args.hybrid_search_root_k,
            search_samples=args.hybrid_search_samples,
            search_continuation_decisions=args.hybrid_search_continuation,
            search_value_weight=args.hybrid_search_value_weight,
            search_potential_weight=args.hybrid_search_potential_weight,
            search_vp_gain_weight=args.hybrid_search_vp_gain_weight,
            search_building_gain_weight=args.hybrid_search_building_gain_weight,
            search_road_control_weight=args.hybrid_search_road_control_weight,
            search_min_vp=args.hybrid_search_min_vp,
            search_max_vp=args.hybrid_search_max_vp,
            search_max_hidden_vp=args.hybrid_search_max_hidden_vp,
            search_to_terminal=args.hybrid_search_to_terminal,
            search_max_decisions=args.hybrid_search_max_decisions,
            search_opponent_root_k=args.hybrid_search_opponent_root_k,
            search_opponent_samples=args.hybrid_search_opponent_samples,
            search_opponent_depth=args.hybrid_search_opponent_depth,
            independent_rollout_seeds=args.independent_rollout_seeds,
        )
    wins = 0
    vp = 0.0
    game_details: list[dict[str, Any]] = []
    topology = BoardTopology()
    for game in range(args.games):
        candidate_seat = game % 4
        board_seed = args.seed + game // 4
        seats = [args.opponent] * 4
        seats[candidate_seat] = "policy"
        env = catan_py.Env(
            victory_target=7,
            visibility="realistic",
            seed=board_seed,
            seats=seats,
            alpha_net=args.opponent_net if args.opponent == "alpha" else None,
        )
        initial_obs = np.asarray(env.obs())
        tile_resources, tile_probabilities = board_from_obs(initial_obs)
        candidate_actions: Counter[str] = Counter()
        opening_vertices: list[int] = []
        for decision in range(5000):
            search_seed = board_seed * 10000 + decision
            if args.planner == "state":
                action = planner.action(env.snapshot(), search_seed)
            elif args.planner.startswith("hybrid-"):
                action = planner.action(env)
            elif (
                args.search_opening_decisions >= 0
                and decision >= args.search_opening_decisions
            ):
                action = greedy.action(env, search_seed)
            else:
                action = planner.action(env, search_seed)
            action = int(action)
            candidate_actions[action_type_name(action)] += 1
            if action < 54 and len(opening_vertices) < 2:
                opening_vertices.append(action)
            _, _, done, winner, _ = env.step(action)
            if done:
                final_vp = np.asarray(env.final_vp())
                final_vp *= 7.0
                wins += int(winner == candidate_seat)
                vp += float(final_vp[candidate_seat])
                final_obs = np.asarray(env.obs())
                observer_seat = env.current_seat()
                players = [
                    player_public_metrics(
                        final_obs,
                        observer_seat,
                        seat,
                        float(final_vp[seat]),
                    )
                    for seat in range(4)
                ]
                income = opening_income(
                    opening_vertices,
                    tile_resources,
                    tile_probabilities,
                    topology,
                )
                opponent_vp = max(
                    float(final_vp[seat])
                    for seat in range(4)
                    if seat != candidate_seat
                )
                game_details.append(
                    {
                        "game": game,
                        "board_seed": board_seed,
                        "candidate_seat": candidate_seat,
                        "winner": int(winner),
                        "won": winner == candidate_seat,
                        "turns": round(float(final_obs[CONTEXT_OFFSET + 15]) * 1000.0),
                        "vp_margin": round(
                            float(final_vp[candidate_seat]) - opponent_vp,
                            3,
                        ),
                        "opening": {
                            "vertices": opening_vertices,
                            "income": income,
                            "income_by_resource": dict(zip(RESOURCE_NAMES, income)),
                            "resource_diversity": sum(value > 0 for value in income),
                            "strategy": strategy_scores(income),
                        },
                        "candidate_actions": dict(candidate_actions),
                        "candidate": players[candidate_seat],
                        "winner_metrics": players[winner] if winner >= 0 else None,
                        "players": players,
                    }
                )
                break
    result = {
        "games": args.games,
        "wins": wins,
        "win_rate": wins / args.games,
        "avg_vp": vp / args.games,
        "seed": args.seed,
        "planner_seed": planner_seed,
        "victory_target": 7,
        "visibility": "realistic",
        "candidate_net": args.candidate_net,
        "specialist_policy_net": args.specialist_policy_net,
        "specialist_policy_min_vp": args.specialist_policy_min_vp,
        "specialist_policy_seat_mask": args.specialist_policy_seat_mask,
        "late_policy_net": args.late_policy_net,
        "late_policy_min_vp": args.late_policy_min_vp,
        "late_policy_seat_mask": args.late_policy_seat_mask,
        "final_policy_net": args.final_policy_net,
        "final_policy_min_vp": args.final_policy_min_vp,
        "final_policy_min_cities": args.final_policy_min_cities,
        "final_policy_max_cities": args.final_policy_max_cities,
        "final_policy_seat_mask": args.final_policy_seat_mask,
        "opponent_net": args.opponent_net,
        "opponent": args.opponent,
        "root_k": args.root_k,
        "samples": args.samples,
        "depth": args.depth,
        "planner": args.planner,
        "continuation_decisions": args.continuation_decisions,
        "value_weight": args.value_weight,
        "potential_weight": args.potential_weight,
        "opponent_root_k": args.opponent_root_k,
        "opponent_samples": args.opponent_samples,
        "opponent_depth": args.opponent_depth,
        "settlement_neural_mix": args.settlement_neural_mix,
        "strategy_settlement_weight": args.strategy_settlement_weight,
        "opening_production_weight": args.opening_production_weight,
        "opening_wheat_weight": args.opening_wheat_weight,
        "opening_wheat_seat_mask": args.opening_wheat_seat_mask,
        "opening_city_weight": args.opening_city_weight,
        "opening_city_seat_mask": args.opening_city_seat_mask,
        "opening_settlement_lookahead": args.opening_settlement_lookahead,
        "opening_rollout_candidates": args.opening_rollout_candidates,
        "opening_rollout_samples": args.opening_rollout_samples,
        "opening_rollout_finalists": args.opening_rollout_finalists,
        "opening_rollout_final_samples": args.opening_rollout_final_samples,
        "opening_rollout_prior_weight": args.opening_rollout_prior_weight,
        "rollout_vp_margin_weight": args.rollout_vp_margin_weight,
        "common_rollout_random_numbers": args.common_rollout_random_numbers,
        "search_common_random_numbers": args.search_common_random_numbers,
        "independent_rollout_seeds": args.independent_rollout_seeds,
        "second_settlement_rollout_samples": args.second_settlement_rollout_samples,
        "heuristic_refinement": args.heuristic_refinement,
        "endgame_conversion": args.endgame_conversion,
        "prefer_city_conversion": args.prefer_city_conversion,
        "prefer_city_conversion_seat_mask": args.prefer_city_conversion_seat_mask,
        "immediate_vp_min": args.immediate_vp_min,
        "conversion_min_vp": args.conversion_min_vp,
        "proposal_conversion_min_vp": args.proposal_conversion_min_vp,
        "conversion_saving_min_vp": args.conversion_saving_min_vp,
        "conversion_saving_max_deficit": args.conversion_saving_max_deficit,
        "endgame_road_push": args.endgame_road_push,
        "endgame_road_push_seat_mask": args.endgame_road_push_seat_mask,
        "opening_road_planning": args.opening_road_planning,
        "road_refinement": args.road_refinement,
        "road_length_weight": args.road_length_weight,
        "road_settlement_weight": args.road_settlement_weight,
        "knight_pressure": args.knight_pressure,
        "knight_pressure_min_vp": args.knight_pressure_min_vp,
        "knight_pressure_seat_mask": args.knight_pressure_seat_mask,
        "leader_robber_weight": args.leader_robber_weight,
        "blocking_settlement_weight": args.blocking_settlement_weight,
        "trade_refinement": args.trade_refinement,
        "resource_tactics": args.resource_tactics,
        "end_turn_trade_sweep": args.end_turn_trade_sweep,
        "end_turn_trade_sweep_max_vp": args.end_turn_trade_sweep_max_vp,
        "evolved_state_refinement": args.evolved_state_refinement,
        "state_refinement_mix": args.state_refinement_mix,
        "hybrid_search_root_k": args.hybrid_search_root_k,
        "hybrid_search_samples": args.hybrid_search_samples,
        "hybrid_search_continuation": args.hybrid_search_continuation,
        "hybrid_search_value_weight": args.hybrid_search_value_weight,
        "hybrid_search_potential_weight": args.hybrid_search_potential_weight,
        "hybrid_search_vp_gain_weight": args.hybrid_search_vp_gain_weight,
        "hybrid_search_building_gain_weight": args.hybrid_search_building_gain_weight,
        "hybrid_search_road_control_weight": args.hybrid_search_road_control_weight,
        "hybrid_search_min_vp": args.hybrid_search_min_vp,
        "hybrid_search_max_vp": args.hybrid_search_max_vp,
        "hybrid_search_max_hidden_vp": args.hybrid_search_max_hidden_vp,
        "hybrid_search_to_terminal": args.hybrid_search_to_terminal,
        "hybrid_search_max_decisions": args.hybrid_search_max_decisions,
        "hybrid_search_opponent_root_k": args.hybrid_search_opponent_root_k,
        "hybrid_search_opponent_samples": args.hybrid_search_opponent_samples,
        "hybrid_search_opponent_depth": args.hybrid_search_opponent_depth,
        "search_opening_decisions": args.search_opening_decisions,
    }
    if args.details_out:
        details = {
            "evaluation": result,
            "summary": summarize_games(game_details),
            "games": game_details,
        }
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        args.details_out.write_text(json.dumps(details, indent=2) + "\n")
        result["details_out"] = str(args.details_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
