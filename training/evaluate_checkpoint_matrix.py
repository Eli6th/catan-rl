"""Run a larger fresh-seed evaluation for a promoted CatanZero checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catanzero import (
    evaluate_balanced_match,
    evaluate_match,
    load_catanzero,
    load_legacy,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluate_fixed(model, legacy, games: int, seed: int, alpha_net: str):
    return {
        "heuristic_v1": evaluate_match(
            model, None, "heuristic", games, seed
        ),
        "heuristic_v2": evaluate_match(
            model, None, "heuristic_v2", games, seed + 1_000
        ),
        "legacy_ppo": evaluate_match(
            model, legacy, "policy", games, seed + 2_000
        ),
        "alphabot": evaluate_match(
            model, None, "alpha", max(12, games // 2), seed + 3_000, alpha_net
        ),
        "heuristic_v2_search8": evaluate_match(
            model,
            None,
            "heuristic_v2",
            max(12, games // 2),
            seed + 4_000,
            search_simulations=8,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7_120_026)
    parser.add_argument("--legacy", type=Path, default=ROOT / "models/catan-512-best.pt")
    parser.add_argument("--alpha-net", default=str(ROOT / "models/catan-512.ctnn"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_catanzero(args.candidate)
    baseline = load_catanzero(args.baseline)
    legacy = load_legacy(args.legacy)
    results = {
        "config": {
            "candidate": str(args.candidate),
            "baseline": str(args.baseline),
            "previous": str(args.previous) if args.previous else None,
            "games": args.games,
            "seed": args.seed,
        },
        "candidate_fixed": evaluate_fixed(
            candidate, legacy, args.games, args.seed, args.alpha_net
        ),
        "baseline_fixed": evaluate_fixed(
            baseline, legacy, args.games, args.seed, args.alpha_net
        ),
        "candidate_vs_baseline": evaluate_match(
            candidate, baseline, "policy", args.games * 2, args.seed + 10_000
        ),
        "baseline_vs_candidate": evaluate_match(
            baseline, candidate, "policy", args.games * 2, args.seed + 20_000
        ),
        "candidate_vs_baseline_balanced": evaluate_balanced_match(
            candidate, baseline, max(8, args.games // 3), args.seed + 25_000
        ),
        "candidate_vs_baseline_search8": evaluate_match(
            candidate,
            baseline,
            "policy",
            max(24, args.games // 2),
            args.seed + 30_000,
            search_simulations=8,
        ),
    }
    if args.previous:
        previous = load_catanzero(args.previous)
        results["candidate_vs_previous"] = evaluate_match(
            candidate, previous, "policy", args.games, args.seed + 40_000
        )
        results["previous_vs_candidate"] = evaluate_match(
            previous, candidate, "policy", args.games, args.seed + 50_000
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
