"""Re-evaluate a run's checkpoints with paired seats and balanced lineups."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from catanzero import (
    benchmark_score,
    evaluate_balanced_match,
    evaluate_match,
    load_catanzero,
    load_legacy,
)


ROOT = Path(__file__).resolve().parents[1]


def fixed_suite(model, legacy, games: int, seed: int, alpha_net: str):
    return {
        "heuristic_v1": evaluate_match(model, None, "heuristic", games, seed),
        "heuristic_v2": evaluate_match(
            model, None, "heuristic_v2", games, seed + 1_000
        ),
        "legacy_ppo": evaluate_match(
            model, legacy, "policy", games, seed + 2_000
        ),
        "alphabot": evaluate_match(
            model, None, "alpha", max(12, games // 2), seed + 3_000, alpha_net
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--boards", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9_120_026)
    parser.add_argument("--legacy", type=Path, default=ROOT / "models/catan-512-best.pt")
    parser.add_argument("--alpha-net", default=str(ROOT / "models/catan-512.ctnn"))
    args = parser.parse_args()

    checkpoints = sorted((args.run_dir / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise ValueError(f"no checkpoints found under {args.run_dir}")
    baseline = load_catanzero(args.baseline)
    legacy = load_legacy(args.legacy)
    baseline_fixed = fixed_suite(
        baseline, legacy, args.games, args.seed, args.alpha_net
    )
    baseline_score = benchmark_score(baseline_fixed)
    results = []
    for index, checkpoint in enumerate(checkpoints):
        model = load_catanzero(checkpoint)
        fixed = fixed_suite(model, legacy, args.games, args.seed, args.alpha_net)
        fixed_score = benchmark_score(fixed)
        balanced = evaluate_balanced_match(
            model,
            baseline,
            args.boards,
            args.seed + 10_000 + index * 1_000,
        )
        ranking_score = fixed_score - baseline_score + balanced["win_rate"] - 0.5
        results.append(
            {
                "checkpoint": str(checkpoint),
                "fixed": fixed,
                "fixed_score": fixed_score,
                "fixed_delta": fixed_score - baseline_score,
                "balanced_vs_baseline": balanced,
                "ranking_score": ranking_score,
            }
        )
        print(
            f"{checkpoint.name}: fixed {fixed_score:.3f} "
            f"({fixed_score - baseline_score:+.3f}), "
            f"balanced {balanced['win_rate']:.1%}"
        )

    selected = max(results, key=lambda item: item["ranking_score"])
    payload = {
        "baseline": str(args.baseline),
        "baseline_fixed": baseline_fixed,
        "baseline_score": baseline_score,
        "results": results,
        "selected": selected,
    }
    output = args.run_dir / "paired_selection.json"
    output.write_text(json.dumps(payload, indent=2))
    shutil.copy2(selected["checkpoint"], args.run_dir / "paired_best.pt")
    print(f"selected {selected['checkpoint']} -> {args.run_dir / 'paired_best.pt'}")


if __name__ == "__main__":
    main()
