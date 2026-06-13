"""Run a reproducible CatanZero ablation and incumbent-promotion campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from catanzero import (
    benchmark_score,
    evaluate_balanced_match,
    evaluate_match,
    evaluate_suite,
    load_catanzero,
    load_legacy,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_CHAMPION = ROOT / "training/runs/20260611-512-bootstrap-v2/best.pt"
DEFAULT_BOOTSTRAP = (
    ROOT / "training/runs/20260611-512-bootstrap-v2/bootstrap_v2_dataset.pt"
)
DEFAULT_LEGACY = ROOT / "models/catan-512-best.pt"
DEFAULT_ALPHA = ROOT / "models/catan-512.ctnn"


@dataclass(frozen=True)
class Experiment:
    name: str
    profile: str
    lr: float = 1e-4
    curriculum_start: float = 0.34
    champion_probability: float = 0.65
    policy_mix: float = 0.75
    exploration_max: float = 0.90
    extra: tuple[str, ...] = field(default_factory=tuple)


EXPERIMENTS = (
    Experiment("league-focus", "league_focus"),
    Experiment("league-anchor", "league_anchor", lr=7.5e-5),
    Experiment("policy-heavy", "policy_heavy", champion_probability=0.50),
    Experiment(
        "exploration-reserve",
        "exploration",
        policy_mix=0.65,
        exploration_max=0.80,
        champion_probability=0.50,
    ),
    Experiment(
        "league-explore",
        "league_focus",
        policy_mix=0.65,
        exploration_max=0.80,
        champion_probability=0.65,
    ),
    Experiment(
        "league-explore-low-lr",
        "league_focus",
        lr=5e-5,
        policy_mix=0.65,
        exploration_max=0.80,
        champion_probability=0.65,
    ),
    Experiment(
        "policy-explore",
        "policy_heavy",
        policy_mix=0.65,
        exploration_max=0.80,
        champion_probability=0.65,
    ),
    Experiment("more-search", "more_search", lr=7.5e-5, champion_probability=0.50),
    Experiment(
        "league-only-low-lr",
        "league_focus",
        lr=5e-5,
        curriculum_start=0.67,
    ),
)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if (
                line.startswith("checkpoint ")
                or line.startswith("finished ")
                or "Traceback" in line
                or "Error" in line
            ):
                print(f"  {line.rstrip()}", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def fixed_evaluation(checkpoint: Path, games: int, seed: int) -> dict:
    candidate = load_catanzero(checkpoint)
    legacy = load_legacy(DEFAULT_LEGACY)
    results = evaluate_suite(
        candidate,
        legacy,
        None,
        None,
        games,
        seed,
        str(DEFAULT_ALPHA),
    )
    return {
        "benchmark_score": benchmark_score(results),
        "results": results,
    }


def compare_checkpoints(
    candidate_path: Path,
    incumbent_path: Path,
    games: int,
    seed: int,
    fixed_baseline: dict,
    fixed_seed: int,
) -> dict:
    candidate = load_catanzero(candidate_path)
    incumbent = load_catanzero(incumbent_path)
    candidate_head = evaluate_match(
        candidate, incumbent, "policy", games, seed
    )
    incumbent_head = evaluate_match(
        incumbent, candidate, "policy", games, seed + 10_000
    )
    fixed = fixed_evaluation(candidate_path, games // 2, fixed_seed)
    balanced = evaluate_balanced_match(
        candidate, incumbent, max(8, games // 3), seed + 15_000
    )
    margin = candidate_head["win_rate"] - incumbent_head["win_rate"]
    fixed_delta = fixed["benchmark_score"] - fixed_baseline["benchmark_score"]
    promotion = (
        balanced["win_rate"] >= 0.5
        and fixed_delta >= -0.025
    )
    return {
        "candidate": str(candidate_path),
        "incumbent": str(incumbent_path),
        "candidate_head_to_head": candidate_head,
        "incumbent_head_to_head": incumbent_head,
        "head_to_head_margin": margin,
        "balanced_head_to_head": balanced,
        "fixed": fixed,
        "fixed_delta": fixed_delta,
        "promotion_eligible": promotion,
        "ranking_score": fixed_delta + balanced["win_rate"] - 0.5,
    }


def train_experiment(
    experiment: Experiment,
    run_dir: Path,
    champion: Path,
    minutes: float,
    seed: int,
    eval_games: int,
) -> Path | None:
    command = [
        str(PYTHON),
        "-u",
        "training/catanzero.py",
        "train",
        "--minutes",
        str(minutes),
        "--checkpoint-minutes",
        str(max(1.5, minutes / 3)),
        "--eval-games",
        str(eval_games),
        "--hidden",
        "512",
        "--threads",
        "10",
        "--selfplay-workers",
        "6",
        "--bootstrap-games",
        "0",
        "--dagger-games",
        "0",
        "--training-profile",
        experiment.profile,
        "--curriculum-start",
        str(experiment.curriculum_start),
        "--champion-sample-probability",
        str(experiment.champion_probability),
        "--policy-mix",
        str(experiment.policy_mix),
        "--exploration-max",
        str(experiment.exploration_max),
        "--lr",
        str(experiment.lr),
        "--champion",
        str(champion),
        "--legacy",
        str(DEFAULT_LEGACY),
        "--alpha-net",
        str(DEFAULT_ALPHA),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        *experiment.extra,
    ]
    print(f"training {experiment.name}: {' '.join(command)}", flush=True)
    run_logged(command, run_dir / "train.log")
    best = run_dir / "best.pt"
    return best if best.exists() else None


def write_report(campaign_dir: Path, payload: dict) -> None:
    (campaign_dir / "campaign.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# Overnight CatanZero campaign",
        "",
        f"Incumbent: `{payload['initial_incumbent']}`",
        "",
        "| experiment | candidate H2H | incumbent H2H | balanced 2v2 | fixed delta | eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["experiments"]:
        comparison = result.get("comparison")
        if not comparison:
            lines.append(f"| {result['name']} | n/a | n/a | n/a | n/a | no checkpoint |")
            continue
        lines.append(
            "| {name} | {candidate:.1%} | {incumbent:.1%} | {balanced:.1%} | "
            "{fixed:+.3f} | {eligible} |".format(
                name=result["name"],
                candidate=comparison["candidate_head_to_head"]["win_rate"],
                incumbent=comparison["incumbent_head_to_head"]["win_rate"],
                balanced=comparison["balanced_head_to_head"]["win_rate"],
                fixed=comparison["fixed_delta"],
                eligible="yes" if comparison["promotion_eligible"] else "no",
            )
        )
    lines.extend(
        [
            "",
            f"Selected checkpoint: `{payload.get('selected_checkpoint')}`",
            "",
            "Candidates are promoted only when they reach at least 50% in paired",
            "2-vs-2 lineups and remain within 0.025 of the incumbent",
            "fixed-opponent benchmark.",
        ]
    )
    (campaign_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=8.0)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--validation-games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=6122026)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment names; default runs the full matrix.",
    )
    parser.add_argument("--skip-reanalysis", action="store_true")
    args = parser.parse_args()

    campaign_dir = args.run_dir or ROOT / "training/runs" / (
        time.strftime("%Y%m%d-%H%M") + "-overnight"
    )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    champion = args.champion.resolve()
    fixed_baseline = fixed_evaluation(
        champion, args.validation_games // 2, args.seed + 500_000
    )
    fixed_seed = args.seed + 500_000
    payload = {
        "initial_incumbent": str(champion),
        "fixed_baseline": fixed_baseline,
        "config": vars(args) | {"champion": str(champion), "run_dir": str(campaign_dir)},
        "experiments": [],
    }

    selected_names = (
        set(args.experiments.split(",")) if args.experiments else None
    )
    experiments = [
        experiment
        for experiment in EXPERIMENTS
        if selected_names is None or experiment.name in selected_names
    ]
    if selected_names and selected_names != {item.name for item in experiments}:
        unknown = selected_names - {item.name for item in experiments}
        raise ValueError(f"unknown experiments: {sorted(unknown)}")

    for index, experiment in enumerate(experiments):
        run_dir = campaign_dir / experiment.name
        candidate = train_experiment(
            experiment,
            run_dir,
            champion,
            args.minutes,
            args.seed + index * 100_000,
            args.eval_games,
        )
        entry = {"name": experiment.name, "run_dir": str(run_dir)}
        if candidate:
            comparison = compare_checkpoints(
                candidate,
                champion,
                args.validation_games,
                args.seed + index * 100_000 + 50_000,
                fixed_baseline,
                fixed_seed,
            )
            entry["comparison"] = comparison
            print(
                f"  validation {experiment.name}: "
                f"H2H {comparison['candidate_head_to_head']['win_rate']:.1%} / "
                f"reverse {comparison['incumbent_head_to_head']['win_rate']:.1%}, "
                f"fixed delta {comparison['fixed_delta']:+.3f}",
                flush=True,
            )
        payload["experiments"].append(entry)
        write_report(campaign_dir, payload)

    eligible = [
        result
        for result in payload["experiments"]
        if result.get("comparison", {}).get("promotion_eligible")
    ]
    selected = max(
        eligible,
        key=lambda item: item["comparison"]["ranking_score"],
        default=None,
    )
    selected_checkpoint = (
        Path(selected["comparison"]["candidate"]) if selected else champion
    )

    if not args.skip_reanalysis:
        reanalysis = Experiment(
            "targeted-reanalysis",
            "league_anchor",
            lr=7.5e-5,
            policy_mix=0.65,
            exploration_max=0.80,
            extra=(
                "--bootstrap-dataset",
                str(DEFAULT_BOOTSTRAP),
                "--bootstrap-sample-limit",
                "12000",
                "--bootstrap-max-epochs",
                "12",
                "--bootstrap-patience",
                "3",
                "--dagger-games",
                "64",
                "--teacher-determinizations",
                "8",
            ),
        )
        run_dir = campaign_dir / reanalysis.name
        candidate = train_experiment(
            reanalysis,
            run_dir,
            selected_checkpoint,
            args.minutes,
            args.seed + 900_000,
            args.eval_games,
        )
        entry = {"name": reanalysis.name, "run_dir": str(run_dir)}
        selected_fixed_seed = args.seed + 1_400_000
        selected_fixed = fixed_evaluation(
            selected_checkpoint, args.validation_games // 2, selected_fixed_seed
        )
        if candidate:
            entry["comparison"] = compare_checkpoints(
                candidate,
                selected_checkpoint,
                args.validation_games,
                args.seed + 950_000,
                selected_fixed,
                selected_fixed_seed,
            )
            if entry["comparison"]["promotion_eligible"]:
                selected_checkpoint = candidate
        payload["experiments"].append(entry)

    payload["selected_checkpoint"] = str(selected_checkpoint)
    write_report(campaign_dir, payload)
    print(f"campaign complete: {campaign_dir}", flush=True)
    print(f"selected checkpoint: {selected_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
