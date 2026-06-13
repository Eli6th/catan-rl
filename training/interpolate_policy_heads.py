"""Interpolate PPO policy heads while preserving the base trunk and value head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


POLICY_KEYS = (
    "policy.weight",
    "policy.bias",
    "policy_type.weight",
    "policy_type.bias",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("specialist", type=Path)
    parser.add_argument("--fractions", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    specialist = torch.load(
        args.specialist,
        map_location="cpu",
        weights_only=False,
    )
    if base["config"]["hidden"] != specialist["config"]["hidden"]:
        raise ValueError("checkpoints use different hidden sizes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fraction in args.fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1]: {fraction}")
        checkpoint = {
            **specialist,
            "model_state": {
                key: value.clone()
                for key, value in specialist["model_state"].items()
            },
            "interpolation": {
                "base": str(args.base),
                "specialist": str(args.specialist),
                "policy_fraction": fraction,
            },
        }
        for key in POLICY_KEYS:
            specialist_value = specialist["model_state"][key]
            base_value = base["model_state"].get(
                key,
                torch.zeros_like(specialist_value),
            )
            checkpoint["model_state"][key] = torch.lerp(
                base_value,
                specialist_value,
                fraction,
            )

        label = f"{fraction:.2f}".replace(".", "p")
        output = args.output_dir / f"policy_{label}.pt"
        torch.save(checkpoint, output)
        print(output)


if __name__ == "__main__":
    main()
