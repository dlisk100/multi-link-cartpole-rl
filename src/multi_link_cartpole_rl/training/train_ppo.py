"""PPO training placeholder.

The future version should:
- Load a YAML config.
- Create the requested cartpole environment.
- Train PPO from Stable-Baselines3.
- Save checkpoints and basic training logs.

This file avoids importing Stable-Baselines3 at module import time so the project remains
easy to inspect before the environment is implemented.
"""

from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    """Create the command-line parser for future PPO training runs."""
    parser = ArgumentParser(description="Train PPO on a staged cartpole environment.")
    parser.add_argument(
        "--config",
        default="configs/single_link.yaml",
        help="Path to a YAML training config.",
    )
    return parser


def main() -> None:
    """Training entry point."""
    args = build_parser().parse_args()
    print(
        "PPO training is not implemented yet. "
        f"Next step: load config '{args.config}' and create the environment."
    )


if __name__ == "__main__":
    main()
