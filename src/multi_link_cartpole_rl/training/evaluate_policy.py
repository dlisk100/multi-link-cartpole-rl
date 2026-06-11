"""Policy evaluation placeholder.

Evaluation will eventually load a trained policy, run several episodes, and report reward
statistics such as mean return, standard deviation, and episode length.
"""

from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    """Create the command-line parser for future policy evaluation."""
    parser = ArgumentParser(description="Evaluate a trained cartpole policy.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--model-path", default="models/latest.zip")
    parser.add_argument("--episodes", type=int, default=10)
    return parser


def main() -> None:
    """Evaluation entry point."""
    args = build_parser().parse_args()
    print(
        "Policy evaluation is not implemented yet. "
        f"Next step: load '{args.model_path}' and run {args.episodes} episodes."
    )


if __name__ == "__main__":
    main()
