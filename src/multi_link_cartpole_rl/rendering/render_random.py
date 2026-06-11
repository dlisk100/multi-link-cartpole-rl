"""Render random actions placeholder.

Random-action rendering is useful before training because it verifies that the environment
dynamics and visualization are understandable.
"""

from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    """Create the command-line parser for future random-action rendering."""
    parser = ArgumentParser(description="Render random actions in a cartpole environment.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--steps", type=int, default=500)
    return parser


def main() -> None:
    """Random rendering entry point."""
    args = build_parser().parse_args()
    print(
        "Random-action rendering is not implemented yet. "
        f"Next step: run {args.steps} random steps using '{args.config}'."
    )


if __name__ == "__main__":
    main()
