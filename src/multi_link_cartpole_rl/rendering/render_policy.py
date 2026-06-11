"""Render a trained policy placeholder.

The future version should load a saved policy, run it in the selected environment, and save
an inspectable demo video or GIF.
"""

from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    """Create the command-line parser for rendering a trained policy."""
    parser = ArgumentParser(description="Render a trained cartpole policy.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--model-path", default="models/latest.zip")
    parser.add_argument("--output", default="videos/policy.mp4")
    return parser


def main() -> None:
    """Policy rendering entry point."""
    args = build_parser().parse_args()
    print(
        "Policy rendering is not implemented yet. "
        f"Next step: load '{args.model_path}' and save a demo to '{args.output}'."
    )


if __name__ == "__main__":
    main()
