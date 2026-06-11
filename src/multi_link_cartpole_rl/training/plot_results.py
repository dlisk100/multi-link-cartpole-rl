"""Generate portfolio-ready static plots from run artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from multi_link_cartpole_rl.training.reporting import (
    build_run_summary,
    plot_random_vs_ppo,
    plot_training_curve,
    write_json_report,
)


def build_parser() -> ArgumentParser:
    """Create the command-line parser for result plotting."""
    parser = ArgumentParser(description="Plot cartpole training and evaluation results.")
    parser.add_argument(
        "--run-dir",
        default="runs/single_link_ppo",
        help="Run directory containing monitor.csv and receiving plot outputs.",
    )
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Path to evaluation JSON. Defaults to <run-dir>/evaluation.json.",
    )
    return parser


def plot_results(*, run_dir: str, evaluation: str | None = None) -> tuple[Path, Path, Path]:
    """Generate plots and a compact run summary."""
    run_path = Path(run_dir)
    evaluation_path = Path(evaluation) if evaluation is not None else run_path / "evaluation.json"
    monitor_path = run_path / "monitor.csv"

    training_curve_path = plot_training_curve(
        monitor_csv=monitor_path,
        output_path=run_path / "training_curve.png",
    )
    comparison_path = plot_random_vs_ppo(
        evaluation_json=evaluation_path,
        output_path=run_path / "random_vs_ppo.png",
    )
    summary_path = write_json_report(
        run_path / "run_summary.json",
        build_run_summary(monitor_csv=monitor_path, evaluation_json=evaluation_path),
    )

    return training_curve_path, comparison_path, summary_path


def main() -> None:
    """Plotting entry point."""
    args = build_parser().parse_args()
    training_curve_path, comparison_path, summary_path = plot_results(
        run_dir=args.run_dir,
        evaluation=args.evaluation,
    )
    print("Result plots written:")
    print(f"training_curve={training_curve_path}")
    print(f"random_vs_ppo={comparison_path}")
    print(f"run_summary={summary_path}")


if __name__ == "__main__":
    main()
