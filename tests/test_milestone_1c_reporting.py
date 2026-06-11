import json
from pathlib import Path

from multi_link_cartpole_rl.training.evaluate_policy import (
    EvaluationStats,
    build_evaluation_report,
)
from multi_link_cartpole_rl.training.evaluate_policy import (
    build_parser as build_evaluate_parser,
)
from multi_link_cartpole_rl.training.plot_results import (
    build_parser as build_plot_parser,
)
from multi_link_cartpole_rl.training.plot_results import (
    plot_results,
)
from multi_link_cartpole_rl.training.reporting import (
    build_run_summary,
    monitor_summary,
    read_monitor_csv,
    write_json_report,
)


def write_monitor_fixture(tmp_path: Path) -> Path:
    """Create a tiny Stable-Baselines3 Monitor CSV fixture."""
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        "\n".join(
            [
                '#{"t_start": 123.0, "env_id": "None"}',
                "r,l,t",
                "1.5,10,0.1",
                "3.0,20,0.2",
                "2.5,15,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return monitor_path


def write_evaluation_fixture(tmp_path: Path) -> Path:
    """Create a tiny evaluation report fixture."""
    evaluation_path = tmp_path / "evaluation.json"
    write_json_report(
        evaluation_path,
        {
            "task": "single_link_stabilization",
            "config_path": "configs/single_link.yaml",
            "model_path": "models/single_link_ppo.zip",
            "episodes": 2,
            "seed": 1,
            "policies": {
                "random": {
                    "mean_episode_length": 12.0,
                    "mean_return": 8.5,
                    "return_std": 1.0,
                    "best_episode_length": 14,
                },
                "trained_ppo": {
                    "mean_episode_length": 500.0,
                    "mean_return": 499.5,
                    "return_std": 0.2,
                    "best_episode_length": 500,
                },
            },
        },
    )
    return evaluation_path


def test_monitor_csv_parsing_and_summary(tmp_path: Path) -> None:
    monitor_path = write_monitor_fixture(tmp_path)

    episodes = read_monitor_csv(monitor_path)
    summary = monitor_summary(episodes)

    assert len(episodes) == 3
    assert episodes[0].reward == 1.5
    assert episodes[1].length == 20
    assert summary["episode_count"] == 3
    assert summary["best_episode_length"] == 20
    assert summary["final_episode_reward"] == 2.5


def test_evaluation_report_serializes_expected_fields() -> None:
    random_stats = EvaluationStats(
        label="random",
        episode_lengths=[10, 14],
        episode_returns=[8.0, 9.0],
    )
    trained_stats = EvaluationStats(
        label="trained_ppo",
        episode_lengths=[500, 500],
        episode_returns=[499.0, 500.0],
    )

    report = build_evaluation_report(
        random_stats=random_stats,
        trained_stats=trained_stats,
        config_path="configs/single_link.yaml",
        model_path="models/single_link_ppo.zip",
        episodes=2,
        seed=1,
    )

    assert report["task"] == "single_link_stabilization"
    assert report["episodes"] == 2
    assert report["seed"] == 1
    assert report["policies"]["random"]["mean_episode_length"] == 12.0
    assert report["policies"]["trained_ppo"]["best_episode_length"] == 500


def test_reporting_cli_parsers_accept_milestone_1c_arguments() -> None:
    evaluate_args = build_evaluate_parser().parse_args(
        [
            "--config",
            "configs/single_link.yaml",
            "--model-path",
            "models/single_link_ppo.zip",
            "--episodes",
            "20",
            "--seed",
            "1",
            "--output",
            "runs/single_link_ppo/evaluation.json",
        ]
    )
    plot_args = build_plot_parser().parse_args(
        [
            "--run-dir",
            "runs/single_link_ppo",
            "--evaluation",
            "runs/single_link_ppo/evaluation.json",
        ]
    )

    assert evaluate_args.output == "runs/single_link_ppo/evaluation.json"
    assert plot_args.run_dir == "runs/single_link_ppo"
    assert plot_args.evaluation == "runs/single_link_ppo/evaluation.json"


def test_plot_results_writes_portfolio_artifacts(tmp_path: Path) -> None:
    monitor_path = write_monitor_fixture(tmp_path)
    evaluation_path = write_evaluation_fixture(tmp_path)

    training_curve_path, comparison_path, summary_path = plot_results(
        run_dir=str(tmp_path),
        evaluation=str(evaluation_path),
    )

    assert monitor_path.exists()
    assert training_curve_path.exists()
    assert comparison_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["training"]["best_episode_length"] == 20
    assert summary["evaluation"]["trained_ppo"]["mean_episode_length"] == 500.0


def test_build_run_summary_combines_training_and_evaluation(tmp_path: Path) -> None:
    monitor_path = write_monitor_fixture(tmp_path)
    evaluation_path = write_evaluation_fixture(tmp_path)

    summary = build_run_summary(
        monitor_csv=monitor_path,
        evaluation_json=evaluation_path,
    )

    assert summary["task"] == "single_link_stabilization"
    assert summary["training"]["episode_count"] == 3
    assert summary["evaluation"]["random"]["best_episode_length"] == 14
