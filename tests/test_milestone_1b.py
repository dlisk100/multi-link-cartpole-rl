from pathlib import Path

import numpy as np

from multi_link_cartpole_rl.rendering.render_policy import build_parser as build_render_parser
from multi_link_cartpole_rl.training.evaluate_policy import (
    build_parser as build_evaluate_parser,
)
from multi_link_cartpole_rl.training.evaluate_policy import evaluate_policies, run_policy_episodes
from multi_link_cartpole_rl.training.train_ppo import build_parser as build_train_parser
from multi_link_cartpole_rl.training.train_ppo import train_ppo
from multi_link_cartpole_rl.utils.experiment import (
    load_single_link_experiment_config,
    make_single_link_env,
)


def write_test_config(tmp_path: Path) -> Path:
    """Create a tiny single-link config for fast tests."""
    config_path = tmp_path / "single_link_test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env_name: SingleLinkCartPole",
                "num_links: 1",
                "max_episode_steps: 20",
                "force_limit: 7.5",
                "learning_rate: 0.0003",
                "total_timesteps: 8",
                f"model_path: {tmp_path / 'models' / 'test_ppo.zip'}",
                f"run_dir: {tmp_path / 'runs' / 'test_ppo'}",
                "evaluation_episodes: 2",
                "ppo_n_steps: 8",
                "ppo_batch_size: 4",
                "ppo_n_epochs: 1",
                "ppo_gamma: 0.98",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_single_link_experiment_config_builds_env(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)

    experiment_config = load_single_link_experiment_config(config_path)
    env = make_single_link_env(experiment_config)
    observation, info = env.reset(seed=1)

    assert experiment_config.env_config.max_episode_steps == 20
    assert experiment_config.env_config.force_limit == 7.5
    assert experiment_config.total_timesteps == 8
    assert experiment_config.model_path == tmp_path / "models" / "test_ppo.zip"
    assert env.observation_space.contains(observation)
    assert info["angle_limit_degrees"] == 12.0
    env.close()


def test_cli_parsers_accept_milestone_1b_arguments() -> None:
    train_args = build_train_parser().parse_args(
        [
            "--config",
            "configs/single_link.yaml",
            "--seed",
            "1",
            "--total-timesteps",
            "8",
            "--model-path",
            "models/test.zip",
            "--run-dir",
            "runs/test",
        ]
    )
    evaluate_args = build_evaluate_parser().parse_args(
        [
            "--config",
            "configs/single_link.yaml",
            "--model-path",
            "models/test.zip",
            "--episodes",
            "3",
            "--seed",
            "2",
        ]
    )
    render_args = build_render_parser().parse_args(
        [
            "--config",
            "configs/single_link.yaml",
            "--model-path",
            "models/test.zip",
            "--seed",
            "3",
            "--steps",
            "4",
        ]
    )

    assert train_args.seed == 1
    assert train_args.total_timesteps == 8
    assert evaluate_args.episodes == 3
    assert evaluate_args.seed == 2
    assert render_args.steps == 4
    assert render_args.seed == 3


def test_run_policy_episodes_collects_statistics(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    experiment_config = load_single_link_experiment_config(config_path)
    env = make_single_link_env(experiment_config)

    def zero_policy(_observation: np.ndarray, env) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    stats = run_policy_episodes(
        label="zero",
        env=env,
        policy=zero_policy,
        episodes=2,
        seed=1,
    )

    assert stats.label == "zero"
    assert len(stats.episode_lengths) == 2
    assert len(stats.episode_returns) == 2
    assert stats.best_episode_length <= 20
    env.close()


def test_ppo_smoke_train_save_load_and_evaluate(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)

    saved_model_path = train_ppo(config_path=str(config_path), seed=1)
    random_stats, trained_stats = evaluate_policies(
        config_path=str(config_path),
        model_path=str(saved_model_path),
        episodes=1,
        seed=1,
    )

    assert saved_model_path.exists()
    assert random_stats.label == "random"
    assert trained_stats.label == "trained_ppo"
    assert len(random_stats.episode_lengths) == 1
    assert len(trained_stats.episode_lengths) == 1
