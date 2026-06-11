"""Evaluate random actions against a trained PPO policy."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from multi_link_cartpole_rl.envs.single_link_cartpole import SingleLinkCartPoleEnv
from multi_link_cartpole_rl.training.reporting import write_json_report
from multi_link_cartpole_rl.utils.experiment import (
    load_single_link_experiment_config,
    make_single_link_env,
)


class Policy(Protocol):
    """Callable policy interface used by the evaluation loop."""

    def __call__(self, observation: np.ndarray, env: SingleLinkCartPoleEnv) -> np.ndarray:
        """Return one action for the current observation."""
        ...


@dataclass(frozen=True)
class EvaluationStats:
    """Episode statistics for one policy."""

    label: str
    episode_lengths: list[int]
    episode_returns: list[float]

    @property
    def mean_episode_length(self) -> float:
        """Return the average episode length."""
        return float(np.mean(self.episode_lengths))

    @property
    def mean_return(self) -> float:
        """Return the average episode return."""
        return float(np.mean(self.episode_returns))

    @property
    def return_std(self) -> float:
        """Return the population standard deviation of episode returns."""
        return float(np.std(self.episode_returns))

    @property
    def best_episode_length(self) -> int:
        """Return the longest episode."""
        return max(self.episode_lengths)

    def to_dict(self) -> dict[str, object]:
        """Serialize summary and raw episode values for reports."""
        return {
            "label": self.label,
            "episode_lengths": self.episode_lengths,
            "episode_returns": self.episode_returns,
            "mean_episode_length": self.mean_episode_length,
            "mean_return": self.mean_return,
            "return_std": self.return_std,
            "best_episode_length": self.best_episode_length,
        }


def build_parser() -> ArgumentParser:
    """Create the command-line parser for future policy evaluation."""
    parser = ArgumentParser(description="Evaluate a trained cartpole policy.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a machine-readable JSON evaluation report.",
    )
    return parser


def run_policy_episodes(
    *,
    label: str,
    env: SingleLinkCartPoleEnv,
    policy: Policy,
    episodes: int,
    seed: int,
) -> EvaluationStats:
    """Run policy episodes and collect return and survival statistics."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    episode_lengths: list[int] = []
    episode_returns: list[float] = []
    env.action_space.seed(seed)

    for episode_index in range(episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy(observation, env)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        episode_lengths.append(env.steps)
        episode_returns.append(total_reward)

    return EvaluationStats(
        label=label,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
    )


def format_stats(stats: EvaluationStats) -> str:
    """Format evaluation statistics for the command-line report."""
    return (
        f"{stats.label}: "
        f"mean_episode_length={stats.mean_episode_length:.1f}, "
        f"mean_return={stats.mean_return:.2f}, "
        f"return_std={stats.return_std:.2f}, "
        f"best_episode_length={stats.best_episode_length}"
    )


def evaluate_policies(
    *,
    config_path: str,
    model_path: str | None,
    episodes: int | None,
    seed: int,
) -> tuple[EvaluationStats, EvaluationStats]:
    """Evaluate random actions and a trained PPO model."""
    from stable_baselines3 import PPO

    experiment_config = load_single_link_experiment_config(config_path)
    resolved_model_path = (
        Path(model_path) if model_path is not None else experiment_config.model_path
    )
    resolved_episodes = (
        episodes if episodes is not None else experiment_config.evaluation_episodes
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {resolved_model_path}. Train first with scripts/train.py."
        )

    random_env = make_single_link_env(experiment_config)
    model_env = make_single_link_env(experiment_config)
    model = PPO.load(resolved_model_path)

    def random_policy(_observation: np.ndarray, env: SingleLinkCartPoleEnv) -> np.ndarray:
        return env.action_space.sample()

    def trained_policy(observation: np.ndarray, _env: SingleLinkCartPoleEnv) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    try:
        random_stats = run_policy_episodes(
            label="random",
            env=random_env,
            policy=random_policy,
            episodes=resolved_episodes,
            seed=seed,
        )
        trained_stats = run_policy_episodes(
            label="trained_ppo",
            env=model_env,
            policy=trained_policy,
            episodes=resolved_episodes,
            seed=seed,
        )
    finally:
        random_env.close()
        model_env.close()

    return random_stats, trained_stats


def build_evaluation_report(
    *,
    random_stats: EvaluationStats,
    trained_stats: EvaluationStats,
    config_path: str,
    model_path: str | None,
    episodes: int | None,
    seed: int,
) -> dict[str, object]:
    """Build a machine-readable evaluation report."""
    experiment_config = load_single_link_experiment_config(config_path)
    resolved_model_path = (
        Path(model_path) if model_path is not None else experiment_config.model_path
    )
    resolved_episodes = (
        episodes if episodes is not None else experiment_config.evaluation_episodes
    )

    return {
        "task": "single_link_stabilization",
        "config_path": config_path,
        "model_path": str(resolved_model_path),
        "episodes": resolved_episodes,
        "seed": seed,
        "policies": {
            "random": random_stats.to_dict(),
            "trained_ppo": trained_stats.to_dict(),
        },
    }


def main() -> None:
    """Evaluation entry point."""
    args = build_parser().parse_args()
    random_stats, trained_stats = evaluate_policies(
        config_path=args.config,
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
    )
    print("Policy evaluation finished:")
    print(format_stats(random_stats))
    print(format_stats(trained_stats))

    if args.output is not None:
        report = build_evaluation_report(
            random_stats=random_stats,
            trained_stats=trained_stats,
            config_path=args.config,
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
        )
        output_path = write_json_report(args.output, report)
        print(f"Wrote evaluation report: {output_path}")


if __name__ == "__main__":
    main()
