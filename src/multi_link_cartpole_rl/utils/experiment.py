"""Shared experiment configuration helpers for single-link milestones."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

from multi_link_cartpole_rl.envs.single_link_cartpole import (
    SingleLinkCartPoleConfig,
    SingleLinkCartPoleEnv,
)
from multi_link_cartpole_rl.utils.config import load_yaml_config


@dataclass(frozen=True)
class SingleLinkExperimentConfig:
    """YAML-backed settings shared by training, evaluation, and rendering."""

    env_config: SingleLinkCartPoleConfig
    learning_rate: float = 0.0003
    total_timesteps: int = 100_000
    model_path: Path = Path("models/single_link_ppo.zip")
    run_dir: Path = Path("runs/single_link_ppo")
    evaluation_episodes: int = 20
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99


def load_single_link_experiment_config(config_path: str | Path) -> SingleLinkExperimentConfig:
    """Load single-link environment and PPO settings from YAML."""
    data = load_yaml_config(config_path)
    if data.get("num_links") != 1:
        raise NotImplementedError("Milestone 1B only supports num_links: 1.")

    default_env = SingleLinkCartPoleConfig()
    env_config = SingleLinkCartPoleConfig(
        num_links=int(data.get("num_links", default_env.num_links)),
        max_episode_steps=int(data.get("max_episode_steps", default_env.max_episode_steps)),
        force_limit=float(data.get("force_limit", default_env.force_limit)),
        cart_position_limit=float(
            data.get("cart_position_limit", default_env.cart_position_limit)
        ),
        angle_limit_radians=float(
            data.get("angle_limit_radians", default_env.angle_limit_radians)
        ),
        gravity=float(data.get("gravity", default_env.gravity)),
        cart_mass=float(data.get("cart_mass", default_env.cart_mass)),
        pole_mass=float(data.get("pole_mass", default_env.pole_mass)),
        pole_half_length=float(data.get("pole_half_length", default_env.pole_half_length)),
        time_step=float(data.get("time_step", default_env.time_step)),
        initial_state_noise=float(
            data.get("initial_state_noise", default_env.initial_state_noise)
        ),
    )

    return SingleLinkExperimentConfig(
        env_config=env_config,
        learning_rate=float(data.get("learning_rate", 0.0003)),
        total_timesteps=int(data.get("total_timesteps", 100_000)),
        model_path=Path(data.get("model_path", "models/single_link_ppo.zip")),
        run_dir=Path(data.get("run_dir", "runs/single_link_ppo")),
        evaluation_episodes=int(data.get("evaluation_episodes", 20)),
        ppo_n_steps=int(data.get("ppo_n_steps", 256)),
        ppo_batch_size=int(data.get("ppo_batch_size", 64)),
        ppo_n_epochs=int(data.get("ppo_n_epochs", 10)),
        ppo_gamma=float(data.get("ppo_gamma", 0.99)),
    )


def make_single_link_env(
    experiment_config: SingleLinkExperimentConfig,
    *,
    render_mode: str | None = None,
) -> SingleLinkCartPoleEnv:
    """Create the configured single-link environment."""
    return SingleLinkCartPoleEnv(
        config=experiment_config.env_config,
        render_mode=render_mode,
    )


def ppo_kwargs(experiment_config: SingleLinkExperimentConfig) -> dict[str, Any]:
    """Return PPO constructor keyword arguments from the shared config."""
    return {
        "learning_rate": experiment_config.learning_rate,
        "n_steps": experiment_config.ppo_n_steps,
        "batch_size": experiment_config.ppo_batch_size,
        "n_epochs": experiment_config.ppo_n_epochs,
        "gamma": experiment_config.ppo_gamma,
    }


def default_angle_limit_radians() -> float:
    """Expose the classic CartPole angle limit for tests and docs."""
    return 12.0 * 2.0 * pi / 360.0
