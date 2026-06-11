"""Render a trained PPO policy in a Matplotlib window."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from multi_link_cartpole_rl.utils.experiment import (
    load_single_link_experiment_config,
    make_single_link_env,
)


def build_parser() -> ArgumentParser:
    """Create the command-line parser for rendering a trained policy."""
    parser = ArgumentParser(description="Render a trained cartpole policy.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum playback steps. Defaults to the environment episode limit.",
    )
    return parser


def render_trained_policy(
    *,
    config_path: str,
    model_path: str | None,
    seed: int,
    steps: int | None = None,
) -> tuple[int, float, bool, bool]:
    """Load a PPO model and play one deterministic episode graphically."""
    from stable_baselines3 import PPO

    experiment_config = load_single_link_experiment_config(config_path)
    resolved_model_path = (
        Path(model_path) if model_path is not None else experiment_config.model_path
    )
    max_steps = steps if steps is not None else experiment_config.env_config.max_episode_steps

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {resolved_model_path}. Train first with scripts/train.py."
        )

    model = PPO.load(resolved_model_path)
    env = make_single_link_env(experiment_config, render_mode="human")
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    try:
        env.render()
        for _ in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32)
            )
            total_reward += reward
            if terminated or truncated:
                break
    finally:
        env.close()

    return env.steps, total_reward, terminated, truncated


def main() -> None:
    """Policy rendering entry point."""
    args = build_parser().parse_args()
    steps, total_reward, terminated, truncated = render_trained_policy(
        config_path=args.config,
        model_path=args.model_path,
        seed=args.seed,
        steps=args.steps,
    )
    print(
        "Policy playback finished: "
        f"steps={steps}, "
        f"total_reward={total_reward:.2f}, "
        f"terminated={terminated}, "
        f"truncated={truncated}"
    )


if __name__ == "__main__":
    main()
