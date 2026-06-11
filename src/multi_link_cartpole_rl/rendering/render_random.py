"""Render random actions placeholder.

Random-action rendering is useful before training because it verifies that the environment
dynamics and visualization are understandable.
"""

from __future__ import annotations

from argparse import ArgumentParser

from multi_link_cartpole_rl.envs.single_link_cartpole import (
    SingleLinkCartPoleConfig,
    SingleLinkCartPoleEnv,
)
from multi_link_cartpole_rl.utils.config import load_yaml_config


def build_parser() -> ArgumentParser:
    """Create the command-line parser for random-action runs."""
    parser = ArgumentParser(description="Render random actions in a cartpole environment.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a graphical Matplotlib window while random actions run.",
    )
    return parser


def build_single_link_config(config_path: str) -> SingleLinkCartPoleConfig:
    """Load the YAML fields currently used by the single-link environment."""
    data = load_yaml_config(config_path)
    if data.get("num_links") != 1:
        raise NotImplementedError("Milestone 1A only supports num_links: 1.")

    return SingleLinkCartPoleConfig(
        max_episode_steps=int(data.get("max_episode_steps", 500)),
        force_limit=float(data.get("force_limit", 10.0)),
    )


def main() -> None:
    """Run one random-action episode in the single-link cartpole environment."""
    args = build_parser().parse_args()
    config = build_single_link_config(args.config)
    env = SingleLinkCartPoleEnv(
        config=config,
        render_mode="human" if args.render else None,
    )

    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(args.steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print(
        "Random episode finished: "
        f"steps={info['step']}, "
        f"total_reward={total_reward:.2f}, "
        f"terminated={terminated}, "
        f"truncated={truncated}, "
        f"failure_reason={info['failure_reason']}, "
        f"final_observation={observation.tolist()}"
    )


if __name__ == "__main__":
    main()
