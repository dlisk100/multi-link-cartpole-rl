"""Train PPO on the single-link stabilization environment."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from multi_link_cartpole_rl.utils.experiment import (
    load_single_link_experiment_config,
    make_single_link_env,
    ppo_kwargs,
)


def build_parser() -> ArgumentParser:
    """Create the command-line parser for PPO training runs."""
    parser = ArgumentParser(description="Train PPO on a staged cartpole environment.")
    parser.add_argument(
        "--config",
        default="configs/single_link.yaml",
        help="Path to a YAML training config.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total PPO timesteps from the config.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the saved model path from the config.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override the Monitor log directory from the config.",
    )
    return parser


def train_ppo(
    *,
    config_path: str,
    seed: int,
    total_timesteps: int | None = None,
    model_path: str | None = None,
    run_dir: str | None = None,
) -> Path:
    """Train PPO and return the saved model path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    experiment_config = load_single_link_experiment_config(config_path)
    resolved_model_path = (
        Path(model_path) if model_path is not None else experiment_config.model_path
    )
    resolved_run_dir = Path(run_dir) if run_dir is not None else experiment_config.run_dir
    resolved_timesteps = (
        total_timesteps if total_timesteps is not None else experiment_config.total_timesteps
    )

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    env = make_single_link_env(experiment_config)
    env.action_space.seed(seed)
    monitored_env = Monitor(env, filename=str(resolved_run_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        monitored_env,
        seed=seed,
        verbose=1,
        **ppo_kwargs(experiment_config),
    )
    model.learn(total_timesteps=resolved_timesteps)
    model.save(resolved_model_path)
    monitored_env.close()

    return resolved_model_path


def main() -> None:
    """Training entry point."""
    args = build_parser().parse_args()
    saved_model_path = train_ppo(
        config_path=args.config,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        model_path=args.model_path,
        run_dir=args.run_dir,
    )
    print(
        "PPO training finished: "
        f"model_path={saved_model_path}, "
        f"config={args.config}, "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
