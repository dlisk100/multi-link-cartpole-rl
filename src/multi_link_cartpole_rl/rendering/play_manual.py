"""Manual keyboard control demo for the single-link cartpole.

This is a learning tool, not a training loop. It lets you feel the control problem directly:
small delayed pushes are usually better than slamming the cart at full force.
"""

from __future__ import annotations

from argparse import ArgumentParser
from math import radians

import numpy as np

from multi_link_cartpole_rl.envs.single_link_cartpole import (
    SingleLinkCartPoleConfig,
    SingleLinkCartPoleEnv,
)
from multi_link_cartpole_rl.utils.config import load_yaml_config


def build_parser() -> ArgumentParser:
    """Create the command-line parser for manual control."""
    parser = ArgumentParser(description="Manually control the single-link cartpole.")
    parser.add_argument("--config", default="configs/single_link.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--difficulty",
        choices=["learning", "classic"],
        default="learning",
        help="Use learning mode for human practice or classic mode for the strict environment.",
    )
    return parser


def build_single_link_config(config_path: str, difficulty: str) -> SingleLinkCartPoleConfig:
    """Load the YAML fields currently used by the single-link environment."""
    data = load_yaml_config(config_path)
    if data.get("num_links") != 1:
        raise NotImplementedError("Manual control currently supports num_links: 1.")

    max_episode_steps = int(data.get("max_episode_steps", 500))
    angle_limit_radians = SingleLinkCartPoleConfig().angle_limit_radians
    initial_state_noise = SingleLinkCartPoleConfig().initial_state_noise

    if difficulty == "learning":
        max_episode_steps = max(max_episode_steps, 1000)
        angle_limit_radians = radians(45.0)
        initial_state_noise = 0.01

    return SingleLinkCartPoleConfig(
        max_episode_steps=max_episode_steps,
        force_limit=float(data.get("force_limit", 10.0)),
        angle_limit_radians=angle_limit_radians,
        initial_state_noise=initial_state_noise,
    )


class ManualCartPoleViewer:
    """Matplotlib viewer with keyboard control."""

    def __init__(self, env: SingleLinkCartPoleEnv, seed: int = 0) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.env = env
        self.seed = seed
        self.episode = 1
        self.force = 0.0
        self.total_reward = 0.0
        self.done = False
        self.observation, _ = self.env.reset(seed=self.seed)
        self.last_failure_reason: str | None = None

        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(
            -self.env.config.cart_position_limit - 0.7,
            self.env.config.cart_position_limit + 0.7,
        )
        self.ax.set_ylim(-0.35, 1.45)
        self.ax.set_xlabel("cart position")
        self.ax.set_yticks([])
        self.ax.grid(axis="x", alpha=0.2)

        (self.track_line,) = self.ax.plot(
            [-self.env.config.cart_position_limit, self.env.config.cart_position_limit],
            [0.0, 0.0],
            color="black",
            linewidth=2,
        )
        self.cart = Rectangle((-0.2, -0.1), 0.4, 0.2, color="#4c78a8")
        self.ax.add_patch(self.cart)
        (self.pole_line,) = self.ax.plot([], [], color="#f58518", linewidth=4)
        (self.pivot_dot,) = self.ax.plot([], [], marker="o", color="#222222")
        self.status_text = self.ax.text(
            0.02,
            0.96,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            family="monospace",
        )
        self.help_text = self.ax.text(
            0.02,
            0.04,
            "Hold Left/A or Right/D to push. Space releases force. R resets. Q quits.",
            transform=self.ax.transAxes,
            family="monospace",
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        interval_ms = int(1000 / self.env.metadata["render_fps"])
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.update)
        self.timer.start()
        self.draw()

    def on_key_press(self, event: object) -> None:
        """Update force or viewer state when a key is pressed."""
        key = getattr(event, "key", None)
        if key in {"left", "a"}:
            self.force = -self.env.config.force_limit
        elif key in {"right", "d"}:
            self.force = self.env.config.force_limit
        elif key == " ":
            self.force = 0.0
        elif key == "r":
            self.reset_episode()
        elif key == "q":
            self.plt.close(self.fig)

    def on_key_release(self, event: object) -> None:
        """Release the cart force when the steering key is released."""
        key = getattr(event, "key", None)
        if key in {"left", "right", "a", "d"}:
            self.force = 0.0

    def reset_episode(self) -> None:
        """Start a fresh manual-control episode."""
        self.episode += 1
        self.force = 0.0
        self.total_reward = 0.0
        self.done = False
        self.observation, _ = self.env.reset(seed=self.seed + self.episode)
        self.draw()

    def update(self) -> bool:
        """Advance the simulation and redraw the viewer."""
        if not self.done:
            action = np.array([self.force], dtype=np.float32)
            self.observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.done = terminated or truncated
            self.last_failure_reason = info["failure_reason"]

        self.draw()
        return True

    def draw(self) -> None:
        """Draw the cart, pole, and status text."""
        x, x_dot, theta, theta_dot = self.observation
        cart_width = 0.4
        cart_height = 0.2
        pivot_y = cart_height / 2
        pole_length = 2.0 * self.env.config.pole_half_length

        self.cart.set_xy((float(x) - cart_width / 2, -cart_height / 2))
        pole_x = float(x) + pole_length * np.sin(float(theta))
        pole_y = pivot_y + pole_length * np.cos(float(theta))
        self.pole_line.set_data([float(x), pole_x], [pivot_y, pole_y])
        self.pivot_dot.set_data([float(x)], [pivot_y])

        outcome = "running"
        if self.done:
            outcome = self.last_failure_reason or "time_limit"
            outcome = f"{outcome}; press R"

        self.status_text.set_text(
            f"episode={self.episode} step={self.env.steps} {outcome}\n"
            f"force={self.force:+.1f} total_reward={self.total_reward:+.1f}\n"
            f"x={x:+.2f} x_dot={x_dot:+.2f} theta={theta:+.2f} theta_dot={theta_dot:+.2f}\n"
            f"angle_limit={np.degrees(self.env.config.angle_limit_radians):.1f} deg"
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the Matplotlib window."""
        self.plt.show()


def main() -> None:
    """Open the manual-control viewer."""
    args = build_parser().parse_args()
    config = build_single_link_config(args.config, difficulty=args.difficulty)
    env = SingleLinkCartPoleEnv(config=config)
    viewer = ManualCartPoleViewer(env=env, seed=args.seed)
    viewer.show()
    env.close()


if __name__ == "__main__":
    main()
