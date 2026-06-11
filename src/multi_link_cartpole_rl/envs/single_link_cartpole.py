"""Single-link cartpole environment.

This is the first real learning environment in the project. It intentionally stays close to
the classic CartPole equations so the Gymnasium API and reinforcement learning loop are easy
to inspect.

State:
- cart position
- cart velocity
- pole angle, where 0 means upright
- pole angular velocity

Action:
- one continuous horizontal force applied to the cart
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, degrees, pi, sin
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SingleLinkCartPoleConfig:
    """Readable parameters for the first environment milestone."""

    num_links: int = 1
    max_episode_steps: int = 500
    force_limit: float = 10.0
    cart_position_limit: float = 2.4
    angle_limit_radians: float = 12.0 * 2.0 * pi / 360.0
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    time_step: float = 0.02
    initial_state_noise: float = 0.05


class SingleLinkCartPoleEnv(gym.Env):
    """A minimal continuous-action CartPole environment.

    This environment is deliberately small:
    - one cart
    - one pole
    - one continuous force action
    - simple Euler integration

    It is good enough for learning the environment API and running random actions. Later
    milestones can improve the physics, reward shaping, rendering, and training loop.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: SingleLinkCartPoleConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SingleLinkCartPoleConfig()
        self.render_mode = render_mode
        self.state: np.ndarray | None = None
        self.steps = 0
        self._fig: Any | None = None
        self._ax: Any | None = None
        self._canvas: Any | None = None
        self._plt: Any | None = None
        self._cart_patch: Any | None = None
        self._pole_line: Any | None = None
        self._pivot_dot: Any | None = None
        self._status_text: Any | None = None

        if self.config.num_links != 1:
            raise ValueError("SingleLinkCartPoleEnv requires num_links=1.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        observation_high = np.array(
            [
                self.config.cart_position_limit * 2.0,
                np.inf,
                self.config.angle_limit_radians * 2.0,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-self.config.force_limit], dtype=np.float32),
            high=np.array([self.config.force_limit], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the cartpole and return the first observation.

        Gymnasium expects reset to return `(observation, info)`. The optional `options`
        dictionary accepts a fixed `state` for tests and debugging.
        """
        super().reset(seed=seed)
        self.steps = 0

        if options is not None and "state" in options:
            state = np.asarray(options["state"], dtype=np.float32)
            if state.shape != (4,):
                raise ValueError("reset option 'state' must have shape (4,).")
            self.state = state
        else:
            low = -self.config.initial_state_noise
            high = self.config.initial_state_noise
            self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
                np.float32
            )

        return self._get_obs(), self._get_info(applied_force=0.0)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one time step.

        Gymnasium expects step to return:
        `(observation, reward, terminated, truncated, info)`.
        """
        if self.state is None:
            raise RuntimeError("Call reset before step.")

        force = self._action_to_force(action)
        x, x_dot, theta, theta_dot = self.state

        total_mass = self.config.cart_mass + self.config.pole_mass
        polemass_length = self.config.pole_mass * self.config.pole_half_length

        costheta = cos(float(theta))
        sintheta = sin(float(theta))

        temp = (force + polemass_length * float(theta_dot) ** 2 * sintheta) / total_mass
        theta_acc = (self.config.gravity * sintheta - costheta * temp) / (
            self.config.pole_half_length
            * (4.0 / 3.0 - self.config.pole_mass * costheta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass

        x = x + self.config.time_step * x_dot
        x_dot = x_dot + self.config.time_step * x_acc
        theta = theta + self.config.time_step * theta_dot
        theta_dot = theta_dot + self.config.time_step * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = self._is_out_of_bounds()
        truncated = self.steps >= self.config.max_episode_steps
        reward = self._compute_reward(force=force, terminated=terminated)
        info = self._get_info(applied_force=force)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the current cartpole state.

        `render_mode="human"` opens/updates a Matplotlib window.
        `render_mode="rgb_array"` returns an image array for tests and future video export.
        """
        if self.state is None:
            return None

        self._ensure_render_artists()
        self._draw_render_frame()

        if self.render_mode == "human":
            self._canvas.draw_idle()
            self._plt.pause(1.0 / self.metadata["render_fps"])
            return None

        self._canvas.draw()
        rgba = np.asarray(self._canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        """Release rendering resources when they exist."""
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._canvas = None
        self._plt = None
        self._cart_patch = None
        self._pole_line = None
        self._pivot_dot = None
        self._status_text = None

    def _action_to_force(self, action: Any) -> float:
        """Convert an action array into a clipped scalar cart force."""
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape not in {(1,), ()}:
            raise ValueError("SingleLinkCartPoleEnv action must be a scalar or shape (1,).")

        force = float(action_array.reshape(-1)[0])
        return float(np.clip(force, -self.config.force_limit, self.config.force_limit))

    def _is_out_of_bounds(self) -> bool:
        """Return whether the cart or pole has exceeded the allowed range."""
        return self._failure_reason() is not None

    def _failure_reason(self) -> str | None:
        """Explain why an episode has terminated, if it has."""
        if self.state is None:
            return None
        x, _, theta, _ = self.state
        if abs(float(x)) > self.config.cart_position_limit:
            return "cart_position_limit"
        if abs(float(theta)) > self.config.angle_limit_radians:
            return "pole_angle_limit"
        return None

    def _compute_reward(self, *, force: float, terminated: bool) -> float:
        """Reward upright, centered, low-effort behavior.

        This is intentionally readable reward shaping:
        - start with a survival reward
        - subtract a penalty for pole angle
        - subtract a smaller penalty for cart position
        - subtract a tiny penalty for large force
        """
        if self.state is None:
            return 0.0

        x, _, theta, _ = self.state
        angle_fraction = float(theta) / self.config.angle_limit_radians
        position_fraction = float(x) / self.config.cart_position_limit
        force_fraction = force / self.config.force_limit

        reward = 1.0
        reward -= 0.5 * angle_fraction**2
        reward -= 0.1 * position_fraction**2
        reward -= 0.01 * force_fraction**2

        if terminated:
            reward -= 1.0

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        """Return the current observation in the observation space dtype."""
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        return self.state.astype(np.float32)

    def _get_info(self, *, applied_force: float) -> dict[str, Any]:
        """Return debugging information that is useful while learning."""
        if self.state is None:
            return {}
        return {
            "step": self.steps,
            "applied_force": applied_force,
            "failure_reason": self._failure_reason(),
            "angle_limit_degrees": degrees(self.config.angle_limit_radians),
            "state_variables": [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ],
        }

    def _ensure_render_artists(self) -> None:
        """Create Matplotlib artists the first time graphical rendering is requested."""
        if self._fig is not None:
            return

        from matplotlib.patches import Rectangle

        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(8, 4.5))
            self._canvas = self._fig.canvas
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(8, 4.5), dpi=100)
            self._canvas = FigureCanvasAgg(self._fig)
            self._ax = self._fig.add_subplot(111)

        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlim(
            -self.config.cart_position_limit - 0.7,
            self.config.cart_position_limit + 0.7,
        )
        self._ax.set_ylim(-0.35, 1.45)
        self._ax.set_xlabel("cart position")
        self._ax.set_yticks([])
        self._ax.grid(axis="x", alpha=0.2)
        self._ax.set_title("Single-Link CartPole")

        self._ax.plot(
            [-self.config.cart_position_limit, self.config.cart_position_limit],
            [0.0, 0.0],
            color="black",
            linewidth=2,
        )
        self._cart_patch = Rectangle((-0.2, -0.1), 0.4, 0.2, color="#4c78a8")
        self._ax.add_patch(self._cart_patch)
        (self._pole_line,) = self._ax.plot([], [], color="#f58518", linewidth=4)
        (self._pivot_dot,) = self._ax.plot([], [], marker="o", color="#222222")
        self._status_text = self._ax.text(
            0.02,
            0.96,
            "",
            transform=self._ax.transAxes,
            verticalalignment="top",
            family="monospace",
        )

    def _draw_render_frame(self) -> None:
        """Update Matplotlib artists from the current state."""
        if self.state is None:
            return

        x, x_dot, theta, theta_dot = self.state
        cart_width = 0.4
        cart_height = 0.2
        pivot_y = cart_height / 2
        pole_length = 2.0 * self.config.pole_half_length

        self._cart_patch.set_xy((float(x) - cart_width / 2, -cart_height / 2))
        pole_x = float(x) + pole_length * sin(float(theta))
        pole_y = pivot_y + pole_length * cos(float(theta))
        self._pole_line.set_data([float(x), pole_x], [pivot_y, pole_y])
        self._pivot_dot.set_data([float(x)], [pivot_y])
        self._status_text.set_text(
            f"step={self.steps}\n"
            f"x={x:+.2f} x_dot={x_dot:+.2f}\n"
            f"theta={degrees(float(theta)):+.1f} deg "
            f"theta_dot={theta_dot:+.2f}\n"
            f"limit={degrees(self.config.angle_limit_radians):.1f} deg"
        )
