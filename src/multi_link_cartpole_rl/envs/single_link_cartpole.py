"""Single-link cartpole environment placeholder.

This module is the planned starting point for the project. The future implementation should
follow the Gymnasium environment API, but the physics are intentionally not implemented yet.

Learning goals for this file:
- Understand what state the cartpole needs.
- Define observation and action spaces.
- Implement `reset` and `step`.
- Add a reward that encourages the pendulum to stay upright.
- Add simple rendering before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SingleLinkCartPoleConfig:
    """Readable parameters for the first environment milestone."""

    num_links: int = 1
    max_episode_steps: int = 500
    force_limit: float = 10.0
    cart_position_limit: float = 2.4


class SingleLinkCartPoleEnv:
    """Placeholder for a future Gymnasium-compatible single-link cartpole.

    The final version should likely subclass `gymnasium.Env`. For now, this lightweight
    placeholder keeps imports working and makes the intended API visible without pretending
    that the environment is ready to train.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: SingleLinkCartPoleConfig | None = None) -> None:
        self.config = config or SingleLinkCartPoleConfig()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:
        """Reset the environment.

        TODO:
        - Seed the random number generator.
        - Initialize cart position, cart velocity, pole angle, and pole angular velocity.
        - Return `(observation, info)`.
        """
        raise NotImplementedError("SingleLinkCartPoleEnv.reset is not implemented yet.")

    def step(self, action: Any) -> tuple:
        """Advance the simulation one step.

        TODO:
        - Clamp or validate the cart force action.
        - Integrate cartpole dynamics.
        - Compute reward.
        - Return `(observation, reward, terminated, truncated, info)`.
        """
        raise NotImplementedError("SingleLinkCartPoleEnv.step is not implemented yet.")

    def render(self) -> Any:
        """Render the environment state.

        TODO:
        - Start with Matplotlib for simple debugging.
        - Later add video-friendly RGB array rendering.
        """
        raise NotImplementedError("SingleLinkCartPoleEnv.render is not implemented yet.")

    def close(self) -> None:
        """Release rendering resources when they exist."""
