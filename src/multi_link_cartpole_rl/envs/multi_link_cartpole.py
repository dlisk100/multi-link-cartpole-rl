"""N-link cartpole environment placeholder.

This module will eventually generalize the single-link cartpole environment to two, three,
and five linked pendulums. It should stay simple until the single-link implementation is
well understood and tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultiLinkCartPoleConfig:
    """Shared parameters for future multi-link environments."""

    num_links: int = 2
    max_episode_steps: int = 500
    force_limit: float = 10.0


class MultiLinkCartPoleEnv:
    """Placeholder for a future Gymnasium-compatible N-link cartpole.

    Planned progression:
    - Reuse ideas from `SingleLinkCartPoleEnv`.
    - Represent observations consistently for any number of links.
    - Keep reward terms interpretable.
    - Add difficulty settings for five-link easy and hard modes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: MultiLinkCartPoleConfig | None = None) -> None:
        self.config = config or MultiLinkCartPoleConfig()
        if self.config.num_links < 2:
            raise ValueError("MultiLinkCartPoleEnv is intended for two or more links.")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:
        """Reset the future N-link environment."""
        raise NotImplementedError("MultiLinkCartPoleEnv.reset is not implemented yet.")

    def step(self, action: Any) -> tuple:
        """Advance the future N-link simulation one step."""
        raise NotImplementedError("MultiLinkCartPoleEnv.step is not implemented yet.")

    def render(self) -> Any:
        """Render the future N-link environment."""
        raise NotImplementedError("MultiLinkCartPoleEnv.render is not implemented yet.")

    def close(self) -> None:
        """Release rendering resources when they exist."""
