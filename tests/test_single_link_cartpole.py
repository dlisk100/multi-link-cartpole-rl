import numpy as np

from multi_link_cartpole_rl.envs.single_link_cartpole import (
    SingleLinkCartPoleConfig,
    SingleLinkCartPoleEnv,
)


def test_reset_returns_gymnasium_tuple() -> None:
    env = SingleLinkCartPoleEnv()

    observation, info = env.reset(seed=123)

    assert observation.shape == (4,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["step"] == 0
    assert info["state_variables"] == [
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_angular_velocity",
    ]


def test_step_returns_gymnasium_step_tuple() -> None:
    env = SingleLinkCartPoleEnv()
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(np.array([0.0]))

    assert observation.shape == (4,)
    assert observation.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step"] == 1
    assert info["applied_force"] == 0.0
    assert info["failure_reason"] is None
    assert info["angle_limit_degrees"] == 12.0


def test_episode_truncates_at_max_steps() -> None:
    env = SingleLinkCartPoleEnv(config=SingleLinkCartPoleConfig(max_episode_steps=3))
    env.reset(seed=123, options={"state": np.array([0.0, 0.0, 0.0, 0.0])})

    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        assert not terminated
        assert not truncated

    _, _, terminated, truncated, info = env.step(np.array([0.0]))

    assert not terminated
    assert truncated
    assert info["step"] == 3


def test_action_force_is_clipped() -> None:
    env = SingleLinkCartPoleEnv(config=SingleLinkCartPoleConfig(force_limit=5.0))
    env.reset(seed=123, options={"state": np.array([0.0, 0.0, 0.0, 0.0])})

    _, _, _, _, info = env.step(np.array([100.0]))

    assert info["applied_force"] == 5.0


def test_rgb_array_render_returns_image() -> None:
    env = SingleLinkCartPoleEnv(render_mode="rgb_array")
    env.reset(seed=123)

    frame = env.render()

    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()
