from math import cos, pi, sin

import numpy as np

from multi_link_cartpole_rl.envs.single_link_cartpole import (
    SingleLinkCartPoleConfig,
    SingleLinkCartPoleEnv,
    wrap_angle_radians,
)
from multi_link_cartpole_rl.training.evaluate_policy import (
    EvaluationStats,
    build_evaluation_report,
)


def make_swingup_env(**overrides: object) -> SingleLinkCartPoleEnv:
    """Create a swing-up environment for tests."""
    config = SingleLinkCartPoleConfig(task="swing_up", **overrides)
    return SingleLinkCartPoleEnv(config=config)


def test_swingup_reset_starts_near_downward() -> None:
    env = make_swingup_env(
        initial_state_noise=0.0,
        swingup_initial_angle_noise=0.01,
        swingup_initial_velocity_noise=0.0,
    )

    observation, info = env.reset(seed=123)

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert abs(wrap_angle_radians(float(env.state[2]) - pi)) <= 0.01
    assert abs(observation[2] - sin(float(env.state[2]))) < 1e-6
    assert abs(observation[3] - cos(float(env.state[2]))) < 1e-6
    assert info["task"] == "swing_up"
    assert info["state_variables"] == [
        "cart_position",
        "cart_velocity",
        "sin_pole_angle",
        "cos_pole_angle",
        "pole_angular_velocity",
    ]
    env.close()


def test_swingup_reset_angle_center_is_configurable() -> None:
    env = make_swingup_env(
        initial_state_noise=0.0,
        swingup_initial_angle_center=0.0,
        swingup_initial_angle_noise=0.01,
        swingup_initial_velocity_noise=0.0,
    )

    _, info = env.reset(seed=123)

    assert abs(info["angle_error_radians"]) <= 0.01
    assert info["is_upright"]
    env.close()


def test_swingup_does_not_terminate_on_pole_angle() -> None:
    env = make_swingup_env()
    env.reset(seed=123, options={"state": np.array([0.0, 0.0, pi, 0.0])})

    _, _, terminated, truncated, info = env.step(np.array([0.0]))

    assert not terminated
    assert not truncated
    assert info["failure_reason"] is None
    env.close()


def test_swingup_terminates_on_cart_position() -> None:
    env = make_swingup_env()
    env.reset(seed=123, options={"state": np.array([2.5, 0.0, pi, 0.0])})

    _, _, terminated, _, info = env.step(np.array([0.0]))

    assert terminated
    assert info["failure_reason"] == "cart_position_limit"
    env.close()


def test_wrapped_angle_error_handles_pi_boundary() -> None:
    assert abs(wrap_angle_radians(pi + 0.1) - (-pi + 0.1)) < 1e-12
    assert abs(wrap_angle_radians(-pi - 0.1) - (pi - 0.1)) < 1e-12
    assert abs(wrap_angle_radians(2 * pi)) < 1e-12


def test_swingup_reward_is_higher_near_upright_than_downward() -> None:
    env = make_swingup_env()

    env.reset(seed=123, options={"state": np.array([0.0, 0.0, pi, 0.0])})
    downward_reward = env._compute_reward(force=0.0, terminated=False)

    env.reset(seed=123, options={"state": np.array([0.0, 0.0, 0.0, 0.0])})
    upright_reward = env._compute_reward(force=0.0, terminated=False)

    assert upright_reward > downward_reward
    assert env._get_info(applied_force=0.0)["is_upright"]
    env.close()


def test_swingup_evaluation_report_includes_success_metrics() -> None:
    random_stats = EvaluationStats(
        label="random",
        episode_lengths=[100, 120],
        episode_returns=[10.0, 12.0],
        swingup_successes=[False, False],
        max_consecutive_upright_steps=[0, 2],
    )
    trained_stats = EvaluationStats(
        label="trained_ppo",
        episode_lengths=[1000, 1000],
        episode_returns=[3500.0, 3600.0],
        swingup_successes=[True, True],
        max_consecutive_upright_steps=[320, 340],
    )

    report = build_evaluation_report(
        random_stats=random_stats,
        trained_stats=trained_stats,
        config_path="configs/single_link_swingup.yaml",
        model_path="models/single_link_swingup_ppo.zip",
        episodes=2,
        seed=1,
    )

    assert report["task"] == "swing_up"
    trained_payload = report["policies"]["trained_ppo"]
    assert trained_payload["swingup_success_rate"] == 1.0
    assert trained_payload["mean_max_consecutive_upright_steps"] == 330.0
