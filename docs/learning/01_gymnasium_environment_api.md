# Gymnasium Environment API

Gymnasium gives RL projects a standard way to describe an environment. Algorithms like PPO can train on your environment if it follows this interface.

## Minimal Pieces

A minimal environment needs:

- `reset()`
- `step(action)`
- `observation_space`
- `action_space`

Most useful environments also implement:

- `render()`
- `close()`
- an `info` dictionary with debugging details

## `reset()`

`reset()` starts a new episode.

It returns:

```text
(observation, info)
```

In cartpole, reset places the cart and pole near an initial state, often with small random noise. This prevents the agent from only memorizing one exact starting condition.

Use `reset(seed=...)` when you want repeatable behavior while debugging.

## `step(action)`

`step(action)` applies one action and advances the environment by one time step.

It returns:

```text
(observation, reward, terminated, truncated, info)
```

For this project, one step roughly means:

1. Convert the action into a cart force.
2. Update the cartpole physics.
3. Compute the reward.
4. Check whether the episode is over.
5. Return the new observation and debugging info.

## Spaces

`observation_space` describes valid observations.

For single-link cartpole, the observation has four numbers:

```text
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

`action_space` describes valid actions.

In this repo's single-link environment, the action space is continuous: one force value between negative and positive force limits.

## Reward

The reward is the learning signal. PPO does not know what "balance the pole" means. It only sees numbers.

A useful reward should roughly match the behavior you want:

- pole upright is good
- cart near center is good
- surviving longer is good
- excessive force may be mildly discouraged

## `terminated` vs `truncated`

These two flags both end an episode, but they mean different things.

`terminated` means the task ended because the environment reached a natural success/failure condition.

Cartpole example: the pole angle exceeded the allowed limit.

`truncated` means the episode was stopped because of an outside limit.

Cartpole example: the environment reached `max_episode_steps`.

This distinction helps training code understand whether the episode truly failed or just hit a time cap.

## Rendering

Rendering lets you inspect behavior visually.

Common modes:

- `human`: show a window for a person to watch
- `rgb_array`: return image pixels for tests, videos, or saved demos

Rendering is not required for learning the API, but it is extremely helpful for debugging physics and reward behavior.

## Mental Model

Gymnasium environments are like small physics labs with a standard control panel:

- `reset()` starts an experiment
- `step(action)` applies one control input
- spaces describe legal inputs and outputs
- reward says how well that step went
- done flags say why the experiment stopped
