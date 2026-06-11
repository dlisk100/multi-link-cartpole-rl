# Multi-Link CartPole RL

This is a learning-focused reinforcement learning and controls project. The goal is to
build a custom cartpole environment, train an RL agent to balance one pendulum, then
progressively increase the difficulty to two, three, and eventually five linked pendulums.
The project is inspired by viral multi-pendulum cartpole demos, but the purpose is learning
custom environment design, physics modeling, reward shaping, PPO training, experiment
tracking, and rendering.

## Why this project exists

CartPole is a classic reinforcement learning problem because it is simple enough to
understand but rich enough to teach core ideas: state, action, reward, termination, control,
and stability. This project starts there, then grows toward a harder multi-link balancing
problem one milestone at a time.

The codebase is intentionally small and readable. Early files are placeholders with clear
TODOs so the environment can be built deliberately instead of hiding the learning inside a
large framework.

## Learning progression

- [ ] Understand the Gymnasium environment API
- [ ] Run random actions in a single-link cartpole
- [ ] Train PPO on single-link cartpole
- [ ] Render a trained single-link policy
- [ ] Add logging for reward and episode length
- [ ] Generalize the environment to multiple links
- [ ] Train two-link cartpole
- [ ] Train three-link cartpole
- [ ] Train five-link easy mode
- [ ] Run hyperparameter/reward sweeps
- [ ] Render final five-link demo

## Milestone 1: single-link cartpole

Implement a custom Gymnasium-compatible environment for the standard cart and one pendulum.
The first goal is not to beat any benchmark. The first goal is to understand the environment
API, basic dynamics, reward shaping, and how PPO interacts with the environment.

## Milestone 2: two-link cartpole

Extend the system to two linked pendulums. This milestone should reveal what must change in
the observation space, action handling, reward function, termination logic, and rendering.

## Milestone 3: three-link cartpole

Increase the difficulty again and make the code more general. By this point, the project
should have reusable environment parameters, cleaner configs, and useful training logs.

## Milestone 4: five-link easy mode

Train a five-link version with forgiving settings: shorter episodes, gentler initial states,
limited randomization, and reward terms that make early progress visible.

## Milestone 5: five-link hard mode

Move toward a more impressive demo with stricter initial conditions, longer episodes, and
less forgiving rewards. This is where experiment tracking and rendering become especially
important.

## Environment design notes

The initial environment files are placeholders. The intended direction is:

- Use the Gymnasium API: `reset`, `step`, `render`, `close`, `observation_space`, and
  `action_space`.
- Start with a simple force applied to the cart as the action.
- Keep state variables explicit and documented.
- Add physics carefully, with tests for shapes, bounds, and termination behavior.
- Generalize only after the single-link version is understandable.

## Training approach

Training will begin with PPO from Stable-Baselines3 because it is widely used, documented,
and suitable for continuous-control learning experiments. The first training script will
load a YAML config, create the environment, train PPO, and save checkpoints.

## Rendering/demo plan

Rendering will start with random actions so the environment can be visually inspected before
training. Later milestones will render trained policies and save videos or GIFs for demos.

## Current status

Initial scaffold only. The project imports cleanly, but the cartpole physics and Gymnasium
environment are intentionally not implemented yet.
