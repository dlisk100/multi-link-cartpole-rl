# Architecture

The repo should stay small, explicit, and pedagogical. Each area has a narrow job so future
milestones can improve one layer at a time.

## Intended Layout

```text
configs/
docs/
scripts/
src/multi_link_cartpole_rl/
  envs/
  rendering/
  training/
  utils/
tests/
```

## `envs/`

Owns Gymnasium-compatible environments and environment-specific configs.

Belongs here:

- `reset`, `step`, `render`, `close`, `observation_space`, and `action_space`.
- Cartpole dynamics and integration.
- Reward functions, termination rules, truncation rules, and info dictionaries.
- Single-link and future multi-link environment implementations.

Does not belong here:

- PPO training scripts.
- CLI argument parsing beyond tiny debug helpers.
- Plotting dashboards or final demo assembly.

## `training/`

Owns training and evaluation logic.

Belongs here:

- Creating environments from configs for training.
- Stable-Baselines3 PPO setup.
- Training loops, checkpoint saving, evaluation, and policy loading.
- Metrics emitted during training.

Does not belong here:

- Environment physics.
- Rendering-specific UI code.
- Hard-coded experiment constants that should live in configs.

## `rendering/`

Owns human inspection and demo rendering.

Belongs here:

- Random rollout visualization.
- Manual play or force-control tools.
- Trained policy rendering.
- Future video/GIF frame export.

Does not belong here:

- Training loops.
- Reward design.
- Environment state mutation outside normal `env.step(...)`.

## `utils/`

Owns small shared helpers when duplication becomes real.

Belongs here:

- Config loading.
- Experiment directory naming.
- Small validation helpers.

Does not belong here:

- Catch-all business logic.
- Core physics, reward, training, or rendering code.

## `configs/`

Owns YAML configuration for environments, training runs, rendering, and milestone settings.

Belongs here:

- Environment parameters such as link count, episode length, force limits, and difficulty.
- PPO hyperparameters once training begins.
- Evaluation and rendering settings.

Does not belong here:

- Secrets.
- Python code.
- Values that silently override CLI arguments without being documented.

## `scripts/`

Owns thin CLI entrypoints.

Belongs here:

- Argument parsing.
- Calling package functions from `envs/`, `training/`, or `rendering/`.
- Simple user-facing commands.

Does not belong here:

- Large implementation logic.
- Duplicated training or rendering code.

## `tests/`

Owns quick checks that protect the learning loop.

Belongs here:

- Import checks.
- Gymnasium API shape checks.
- Deterministic reset/step behavior where possible.
- Config loading checks.
- Fast smoke tests for scripts.

Does not belong here:

- Long PPO training jobs.
- Tests that require a display unless guarded or skipped.

