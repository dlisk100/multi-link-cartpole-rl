# Milestones

Use this file to scope implementation threads. A milestone is done only when the code,
docs, and checks for that milestone are complete.

## Milestone 0: Repo Scaffold And Learning Plan

Goal: Establish the repo structure, dependency workflow, and project guidance.

Concept learned: How a small Python RL project is organized before implementation starts.

Files likely touched: `README.md`, `AGENTS.md`, `docs/`, `pyproject.toml`, `tests/`.

Done means: The package imports, the docs define the staged plan, and future threads know
which files to read first.

How to validate: Run `uv run pytest` and `uv run ruff check .` if configured.

Risks / traps: Writing a large architecture manifesto instead of practical guidance; adding
dependencies before a milestone needs them.

## Milestone 1A: Understand Gymnasium API With Single-Link Random Actions

Goal: Build or audit a minimal single-link cartpole environment that supports random
actions without graphical rendering.

Concept learned: Gymnasium `Env`, `reset`, `step`, observation spaces, action spaces,
termination, truncation, rewards, and `info`.

Files likely touched: `src/multi_link_cartpole_rl/envs/single_link_cartpole.py`,
`src/multi_link_cartpole_rl/envs/__init__.py`, `configs/single_link.yaml`,
`scripts/render_random.py`, `tests/test_single_link_cartpole.py`, `README.md`.

Done means: A random rollout can reset and step through the environment. Observations and
actions match declared spaces. Tests cover the basic API and failure conditions.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/render_random.py --config configs/single_link.yaml --steps 100 --seed 1
```

Risks / traps: Overbuilding multi-link abstractions too early; hiding dynamics in unclear
helpers; confusing terminated with truncated.

## Milestone 1B: Render Random Single-Link Behavior

Goal: Render random single-link behavior so physics can be visually inspected.

Concept learned: Visual debugging, coordinate transforms, render modes, and why seeing
dynamics matters before training.

Files likely touched: `src/multi_link_cartpole_rl/envs/single_link_cartpole.py`,
`src/multi_link_cartpole_rl/rendering/render_random.py`, `scripts/render_random.py`,
`tests/test_single_link_cartpole.py`, `README.md`.

Done means: `render_mode="human"` displays a live view and `render_mode="rgb_array"`
returns image arrays suitable for tests and future video export.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/render_random.py --config configs/single_link.yaml --steps 100 --seed 1 --render
```

Risks / traps: Making rendering drive environment state directly; requiring a display for
all tests; polishing visuals before the physics are understandable.

## Milestone 1C: Train PPO On Single-Link Cartpole

Goal: Train a first PPO policy on the single-link environment.

Concept learned: How an RL library wraps an environment, samples rollouts, optimizes a
policy, logs progress, and saves checkpoints.

Files likely touched: `src/multi_link_cartpole_rl/training/train_ppo.py`,
`scripts/train.py`, `configs/single_link.yaml`, `src/multi_link_cartpole_rl/utils/config.py`,
`tests/`, `README.md`.

Done means: A short training run starts from config, trains without crashing, saves a model
checkpoint, and documents the command used.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/train.py --config configs/single_link.yaml --total-timesteps 1000
```

Risks / traps: Treating first training quality as the main milestone; adding many logging
tools before a checkpoint can be trained; long default training runs.

## Milestone 1D: Evaluate And Render Trained Single-Link Policy

Goal: Load a trained single-link checkpoint, evaluate it, and render policy behavior.

Concept learned: Separating training from evaluation, deterministic rollouts, and visual
inspection of learned control.

Files likely touched: `src/multi_link_cartpole_rl/training/evaluate_policy.py`,
`src/multi_link_cartpole_rl/rendering/render_policy.py`, `scripts/evaluate.py`,
`scripts/render_policy.py`, `README.md`, `tests/`.

Done means: A saved model can be loaded, evaluated for one or more episodes, and rendered
without retraining.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/evaluate.py --config configs/single_link.yaml --model-path <checkpoint>
uv run python scripts/render_policy.py --config configs/single_link.yaml --model-path <checkpoint>
```

Risks / traps: Mixing evaluation into training scripts; requiring a specific local
checkpoint path in tests; overinterpreting one lucky episode.

## Milestone 2A: Clean Up Metrics, Logging, And Configs

Goal: Make runs easier to reproduce and compare.

Concept learned: Experiment hygiene, config boundaries, metrics, and run directories.

Files likely touched: `configs/`, `src/multi_link_cartpole_rl/utils/experiment.py`,
`src/multi_link_cartpole_rl/training/`, `tests/`, `README.md`.

Done means: Training and evaluation write clear run artifacts with config snapshots and
basic metrics.

How to validate: Run tests, lint, a tiny training run, and inspect the generated run
directory.

Risks / traps: Building a full experiment tracker; making configs too magical; breaking
simple CLI overrides.

## Milestone 2B: Generalize Toward A Configurable Multi-Link Environment

Goal: Prepare shared structure for N-link cartpoles while keeping the single-link path
stable.

Concept learned: Generalizing state representation and dynamics without losing clarity.

Files likely touched: `src/multi_link_cartpole_rl/envs/`, `configs/`, `tests/`,
`docs/ARCHITECTURE.md`.

Done means: The intended N-link observation/action/config shape is explicit, documented,
and tested enough to support the two-link milestone.

How to validate: Run tests and a single-link random rollout to confirm no regression.

Risks / traps: Abstracting before the two-link needs are concrete; breaking single-link
training; hiding physics behind clever indexing.

## Milestone 3A: Two-Link Cartpole

Goal: Implement two-link environment behavior.

Concept learned: Coupled pendulum dynamics, larger state spaces, and multi-link reward
design.

Files likely touched: `src/multi_link_cartpole_rl/envs/multi_link_cartpole.py`,
`configs/two_link.yaml`, `tests/`, `rendering/`.

Done means: Two-link reset/step/render works with tests and a random rollout command.

How to validate: Run tests, lint, and a two-link random rollout.

Risks / traps: Copy-pasting single-link physics incorrectly; unclear angle conventions;
reward terms that mask obvious failures.

## Milestone 3B: Train Two-Link PPO

Goal: Train PPO on the two-link environment.

Concept learned: Difficulty scaling, initialization sensitivity, and reward debugging.

Files likely touched: `training/`, `configs/two_link.yaml`, `README.md`, `tests/`.

Done means: A short two-link training run saves a checkpoint and evaluation can load it.

How to validate: Run tests, lint, tiny train, and evaluate the checkpoint.

Risks / traps: Assuming hyperparameters from single-link will transfer cleanly; making
episodes too hard too early.

## Milestone 4A: Three-Link Cartpole

Goal: Extend the environment and rendering to three links.

Concept learned: How control difficulty grows with additional coupled degrees of freedom.

Files likely touched: `envs/`, `rendering/`, `configs/three_link.yaml`, `tests/`.

Done means: Three-link random rollouts run and render with consistent observations.

How to validate: Run tests, lint, and a three-link random rollout.

Risks / traps: Losing track of state ordering; making rendering visually misleading.

## Milestone 4B: Train Three-Link PPO

Goal: Train PPO on three links.

Concept learned: Curriculum, reward shaping, and experiment comparison.

Files likely touched: `training/`, `configs/three_link.yaml`, `utils/`, `README.md`.

Done means: A checkpoint can be trained, evaluated, and compared with two-link runs.

How to validate: Run tests, lint, tiny train, and evaluate.

Risks / traps: Chasing long training runs before basic signals improve; changing too many
reward terms at once.

## Milestone 5A: Five-Link Easy Mode

Goal: Create a forgiving five-link setup that can show progress.

Concept learned: Curriculum design and reducing task difficulty without changing the goal.

Files likely touched: `envs/`, `configs/five_link_easy.yaml`, `training/`, `rendering/`,
`tests/`.

Done means: Five-link easy random/evaluation commands work, and training has a plausible
path to improvement.

How to validate: Run tests, lint, random rollout, tiny training smoke, and evaluation.

Risks / traps: Making easy mode so forgiving it no longer represents five-link balancing;
overfitting to one seed.

## Milestone 5B: Five-Link Hard Mode

Goal: Move toward the final impressive five-link demo.

Concept learned: Robustness, evaluation discipline, and final control performance.

Files likely touched: `configs/five_link_hard.yaml`, `training/`, `rendering/`, `README.md`.

Done means: A hard-mode policy can be trained or selected, evaluated across seeds, and
rendered for demo review.

How to validate: Run tests, lint, evaluation across multiple seeds, and render the chosen
policy.

Risks / traps: Optimizing for one lucky video; skipping evaluation; changing hard mode
after selecting a policy.

## Milestone 6: Experiment Sweeps And Final Demo Assets

Goal: Produce final comparison results and demo assets for the current highest reliable
link count, expected to be five links before any stretch work.

Concept learned: Experiment reporting and communicating control/RL results clearly.

Files likely touched: `scripts/`, `training/`, `rendering/`, `docs/`, `README.md`.

Done means: The repo has final commands, selected artifacts, a demo render, and concise
notes explaining what worked.

How to validate: Re-run the documented final commands from a clean checkout or fresh
environment where practical.

Risks / traps: Running broad sweeps without a question; keeping large generated artifacts
in git; overstating results.

## Stretch Milestone 7: Six-Link Or Highest Feasible Cartpole

Goal: Push beyond five links if the project has earned that complexity.

Concept learned: Scaling RL experiments, throughput, curriculum design, and how training
infrastructure choices become important only after the baseline loop works.

Files likely touched: `envs/`, `training/`, `rendering/`, `configs/`, `scripts/`, `docs/`.

Done means: The chosen high-link target has a documented environment config, training path,
evaluation path, and demo render. The result is evaluated across seeds rather than selected
from one lucky rollout.

How to validate: Run tests and lint, then run the documented training/evaluation/rendering
commands for the selected high-link target. For expensive runs, record exact commands,
hardware assumptions, seeds, wall-clock time, and artifact paths.

Risks / traps: Switching to a faster RL stack before knowing the bottleneck; treating a
viral demo as the minimum acceptable outcome; doing broad sweeps without enough metrics to
learn from failures.
