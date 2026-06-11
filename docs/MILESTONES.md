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

## Milestone 1B: Train, Evaluate, And Render Single-Link PPO

Goal: Train a first PPO policy on the single-link stabilization task, compare it with
random actions, and render the trained behavior.

Concept learned: How an RL library wraps an environment, samples rollouts, optimizes a
policy, saves checkpoints, evaluates deterministically, and supports visual inspection.

Files likely touched: `src/multi_link_cartpole_rl/training/`,
`src/multi_link_cartpole_rl/rendering/`, `src/multi_link_cartpole_rl/utils/experiment.py`,
`configs/single_link.yaml`, `tests/`, `README.md`.

Done means: PPO training saves a checkpoint, evaluation shows trained PPO surviving longer
than random actions, and graphical playback shows stable control.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/train.py --config configs/single_link.yaml --seed 1
uv run python scripts/evaluate.py --config configs/single_link.yaml --model-path models/single_link_ppo.zip --episodes 20 --seed 1
uv run python scripts/render_policy.py --config configs/single_link.yaml --model-path models/single_link_ppo.zip --seed 1
```

Risks / traps: Overinterpreting one lucky episode; requiring committed local checkpoints in
tests; changing task difficulty while judging PPO quality.

## Milestone 1C: Portfolio-Ready Metrics And Reproducible Evidence

Goal: Package the solved single-link PPO result into reproducible metrics and plots.

Concept learned: Experiment evidence, run artifacts, machine-readable metrics, and
portfolio-friendly result communication.

Files likely touched: `src/multi_link_cartpole_rl/training/`, `scripts/plot_results.py`,
`tests/`, `README.md`.

Done means: Evaluation can write a JSON report, training/evaluation plots can be generated
from ignored run artifacts, and README documents the exact reproduction commands.

How to validate:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/evaluate.py --config configs/single_link.yaml --model-path models/single_link_ppo.zip --episodes 20 --seed 1 --output runs/single_link_ppo/evaluation.json
uv run python scripts/plot_results.py --run-dir runs/single_link_ppo --evaluation runs/single_link_ppo/evaluation.json
```

Risks / traps: Building a full experiment tracker; committing large generated artifacts;
making portfolio plots before the metrics are reproducible.

## Historical Milestone 1D: Folded Into 1B

Evaluation and graphical playback were completed as part of Milestone 1B. Keep this note so
older thread plans that mention 1D can be mapped to the current repo state.

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
