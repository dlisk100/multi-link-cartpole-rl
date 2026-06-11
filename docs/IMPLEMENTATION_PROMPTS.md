# Implementation Prompts

Copy one prompt into a fresh implementation thread. Each thread should stay scoped to the
requested milestone and should not overbuild.

## Milestone 1A Prompt

```text
You are an implementation thread for /Users/davidlisk/Development/multi-link-cartpole-rl.

Implement or finish Milestone 1A: understand the Gymnasium API using a single-link cartpole
with random actions.

Before editing, read:
- AGENTS.md
- docs/PROJECT_BRIEF.md
- docs/ROADMAP.md
- docs/MILESTONES.md
- docs/ARCHITECTURE.md

Scope:
- Inspect the current single-link environment and random-action script.
- Ensure the single-link environment has a clear Gymnasium-compatible reset/step API.
- Ensure observation_space and action_space match returned observations/actions.
- Keep dynamics simple and readable; this is a learning milestone, not a perfect physics
  simulator.
- Add or update focused tests for reset, step, action clipping or validation, termination,
  truncation, and imports.
- Add or update a simple random rollout command if needed.
- Update README/docs only for commands or behavior that actually exist.

Do not:
- Implement two-link, three-link, or five-link dynamics.
- Add a large abstraction layer for future N-link support.
- Add heavy dependencies unless you explain why they are necessary.
- Start PPO training work except for preserving existing imports if present.

Validate with:
- uv run pytest
- uv run ruff check .
- uv run python scripts/render_random.py --config configs/single_link.yaml --steps 100 --seed 1

End with:
- Summary of changed files.
- Checks run.
- Any risks or follow-up notes for Milestone 1B.
```

## Milestone 1B Prompt

```text
You are an implementation thread for /Users/davidlisk/Development/multi-link-cartpole-rl.

Implement or finish Milestone 1B: render random single-link cartpole behavior.

Before editing, read:
- AGENTS.md
- docs/PROJECT_BRIEF.md
- docs/ROADMAP.md
- docs/MILESTONES.md
- docs/ARCHITECTURE.md

Scope:
- Inspect the current single-link environment and rendering code.
- Add or complete simple rendering for random single-link rollouts.
- Support human visual inspection and an rgb_array path that can be tested without requiring
  an interactive display.
- Keep rendering separate from training logic.
- Add or update focused tests for render output where practical.
- Update README/docs for the actual render command.

Do not:
- Rewrite the environment architecture.
- Add policy training.
- Implement multi-link rendering.
- Spend time on final-demo polish.

Validate with:
- uv run pytest
- uv run ruff check .
- uv run python scripts/render_random.py --config configs/single_link.yaml --steps 100 --seed 1 --render

End with:
- Summary of changed files.
- Checks run.
- Any risks or follow-up notes for Milestone 1C.
```

## Milestone 1C Prompt

```text
You are an implementation thread for /Users/davidlisk/Development/multi-link-cartpole-rl.

Implement or finish Milestone 1C: train PPO on the single-link cartpole.

Before editing, read:
- AGENTS.md
- docs/PROJECT_BRIEF.md
- docs/ROADMAP.md
- docs/MILESTONES.md
- docs/ARCHITECTURE.md

Scope:
- Inspect the existing training code, configs, and scripts.
- Add or complete a minimal Stable-Baselines3 PPO training path for the single-link
  environment.
- Load settings from config where the repo already has config support.
- Save a model checkpoint in a clear run directory.
- Keep default or test training runs short enough for smoke validation.
- Add focused tests for config parsing or training setup where practical without running a
  long training job.
- Update README/docs with the real command.

Do not:
- Tune PPO extensively.
- Add experiment sweep infrastructure.
- Implement multi-link training.
- Require long training as part of the normal test suite.

Validate with:
- uv run pytest
- uv run ruff check .
- uv run python scripts/train.py --config configs/single_link.yaml --total-timesteps 1000

End with:
- Summary of changed files.
- Checks run.
- Path to any generated checkpoint, if one was created.
- Any risks or follow-up notes for Milestone 1D.
```

## Milestone 1D Prompt

```text
You are an implementation thread for /Users/davidlisk/Development/multi-link-cartpole-rl.

Implement or finish Milestone 1D: evaluate and render a trained single-link policy.

Before editing, read:
- AGENTS.md
- docs/PROJECT_BRIEF.md
- docs/ROADMAP.md
- docs/MILESTONES.md
- docs/ARCHITECTURE.md

Scope:
- Inspect the existing evaluation, rendering, and training artifact paths.
- Add or complete a deterministic evaluation command for a saved single-link PPO model.
- Add or complete a policy rendering command that loads a checkpoint and renders behavior.
- Keep evaluation separate from training.
- Add focused tests for argument/config handling where practical without requiring a large
  committed model file.
- Update README/docs with the actual commands.

Do not:
- Retrain inside evaluation.
- Commit large model artifacts.
- Implement multi-link evaluation.
- Add final-demo export polish yet.

Validate with:
- uv run pytest
- uv run ruff check .
- uv run python scripts/evaluate.py --config configs/single_link.yaml --model-path <checkpoint>
- uv run python scripts/render_policy.py --config configs/single_link.yaml --model-path <checkpoint>

End with:
- Summary of changed files.
- Checks run.
- Any risks or follow-up notes for Milestone 2A.
```

