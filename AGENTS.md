# AGENTS.md

This is a learning-oriented reinforcement learning and controls project. The goal is to
understand the mechanics of custom Gymnasium environments, training loops, rendering, and
control intuition while progressing from a single-link cartpole toward a five-link demo.

## Working Style

- Prefer simple, readable, well-documented code over clever abstractions.
- Work milestone-by-milestone. Do not skip directly to five-link cartpole before the
  single-link environment is understood, tested, trained, and rendered.
- Keep changes scoped to the current task.
- Keep physics, training, rendering, configs, scripts, utilities, and tests separated.
- Avoid adding heavy dependencies unless the implementation thread explains why they are
  needed and what tradeoff they introduce.
- Do not handle, print, request, or commit private tokens, API keys, or secrets.

## Project Commands

This project uses `uv`. Prefer:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/render_random.py --config configs/single_link.yaml --steps 100 --seed 1
```

Run tests and lint when they are available and quick. If a check cannot be run, report why.

## Documentation

- Read `docs/PROJECT_BRIEF.md`, `docs/ROADMAP.md`, and `docs/MILESTONES.md` before
  implementing milestone work.
- Update docs when architecture, milestone definitions, commands, or project assumptions
  change.
- Implementation threads should finish with a concise summary of changed files and checks.

## Architecture Guardrails

- `envs/`: environment dynamics, Gymnasium API, reward, termination, observation/action spaces.
- `training/`: model training and evaluation logic.
- `rendering/`: visual inspection, manual play, videos, and policy rendering.
- `configs/`: YAML experiment and environment settings.
- `scripts/`: thin command-line entrypoints.
- `utils/`: small shared helpers only when duplication is real.
- `tests/`: behavior checks for environment API, physics invariants, configs, and scripts.

