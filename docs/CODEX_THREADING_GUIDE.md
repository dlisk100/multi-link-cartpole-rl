# Codex Threading Guide

Use Codex threads deliberately so the project keeps its learning progression.

## Thread Roles

Architecture thread:

- Owns roadmap, architecture guidance, milestone definitions, and decision log updates.
- Does not implement major milestones.
- Produces focused prompts for future implementation threads.

Tutor thread:

- Owns explanations, learning notes, controls intuition, and RL concept walkthroughs.
- Can read code and docs, but should avoid editing implementation files unless explicitly
  asked.

Implementation threads:

- Own one milestone at a time.
- Read `AGENTS.md`, `docs/ROADMAP.md`, `docs/MILESTONES.md`, and the relevant prompt in
  `docs/IMPLEMENTATION_PROMPTS.md` before editing.
- Make scoped code, tests, and docs changes for that milestone.
- End with checks run and a summary of changed files.

## Collaboration Rules

- Avoid parallel edits to the same files.
- Prefer separate branches or worktrees for isolated implementation work.
- Keep each implementation thread small enough to review.
- Do not move ahead to later milestones just because the next abstraction looks tempting.
- When architecture or milestone definitions change, update the docs in the architecture
  thread.

## Implementation Thread Checklist

1. Read the repo guidance files.
2. Inspect current code before assuming milestone status.
3. Implement only the requested milestone.
4. Add or update focused tests.
5. Update README/docs only where behavior or commands changed.
6. Run quick checks:

```bash
uv run pytest
uv run ruff check .
```

7. Report changed files, checks, and any follow-up risks.

