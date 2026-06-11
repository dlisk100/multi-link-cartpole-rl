# Decision Log

This is an Architecture Decision Record style log. Keep entries short and append new
decisions when project direction changes.

## ADR-0001: Use Python First

Date: 2026-06-11

Status: Accepted

Decision: Use Python as the primary implementation language.

Context: Gymnasium, Stable-Baselines3, MuJoCo, NumPy, and plotting tools are Python-native
or Python-friendly. The project is also learning-oriented, so readable scripts matter.

Consequences: The repo should optimize for clear Python modules and `uv run` workflows.

## ADR-0002: Use Gymnasium-Style Environments

Date: 2026-06-11

Status: Accepted

Decision: Implement environments using the Gymnasium API.

Context: Gymnasium defines the standard `reset` and `step` contract used by common RL
libraries.

Consequences: Tests should protect observation spaces, action spaces, `reset`, `step`,
termination, truncation, and `info`.

## ADR-0003: Start Single-Link Before Multi-Link

Date: 2026-06-11

Status: Accepted

Decision: Build and understand the single-link cartpole before implementing two, three, or
five links.

Context: Multi-link dynamics and training are much harder to debug if the core environment
and training loop are not already understood.

Consequences: Implementation threads should not leap to five-link work until earlier
milestones are validated.

## ADR-0004: Likely First Stack Is MuJoCo/Gymnasium/Stable-Baselines3

Date: 2026-06-11

Status: Accepted

Decision: Use Gymnasium-style environments and Stable-Baselines3 PPO as the likely first
training stack. MuJoCo remains a likely option for richer dynamics if the hand-built model
becomes limiting.

Context: The stack is well documented and appropriate for learning before optimizing
throughput.

Consequences: Keep training APIs compatible with Stable-Baselines3 and document any future
move toward MuJoCo-based modeling.

## ADR-0005: Defer PufferLib

Date: 2026-06-11

Status: Accepted

Decision: Defer PufferLib until the core learning loop works.

Context: PufferLib can be useful for throughput and scaling, but it adds another layer of
abstraction before the project needs it.

Consequences: Revisit only after single-link and early multi-link training are working and
the bottleneck is clearly environment throughput or parallel training.

## ADR-0006: Treat Six-Link Cartpole As A Stretch Target

Date: 2026-06-11

Status: Accepted

Decision: Keep five-link hard mode as the first major proof-of-work target, but explicitly
allow six links or the highest feasible link count as a later stretch target.

Context: The project owner is inspired by a June 2026 six-pendulum cartpole demo and wants
to work toward similarly high-link balancing while learning the RL process.

Consequences: Architecture should avoid dead ends that prevent high-link experiments later,
but implementation threads should still move through single-link, two-link, three-link, and
five-link milestones before attempting six-link work.
