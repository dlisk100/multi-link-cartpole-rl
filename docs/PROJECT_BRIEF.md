# Project Brief

## What We Are Building

`multi-link-cartpole-rl` is a staged reinforcement learning and controls project. It starts
with a single-link cartpole, then gradually grows to two, three, and five linked pendulums.
The final proof-of-work artifact is a clean demo of an RL policy balancing a five-link
cartpole.

The project is also a learning lab. Each milestone should make one part of the RL/control
stack understandable: the Gymnasium API, state and action spaces, dynamics, reward shaping,
rendering, PPO training, evaluation, metrics, and experiment design.

## Why We Are Building It

Cartpole is a compact bridge between mechanical intuition and reinforcement learning. The
cart applies force, the pendulum dynamics respond, and the policy learns a control strategy
from rewards. Scaling to multiple links adds real difficulty while keeping the system easy
to reason about as a mechanical system.

The point is not to hide complexity behind a framework. The point is to build enough of the
loop by hand to understand what is happening, then use mature libraries where they help.

## What Counts As Success

Success means:

- A readable Python codebase with separated environment, training, rendering, configs, and
  tests.
- A single-link cartpole that can be stepped, rendered, trained, evaluated, and explained.
- A staged path from one link to two, three, and five links without losing clarity.
- Reproducible commands for random rollouts, training, evaluation, and final demo rendering.
- Learning notes and docs that help future implementation threads move without guessing.

The final public artifact should be a short demo and a concise explanation of what was
learned while building it.

## Out Of Scope For Now

- Jumping straight to a five-link environment before the single-link loop is solid.
- Large framework rewrites or abstract simulation engines before they are justified.
- Distributed training, GPU optimization, or high-throughput RL infrastructure.
- PufferLib or other advanced RL stacks before the core Gymnasium/PPO loop works.
- Private service integrations, API keys, remote training services, or secret handling.

