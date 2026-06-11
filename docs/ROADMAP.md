# Roadmap

The roadmap is staged so each milestone teaches one layer of the system before the next
layer adds complexity. The practical proof-of-work target remains five links first. After
that, the project can attempt six links or the highest feasible link count if the earlier
milestones show that the environment, training loop, and experiment process are ready.

## Milestone 0: Repo Scaffold And Learning Plan

Set up the project structure, dependency management, basic package imports, docs, and a
shared plan for how Codex threads should work.

## Milestone 1A: Understand Gymnasium API With Single-Link Random Actions

Create or audit the single-link Gymnasium environment. Confirm `reset` and `step` return the
right shapes and metadata. Run random actions without graphical rendering.

## Milestone 1B: Train, Evaluate, And Render Single-Link PPO

Train a first Stable-Baselines3 PPO policy on the single-link stabilization task. Evaluate
it against random actions and render the trained policy for visual inspection.

## Milestone 1C: Portfolio-Ready Metrics And Reproducible Evidence

Package the solved single-link PPO result into JSON metrics and static plots. Keep generated
artifacts ignored, but make the improvement over random policy easy to reproduce.

## Milestone 1D: Single-Link Swing-Up

Add a downward-start single-link swing-up task. Train PPO to swing the pole upright and
stabilize it before moving the same idea toward multi-link swing-up. Track survival,
success rate, and max consecutive upright steps separately so long episodes do not get
mistaken for solved swing-up behavior.

## Milestone 2A: Clean Up Metrics, Logging, And Configs

Make training results easier to compare. Add clear run directories, metrics files, and
config-driven settings without turning the repo into an experiment platform.

## Milestone 2B: Generalize Toward A Configurable Multi-Link Environment

Design the shared representation needed for N-link cartpoles. Keep the single-link path
working while introducing reusable state/action/reward structure.

## Milestone 3A: Two-Link Cartpole

Implement two-link dynamics, observations, rewards, termination logic, and tests.

## Milestone 3B: Train Two-Link PPO

Train and evaluate PPO on the two-link environment. Compare failure modes against
single-link.

## Milestone 4A: Three-Link Cartpole

Extend the environment and rendering to three links. Keep the API consistent.

## Milestone 4B: Train Three-Link PPO

Train and evaluate PPO on three links. Improve rewards and initialization only as needed.

## Milestone 5A: Five-Link Easy Mode

Create a forgiving five-link setup with easier initial states, shorter episodes, and reward
terms that show progress.

## Milestone 5B: Five-Link Hard Mode

Increase difficulty toward an impressive final demo with stricter initialization, longer
episodes, and cleaner evaluation.

## Milestone 6: Experiment Sweeps And Final Demo Assets

Run targeted sweeps, select a final policy, and produce demo assets plus a short technical
writeup.

## Stretch Milestone 7: Six-Link Or Highest Feasible Cartpole

Attempt a six-link cartpole or another clearly justified high-link target. This milestone is
not active until five-link hard mode has a working evaluation and rendering path.

Likely work:

- Revisit environment performance and vectorization.
- Run broader hyperparameter and reward sweeps.
- Consider PufferLib or another higher-throughput training stack if SB3 wall-clock time is
  the limiting factor.
- Produce a demo only after evaluating across seeds, not from one lucky rollout.
