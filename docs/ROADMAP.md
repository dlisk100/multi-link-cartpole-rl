# Roadmap

The roadmap is staged so each milestone teaches one layer of the system before the next
layer adds complexity.

## Milestone 0: Repo Scaffold And Learning Plan

Set up the project structure, dependency management, basic package imports, docs, and a
shared plan for how Codex threads should work.

## Milestone 1A: Understand Gymnasium API With Single-Link Random Actions

Create or audit the single-link Gymnasium environment. Confirm `reset` and `step` return the
right shapes and metadata. Run random actions without graphical rendering.

## Milestone 1B: Render Random Single-Link Behavior

Add simple rendering for the single-link environment so the dynamics can be inspected by
eye before training.

## Milestone 1C: Train PPO On Single-Link Cartpole

Train a first Stable-Baselines3 PPO policy on the single-link environment. Keep the training
loop minimal and readable.

## Milestone 1D: Evaluate And Render Trained Single-Link Policy

Load a trained checkpoint, evaluate it deterministically, and render policy behavior for
inspection.

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

