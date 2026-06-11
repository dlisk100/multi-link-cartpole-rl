# PPO In Plain English

PPO stands for Proximal Policy Optimization. You do not need every detail yet.

For this project, think of PPO as a practical algorithm that repeatedly:

1. Runs the current policy in the environment.
2. Records what happened.
3. Estimates which actions were better or worse than expected.
4. Updates the policy, but not too violently.
5. Repeats.

## Policy

The policy is the decision-maker.

It looks at an observation and chooses an action. In this repo, that means looking at cart/pole state and choosing a cart force.

Early in training, the policy is usually clumsy. Over time, PPO nudges it toward actions that led to better rewards.

## Value Function

The value function estimates how promising a state is.

Plain version: "From this situation, how much future reward do I expect?"

This helps PPO judge whether an action was surprisingly good or bad.

## Rollout

A rollout is a batch of experience collected by running the policy.

It contains many steps like:

```text
observation -> action -> reward -> next observation
```

PPO learns from rollouts instead of from one step at a time.

## Advantage

Advantage means "better or worse than expected."

If the value function expected a situation to go poorly, but the agent survived and earned good reward, that action may have positive advantage.

If the value function expected a situation to go well, but the pole fell, that action may have negative advantage.

This is useful because PPO cares about relative improvement, not just raw reward.

## Clipping

PPO updates the policy carefully. Clipping limits how big each policy update can be.

The intuition is:

- small learning steps are usually stable
- giant policy changes can destroy behavior that was starting to work

Clipping is one reason PPO is popular: it is often forgiving enough for first serious RL projects.

## Why PPO Is A Reasonable First Algorithm

PPO is a good first choice here because it:

- works with continuous actions
- is widely used and documented
- is available in Stable-Baselines3
- handles noisy trial-and-error learning reasonably well
- has enough knobs for experiments without requiring you to build an RL algorithm from scratch

It will not magically fix a broken environment or reward. But once the environment behaves sensibly, PPO is a solid first training tool.

## What Not To Worry About Yet

For Milestone 1A and the first PPO milestone, do not worry deeply about:

- deriving the PPO objective
- proving convergence
- tuning every hyperparameter
- comparing every RL algorithm
- building neural networks by hand

Focus first on:

- what the environment returns
- whether random/manual behavior makes sense
- whether rewards match visible behavior
- whether episode length improves during training

The algorithm matters, but the environment is the lesson at this stage.
