# Training Debugging Checklist

Use this when training looks confusing. Most RL bugs are easier to find by checking the simple things first.

## Environment Sanity Checks

- Can you call `reset()` and get an observation with the expected shape?
- Can you call `step(action)` many times without crashing?
- Do `terminated` and `truncated` become true for understandable reasons?
- Does `info["failure_reason"]` match what you saw?
- Does rendering match the numeric state?

## Random Policy Behavior

- Random actions should usually fail.
- Failure should look physically plausible.
- Episode lengths should vary with the seed.
- Rewards should be lower when the pole falls quickly.

If random actions balance well for a long time, the task may be too easy or the physics may be wrong.

## Reward Scale

- Are rewards usually in a readable range?
- Is good behavior clearly rewarded more than bad behavior?
- Is failure penalized enough to matter?
- Are force penalties small enough that the agent is still willing to move?

For early cartpole, simple rewards around step-sized values like `1.0` are easier to reason about than extremely tiny or huge numbers.

## Episode Length

Track episode length separately from reward.

For cartpole, longer episodes often mean the agent is learning, even before the motion looks graceful.

Watch for:

- episode length stuck near the minimum
- episode length improving, then collapsing
- many episodes ending by time limit
- many episodes ending for the same failure reason

## Observation Values

Print or log a few observations.

Check:

- shape is correct
- dtype is numeric, usually float32
- values are finite
- angle units are what you expect
- velocities are not exploding instantly
- observations match what rendering shows

If observations are wrong, PPO is learning from bad sensor data.

## Action Limits

Check:

- actions are inside `action_space`
- environment clips actions safely if needed
- force signs match left/right behavior
- max force is strong enough to influence the pole
- max force is not so strong that one action launches the system unrealistically

For continuous control, action scaling problems are very common.

## Training Curves

Useful curves:

- mean episode reward
- mean episode length
- failure reason counts
- value loss and policy loss if available

Do not expect a perfectly smooth line. RL curves are noisy. Look for trends over many episodes.

## Signs The Agent Is Learning

- average episode length increases
- pole stays near upright longer
- failures happen later
- cart movement becomes less random
- actions respond to pole angle and angular velocity
- trained policy visibly outperforms random actions

Learning may appear as "less bad" before it appears as "good."

## Signs The Environment Or Reward Is Broken

- random policy gets high reward for obvious failure
- trained policy learns to fail immediately
- rewards do not change when behavior changes
- observations contain `nan` or huge values
- actions have no visible effect
- termination happens immediately from normal reset states
- rendering and numeric state disagree

When these happen, pause training and debug the environment. More PPO timesteps usually will not fix a broken task definition.

## First Debugging Habit

Before changing hyperparameters, run one short episode and inspect:

```text
observation, action, reward, terminated, truncated, info
```

RL feels less mysterious when you can explain one step at a time.
