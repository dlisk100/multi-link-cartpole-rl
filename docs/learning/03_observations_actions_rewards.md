# Observations, Actions, And Rewards

The agent only learns from what the environment gives it. Good observations, actions, and rewards make the task learnable.

## Single-Link Observations

For this repo's single-link cartpole, a practical observation is:

```text
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

This gives the agent enough information to estimate:

- where the cart is
- whether the cart is drifting
- where the pole is leaning
- whether the pole is falling or recovering

If any of these are missing, the agent may act like it is guessing.

## Why `sin(theta)` And `cos(theta)` Can Help Later

Angles wrap around. For example, `pi` and `-pi` are close physically, but far apart numerically.

Using `sin(theta)` and `cos(theta)` can represent angle direction without a sudden jump at the wrap boundary.

For early single-link cartpole near upright, raw `theta` is fine because the angle stays small. For larger swings or multi-link systems, sine/cosine observations may become more useful.

## Discrete vs Continuous Actions

Discrete actions are choices from a small menu.

Example:

```text
push left, do nothing, push right
```

Continuous actions are real-valued controls.

Example from this repo:

```text
force = -10.0 to +10.0
```

Continuous force is closer to a physical control input, but it can make training slightly more sensitive because the agent must learn both direction and magnitude.

## Reward Shaping

Reward shaping means adding helpful reward terms that point the agent toward the behavior you want.

For cartpole, a shaped reward might include:

- positive reward for surviving each step
- penalty for large pole angle
- smaller penalty for cart position far from center
- small penalty for excessive force

The goal is not to write the perfect physics objective. The goal is to give the agent a learning signal that is clear enough to improve.

## Common Bad Rewards

Bad rewards often make the agent learn the wrong thing.

Watch out for rewards that:

- give high reward even after obvious failure
- punish all movement so strongly that the agent becomes passive
- reward force magnitude instead of balance
- ignore cart position until it is too late
- are so tiny or huge that training curves are hard to interpret
- conflict with termination rules

## Why Reward Hacking Happens

Reward hacking happens when the agent finds a way to score well without doing what you intended.

The agent does not know your real goal. It only optimizes the reward numbers.

Example patterns:

- the agent learns to end episodes quickly if failure is not penalized
- the agent shakes the cart because movement accidentally gives reward
- the agent camps near a boundary if the reward ignores position
- the agent uses huge forces if effort is free

This is normal. When it happens, treat it as feedback that the reward or termination rules need clearer teaching signals.

## Practical Debugging Question

After watching an episode, ask:

"If I only saw the reward values, would I be able to tell that upright, centered balancing is better than the behavior I just watched?"

If the answer is no, improve the reward before blaming PPO.
