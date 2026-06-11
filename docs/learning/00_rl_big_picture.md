# RL Big Picture

Reinforcement learning is about learning by interaction.

Instead of giving the program the correct action for every situation, we give it:

- a world it can interact with
- observations about that world
- actions it is allowed to take
- rewards that say whether behavior was useful

The agent tries things, sees what happens, and gradually changes its behavior to get more reward.

## Core Words

**Agent**: The learner. In this project, the agent will eventually be a PPO policy deciding how much force to apply to the cart.

**Environment**: The simulated world. Here, it is the cart, pole, physics update, reward function, and episode rules.

**Observation**: What the agent gets to see at each step. For single-link cartpole, the current observation is:

```text
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

**Action**: What the agent can do. In this repo's single-link environment, the action is one continuous value: horizontal force on the cart.

**Reward**: A number returned after each action. Higher should mean "that was better." The current reward mostly encourages staying alive, keeping the pole upright, staying near the center, and not using excessive force.

**Episode**: One attempt. It starts with `reset()` and ends when the pole/cart goes out of bounds or the maximum step count is reached.

**Policy**: The agent's decision rule. It maps observations to actions. Early on, a random policy samples actions blindly. During training, PPO adjusts the policy so actions become more useful.

**Training loop**: Repeated interaction:

1. Reset the environment.
2. Observe the state.
3. Choose an action.
4. Step the environment.
5. Receive the next observation, reward, and done flags.
6. Use that experience to improve the policy.
7. Repeat many times.

## Why Cartpole Is A Good First Task

Cartpole is simple enough to understand physically, but rich enough to teach the main RL/control ideas:

- the system is unstable, so doing nothing fails
- actions have delayed effects, so timing matters
- velocity matters, not just position
- rewards must describe the behavior you want
- termination rules strongly affect learning

From a controls perspective, this is a classic balancing problem. From an RL perspective, it is a small world where you can see the whole loop before scaling up to two, three, or five links.

Before Milestone 1A feels natural, you should be able to say: "The agent sees the cartpole state, chooses a force, the environment advances physics, returns a reward, and the episode continues or ends."
