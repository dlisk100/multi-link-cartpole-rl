# Cartpole Physics Intuition

An inverted pendulum is unstable because gravity helps it fall away from upright.

If the pole leans slightly right, gravity creates a torque that makes it lean farther right. Upright balance is possible, but it requires active correction.

## What The Cart Can Control

The cart cannot directly grab the pole and rotate it.

It can only move left or right. That cart acceleration changes the pivot point under the pole. Good control means moving the base in a way that catches the falling pole and keeps it near vertical.

Plain version:

- pole falling right: often push/move right to get back under it
- pole falling left: often push/move left
- but if the cart is near a track limit, the best action may change

This is why the problem is more interesting than "push opposite the angle."

## Why Velocities Matter

Position tells you where the system is now.

Velocity tells you where it is going next.

For cartpole:

- `cart_position` tells whether the cart is near the center or track edge
- `cart_velocity` tells whether it is drifting toward danger
- `pole_angle` tells whether the pole is leaning
- `pole_angular_velocity` tells whether the pole is falling quickly or recovering

A pole that is slightly tilted but moving back upright is very different from a pole with the same angle falling rapidly away from upright.

## Underactuated In Plain English

A system is underactuated when it has fewer direct controls than moving parts you care about.

In single-link cartpole:

- things you care about: cart position, cart velocity, pole angle, pole angular velocity
- thing you directly control: one horizontal cart force

You do not control the pole angle directly. You influence it indirectly through cart motion.

That indirectness is the heart of the control problem.

## Why Multi-Link Gets Hard Quickly

Adding links adds more angles and angular velocities. Each link affects the others through the joints.

With one pole, the agent mostly learns to keep one unstable angle near upright.

With two or more links:

- there are more ways to fail
- motion in one link can disturb another
- useful actions depend on more history and timing
- rewards become harder to design
- small physics errors become more visible

The cart still has only one horizontal force input, but the number of moving pieces grows. That is why the project scales in stages instead of jumping straight to five links.

## What To Watch In The Simulator

When running random or manual actions, look for:

- Does the pole fall in a physically believable way?
- Does cart movement affect the pole direction?
- Do velocities grow smoothly, or do they explode instantly?
- Does the episode end for understandable reasons?

If the physics does not make visual sense, training will usually struggle no matter how good the RL algorithm is.
