# Gymnasium Car Racing (v3)

## Overview

**Car Racing** is a top-down racing environment in Gymnasium designed as a control task learned directly from image pixels. The track is randomly generated for each episode.

The environment belongs to Gymnasium's Box2D environments.

## Environment Setup

```python
import gymnasium as gym

env = gym.make("CarRacing-v3")
```

### Spaces

| Property | Specification |
|---|---|
| Action space (continuous) | `Box(-1.0, 1.0, (3,), float32)` with per-dimension bounds |
| Observation space | `Box(0, 255, (96, 96, 3), uint8)` |
| Observation shape | `(96, 96, 3)` |
| Observation data type | `uint8` |
| Observation format | RGB image |

## Action Space

The environment supports continuous and discrete actions.

### Continuous Actions

The default action space has three values:

| Index | Action | Range | Meaning |
|---|---|---|---|
| `0` | Steering | `-1` to `1` | `-1` = full left, `1` = full right |
| `1` | Gas | `0` to `1` | Controls acceleration |
| `2` | Brake | `0` to `1` | Controls braking |

Example:

```python
action = [-0.5, 0.7, 0.0]

observation, reward, terminated, truncated, info = env.step(action)
```

This steers left while applying gas and no brake.

### Discrete Actions

Set `continuous=False` to use five discrete actions.

```python
env = gym.make("CarRacing-v3", continuous=False)
```

| Action | Meaning |
|---|---|
| `0` | Do nothing |
| `1` | Steer left |
| `2` | Steer right |
| `3` | Gas |
| `4` | Brake |

This mode is suitable for a basic DQN agent because DQN selects from a finite set of actions.

## Observation Space

The observation is a **96 × 96 RGB image** containing a top-down view of the car and race track.

| Property | Value |
|---|---|
| Shape | `(96, 96, 3)` |
| Data type | `uint8` |
| Pixel range | `0–255` |
| Channels | Red, Green, Blue |

Example:

```python
observation, info = env.reset()

print(observation.shape)  # (96, 96, 3)
print(observation.dtype)  # uint8
```

### Does the Environment Provide Numerical Car State?

**No. The observation contains only the RGB image.**

The agent does not directly receive separate numerical values for:

- Car speed
- Steering angle
- Wheel sensor readings
- Car coordinates
- Track geometry

The environment's rendered window includes indicators such as speed, ABS sensors, steering wheel position, and gyroscope readings. These are visual elements in the rendered display, not separate numerical fields in the observation.

The Gymnasium API also returns other information, but that should not be confused with the observation:

| Output | Purpose |
|---|---|
| `observation` | RGB image used as the agent's input |
| `reward` | Feedback received after an action |
| `terminated` | Whether the episode ended naturally |
| `truncated` | Whether the episode stopped because of an external limit |
| `info` | Additional diagnostic information |

This makes CarRacing a **visual reinforcement learning task**.

## Reward Function

The environment provides:

- `-0.1` per frame.
- `+1000 / N` for each newly visited track tile, where `N` is the total number of tiles in the track.

For example, if the agent completes the track in 732 frames:

\[
R = 1000 - 0.1 \times 732 = 926.8
\]

The agent must balance two objectives:

1. Visit all required track tiles.
2. Minimize the number of frames needed to complete the track.

## Starting State

The car starts at rest in the center of the road.

The track is randomly generated, so the agent can encounter a different layout in each episode.

## Episode Termination

An episode can end when:

- The required track tiles have been visited.
- The car goes far outside the playfield, resulting in a penalty of `-100` and the car being considered dead.

An episode may also be truncated by an external limit, such as a time limit.

Always handle both termination flags:

```python
observation, reward, terminated, truncated, info = env.step(action)

done = terminated or truncated
```

## Environment Arguments

### `continuous`

Controls the action-space type.

```python
# Continuous actions
env = gym.make("CarRacing-v3", continuous=True)

# Discrete actions
env = gym.make("CarRacing-v3", continuous=False)
```

### `lap_complete_percent`

Specifies the percentage of track tiles that must be visited before a lap is considered complete.

### `domain_randomize`

Enables domain randomization, changing the background and track colors between resets.

```python
env = gym.make(
    "CarRacing-v3",
    domain_randomize=True,
)
```

This can help an agent learn visual features that generalize across different color schemes.

## Reset Arguments

When `domain_randomize=True`, the `randomize` reset option controls whether the color scheme changes during a reset.

```python
import gymnasium as gym

env = gym.make(
    "CarRacing-v3",
    domain_randomize=True,
)

# Reset normally
observation, info = env.reset()

# Randomize the color scheme
observation, info = env.reset(
    options={"randomize": True}
)

# Keep the current color scheme
observation, info = env.reset(
    options={"randomize": False}
)
```

The `randomize` option requires `domain_randomize=True` during environment initialization.

## Important Gameplay Considerations

The environment simulates a powerful rear-wheel-drive car.

- Avoid accelerating aggressively while turning.
- Excessive steering can cause the car to lose control.
- Leaving the track can result in a substantial penalty.
- Randomly generated tracks require the agent to generalize rather than memorize a fixed layout.

## Reinforcement Learning Considerations

### DQN Baseline

For a basic DQN implementation, use the discrete action space:

```python
env = gym.make(
    "CarRacing-v3",
    continuous=False,
)
```

The agent selects one of five actions at each time step.

Because the observation is an image, a CNN is generally more suitable than a small fully connected network for learning spatial features from raw pixels.

Potential preprocessing techniques include:

- Resizing images.
- Converting RGB images to grayscale.
- Normalizing pixel values.
- Stacking consecutive frames to provide temporal information.

These techniques are optional design choices, not requirements imposed by the environment.

### Continuous Control

For continuous control, the agent outputs three action values:

\[
a = [\text{steering}, \text{gas}, \text{brake}]
\]

Algorithms designed for continuous action spaces, such as **TD3** and **SAC**, are suitable candidates for later experiments.

### Experimentation Suggestions

- Use multiple training seeds.
- Evaluate using separate, fixed seeds.
- Record episode rewards, episode lengths, and completion rates.
- Save model checkpoints and experiment configurations.
- Compare learning curves across experiments.
- Keep training and evaluation procedures consistent.

## Version Notes

This reference targets **`CarRacing-v3`**. Gymnasium environment versions can differ in their API and implementation details. Check the documentation for the version installed in your environment.

## Reference

- [Gymnasium Car Racing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)
- Chris Campbell (2014): [Top-down car tutorial for Box2D](http://www.iforce2d.net/b2dtut/top-down-car)

## Credits

Created by **Oleg Klimov**.
