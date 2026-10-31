# Experiment 003.c — PReLU Double DQN

**Date:** 2026-10-31  
**Experiment:** `dd_prelu`  
**Training seed:** 42  
**Training duration:** 1,000 episodes

### Objective

Evaluate Parametric ReLU (PReLU) as an alternative activation function for the existing Double DQN agent and compare its performance against the Leaky ReLU implementation.

### Hypothesis

PReLU's learnable negative slopes may allow the network to adapt its activation behavior to the features it learns, potentially improving driving performance and training stability compared with the fixed negative slope of Leaky ReLU.

### Configuration and implementation

Kept the existing Double DQN architecture and training configuration, changing the activation function to PReLU.

Used channel-wise PReLU in convolutional layers and feature-wise PReLU in the hidden fully connected layer:

- First convolution: 16 learnable slopes.
- Second convolution: 32 learnable slopes.
- Hidden fully connected layer: `hidden_dim` learnable slopes.
- Initial slope: `0.25` (PyTorch default).

### Evaluation

Evaluated the episode 1000 checkpoint across 50 fixed seeds (`10000–10049`), with greedy action selection and no recording.

| Metric | Result |
|---|---:|
| Mean reward | **575.70** |
| Standard deviation | 107.92 |
| Minimum reward | 272.09 |
| Maximum reward | 804.11 |
| Episodes reaching 250 steps | 50/50 |

### Comparison with Leaky ReLU

Compared against the `dd_leaky_relu` episode 1000 checkpoint using the same 50 evaluation seeds.

| Metric | PReLU | Leaky ReLU |
|---|---:|---:|
| Mean reward | 575.70 | **618.67** |
| Standard deviation | **107.92** | 159.81 |
| Minimum reward | **272.09** | 242.42 |
| Maximum reward | 804.11 | **865.65** |
| Episodes reaching 250 steps | 50/50 | 50/50 |

Leaky ReLU achieved a 42.97-point higher mean reward, approximately 7.46% above PReLU's mean. PReLU showed lower variation across evaluation seeds, while Leaky ReLU achieved a higher maximum reward.

### Findings

- PReLU successfully learned a capable driving policy within 1,000 episodes.
- Its learnable negative slopes did not produce a higher average evaluation reward than Leaky ReLU at the same training duration.
- PReLU had more consistent evaluation results, as indicated by its lower standard deviation.
- Both checkpoints completed all 50 evaluation episodes without early termination.

### Conclusion

PReLU did not demonstrate an improvement over Leaky ReLU in this 1,000-episode comparison. Although it produced lower variation across evaluation seeds, Leaky ReLU achieved a higher mean reward and maximum reward.

For the current CarRacing Double DQN setup, Leaky ReLU remains the preferred activation. The PReLU experiment is considered complete without extending training.