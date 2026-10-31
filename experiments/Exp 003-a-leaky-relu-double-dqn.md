# Experiment 003.a — Leaky ReLU Double DQN

**Experiment:** `dd_leaky_relu`  
**Date:** 2026-10-30  
**Training seed:** 42  
**Status:** Completed

## Objective

Investigate whether replacing ReLU with Leaky ReLU improves the learning behavior and performance of the Double DQN agent in the Gymnasium CarRacing environment.

## Hypothesis

Leaky ReLU may work better than standard ReLU for this agent because its non-zero gradient for negative inputs could help preserve gradient flow and improve learning. The expectation was that this would lead to more stable learning, better driving behavior, and higher evaluation rewards in this particular CarRacing setup.

## Configuration

The experiment retained the existing Double DQN setup and changed the activation function to Leaky ReLU.

- **Activation:** Leaky ReLU
- **Agent:** Double DQN
- **Training episodes:** 2,300
- **Training seed:** 42
- **Environment:** 10 asynchronous vectorized environments
- **Evaluation:** Greedy policy (`epsilon = 0.0`)

No additional optimizer smoothing was introduced, keeping the experiment focused on the activation-function change.

## Training Summary

| Metric | Result |
|---|---:|
| Completed runs | 1 |
| Training seed | 42 |
| Final reward (last window) | 708.7 |
| Best episode reward | 914.1 |
| Worst episode reward | -163.0 |
| Mean episode length | 250.6 |
| Environment steps | 577,222 |
| Training time | 8,106 seconds |

The learning curve continued trending upward toward the end of training, unlike earlier experiments where performance tended to plateau or decline.

## Checkpoint Evaluation

A checkpoint sweep was performed using 10 fixed evaluation seeds (10000–10009), with 10 episodes per checkpoint.

| Episode | Mean reward | Std | Minimum | Maximum | Stuck |
|---:|---:|---:|---:|---:|---:|
| 2000 | 561.9 | 166.4 | 398.3 | 849.8 | 0/10 |
| 2050 | 635.8 | 212.0 | 360.6 | 896.8 | 0/10 |
| 2100 | 532.8 | 50.2 | 432.1 | 598.8 | 0/10 |
| 2150 | 587.3 | 65.2 | 513.6 | 702.9 | 0/10 |
| 2200 | 504.8 | 349.8 | -93.5 | 872.7 | 2/10 |
| 2250 | 762.1 | 154.7 | 373.3 | 913.7 | 0/10 |
| **2300** | **758.2** | **97.0** | **573.0** | **884.6** | **0/10** |

Episode 2300 was selected as the best checkpoint, with the highest checkpoint score (76.13).

### Expanded Evaluation — Episode 2300

The selected checkpoint was evaluated across 50 fixed seeds (10000–10049).

| Metric | Result |
|---|---:|
| Evaluation episodes | 50 |
| Mean reward | **809.08** |
| Standard deviation | **89.98** |
| Minimum reward | 524.16 |
| Maximum reward | 917.30 |
| Stuck episodes | 0/50 |

All 50 evaluation episodes completed without being classified as stuck. The results indicate strong performance across a broader set of environment seeds.

## Observed Driving Behavior

During human-rendered evaluation, the agent displayed more purposeful driving behavior compared with earlier models:

- Used braking more purposefully to control speed.
- Demonstrated more stable car handling, particularly when navigating corners.
- Showed less reliance on repeatedly applying the accelerator.

These observations suggest that the agent learned more controlled and consistent driving behavior rather than relying primarily on repetitive acceleration. They are qualitative observations, not separately measured metrics.

## Findings

1. The Leaky ReLU experiment achieved the strongest overall evaluation result among the models tested so far.
2. The selected checkpoint achieved a mean reward of 809.08 with a standard deviation of 89.98 across 50 evaluation seeds.
3. The checkpoint had no stuck episodes in the expanded evaluation.
4. The training curve was still trending upward at episode 2300, suggesting that performance had not clearly plateaued.
5. Human-rendered evaluation showed more purposeful braking and more stable car handling.
6. The checkpoint performed favorably compared with the earlier Double DQN model, which achieved 777.71 ± 187.61 across 40 evaluation seeds. The evaluation seed sets differed, so this is not a perfectly controlled comparison.

## Conclusion

The results strongly support the hypothesis that **Leaky ReLU is a better activation function for this particular Double DQN agent and CarRacing use case**.

The improvement is supported by the expanded evaluation, the upward training trend, and qualitative observations of driving behavior. The experiment does not establish that Leaky ReLU is universally better than ReLU, but it provides sufficient practical evidence to carry Leaky ReLU forward as the preferred activation for the current project.

**Selected checkpoint:** `dd_leaky_relu_checkpoint_epi2300_seed42.pth`

**Decision:** Keep this checkpoint as the current best reference model and use Leaky ReLU in subsequent experiments.