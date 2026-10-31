# Experiment 2.a — Double DQN Target Network Update Frequency and Tau

**Date:** 2026-10-30  
**Experiment:** `dd_update_req_tau`

## Objective

Evaluate whether more frequent but softer target-network updates can maintain competitive Double DQN performance compared with the previous successful configuration.

## Configuration

| Parameter | Value |
|---|---:|
| Episodes | 2,300 |
| Target update frequency | 250 |
| Input channels | 4 |
| Hidden dimension | 224 |
| Learning rate | 0.00015 |
| Discount factor | 0.99 |
| Tau | 0.70 |
| Initial / Minimum epsilon | 1.0 / 0.05 |
| Epsilon decay | 0.998 |
| Replay capacity | 50,000 |
| Replay batch size | 32 |
| Replay warmup | 100 |
| Device | CUDA |
| Training environments | 10 (AsyncVectorEnv) |
| Gradient update interval | 3 environment steps |

The hidden dimension remained unchanged from previous experiments. Replay was enabled throughout training.

## Training Run Summary

*One training run, seed 42.*

| Metric | Result |
|---|---:|
| Final reward (last window) | 529.0 |
| Best episode reward | 776.9 |
| Worst episode reward | -148.6 |
| Mean episode length | 250.9 |
| Environment steps | 579,777 |
| Training time | 8,197 s |

## Checkpoint Sweep

Ten checkpoints were evaluated using five fixed seeds (`10000–10004`).

| Episode | Score | Mean | Std. dev. | Min | Max |
|---:|---:|---:|---:|---:|---:|
| 1900 | 68.98 | 608.1 | 71.2 | 479.2 | 698.7 |
| 1950 | 54.57 | 466.2 | 191.9 | 310.3 | 834.4 |
| 2000 | 62.87 | 488.8 | 74.0 | 421.2 | 629.7 |
| 2050 | 70.56 | 602.5 | 72.5 | 550.9 | 745.6 |
| 2100 | 56.37 | 394.9 | 75.6 | 308.8 | 529.3 |
| 2150 | 63.48 | 487.4 | 64.3 | 433.3 | 606.6 |
| 2200 | 53.94 | 562.4 | 209.8 | 173.6 | 807.3 |
| 2202 | 67.28 | 551.1 | 56.2 | 479.2 | 614.7 |
| 2250 | 41.15 | 277.7 | 159.5 | 25.8 | 456.0 |
| 2300 | 57.85 | 488.9 | 114.5 | 286.8 | 641.3 |

## Expanded Evaluation

Episodes 1900 and 2202 were evaluated on the same 40 seeds (`10000–10039`).

| Metric | Episode 1900 | Episode 2202 |
|---|---:|---:|
| Mean reward | **708.30** | 604.94 |
| Standard deviation | **85.72** | 136.08 |
| Minimum reward | 479.15 | 243.43 |
| Maximum reward | 877.58 | 895.52 |

Both checkpoints completed all 40 episodes at the 250-step duration.

## Findings and Conclusion

The modified target-update configuration maintained performance comparable to the previous successful Double DQN experiment, demonstrating that more frequent updates with `tau=0.70` are a viable alternative.

An accidental training interruption caused the accumulated replay buffer to be lost. Training resumed from the saved model checkpoint with a fresh replay buffer, disrupting training continuity and contributing to the observed checkpoint fluctuations.

The expanded evaluation identified episode 1900 as the strongest candidate, achieving a mean reward of **708.30 ± 85.72** across 40 seeds.

**Selected checkpoint: Episode 1900**

## Training Infrastructure

Added interrupt-based replay persistence to `train.py`.

- Regular checkpoints exclude replay data to remain lightweight.
- On `Ctrl+C`, the handler saves the agent checkpoint with `include_replay=True`.
- This preserves accumulated experience when training is interrupted, allowing recovery without rebuilding the replay buffer.