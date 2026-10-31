# EXP-001.c — CarRacing Baseline Recovery

**Date:** 2026-10-29  
**Project:** EvolutionaryRLDriving  
**Environment:** Gymnasium CarRacing  
**Status:** Completed

## Objective

Resolve a late-training failure in the CarRacing DQN baseline and establish a functional configuration for subsequent model-improvement experiments.

## 1. Problem: Policy Degeneration

**Original run:** `dqn_carracing_baseline`  
**Seed:** 42 · **Learning rate:** 0.00025 · **Double DQN:** Disabled

The original agent learned to drive but its policy deteriorated late in training. Greedy evaluation across eight seeds showed a sharp decline between episodes 2200 and 2300.

| Metric | Episode 2200 | Episode 2300 |
|---|---:|---:|
| Mean reward | 671.51 | -93.49 |
| Standard deviation | 97.28 | 0.52 |
| Gas action | Present | Absent |

Action inspection at episode 2300 confirmed that the policy predominantly selected `right` or `noop`, with no gas action across the five inspected evaluation episodes.

An additional evaluation with epsilon set to 0.05 produced mean rewards of 538.89 ± 81.92 at episode 2200 and 38.54 ± 99.94 at episode 2300. Exploration did not restore the original driving performance.

## 2. Solution: Enable Double DQN

Double DQN was enabled to address the observed failure. Two runs were conducted.

### Run A — Lower learning rate

- **Run:** `double_dqn`
- **Learning rate:** 0.0001
- **Double DQN:** Enabled
- **Evaluated through:** Episode 2050

At episode 2050, evaluation on seeds 10000–10004 produced a mean reward of 543.96 ± 79.96. Gas actions remained present, but the run did not reach episode 2300.

### Run B — Original learning rate

- **Run:** `double_dqn_no_low_lr`
- **Learning rate:** 0.00025
- **Double DQN:** Enabled
- **Training horizon:** 2300 episodes

This run started fresh after the previous artifacts were deleted. It completed 576,975 environment steps in 7,954 seconds, with a final-window training reward of 630.0 and a best episode reward of 917.3.

At episode 2300, evaluation on seeds 10000–10004 produced:

| Metric | Original DQN | Double DQN enabled |
|---|---:|---:|
| Learning rate | 0.00025 | 0.00025 |
| Training horizon | 2300 episodes | 2300 episodes |
| Mean evaluation reward | -93.45 | 431.53 |
| Standard deviation | 0.59 | 70.40 |
| Minimum reward | -93.94 | 300.00 |
| Maximum reward | -92.28 | 506.18 |
| Gas action | Absent | Present in all 5 episodes |

The Double DQN rewards were 428.85, 300.00, 453.46, 506.18, and 469.18. Its mean was 524.98 points higher than the original DQN's result, and its action logs showed continued acceleration and steering.

## 3. Outcome

Enabling Double DQN allowed the agent to retain functional driving at episode 2300 under the same learning rate and training horizon where the original DQN had degenerated.

The result is consistent with the possibility that value-estimation instability contributed to the original failure, but the underlying cause was not directly verified. Further investigation into that mechanism is not necessary for the immediate goal of improving the CarRacing model.

## 4. Decision

- Keep the original DQN run as a record of the failure.
- Adopt the Double DQN-enabled configuration with learning rate 0.00025 as the current baseline.
- Consider the baseline-recovery experiment complete.
- Move on to experiments focused on improving driving performance.

## Run References

- Original DQN: `runs/dqn_carracing_baseline/2026-10-28_162515_seed42`
- Double DQN, learning rate 0.0001: `double_dqn` run artifacts
- Double DQN, learning rate 0.00025: `runs/double_dqn_no_low_lr/2026-10-29_123227_seed42`

**Conclusion:** A single algorithm configuration change—enabling Double DQN—was sufficient to retain functional driving at the original training horizon. The modified baseline is ready for the next stage of CarRacing experiments.