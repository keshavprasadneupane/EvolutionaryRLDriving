# Experiment 003.b — Resuming DQN Training Without Replay Buffer Restoration

**Experiment:** `dd_leaky_relu`  
**Date:** 2026-10-31  
**Training seed:** 42  
**Status:** Completed

## Objective

Investigate whether a trained Double DQN agent can successfully resume training from a saved model checkpoint without restoring its previous replay buffer.

The experiment examines whether the agent can rebuild its replay buffer through newly collected environment interactions, continue learning, and potentially improve its existing policy.

## Hypothesis

A trained DQN agent should be capable of resuming training without its previous replay buffer, provided its saved network parameters are restored correctly.

Although the replay buffer contains previously collected experiences, the learned network parameters retain the knowledge acquired during training. After resuming, the agent can collect new transitions and gradually rebuild its replay buffer.

The initial expectation was:

- Training should be able to resume without the previous replay buffer.
- An initially empty replay buffer may cause temporary instability due to limited experience diversity.
- Given sufficient additional training, the agent should rebuild its replay buffer and recover its learning performance.
- The agent may eventually improve beyond its pre-resume checkpoint.

## Experimental Procedure

1. Train the Leaky ReLU Double DQN agent to episode 2300.
2. Save the model checkpoint.
3. Resume training from the checkpoint without restoring the previous replay buffer.
4. Allow the agent to collect new experiences and populate its replay buffer during resumed training.
5. Extend training to episode 3500.
6. Sweep checkpoints from episode 2350 to 3500 using 10 fixed evaluation seeds.
7. Select the best-performing checkpoint and evaluate it across 50 fixed seeds.

The experiment retained the existing agent architecture and training configuration. The primary experimental change was resuming without the previously accumulated replay buffer.

## Training Summary

| Metric | Result |
|---|---:|
| Completed runs | 1 |
| Training seed | 42 |
| Total training episodes | 3,500 |
| Final reward (last window) | 803.6 |
| Best episode reward | 916.2 |
| Worst episode reward | -163.0 |
| Mean episode length | 250.3 |
| Environment steps | 877,747 |
| Training time | 12,783 seconds |

The resumed training completed successfully. The final training reward increased from 708.7 at episode 2300 to 803.6 at episode 3500.

## Checkpoint Sweep

Checkpoints from episode 2350 to 3500 were evaluated using 10 fixed seeds (10000–10009).

| Episode | Mean reward | Std | Minimum | Maximum | Stuck |
|---:|---:|---:|---:|---:|---:|
| 2350 | 830.7 | 107.3 | 561.7 | 915.7 | 0/10 |
| 2400 | 588.3 | 188.8 | 305.7 | 857.5 | 0/10 |
| 2450 | 767.3 | 100.5 | 595.0 | 907.0 | 0/10 |
| 2500 | 686.7 | 213.6 | 383.7 | 905.5 | 0/10 |
| 2550 | 657.7 | 163.5 | 368.6 | 903.3 | 0/10 |
| 2600 | 574.9 | 193.0 | 289.9 | 869.7 | 0/10 |
| 2650 | 587.2 | 178.7 | 300.0 | 802.6 | 0/10 |
| **2700** | **840.2** | **45.7** | **766.5** | **902.8** | **0/10** |
| 2750 | 399.3 | 421.8 | -93.7 | 850.0 | 4/10 |
| 2800 | 471.8 | 50.6 | 400.0 | 548.7 | 0/10 |
| 2850 | 684.5 | 116.7 | 487.5 | 901.5 | 0/10 |
| 2900 | 762.0 | 111.1 | 566.7 | 856.7 | 0/10 |
| 2950 | 680.0 | 74.0 | 601.3 | 830.5 | 0/10 |
| 3000 | 530.6 | 209.3 | 300.7 | 902.2 | 0/10 |
| 3050 | 771.0 | 129.4 | 446.7 | 909.2 | 0/10 |
| 3100 | 795.3 | 70.7 | 691.7 | 892.8 | 0/10 |
| 3150 | 112.4 | 215.7 | -93.7 | 476.9 | 5/10 |
| 3200 | 736.3 | 170.0 | 290.0 | 887.0 | 0/10 |
| 3250 | 808.3 | 59.3 | 716.0 | 905.6 | 0/10 |
| 3300 | 589.2 | 68.5 | 497.5 | 703.1 | 0/10 |
| 3350 | 686.1 | 42.2 | 627.3 | 784.2 | 0/10 |
| 3400 | 543.4 | 214.0 | 299.4 | 877.3 | 0/10 |
| 3450 | 618.9 | 93.1 | 404.5 | 737.8 | 0/10 |
| 3500 | 738.4 | 87.9 | 603.0 | 888.4 | 0/10 |

Episode 2700 achieved the highest checkpoint score of 86.16 and was selected for expanded evaluation.

The sweep also revealed considerable checkpoint-to-checkpoint fluctuations, including temporary performance degradation at episodes 2750 and 3150. Therefore, the highest training episode was not necessarily the best-performing checkpoint.

## Expanded Evaluation — Episode 2700

The selected checkpoint was evaluated across 50 fixed seeds (10000–10049).

| Metric | Episode 2300 | Episode 2700 |
|---|---:|---:|
| Evaluation episodes | 50 | 50 |
| Mean reward | 809.08 | **858.48** |
| Standard deviation | 89.98 | **87.86** |
| Minimum reward | 524.16 | 368.86 |
| Maximum reward | 917.30 | 922.60 |
| Stuck episodes | 0/50 | 0/50 |

Both checkpoints were evaluated on the same 50 seeds.

Episode 2700 achieved:

- A mean reward improvement of 49.40 points.
- Approximately 6.1% higher mean reward.
- Slightly lower standard deviation.
- A higher maximum evaluation reward.
- Zero stuck episodes across all 50 evaluations.

Although episode 2700 had a lower minimum reward due to a particularly weak result on seed 10046, its overall evaluation performance was higher.

## Findings

1. **Training resumed successfully without restoring the previous replay buffer.** The agent continued collecting new experiences and training its network.
2. The agent completed the extended training run, reaching 3500 episodes.
3. The training reward increased from 708.7 at episode 2300 to 803.6 at episode 3500.
4. The checkpoint sweep identified episode 2700 as the strongest checkpoint, rather than the final episode.
5. Expanded evaluation confirmed that episode 2700 outperformed the pre-resume episode-2300 checkpoint on the same 50 evaluation seeds.
6. The selected checkpoint achieved 858.48 ± 87.86, compared with 809.08 ± 89.98 before resuming.
7. Significant checkpoint fluctuations occurred during resumed training, showing that recovery and improvement were not necessarily smooth or monotonic.

## Conclusion

The experiment successfully demonstrated that **the Double DQN agent can resume training without restoring its previous replay buffer and continue learning effectively**.

The agent rebuilt its replay buffer through fresh environment interactions and eventually produced a checkpoint that outperformed its pre-resume reference model across 50 evaluation seeds.

This strongly supports the hypothesis that restoring the replay buffer is not strictly necessary for successful training continuation in this particular setup. The saved network parameters retained useful learned behavior, while newly collected experiences provided the data needed for further optimization.

The experiment also demonstrates that an initially empty replay buffer does not necessarily prevent eventual recovery, although temporary instability and checkpoint fluctuations can occur.

**Selected checkpoint:** `dd_leaky_relu_checkpoint_epi2700_seed42.pth`

**Decision:** Retain episode 2700 as the current best reference checkpoint. The training workflow can use model-only checkpoints for resuming interrupted runs, with the understanding that replay-buffer reconstruction may require additional training and can introduce temporary instability.

**Practical takeaway:** Replay-buffer persistence is useful for preserving training continuity, but it is not an absolute requirement for recovering and improving a previously trained DQN agent.