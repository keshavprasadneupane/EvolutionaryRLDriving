### EXP-001 Follow-up 2: Epsilon Dependence and Checkpoint Re-evaluation

- **Date:** 2026-10-29 | **Phase:** DQN | **Seed:** 42 | **Status:** Evaluation complete

---

### 1. Key Findings

- **Episode-2300 checkpoint:** Evaluated on eight seeds, greedy performance was **-93.49 ± 0.52**. With `ε = 0.05`, the same seeds produced **38.54 ± 99.94**. Exploration improved results but did not produce reliable driving.
- **Episode-2200 checkpoint:** Greedy evaluation across eight seeds achieved **671.51 ± 97.28**. With `ε = 0.05`, it scored **538.89 ± 81.92**.
- **Policy comparison:** Debugging episode 2200 showed the agent selected `gas` in 71 of 250 decisions on seed 10000. The episode-2300 policy's earlier diagnostic showed it never selected `gas`, supporting a late-training change in action selection.
- **Conclusion:** The episode-2200 checkpoint remains the strongest candidate. The results suggest that epsilon noise partially compensates for the episode-2300 policy collapse, but does not explain its underlying cause.

---

### 2. Core Takeaways

- The collapse is specific to the later checkpoint; episode 2200 still performs well under greedy evaluation.
- Exploration is not a substitute for a stable learned policy: epsilon 0.05 reduced episode-2200's mean reward while improving episode-2300's.
- `double_dqn` was disabled in the original run. Its effect remains untested.

---

### 3. Action Items & Next Steps

1. Inspect training metrics around episodes 2100–2300, especially loss, reward, epsilon, and Q-values.
2. Retain episode 2200 as the best evaluated checkpoint from the original run.
3. Run EXP-001b with **Double DQN enabled**, using separate run directories for seeds 42, 43, and 44.
4. Evaluate greedily every 50 episodes on fixed seeds 10000–10004, saving the best checkpoint based on independent evaluation rewards.
