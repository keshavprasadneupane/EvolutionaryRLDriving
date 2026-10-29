### EXP-001 Follow-up: Greedy Evaluation & Late-Training Policy Collapse

- **Date:** 2026-10-29 | **Phase:** DQN | **Seed:** 42 | **Status:** Evaluation complete; collapse under investigation

---

### 1. Key Findings & Checkpoint Sweep

- **Training vs. evaluation:** Training rewards reached 688.9, but the final model (episode 2300) scored **-93.45 ± 0.59** with greedy evaluation (`ε = 0`), exhibiting stationary behavior.
- **Checkpoint sweep:** Evaluated checkpoints on fixed seeds 10000–10004.
  - **Best checkpoint:** Episode 2200 — **712.8 ± 99.0** mean reward.
  - **Policy collapse:** Performance declined at episode 2250 and collapsed by episode 2300 (5/5 episodes stuck).
- **Exploration test:** At `ε = 0.05`, the episode-2300 checkpoint scored **107.92 ± 147.80** across 20 seeds. Exploration improved performance but results remained inconsistent (3/20 episodes scored above 300).
- **Configuration:** `double_dqn` was disabled in the original run.

---

### 2. Core Takeaways

- The `_final` model represents the last training state, not the best checkpoint. Episode 2200 remains the best evaluated checkpoint from this run.
- High training rewards do not guarantee a functional greedy policy.
- The effect of disabling Double DQN on the collapse remains unconfirmed.

---

### 3. Action Items & Next Steps

**Diagnostics — no new training**
1. Inspect `metrics.csv` for episodes 2100–2323, grouping reward, loss, and epsilon into 10-episode bins.
2. Compare episodes 2200 and 2300 at `ε = 0` and `ε = 0.05` using the same seeds.
3. Evaluate episode 2200 on additional seeds and inspect Q-value targets.
4. Fix the invalid `render_mode='none'` warning.

**EXP-001b — Controlled Baseline Comparison**
- Run three training seeds (42, 43, 44) with **Double DQN enabled**, retaining the original run and new artifacts separately.
- Evaluate greedily every 50 episodes on fixed seeds 10000–10004.
- Save the best checkpoint based on independent evaluation performance.
- Base checkpoint selection and stopping decisions on evaluation rewards, not training rewards.
