# EXP-001: DQN Baseline and Optimization on CarRacing

- **Date:** 2026-10-28
- **Status:** Interrupted
- **Phase:** DQN

## 1. Objective

Evaluate how grayscale conversion, frame stacking, and asynchronous vectorization affect DQN training throughput and sample efficiency.

**Hypothesis:** These optimizations improve training efficiency and produce stable evaluation rewards above 300.

## 2. Setup

- **Environment:** Gymnasium CarRacing (`continuous=False`).
- **Algorithm:** DQN with experience replay and hard target-network updates.
- **Optimization phases:**
  1. RGB baseline with a single environment.
  2. Grayscale observations with frame stacking.
  3. Ten asynchronous environments with `TRAIN_EVERY=3`.
- **Expected episode runtime:** Approximately 15 seconds for the RGB baseline, 7 seconds for grayscale with frame stacking, and 3 to 4 seconds for asynchronous vectorization.
- **Fixed settings:** Learning rate `0.00025`, batch size `32`, replay capacity `50000`, discount factor `0.99`, hidden dimension `224`, and epsilon decay `0.998`.
- **Configuration:** `configs/dqn_carracing.json`
- **Seed:** `42`
- **Evaluation:** Independent evaluation with exploration disabled. The original plan specified 10 episodes; the follow-up plan is to evaluate on five fixed seeds.
- **Training budget:** Reduced from 10,000 to 2,300 episodes based on encouraging early training results, not verified evaluation performance.

## 3. Success Criteria

- Mean evaluation reward consistently exceeds 300.
- Optimized episode runtime remains below 5 seconds.
- Evaluation uses fixed seeds with exploration disabled.

## 4. Runs and Artifact Limitations

All three optimization phases reused the same run directory:

`runs/dqn_carracing_baseline/2026-10-28_162515_seed42`

The artifacts from earlier phases were overwritten by subsequent runs. Consequently, the original RGB baseline and grayscale results are unavailable for independent analysis. Only the surviving run's artifacts can be used to report results.

| Phase | Seed | Artifact status |
|---|---:|---|
| RGB baseline | 42 | Overwritten; results unavailable. |
| Grayscale with frame stacking | 42 | Overwritten; results unavailable. |
| Asynchronous vectorization | 42 | Surviving artifacts; run interrupted. |

The surviving run's directory identifies the experiment and seed, but the overwritten artifacts prevent a reliable comparison of the three phases.

## 5. Results

Generated the summary using:

```bash
uv run python -m evaluation.aggregate dqn_carracing_baseline
```

| Metric | Result |
|---|---:|
| Preserved runs | 1 |
| Final 20-episode moving average | 589.8 |
| Best episode reward | 907.5 |
| Worst episode reward | -158.8 |
| Mean episode length | 250.8 steps |
| Environment steps | 361,228 |
| Training time | 5,050 seconds |
| Independent evaluation reward | Unavailable |

**Artifacts:** `results/dqn_carracing_baseline/interrupted/`

The surviving run was interrupted before the planned training budget was completed. The training rewards are promising, but the success criteria remain unverified because independent evaluation results are unavailable.

## 6. Conclusion

The surviving run achieved a best episode reward of 907.5 and a final 20-episode moving average of 589.8 after approximately 84 minutes of training. These results indicate promising learning progress, but consistency and generalization remain uncertain.

The hypothesis cannot be evaluated conclusively. The earlier phases' artifacts were overwritten, preventing a controlled comparison, and independent evaluation has not yet been completed.

## 7. Follow-up

- Inspect the surviving checkpoint and metadata before deciding whether to resume or restart.
- Evaluate the checkpoint on five fixed seeds with exploration disabled.
- Record the evaluation mean and standard deviation.
- Use a unique run directory for every new run to prevent accidental overwrites.
- Repeat the RGB and grayscale phases if a controlled comparison is required.
- Implement rendered trajectory evaluation in `test.py`.
- Generate and review the final summaries and learning curves.
- Commit the finalized experiment documentation and appropriate artifacts.