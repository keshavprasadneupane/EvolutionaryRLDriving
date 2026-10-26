# EXP-XXX: <Experiment title>

Complete this plan before starting the experiment. Once the experiment begins, keep the original plan intact. If the setup changes, add a dated note explaining the change.

- **Start date:**
- **Status:** Planned / Running / Completed / Abandoned
- **Phase:** DQN / Genetic DQN / TD3 / Genetic TD3 / Custom environment

## 1. Research question and hypothesis

What question is this experiment intended to answer? What result do you expect, and why?

## 2. Experimental setup

- **Environment:**
- **Algorithm or method:**
- **Change from baseline:** Ideally, change one major factor at a time.
- **Fixed settings:** Network architecture, replay capacity, batch size, discount factor, optimizer, and other relevant settings.
- **Configuration file:** `configs/...`
- **Training seeds:** Prefer 3–5 seeds when practical.
- **Evaluation procedure:** Number of episodes, fixed evaluation seeds, and exploration settings.
- **Training budget:** Environment steps per run.

## 3. Success criteria

Choose the metric that will determine whether the hypothesis is supported. Specify the target or minimum improvement before running the experiment.

## 4. Runs

| Run folder | Seed | Notes |
|---|---:|---|
| | | |

## 5. Results

Generate the summary from the recorded run data:

```bash
uv run python -m evaluation.aggregate <experiment-name>
```

- **Summary table:** `results/<experiment-name>/summary.md`
- **Learning curve:** `results/<experiment-name>/learning_curve.png`

Summarize the results here only after the aggregation step has completed.

## 6. Conclusion

Was the hypothesis supported, contradicted, or inconclusive? Discuss variation between seeds, practical significance, unexpected results, and limitations.

## 7. Follow-up experiments

List the next questions raised by these results.
